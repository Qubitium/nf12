# SPDX-License-Identifier: Apache-2.0
"""Shared weight loading, GPU timing and result output.

Graph timing divides replay time by launch count. Cold runs rotate at least
max(3x L2 + one copy, 2 GiB), with a separate escape table per weight copy.
Event timing brackets each launch and uses a read-only 2.5x-L2 flush in cold
mode to avoid write-back traffic.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterator

import torch

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"


def import_nf12():
    try:
        import nf12
    except ImportError:
        sys.path.insert(0, str(BENCH_DIR.parent / "python"))
        import nf12
    return nf12

M_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384]


@dataclasses.dataclass(frozen=True)
class Shape:
    k: int
    n: int
    # (HF repo, tensor names concatenated along N) for a real layer; None -> random.
    source: tuple[str, tuple[str, ...]] | None
    note: str
    tag: str | None = None  # distinguishes rows that share K x N with a grid shape

    @property
    def key(self) -> str:
        return self.tag or f"{self.k}x{self.n}"


SHAPES = [
    Shape(4096, 4096, ("Qwen/Qwen3-8B", ("model.layers.18.self_attn.q_proj.weight",)), "Qwen3-8B q_proj"),
    Shape(4096, 14336, ("unsloth/Meta-Llama-3.1-8B-Instruct", ("model.layers.16.mlp.gate_proj.weight",)),
          "Llama-3.1-8B gate_proj"),
    Shape(14336, 4096, ("unsloth/Meta-Llama-3.1-8B-Instruct", ("model.layers.16.mlp.down_proj.weight",)),
          "Llama-3.1-8B down_proj"),
    Shape(4096, 28672,
          ("unsloth/Meta-Llama-3.1-8B-Instruct",
           ("model.layers.16.mlp.gate_proj.weight", "model.layers.16.mlp.up_proj.weight")),
          "Llama-3.1-8B gate_up (fused)"),
    Shape(5120, 25600, ("Qwen/Qwen3-32B", ("model.layers.32.mlp.gate_proj.weight",)), "Qwen3-32B gate_proj"),
    Shape(25600, 5120, ("Qwen/Qwen3-32B", ("model.layers.32.mlp.down_proj.weight",)), "Qwen3-32B down_proj"),
    Shape(8192, 8192, None, "70B-class o_proj"),
    Shape(8192, 57344, None, "70B-class gate_up (fused)"),
    Shape(2048, 8192, ("unsloth/Llama-3.2-1B", ("model.layers.8.mlp.gate_proj.weight",)), "Llama-3.2-1B gate_proj"),
    Shape(896, 4864, ("Qwen/Qwen2.5-0.5B-Instruct", ("model.layers.12.mlp.gate_proj.weight",)),
          "Qwen2.5-0.5B gate_proj"),
]

# Benchmarked by the harnesses but not by the SOL/cuBLAS sweep (same K x N as a grid shape).
EXTRA_SHAPES = [
    # Heavy-escape checkpoint: about 10% of groups.
    Shape(4096, 4096, ("unsloth/mistral-7b-instruct-v0.3", ("model.layers.0.self_attn.q_proj.weight",)),
          "Mistral-7B-v0.3 L0 q_proj (heavy escape)", tag="4096x4096-mistral-l0q"),
]

# Router gates and lm_heads: not in the default grid, selected by key (--shapes).
MODEL_SHAPES = [
    Shape(2048, 128, ("Qwen/Qwen3-30B-A3B", ("model.layers.24.mlp.gate.weight",)), "Qwen3-30B-A3B router gate"),
    Shape(2048, 64, ("deepseek-ai/DeepSeek-V2-Lite", ("model.layers.13.mlp.gate.weight",)),
          "DeepSeek-V2-Lite router gate"),
    Shape(4096, 128, ("Qwen/Qwen3-235B-A22B", ("model.layers.47.mlp.gate.weight",)), "Qwen3-235B-A22B router gate"),
    Shape(4096, 151936, ("Qwen/Qwen3-8B", ("lm_head.weight",)), "Qwen3-8B lm_head"),
    Shape(5120, 151936, ("Qwen/Qwen3-32B", ("lm_head.weight",)), "Qwen3-32B lm_head"),
    Shape(4096, 128256, ("unsloth/Meta-Llama-3.1-8B-Instruct", ("lm_head.weight",)), "Llama-3.1-8B lm_head"),
]


def shape_by_key(key: str) -> Shape:
    for s in SHAPES + EXTRA_SHAPES + MODEL_SHAPES:
        if s.key == key:
            return s
    k, n = (int(v) for v in key.split("x"))
    return Shape(k, n, None, "custom")


def entry_label(entry: dict) -> str:
    """Row label of a result entry (older results carry no "key")."""
    return entry.get("key") or f"{entry['k']}x{entry['n']}"


def _cached_file(repo: str, filename: str) -> Path | None:
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(repo, filename)
    return Path(p) if isinstance(p, str) and os.path.isfile(p) else None


# Range-downloaded tensors avoid fetching whole checkpoint shards.
TENSOR_CACHE = Path(os.environ.get("NF12_BENCH_TENSOR_CACHE", Path.home() / ".cache" / "nf12-bench"))

_SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def _shard_of(repo: str, name: str, remote: bool) -> str | None:
    index = _cached_file(repo, "model.safetensors.index.json")
    if index is None and remote:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            index = Path(hf_hub_download(repo, "model.safetensors.index.json"))
        except EntryNotFoundError:
            index = None
    if index is None:
        return "model.safetensors"
    return json.loads(index.read_text())["weight_map"].get(name)


def fetch_hf_tensor(repo: str, name: str) -> Path:
    """Downloads one tensor of a hub checkpoint with HTTP range reads into TENSOR_CACHE."""
    import struct

    from huggingface_hub import HfFileSystem
    from safetensors.torch import save_file

    out = TENSOR_CACHE / repo.replace("/", "--") / f"{name}.safetensors"
    shard = _shard_of(repo, name, remote=True)
    if shard is None:
        raise KeyError(f"{repo} has no tensor {name}")
    with HfFileSystem().open(f"{repo}/{shard}", "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(header_len))[name]
        start, end = meta["data_offsets"]
        f.seek(8 + header_len + start)
        data = bytearray(f.read(end - start))
    t = torch.frombuffer(data, dtype=_SAFETENSORS_DTYPES[meta["dtype"]]).reshape(meta["shape"])
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: t}, str(out))
    return out


def load_hf_tensor(repo: str, name: str, fetch: bool) -> torch.Tensor | None:
    """Load one tensor from the local HF cache (full shard) or TENSOR_CACHE.
    With ``fetch``, range-download missing tensors. Return None if unavailable."""
    from safetensors import safe_open

    shard = _shard_of(repo, name, remote=False)
    path = _cached_file(repo, shard) if shard else None
    if path is None:
        path = TENSOR_CACHE / repo.replace("/", "--") / f"{name}.safetensors"
        if not path.is_file():
            if not fetch:
                return None
            try:
                path = fetch_hf_tensor(repo, name)
            except Exception as e:  # noqa: BLE001 - any hub failure means "use random weights"
                print(f"nf12-bench: cannot fetch {repo}:{name} ({type(e).__name__}); using random weights",
                      file=sys.stderr)
                return None
    with safe_open(str(path), framework="pt") as f:
        return f.get_tensor(name) if name in f.keys() else None


def load_hf_tensors(repo: str, names: tuple[str, ...], fetch: bool = False) -> torch.Tensor | None:
    """Concatenate the named tensors along dim 0, or return None if any is unavailable."""
    parts = []
    for name in names:
        t = load_hf_tensor(repo, name, fetch)
        if t is None:
            return None
        parts.append(t)
    return torch.cat(parts, dim=0)


def make_weight(shape: Shape, mode: str, device: torch.device, seed: int = 0,
                fetch: bool = False) -> tuple[torch.Tensor, str]:
    """Returns (W [N, K] bf16 on device, description of its origin)."""
    if mode == "real" and shape.source is not None:
        w = load_hf_tensors(*shape.source, fetch=fetch)
        if w is not None:
            if tuple(w.shape) != (shape.n, shape.k):
                raise ValueError(f"{shape.source} has shape {tuple(w.shape)}, expected {(shape.n, shape.k)}")
            origin = f"{shape.source[0]}: {', '.join(shape.source[1])}"
            return w.to(device=device, dtype=torch.bfloat16).contiguous(), origin
    g = torch.Generator(device=device).manual_seed(seed)
    w = torch.randn(shape.n, shape.k, generator=g, device=device, dtype=torch.float32).mul_(0.02)
    return w.to(torch.bfloat16), "random normal(0, 0.02)"


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def is_tegra() -> bool:
    return Path("/sys/class/devfreq/gpu-gpc-0").exists() or Path("/etc/nv_tegra_release").exists()


_SMI_FIELDS = [
    "clocks.sm", "clocks.mem", "clocks.max.sm", "clocks.max.mem", "temperature.gpu", "power.draw", "pstate",
    "clocks_event_reasons.active", "clocks_event_reasons.sw_power_cap", "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown", "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
]


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def gpu_state(device: int | None = None) -> dict:
    """Read GPU clocks and throttle state via nvidia-smi or Tegra sysfs."""
    device = torch.cuda.current_device() if device is None else device
    if is_tegra():
        state = {}
        for name in ("gpu-gpc-0", "gpu-nvd-0", "bwmgr"):
            d = Path("/sys/class/devfreq") / name
            state[name] = {f: _read(d / f) for f in ("cur_freq", "max_freq", "governor")}
        state["nvpmodel"] = _run(["nvpmodel", "-q"])
        return state
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    uuid = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
    out = _run(["nvidia-smi", "-i", uuid, f"--query-gpu={','.join(_SMI_FIELDS)}", "--format=csv,noheader,nounits"])
    if out is None:
        return {"error": "nvidia-smi unavailable"}
    return dict(zip(_SMI_FIELDS, (v.strip() for v in out.split(","))))


def throttled(state: dict) -> list[str]:
    """Names of active throttle reasons that would slow a benchmark."""
    return [f.split(".")[-1] for f in _SMI_FIELDS[8:] if state.get(f) == "Active"]


def state_summary(state: dict | None) -> str:
    """One-line clock/power description of a gpu_state() snapshot for reports."""
    if not state:
        return "n/a"
    if "gpu-gpc-0" in state:
        def mhz(d: dict, f: str) -> str:
            return str(int(d[f]) // 1_000_000) if d.get(f) else "?"

        gpc, emc = state["gpu-gpc-0"], state["bwmgr"]
        mode = state["nvpmodel"].splitlines()[0] if state.get("nvpmodel") else "nvpmodel n/a"
        return (f"GPC {mhz(gpc, 'cur_freq')}/{mhz(gpc, 'max_freq')} MHz, EMC {mhz(emc, 'cur_freq')}/"
                f"{mhz(emc, 'max_freq')} MHz, {mode}")
    if "clocks.sm" not in state:
        return state.get("error", "n/a")
    return (f"SM {state['clocks.sm']}/{state['clocks.max.sm']} MHz, mem {state['clocks.mem']} MHz, "
            f"{state['power.draw']} W, {state['temperature.gpu']} C, throttle: {', '.join(throttled(state)) or 'none'}")


def machine_info() -> dict:
    dev = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(dev)
    return {
        "hostname": socket.gethostname(),
        "arch": platform.machine(),
        "gpu": p.name,
        "sm": p.major * 10 + p.minor,
        "num_sms": p.multi_processor_count,
        "l2_bytes": p.L2_cache_size,
        "total_memory": p.total_memory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nf12_rev": os.environ.get("NF12_BENCH_REV"),  # supplied by the run environment
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver": _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"]),
    }


def machine_tag() -> str:
    if is_tegra():
        return "thor"
    name = torch.cuda.get_device_name()
    return "rtxpro6000" if "RTX PRO 6000" in name else re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def l2_bytes() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size


# Smaller footprints can overestimate model-scale bandwidth on Thor.
ROTATE_MIN_BYTES = 2 << 30


def cold_copies(bytes_per_copy: int) -> int:
    """Rotate max(3x L2 + one copy, 2 GiB). Each copy must own its escape table."""
    span = max(3 * l2_bytes() + bytes_per_copy, ROTATE_MIN_BYTES)
    return max(2, math.ceil(span / bytes_per_copy))


def build_cuda(src: Path, name: str, shared: bool = False) -> Path:
    """Compiles a bench CUDA source for the current device into bench/build (cached by mtime)."""
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    sm = p.major * 10 + p.minor
    out = BENCH_DIR / "build" / (f"lib{name}_sm{sm}.so" if shared else f"{name}_sm{sm}")
    deps = [src, *src.parent.glob("*.cuh")]
    if not out.exists() or out.stat().st_mtime < max(d.stat().st_mtime for d in deps):
        nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        out.parent.mkdir(parents=True, exist_ok=True)
        flags = ["-shared", "-Xcompiler", "-fPIC"] if shared else []
        subprocess.run([nvcc, "-O3", "-std=c++17", f"-arch=sm_{sm}", *flags, "-o", str(out), str(src)], check=True)
    return out


class SolRead:
    """Streaming-read calibration using the benchmark's stream and graph timing."""

    def __init__(self) -> None:
        import ctypes

        self._lib = ctypes.CDLL(str(build_cuda(BENCH_DIR / "sol" / "read_lib.cu", "nf12bench_read", shared=True)))
        self._lib.nf12bench_read.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
                                             ctypes.c_void_p]
        self._lib.nf12bench_read_describe.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self._ctypes = ctypes
        self.num_configs = self._lib.nf12bench_read_num_configs()
        self.sink = torch.zeros(1, dtype=torch.int32, device="cuda")

    def describe(self, cfg: int) -> str:
        buf = self._ctypes.create_string_buffer(128)
        self._lib.nf12bench_read_describe(cfg, buf, len(buf))
        return buf.value.decode()

    def __call__(self, t: torch.Tensor, cfg: int) -> None:
        err = self._lib.nf12bench_read(t.data_ptr(), t.nbytes, cfg, self.sink.data_ptr(),
                                       torch.cuda.current_stream().cuda_stream)
        if err:
            raise RuntimeError(f"nf12bench_read failed: cudaError {err}")

    def best(self, bufs: list[torch.Tensor], budget_s: float = 0.25) -> tuple[int, float]:
        """(config, median us) of the fastest configuration reading ``bufs`` in rotation."""
        res = [(time_graph(lambda i: self(bufs[i], c), len(bufs), min_launches=20, budget_s=budget_s)["median_us"], c)
               for c in range(self.num_configs)]
        us, cfg = min(res)
        return cfg, us


def _percentiles(samples_us: list[float]) -> dict:
    s = sorted(samples_us)

    def q(p: float) -> float:
        i = p * (len(s) - 1)
        lo, hi = math.floor(i), math.ceil(i)
        return s[lo] + (s[hi] - s[lo]) * (i - lo)

    return {"median_us": q(0.5), "p10_us": q(0.1), "p90_us": q(0.9), "samples": len(s)}


def time_graph(fn: Callable[[int], object], copies: int, *, min_launches: int = 100, max_replays: int = 100,
               replay_target_ms: float = 0.5, budget_s: float = 3.0, distinct_outputs: bool = False) -> dict:
    """Per-launch timing of ``fn(i % copies)`` in a CUDA graph.

    Outputs normally reuse allocator storage and remain L2-resident.
    ``distinct_outputs`` keeps them alive during capture so codec writes
    use separate addresses and reach DRAM.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(copies):  # eager warmup: lazy init, allocator, cuBLAS heuristics
            fn(i)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(copies):
            fn(i)
        end.record()
    end.synchronize()
    est_ms = max(start.elapsed_time(end) / copies, 1e-4)
    rounds = max(1, math.ceil(replay_target_ms / (copies * est_ms)))
    n = copies * rounds
    replays = max(math.ceil(min_launches / n), min(max_replays, int(budget_s * 1e3 / (n * est_ms))))
    replays = max(replays, 10)

    graph = torch.cuda.CUDAGraph()
    outs = []
    with torch.cuda.graph(graph, stream=stream):
        for i in range(n):
            y = fn(i % copies)
            if distinct_outputs:
                outs.append(y)
    del outs, y
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(replays)]

    def run() -> tuple[list[float], dict[str, int]]:
        with torch.cuda.stream(stream):
            graph.replay()
            for a, b in events:
                a.record()
                graph.replay()
                b.record()
        emc = wait_sampling_clocks(events[-1][1])
        return [a.elapsed_time(b) * 1e3 / n for a, b in events], emc

    samples, emc = retry_low_clocks(run)
    del graph
    return {**_percentiles(samples), "launches_per_replay": n, "copies": copies, **emc}


# Tegra governors can lower EMC and GPU clocks even in MAXN.
# Sample both during timing and retry dips, keeping the final attempt at the limit.
_TEGRA_CLOCKS = {"emc": Path("/sys/class/devfreq/bwmgr"), "gpc": Path("/sys/class/devfreq/gpu-gpc-0")}


def _mhz(path: Path) -> int:
    return int(_read(path) or 0) // 1_000_000


def wait_sampling_clocks(done: torch.cuda.Event) -> dict[str, int]:
    """Wait for ``done`` and return sampled Tegra clock minima in MHz.

    Returns an empty dict off Tegra or when the event has already completed.
    """
    lo: dict[str, int] = {}
    domains = {k: d for k, d in _TEGRA_CLOCKS.items() if d.exists()}
    while domains and not done.query():
        for k, d in domains.items():
            f = _mhz(d / "cur_freq")
            lo[k] = min(lo.get(k, f), f)
    done.synchronize()
    return lo


def retry_low_clocks(run: Callable[[], tuple[list[float], dict[str, int]]],
                     attempts: int = 4) -> tuple[list[float], dict]:
    """Retry sampled clock dips up to ``attempts`` times, keeping the final result."""
    for attempt in range(1, attempts + 1):
        samples, lo = run()
        if all(f >= _mhz(_TEGRA_CLOCKS[k] / "max_freq") for k, f in lo.items()):
            break
    if not lo:
        return samples, {}
    return samples, {**{f"{k}_min_mhz": f for k, f in lo.items()}, "clock_attempts": attempt}


class L2Flusher:
    def __init__(self) -> None:
        self.buf = torch.empty(int(2.5 * l2_bytes()) // 4, dtype=torch.float32, device="cuda").fill_(1.0)
        self.sink = torch.empty((), dtype=torch.float32, device="cuda")

    def __call__(self) -> None:
        torch.sum(self.buf, dim=0, out=self.sink)


def time_events(fn: Callable[[int], object], *, cold: bool, iters: int = 100, warmup: int = 5) -> dict:
    """Median/p10/p90 of one event-bracketed launch of ``fn(0)``, optionally with an L2 flush before each."""
    flush = L2Flusher() if cold else None
    for _ in range(warmup):
        fn(0)
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]

    def run() -> tuple[list[float], dict[str, int]]:
        for a, b in events:
            if flush is not None:
                flush()
            a.record()
            fn(0)
            b.record()
        emc = wait_sampling_clocks(events[-1][1])
        return [a.elapsed_time(b) * 1e3 for a, b in events], emc

    samples, emc = retry_low_clocks(run)
    return {**_percentiles(samples), "launches_per_replay": 1, "copies": 1, **emc}


def time_fn(fn: Callable[[int], object], copies: int, *, timing: str, cold: bool, min_launches: int = 100,
            budget_s: float = 3.0) -> dict:
    if timing == "graph":
        return time_graph(fn, copies if cold else 1, min_launches=min_launches, budget_s=budget_s)
    return time_events(fn, cold=cold, iters=max(100, min_launches))


@contextlib.contextmanager
def gpu_lock(path: str | None) -> Iterator[None]:
    """Hold an exclusive flock for one measurement, or skip it if path is None."""
    if path is None:
        yield
        return
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def write_report(path: Path, result: dict, render: Callable[[Path], str]) -> str:
    write_json(path, result)
    text = render(path)
    path.with_suffix(".md").write_text(text + "\n")
    return text
