# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-License-Identifier: Apache-2.0
"""Interleaved source-built ROCm codec comparison, including public API costs.

Example (run with HIP_VISIBLE_DEVICES=1 PYTHON_GIL=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1):
  python bench/hip_encode.py --baseline-ref <commit> --build-root /tmp/nf12-encode \
      --output /tmp/encode.json

The baseline uses a distinct SONAME and Torch namespace. Both public functions
execute the checkout's identical Python API code, including validation, loading
fast path and result construction. Only the native op namespace differs.
GPU graphs invoke the real kernels with prevalidated base/capacity, omitting
host selection, transfers, allocation and synchronization. They are a diagnostic
lower bound, not a claim that the public encoder supports graph capture.
"""
from __future__ import annotations

import argparse
import ctypes as C
import importlib.metadata
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time
import types

import torch

from common import MODEL_SHAPES, SHAPES, make_weight, shape_by_key

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import nf12
from nf12 import reference


def run(args):
    subprocess.run([str(x) for x in args], check=True)


def build(args):
    root = args.build_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = root / 'baseline-src'
    if not args.skip_build:
        if base.exists():
            raise ValueError('Use a fresh --build-root or --skip-build; baseline source already exists')
        base.mkdir()
        archive = subprocess.check_output(['git', 'archive', args.baseline_ref], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
            tf.extractall(base, filter='data')
        p = base / 'csrc/torch/ops.cpp'
        p.write_text(p.read_text().replace('STABLE_TORCH_LIBRARY(nf12,', 'STABLE_TORCH_LIBRARY(nf12_baseline,')
                     .replace('STABLE_TORCH_LIBRARY_IMPL(nf12,', 'STABLE_TORCH_LIBRARY_IMPL(nf12_baseline,'))
        p = base / 'csrc/hip/CMakeLists.txt'
        p.write_text(p.read_text() + '\nset_target_properties(nf12_hip PROPERTIES OUTPUT_NAME nf12_baseline_hip)\n'
                     'set_target_properties(nf12_torch PROPERTIES OUTPUT_NAME nf12_baseline_torch)\n')
        for name, source in [('baseline', base), ('current', ROOT)]:
            dst = root / name
            run(['cmake', '-S', source / 'csrc/hip', '-B', dst,
                 f'-DCMAKE_HIP_COMPILER={args.rocm}/lib/llvm/bin/clang++',
                 f'-DCMAKE_HIP_COMPILER_ROCM_ROOT={args.rocm}', f'-DCMAKE_CXX_FLAGS=-I{args.rocm}/include',
                 '-DNF12_HIP_ARCHS=gfx1201', '-DNF12_BUILD_TORCH=ON',
                 f'-DPython_EXECUTABLE={sys.executable}', '-DCMAKE_BUILD_TYPE=Release'])
            run(['cmake', '--build', dst, '-j', '8'])
            run([args.rocm / 'bin/hipcc', '-O3', '-std=c++17', '--offload-arch=gfx1201',
                 '-shared', '-fPIC', f'-I{source}/csrc/hip', f'-I{source}/include',
                 ROOT / 'bench/hip_encode_stages.hip', '-o', dst / 'libencode_stages.so'])
    return root


def public_function(name, namespace):
    fn = getattr(nf12, name)
    env = dict(fn.__globals__)
    env['torch'] = types.SimpleNamespace(ops=types.SimpleNamespace(nf12=namespace))
    return types.FunctionType(fn.__code__, env, fn.__name__, fn.__defaults__)


def stats(values):
    v = sorted(values)
    return dict(median_us=statistics.median(v), p10_us=v[len(v)//10],
                p90_us=v[min(len(v)-1, len(v)*9//10)], samples_us=values)


def paired(functions, warmup, repeats, gpu=False, launches=4):
    stream = torch.cuda.current_stream()
    for _ in range(warmup):
        for fn in functions:
            fn()
    stream.synchronize()
    graphs = []
    if gpu:
        for fn in functions:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(launches):
                    fn()
            graphs.append(graph)
        functions = [g.replay for g in graphs]
        for fn in functions:
            fn()
        stream.synchronize()
    events = [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)]
    samples = [[], []]
    for i in range(repeats):
        for j in ([0, 1] if i % 2 else [1, 0]):
            stream.synchronize()
            if gpu:
                events[0].record()
                functions[j]()
                events[1].record()
                events[1].synchronize()
                elapsed = events[0].elapsed_time(events[1]) * 1000 / launches
            else:
                start = time.perf_counter_ns()
                result = functions[j]()  # retain outputs until the stream completes
                stream.synchronize()
                elapsed = (time.perf_counter_ns() - start) / 1000
                del result
            samples[j].append(elapsed)
    a, b = map(stats, samples)
    return dict(baseline=a, current=b, speedup=a['median_us']/b['median_us'])


class Native:
    def __init__(self, directory, baseline=False):
        libname = 'libnf12_baseline_hip.so' if baseline else 'libnf12_hip.so'
        self.lib = C.CDLL(str(directory / libname))
        self.stages = C.CDLL(str(directory / 'libencode_stages.so'))
        ptr, i64, size = C.c_void_p, C.c_int64, C.c_size_t
        self.lib.nf12_encode_workspace_bytes.argtypes = [i64, i64]
        self.lib.nf12_encode_workspace_bytes.restype = size
        self.lib.nf12_encode_analyze.argtypes = [ptr, i64, i64, i64, ptr, size,
                                                C.POINTER(C.c_uint8), C.POINTER(C.c_uint32), ptr]
        self.lib.nf12_encode.argtypes = [ptr, i64, i64, i64, C.c_uint8, ptr, ptr, C.c_uint32, ptr, size, ptr]
        self.stages.nf12_bench_encode_stages.argtypes = [C.c_int, ptr, i64, i64, C.c_uint8, ptr, ptr, ptr, ptr]

    def bind(self, w, encoded):
        n, k = w.shape
        ws_bytes = self.lib.nf12_encode_workspace_bytes(n, k)
        ws = torch.empty(ws_bytes, dtype=torch.uint8, device=w.device)
        packed = torch.empty((n, 2*k), dtype=torch.uint8, device=w.device)
        escapes = torch.empty_like(encoded.escapes)
        base, count = C.c_uint8(), C.c_uint32()
        stream = torch.cuda.current_stream().cuda_stream

        def analyze():
            rc = self.lib.nf12_encode_analyze(w.data_ptr(), n, k, w.stride(0), ws.data_ptr(), ws_bytes,
                                               C.byref(base), C.byref(count), stream)
            assert rc == 0

        def encode():
            rc = self.lib.nf12_encode(w.data_ptr(), n, k, w.stride(0), encoded.exp_base,
                                      packed.data_ptr(), escapes.data_ptr(), encoded.num_escapes,
                                      ws.data_ptr(), ws_bytes, stream)
            assert rc == 0

        def stage(stage_id):
            rc = self.stages.nf12_bench_encode_stages(stage_id, w.data_ptr(), n, k, encoded.exp_base,
                                                     packed.data_ptr(), escapes.data_ptr(), ws.data_ptr(),
                                                     torch.cuda.current_stream().cuda_stream)
            assert rc == 0

        analyze()
        assert (base.value, count.value) == (encoded.exp_base, encoded.num_escapes)
        stage(2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stage(2)
        packed.fill_(0xa5)
        escapes.zero_()
        graph.replay()
        assert packed.flatten()[:encoded.packed.numel()].equal(encoded.packed.flatten())
        assert escapes.equal(encoded.escapes)
        stage(3)
        assert packed.equal(nf12.pack_bf16t(w))
        return analyze, encode, stage


def weight(shape, pattern):
    w, _ = make_weight(shape, 'random', torch.device('cuda'))
    if pattern == 'zero':
        w.zero_()
    elif pattern == 'heavy':
        # Widely separated exponents in every fourth atom's groups.
        w[:, ::64] = float('inf')
        w[:, 1::64] = 0
    elif pattern == 'special':
        bits = torch.tensor([0, -32768, 1, -32767, 127, 128, 32639, -129,
                             32640, -128, 32704, 32641, -63, -1, 16256, -16512],
                            dtype=torch.int16, device='cuda')
        w.view(torch.int16).reshape(-1, bits.numel()).copy_(bits)
    return w


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-ref', required=True)
    p.add_argument('--build-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rocm', type=Path, default=Path('/opt/rocm'))
    p.add_argument('--skip-build', action='store_true')
    p.add_argument('--shapes', nargs='*')
    p.add_argument('--patterns', nargs='+', default=['normal', 'heavy', 'zero', 'special'])
    p.add_argument('--warmup', type=int, default=8)
    p.add_argument('--repeats', type=int, default=31)
    args = p.parse_args()
    assert torch.version.hip and torch.cuda.device_count() == 1
    prop = torch.cuda.get_device_properties(0)
    assert prop.gcnArchName.split(':')[0] == 'gfx1201'
    assert not sys._is_gil_enabled(), 'Run the free-threaded interpreter with PYTHON_GIL=0'
    print(prop, flush=True)
    root = build(args)
    nf12.__path__.insert(0, str(root / 'current'))
    nf12._library.load()
    torch.ops.load_library(str(root / 'baseline/libnf12_baseline_torch.so'))
    namespaces = [torch.ops.nf12_baseline, torch.ops.nf12]
    public = {op: [public_function(op, ns) for ns in namespaces] for op in ('encode', 'pack_bf16t')}
    natives = [Native(root / 'baseline', True), Native(root / 'current')]
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                if d.metadata['Name'].lower().startswith('rocm')}
    result = dict(metadata=dict(baseline_ref=args.baseline_ref,
        current_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        diff=subprocess.check_output(['git', 'diff', '--', 'csrc/hip/encode.hip'], cwd=ROOT, text=True),
        torch=torch.__version__, hip=torch.version.hip, rocm_packages=packages,
        hipconfig=subprocess.check_output([args.rocm/'bin/hipconfig', '--version'], text=True).strip(),
        gpu=str(prop), visible_devices=os.environ.get('HIP_VISIBLE_DEVICES'), gil=sys._is_gil_enabled(),
        omp=os.environ.get('OMP_NUM_THREADS'), mkl=os.environ.get('MKL_NUM_THREADS'),
        warmup=args.warmup, repeats=args.repeats, graph_launches=4,
        weights='synthetic, no checkpoint download',
        wall='current-stream sync before and after; includes allocation, Python, host validation and D2H waits',
        gpu_timing='HIP events around graph replay / 4; preallocated, known base/capacity; no D2H or host selection',
        cache='warm repeated input, no explicit cache flush; largest inputs exceed cache'), cases=[])
    shapes = [shape_by_key(x) for x in args.shapes] if args.shapes else MODEL_SHAPES[:3] + SHAPES
    for shape in shapes:
        patterns = args.patterns if args.shapes or shape in MODEL_SHAPES[:3] + [SHAPES[0], SHAPES[7]] else ['normal']
        for pattern in patterns:
            w = weight(shape, pattern)
            encoded = [f(w) for f in public['encode']]
            a, b = encoded
            assert a.exp_base == b.exp_base and a.packed.equal(b.packed) and a.escapes.equal(b.escapes)
            assert nf12.decode(b).view(torch.int16).equal(w.view(torch.int16))
            assert public['pack_bf16t'][0](w).equal(public['pack_bf16t'][1](w))
            if w.numel() <= 524288:
                rp, re, rb = reference.encode(w.cpu())
                assert rb == b.exp_base and rp.equal(b.packed.cpu()) and re.equal(b.escapes.cpu())
                assert reference.pack_bf16t(w.cpu()).equal(nf12.pack_bf16t(w).cpu())
            bound = [native.bind(w, enc) for native, enc in zip(natives, encoded)]
            metrics = {}
            for op in public:
                metrics['public_'+op] = paired([lambda f=f: f(w) for f in public[op]], args.warmup, args.repeats)
            for name, idx in [('analyze_preallocated_wall', 0), ('encode_preallocated_wall', 1)]:
                metrics[name] = paired([f[idx] for f in bound], args.warmup, args.repeats)
            for name, stage in [('analyze_gpu', 0), ('encode_gpu', 1), ('full_gpu', 2), ('bf16t_gpu', 3)]:
                metrics[name] = paired([lambda f=f: f[2](stage) for f in bound], args.warmup, args.repeats, gpu=True)
            def allocate():
                return (torch.empty_like(b.packed), torch.empty_like(b.escapes),
                        torch.empty(natives[1].lib.nf12_encode_workspace_bytes(shape.n, shape.k),
                                    dtype=torch.uint8, device=w.device))
            metrics['cached_allocations_and_sync_wall'] = paired([allocate]*2, args.warmup, args.repeats)
            metrics['empty_sync_wall'] = paired([lambda: None]*2, args.warmup, args.repeats)
            case = dict(key=shape.key, n=shape.n, k=shape.k, pattern=pattern,
                        exp_base=b.exp_base, escapes=b.num_escapes, correctness='byte exact', metrics=metrics)
            result['cases'].append(case)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            m = metrics['public_encode']
            print(shape.key, pattern, 'encode us', *(round(m[v]['median_us'],2) for v in ('baseline','current')),
                  'speedup', round(m['speedup'],3), flush=True)
            del bound, encoded, a, b, w
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
