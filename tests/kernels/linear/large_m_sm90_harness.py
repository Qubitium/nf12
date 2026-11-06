# SPDX-License-Identifier: Apache-2.0
"""ctypes access to the SM90 large-M kernel family.

Adds the sibling large_m_sm90_shim.cu to large_m_harness.Shim's recursive
csrc/linear build through NF12_EXTRA_SRCS. One process can run SM90 and
mma.sync kernels. Set NF12_LARGE_M_BUILD_DIR to a private directory.
"""

import ctypes
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import large_m_harness as H  # noqa: E402

SHIM_SRC = os.path.join(HERE, "large_m_sm90_shim.cu")

# Schedule is config // 16 % 16. SPLIT/CLUSTER_SPLIT factor is config // 256
# (large_m_sm90.cu).
HYBRID, DATA_PARALLEL, STREAM_K, SPLIT, CLUSTER_SPLIT = 0, 1, 2, 3, 4


class Sm90Shim(H.Shim):
    def __init__(self):
        extra = list(filter(None, os.environ.get("NF12_EXTRA_SRCS", "").split(",")))
        if SHIM_SRC not in extra:
            os.environ["NF12_EXTRA_SRCS"] = ",".join(extra + [SHIM_SRC])
        super().__init__()
        L = self.lib
        i64, u32, vp, sz = ctypes.c_int64, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t
        L.nf12_test_sm90.restype = ctypes.c_int
        L.nf12_test_sm90.argtypes = [
            ctypes.c_int,                    # config (-1 = heuristic)
            vp, i64, i64,                    # x, m, ldx
            ctypes.c_int, vp, vp, u32, u32,  # source, weight, escapes, num_escapes, exp_base
            i64, i64, vp, vp, i64,           # n, k, bias, y, ldy
            vp, sz, vp,                      # workspace, bytes, stream
        ]
        L.nf12_test_sm90_ws.restype = sz
        L.nf12_test_sm90_ws.argtypes = [ctypes.c_int, i64, i64, i64]
        L.nf12_test_sm90_num_configs.restype = ctypes.c_int
        L.nf12_test_sm90_config_name.restype = ctypes.c_char_p
        L.nf12_test_sm90_config_name.argtypes = [ctypes.c_int]
        L.nf12_test_sm90_supported.restype = ctypes.c_int
        L.nf12_test_sm90_heuristic.restype = ctypes.c_int
        L.nf12_test_sm90_heuristic.argtypes = [i64, i64, i64]
        self._ws90 = None

    def sm90_supported(self):
        return bool(self.lib.nf12_test_sm90_supported())

    def sm90_heuristic(self, m, n, k):
        """Config code the heuristic (config = -1) runs for this shape."""
        return self.lib.nf12_test_sm90_heuristic(m, n, k)

    def sm90_cluster_split_ok(self, config):
        """Cluster split-K requires register-A configs because shared-A configs
        have incompatible producer/consumer stage lifetimes."""
        return "_c" not in self.lib.nf12_test_sm90_config_name(config % 16).decode()

    def sm90_num_configs(self):
        return self.lib.nf12_test_sm90_num_configs()

    def sm90_config_name(self, config):
        if config < 0:
            return "auto"
        sched = ["hyb", "dp", "sk", f"split{config // 256}", f"csplit{config // 256}"][config // 16 % 16]
        return f"{self.lib.nf12_test_sm90_config_name(config % 16).decode()}/{sched}"

    def sm90_workspace(self, config, m, n, k):
        nbytes = self.lib.nf12_test_sm90_ws(config, m, n, k)
        if self._ws90 is None or self._ws90.numel() < nbytes:
            # Arbitrary initial workspace contents must be valid.
            self._ws90 = torch.randint(0, 256, (max(nbytes, 256),), dtype=torch.uint8, device="cuda")
        return self._ws90, nbytes

    def sm90(self, x, w, bias=None, y=None, source="nf12", config=-1, ws=None):
        """Return y [m, n] for bf16 x [m, k] (possibly row-padded) and H.NF12Packed w."""
        m, k = x.shape
        n = w.n
        assert k == w.k and x.stride(1) == 1
        if y is None:
            y = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
        if ws is None:
            ws, nbytes = self.sm90_workspace(config, m, n, k)
        else:
            nbytes = ws.numel()
        wt = w.packed if source == "nf12" else w.bf16t
        esc = w.escapes.data_ptr() if w.num_escapes else None
        rc = self.lib.nf12_test_sm90(
            config, x.data_ptr(), m, x.stride(0), 0 if source == "nf12" else 1, wt.data_ptr(), esc,
            w.num_escapes, w.exp_base, n, k, bias.data_ptr() if bias is not None else None, y.data_ptr(),
            y.stride(0), ws.data_ptr(), nbytes, torch.cuda.current_stream().cuda_stream,
        )
        if rc != 0:
            if rc == 801:  # cudaErrorNotSupported
                raise H.Unsupported(f"sm90 config {config} unsupported on this device")
            raise RuntimeError(f"sm90 kernel failed: cudaError {rc}")
        return y
