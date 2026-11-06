# SPDX-License-Identifier: Apache-2.0
"""ctypes access to the SM120 large-M kernel family.

Adds the sibling large_m_sm120_shim.cu to large_m_harness.Shim's recursive
csrc/linear build through NF12_EXTRA_SRCS. One process can run SM120 and
mma.sync kernels. Set NF12_LARGE_M_BUILD_DIR to a private directory.
"""

import ctypes
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import large_m_harness as H  # noqa: E402

SHIM_SRC = os.path.join(HERE, "large_m_sm120_shim.cu")

# Schedule is config // 16 % 16. SPLIT factor is config // 256 (large_m_sm120.h).
HYBRID, DATA_PARALLEL, STREAM_K, SPLIT = 0, 1, 2, 3


class Sm120Shim(H.Shim):
    def __init__(self):
        extra = list(filter(None, os.environ.get("NF12_EXTRA_SRCS", "").split(",")))
        if SHIM_SRC not in extra:
            os.environ["NF12_EXTRA_SRCS"] = ",".join(extra + [SHIM_SRC])
        super().__init__()
        L = self.lib
        i64, u32, vp, sz = ctypes.c_int64, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t
        L.nf12_test_sm120.restype = ctypes.c_int
        L.nf12_test_sm120.argtypes = [
            ctypes.c_int,                    # config (-1 = heuristic)
            vp, i64, i64,                    # x, m, ldx
            ctypes.c_int, vp, vp, u32, u32,  # source, weight, escapes, num_escapes, exp_base
            i64, i64, vp, vp, i64,           # n, k, bias, y, ldy
            vp, sz, vp,                      # workspace, bytes, stream
        ]
        L.nf12_test_sm120_ws.restype = sz
        L.nf12_test_sm120_ws.argtypes = [ctypes.c_int, i64, i64, i64]
        L.nf12_test_sm120_num_configs.restype = ctypes.c_int
        L.nf12_test_sm120_config_name.restype = ctypes.c_char_p
        L.nf12_test_sm120_config_name.argtypes = [ctypes.c_int]
        L.nf12_test_sm120_supported.restype = ctypes.c_int
        L.nf12_test_sm120_heuristic.restype = ctypes.c_int
        L.nf12_test_sm120_heuristic.argtypes = [i64, i64, i64]
        L.nf12_test_sm120_narrow_supported.restype = ctypes.c_int
        L.nf12_test_sm120_narrow_heuristic.restype = ctypes.c_int
        L.nf12_test_sm120_narrow_heuristic.argtypes = [i64, i64, i64]
        L.nf12_test_sm120_narrow.restype = ctypes.c_int
        L.nf12_test_sm120_narrow.argtypes = [
            ctypes.c_int, vp, i64, i64, ctypes.c_int, vp, vp, u32, u32, i64, i64, vp, vp, i64, vp]
        self._ws120 = None

    def sm120_supported(self):
        return bool(self.lib.nf12_test_sm120_supported())

    def sm120_heuristic(self, m, n, k):
        """Config code the heuristic (config = -1) runs for this shape."""
        return self.lib.nf12_test_sm120_heuristic(m, n, k)

    def sm120_num_configs(self):
        return self.lib.nf12_test_sm120_num_configs()

    def sm120_config_name(self, config):
        if config < 0:
            return "auto"
        sched = ["hyb", "dp", "sk", f"split{config // 256}"][config // 16 % 16]
        return f"{self.lib.nf12_test_sm120_config_name(config % 16).decode()}/{sched}"

    def sm120_workspace(self, config, m, n, k):
        nbytes = self.lib.nf12_test_sm120_ws(config, m, n, k)
        if self._ws120 is None or self._ws120.numel() < nbytes:
            # Arbitrary initial workspace contents must be valid.
            self._ws120 = torch.randint(0, 256, (max(nbytes, 256),), dtype=torch.uint8, device="cuda")
        return self._ws120, nbytes

    def sm120(self, x, w, bias=None, y=None, source="nf12", config=-1, ws=None):
        """Return y [m, n] for bf16 x [m, k] (possibly row-padded) and H.NF12Packed w."""
        m, k = x.shape
        n = w.n
        assert k == w.k and x.stride(1) == 1
        if y is None:
            y = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
        if ws is None:
            ws, nbytes = self.sm120_workspace(config, m, n, k)
        else:
            nbytes = ws.numel()
        wt = w.packed if source == "nf12" else w.bf16t
        esc = w.escapes.data_ptr() if w.num_escapes else None
        rc = self.lib.nf12_test_sm120(
            config, x.data_ptr(), m, x.stride(0), 0 if source == "nf12" else 1, wt.data_ptr(), esc,
            w.num_escapes, w.exp_base, n, k, bias.data_ptr() if bias is not None else None, y.data_ptr(),
            y.stride(0), ws.data_ptr(), nbytes, torch.cuda.current_stream().cuda_stream,
        )
        if rc != 0:
            if rc == 801:  # cudaErrorNotSupported
                raise H.Unsupported(f"sm120 config {config} unsupported on this device")
            raise RuntimeError(f"sm120 kernel failed: cudaError {rc}")
        return y

    def narrow_supported(self):
        return bool(self.lib.nf12_test_sm120_narrow_supported())

    def narrow_heuristic(self, m, n, k):
        return self.lib.nf12_test_sm120_narrow_heuristic(m, n, k)

    @staticmethod
    def narrow_config_name(config):
        return "auto" if config < 0 else f"t{8 << (config & 3)}_w{1 << ((config >> 2) & 3)}_s{config >> 4}"

    def narrow(self, x, w, bias=None, y=None, source="nf12", config=-1):
        """Run the narrow-weight kernel with x and w as for sm120()."""
        m, k = x.shape
        n = w.n
        if y is None:
            y = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
        wt = w.packed if source == "nf12" else w.bf16t
        esc = w.escapes.data_ptr() if w.num_escapes else None
        rc = self.lib.nf12_test_sm120_narrow(
            config, x.data_ptr(), m, x.stride(0), 0 if source == "nf12" else 1, wt.data_ptr(), esc, w.num_escapes,
            w.exp_base, n, k, bias.data_ptr() if bias is not None else None, y.data_ptr(), y.stride(0),
            torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            if rc == 801:
                raise H.Unsupported(f"narrow config {config} unsupported")
            raise RuntimeError(f"narrow kernel failed: cudaError {rc}")
        return y
