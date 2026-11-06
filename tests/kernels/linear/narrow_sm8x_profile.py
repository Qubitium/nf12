# SPDX-License-Identifier: Apache-2.0
"""One marked router invocation for Nsight Compute / torch CUDA timeline."""
import argparse
import ctypes
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'bench'))
import common
import nf12

p = argparse.ArgumentParser()
p.add_argument('--lib', required=True)
p.add_argument('--shape', default='2048x64')
p.add_argument('--m', type=int, default=128)
p.add_argument('--config', type=int, default=1)
p.add_argument('--timeline')
a = p.parse_args()
s = common.shape_by_key(a.shape)
w, origin = common.make_weight(s, 'real', torch.device('cuda'), fetch=True)
nw = nf12.encode(w)
x = torch.randn(a.m, s.k, device='cuda', dtype=torch.bfloat16)
y = torch.empty(a.m, s.n, device='cuda', dtype=torch.bfloat16)
L = ctypes.CDLL(a.lib)
L.nf12_narrow_test.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]

def narrow():
    rc = L.nf12_narrow_test(a.config, 0, x.data_ptr(), a.m, s.k, nw.packed.data_ptr(), nw.escapes.data_ptr(),
        nw.num_escapes, nw.exp_base, s.n, s.k, None, y.data_ptr(), s.n, torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(rc)

for _ in range(3):
    narrow()
    nf12.linear(x, nw)
    torch.mm(x, w.T, out=y)
torch.cuda.synchronize()
if a.timeline:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        for label, fn in [('narrow', narrow), ('production', lambda: nf12.linear(x, nw)),
                          ('cublas', lambda: torch.mm(x, w.T, out=y)), ('decode', lambda: nf12.decode(nw))]:
            with torch.profiler.record_function(label):
                fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(a.timeline)
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=30))
else:
    torch.cuda.nvtx.range_push('narrow_profile')
    narrow()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
print(json.dumps(dict(shape=a.shape, m=a.m, config=a.config, origin=origin, gpu=common.gpu_state())))
