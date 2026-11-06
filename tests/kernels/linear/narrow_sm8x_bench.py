# SPDX-License-Identifier: Apache-2.0
"""SM89 router benchmark. Run under flock /tmp/nf12-gpu0.lock.
Each compressed copy owns its escapes. Every timed format rotates >=2 GiB.
"""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--shapes', default='2048x64,2048x128,4096x128')
    p.add_argument('--ms', default='16,32,64,96,128,192,256,384,512,768,1024,2048')
    p.add_argument('--lib')
    p.add_argument('--compare-lib', help='paired previous narrow implementation with the same config IDs')
    p.add_argument('--configs', default='-1')
    p.add_argument('--heavy', action='store_true')
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--safety', action='store_true', help='check concurrent streams, strided inputs/outputs, graph replay')
    p.add_argument('--checks-only', action='store_true', help='two independent copies, no timing')
    a = p.parse_args()
    torch.manual_seed(1234)
    result = dict(info=common.machine_info(), before=common.gpu_state(), args=vars(a), rows=[])
    L = None
    if not a.baseline:
        L = ctypes.CDLL(a.lib)
        L.nf12_narrow_test.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
        L.nf12_narrow_test.restype = ctypes.c_int
        if hasattr(L, 'nf12_narrow_stats'):
            L.nf12_narrow_stats.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                           ctypes.POINTER(ctypes.c_int)]
    other = ctypes.CDLL(a.compare_lib) if a.compare_lib else None
    if other:
        other.nf12_narrow_test.argtypes = L.nf12_narrow_test.argtypes
        other.nf12_narrow_test.restype = ctypes.c_int
    for key in a.shapes.split(','):
        shape = common.shape_by_key(key)
        w, origin = common.make_weight(shape, 'real', torch.device('cuda'), fetch=True)
        if shape.source and 'random' in origin:
            raise RuntimeError('real checkpoint unavailable: ' + key)
        if a.heavy:
            # Widen exponents every 64 row-major elements to force escapes.
            w = w.clone()
            w.view(-1, 8)[::8, 0] = 2.0 ** -30
        nw = nf12.encode(w)
        bt = nf12.pack_bf16t(w)
        count = 2 if a.checks_only else common.cold_copies(nw.nbytes)
        copies = [nw] + [nf12.NF12Weight(nw.packed.clone(), nw.escapes.clone(), nw.exp_base) for _ in range(count - 1)]
        wc = 2 if a.checks_only else common.cold_copies(w.nbytes)
        weights = [w] + [w.clone() for _ in range(wc - 1)]
        for m in map(int, a.ms.split(',')):
            x = torch.randn(m, shape.k + (64 if a.safety else 0), device='cuda', dtype=torch.bfloat16)[:, :shape.k]
            y = torch.empty(m, shape.n + (16 if a.safety else 0), device='cuda', dtype=torch.bfloat16)[:, :shape.n]
            twin = torch.empty_like(y)
            bias = torch.randn(shape.n, device='cuda', dtype=torch.bfloat16)
            fns = {'production': (lambda i: nf12.linear(x, copies[i]), count),
                   'cublas': (lambda i: torch.mm(x, weights[i].T, out=y), wc),
                   'decode_cublas': (lambda i: torch.mm(x, nf12.decode(copies[i]).T, out=y), count)}
            row = dict(key=key, m=m, origin=origin, escape_rate=nw.num_escapes/(shape.n*shape.k/8),
                       copies=count, compressed_bytes=count*nw.nbytes, bf16_bytes=wc*w.nbytes, correctness={}, orders=[])
            if L:
                def launch(i, cfg, source=0, out=y, b=None, lib=L):
                    q = copies[i]
                    rc = lib.nf12_narrow_test(cfg, source, x.data_ptr(), m, x.stride(0),
                        (q.packed if source == 0 else bt).data_ptr(), q.escapes.data_ptr() if q.num_escapes else None,
                        q.num_escapes, q.exp_base, shape.n, shape.k, b.data_ptr() if b is not None else None,
                        out.data_ptr(), out.stride(0), torch.cuda.current_stream().cuda_stream)
                    if rc:
                        raise RuntimeError(f'config {cfg}: cudaError {rc}')
                    return out
                for cfg in map(int, a.configs.split(',')):
                    if hasattr(L, 'nf12_narrow_stats'):
                        stats = (ctypes.c_int * 7)()
                        rc = L.nf12_narrow_stats(cfg, m, shape.n, shape.k, stats)
                        if rc:
                            raise RuntimeError(f'resource query failed: {rc}')
                        row.setdefault('resources', {})[str(cfg)] = dict(
                            zip(('registers', 'shared_bytes', 'threads', 'resident_ctas_per_sm', 'bm',
                                 'local_bytes', 'resolved_config'), stats))
                    for b in (None, bias):
                        launch(0, cfg, out=y, b=b)
                        launch(0, cfg, source=1, out=twin, b=b)
                        torch.cuda.synchronize()
                        if not torch.equal(y.view(torch.int16), twin.view(torch.int16)):
                            raise AssertionError(f'BF16 twin mismatch {key} M{m} cfg{cfg}')
                    ref = x.float() @ w.float().T + bias.float()
                    rel = ((y.float()-ref).norm()/ref.norm()).item()
                    if rel > 0.01:
                        raise AssertionError(f'numerical mismatch {key} M{m} cfg{cfg}: {rel}')
                    if a.safety:
                        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
                        outs = [torch.empty_like(y), torch.empty_like(y)]
                        for stream, out in zip(streams, outs):
                            stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):
                                launch(count - 1, cfg, out=out, b=bias)
                        for stream in streams:
                            torch.cuda.current_stream().wait_stream(stream)
                        if not all(torch.equal(out.view(torch.int16), twin.view(torch.int16)) for out in outs):
                            raise AssertionError('concurrent stream mismatch')
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            launch(count - 1, cfg, out=y, b=bias)
                        y.zero_()
                        graph.replay()
                        torch.cuda.synchronize()
                        if not torch.equal(y.view(torch.int16), twin.view(torch.int16)):
                            raise AssertionError('graph replay mismatch')
                    row['correctness'][str(cfg)] = dict(matches_twin=True, rel_fp32=rel,
                                                       concurrent_streams=a.safety, graph_replay=a.safety)
                    fns[f'narrow_{cfg}'] = (lambda i, c=cfg: launch(i, c), count)
                    if other:
                        old_out = torch.empty_like(y)
                        launch(0, cfg, out=old_out, b=bias, lib=other)
                        if not torch.equal(old_out.view(torch.int16), twin.view(torch.int16)):
                            raise AssertionError('paired implementation changed the arithmetic')
                        fns[f'previous_{cfg}'] = (lambda i, c=cfg: launch(i, c, lib=other), count)
            for order in (() if a.checks_only else (list(fns), list(reversed(fns)))):
                times = {}
                for name in order:
                    fn, nc = fns[name]
                    times[name] = common.time_graph(fn, nc, min_launches=10, budget_s=0.3, max_replays=10)
                row['orders'].append(dict(times=times, gpu=common.gpu_state()))
            result['rows'].append(row)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(result, indent=1))
            print(key, m, ({k: round(v['median_us'], 3) for k,v in row['orders'][0]['times'].items()}
                           if row['orders'] else row['correctness']), flush=True)
        del copies, weights, nw, bt, w, x, y, twin
        torch.cuda.empty_cache()
    result['after'] = common.gpu_state()
    Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
