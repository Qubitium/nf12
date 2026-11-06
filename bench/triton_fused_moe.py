# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone BF16 Triton MoE baseline.

The GEMM comes from vLLM via Aphrodite b8c04dd
(aphrodite/model_executor/layers/fused_moe/fused_moe.py). Quantization, bias,
expert-parallel (-1 expert), tensor-descriptor and swap-AB branches were
removed. The remaining body and BF16 default config are unchanged.

Layer execution follows upstream fused_experts_impl. Sparse batches use
naive block assignment. Otherwise two Triton kernels replace CUDA alignment,
producing padded assignment IDs (sentinel A), one expert per BLOCK_M block
and a padded count. Within-expert order is arbitrary.
"""

from __future__ import annotations

import dataclasses

import torch
import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    naive_block_assignment: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(offs == 0, pid_m, num_valid_tokens)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _align_plan_kernel(topk_ids_ptr, A, E, cursor_ptr, expert_ids_ptr, ntpp_ptr, sorted_ptr, num_blocks,
                       BLOCK_M: tl.constexpr, E_P2: tl.constexpr, CHUNK: tl.constexpr):
    # Count experts, scan padded counts, assign blocks and initialize scatter cursors.
    counts = tl.zeros([E_P2], dtype=tl.int32)
    for start in range(0, A, CHUNK):
        offs = start + tl.arange(0, CHUNK)
        ids = tl.load(topk_ids_ptr + offs, mask=offs < A, other=-1)
        counts += tl.histogram(ids, E_P2, mask=(offs < A) & (ids >= 0) & (ids < E))
    padded = (counts + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    ends = tl.cumsum(padded, axis=0)
    starts = ends - padded
    e = tl.arange(0, E_P2)
    tl.store(cursor_ptr + e, starts, mask=e < E)
    tl.store(ntpp_ptr, tl.sum(padded, axis=0))
    for b0 in range(0, num_blocks, CHUNK):
        b = b0 + tl.arange(0, CHUNK)
        owner = tl.sum(((ends[None, :] <= b[:, None] * BLOCK_M) & (e[None, :] < E)).to(tl.int32), axis=1)
        tl.store(expert_ids_ptr + b, tl.minimum(owner, E - 1), mask=b < num_blocks)
    for j0 in range(0, BLOCK_M, 32):
        j = j0 + tl.arange(0, 32)
        pad = (j[None, :] >= (counts[:, None] % BLOCK_M)) & (j[None, :] < BLOCK_M) & (counts[:, None] % BLOCK_M != 0)
        tl.store(sorted_ptr + (starts + counts)[:, None] + j[None, :] - counts[:, None] % BLOCK_M, A,
                 mask=pad & (e[:, None] < E))


@triton.jit
def _align_scatter_kernel(topk_ids_ptr, A, E, cursor_ptr, sorted_ptr, CHUNK: tl.constexpr):
    offs = tl.program_id(0) * CHUNK + tl.arange(0, CHUNK)
    ids = tl.load(topk_ids_ptr + offs, mask=offs < A, other=-1)
    valid = (offs < A) & (ids >= 0) & (ids < E)
    dest = tl.atomic_add(cursor_ptr + ids, 1, mask=valid)
    tl.store(sorted_ptr + dest, offs, mask=valid)


@dataclasses.dataclass
class Assignment:
    sorted_ids: torch.Tensor | None  # None: naive block assignment
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor


def max_padded(num_assignments: int, num_experts: int, block_m: int) -> int:
    n = num_assignments + num_experts * (block_m - 1)
    return min(num_assignments * block_m, n) if num_assignments < num_experts else n


def moe_align_block_size(topk_ids: torch.Tensor, block_m: int, num_experts: int) -> Assignment:
    a = topk_ids.numel()
    size = max_padded(a, num_experts, block_m)
    dev = topk_ids.device
    sorted_ids = torch.empty(size, dtype=torch.int32, device=dev)
    num_blocks = triton.cdiv(size, block_m)
    expert_ids = torch.empty(num_blocks, dtype=torch.int32, device=dev)
    ntpp = torch.empty(1, dtype=torch.int32, device=dev)
    cursor = torch.empty(num_experts, dtype=torch.int32, device=dev)
    flat = topk_ids.view(-1)
    _align_plan_kernel[(1,)](flat, a, num_experts, cursor, expert_ids, ntpp, sorted_ids, num_blocks,
                             BLOCK_M=block_m, E_P2=triton.next_power_of_2(num_experts), CHUNK=1024)
    _align_scatter_kernel[(triton.cdiv(a, 1024),)](flat, a, num_experts, cursor, sorted_ids, CHUNK=1024)
    return Assignment(sorted_ids, expert_ids, ntpp)


def prepare_assignment(topk_ids: torch.Tensor, block_m: int, num_experts: int) -> Assignment:
    """Use naive assignment when tokens * topk * 4 <= E, otherwise align."""
    t, topk = topk_ids.shape
    if t * topk * 4 <= num_experts:
        return Assignment(None, topk_ids.view(-1),
                          torch.full((1,), topk_ids.numel() * block_m, dtype=torch.int32, device=topk_ids.device))
    return moe_align_block_size(topk_ids, block_m, num_experts)


def get_default_config(m: int, num_experts: int) -> dict:
    """fused_moe.py get_default_config, BF16 branch (CUDA)."""
    block_m = 16 if m <= 32 else 32 if m <= 96 else 64 if m <= 512 else 128
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": 64 if m <= 64 else 128,
        "BLOCK_SIZE_K": 128 if m <= 64 else 64,
        "GROUP_SIZE_M": 16 if m // max(num_experts, 1) > 128 else 1,
        "num_warps": 4 if m <= 128 else 8,
        "num_stages": 4 if m <= 32 else 3,
    }


def config_key(cfg: dict) -> str:
    return "m{BLOCK_SIZE_M}n{BLOCK_SIZE_N}k{BLOCK_SIZE_K}g{GROUP_SIZE_M}w{num_warps}s{num_stages}".format(**cfg)


def invoke(a: torch.Tensor, w: torch.Tensor, c: torch.Tensor, topk_weights: torch.Tensor | None,
           asg: Assignment, top_k: int, num_valid: int, cfg: dict) -> None:
    """Write a[id // top_k] @ w[expert].T, optionally scaled by topk_weights[id]."""
    bm = cfg["BLOCK_SIZE_M"]
    if asg.sorted_ids is not None:
        em = asg.sorted_ids.numel()
        if a.size(0) < bm:  # vLLM's small-batch grid bound
            em = min(em, a.size(0) * top_k * bm)
    else:
        em = num_valid * bm
    n, k = w.size(1), w.size(2)
    grid = (triton.cdiv(em, bm) * triton.cdiv(n, cfg["BLOCK_SIZE_N"]),)
    fused_moe_kernel[grid](
        a, w, c, topk_weights, asg.sorted_ids, asg.expert_ids, asg.num_tokens_post_padded,
        n, k, em, num_valid,
        a.stride(0), a.stride(1), w.stride(0), w.stride(2), w.stride(1), c.stride(0), c.stride(1),
        naive_block_assignment=asg.sorted_ids is None,
        MUL_ROUTED_WEIGHT=topk_weights is not None, top_k=top_k, compute_type=tl.bfloat16,
        **cfg,
    )


def fused_experts(x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor, cfg13: dict, cfg2: dict) -> torch.Tensor:
    """Gated SiLU MoE with weighted w2 and top-k sum, using per-GEMM tiles."""
    t, topk = topk_ids.shape
    e, n13, _ = w13.shape
    hidden = w2.size(1)
    a = t * topk
    asg13 = prepare_assignment(topk_ids, cfg13["BLOCK_SIZE_M"], e)
    asg2 = asg13 if cfg2["BLOCK_SIZE_M"] == cfg13["BLOCK_SIZE_M"] else prepare_assignment(
        topk_ids, cfg2["BLOCK_SIZE_M"], e)
    c1 = torch.empty(a, n13, dtype=x.dtype, device=x.device)
    invoke(x, w13, c1, None, asg13, topk, a, cfg13)
    h = torch.nn.functional.silu(c1[:, : n13 // 2]) * c1[:, n13 // 2:]
    c3 = torch.empty(a, hidden, dtype=x.dtype, device=x.device)
    invoke(h, w2, c3, topk_weights.view(-1), asg2, 1, a, cfg2)
    return c3.view(t, topk, hidden).sum(dim=1)


def tuning_tiles(rows_per_expert: int) -> list[dict]:
    """Tiles for the first search round, before varying warps/stages."""
    rows = max(1, rows_per_expert)
    bms = [16, 32] if rows <= 16 else [16, 32, 64] if rows <= 64 else [32, 64, 128]
    out = []
    for bm in bms:
        for bn in (32, 64, 128, 256) if bm <= 32 else (64, 128, 256):
            for bk in (64, 128, 256) if bm <= 32 else (32, 64, 128):
                if bm * bk + bk * bn > 128 * 256 + 256 * 64:  # operand tiles of one stage
                    continue
                out.append({"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
                            "GROUP_SIZE_M": 1 if rows <= 128 else 16, "num_warps": 4, "num_stages": 3})
    return out
