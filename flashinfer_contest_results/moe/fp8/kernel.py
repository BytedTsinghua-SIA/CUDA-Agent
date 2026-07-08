"""MoE FP8 VibeCUDA level5 Triton kernel。

功能：提供 FlashInfer-Bench 兼容的 `moe_forward` 入口，用于
`moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`。
参数：由官方 workload 传入 routing logits/bias、hidden states、FP8 weights/scales、
local expert offset 和 routed scaling factor。
"""

import torch
import triton
import triton.language as tl
from typing import Optional

NUM_EXPERTS_GLOBAL = 256
NUM_LOCAL_EXPERTS = 32
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
SCALE_BLOCK = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4

_EPILOGUE_SUBTILE = 1
_GEMM1_BN = 128 // _EPILOGUE_SUBTILE
_GEMM2_BK = _GEMM1_BN
_INTER_SCALE_BLOCKS = INTERMEDIATE_SIZE // _GEMM1_BN


def _ensure_tma_allocator():
    if not hasattr(_ensure_tma_allocator, '_done'):
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)
        triton.set_allocator(alloc_fn)
        _ensure_tma_allocator._done = True


# ============== ROUTING KERNEL ==============

@triton.jit
def _fused_routing_kernel(
    logits_ptr, bias_ptr,
    topk_idx_ptr, topk_weights_ptr,
    perm_buf_ptr, counts_ptr,
    T, routed_scaling_factor,
    offset, num_local,
    E: tl.constexpr, N_GRP: tl.constexpr,
    GRP_SIZE: tl.constexpr, TOP_K_C: tl.constexpr,
    TOPK_GRP: tl.constexpr,
    PERM_BUF_SIZE: tl.constexpr,
    USE_PDL: tl.constexpr,
    COUNT_LOCAL: tl.constexpr = False,
    PDL_LAUNCH: tl.constexpr = False,
):
    tid = tl.program_id(0)

    if USE_PDL:
        if tid == 0:
            buf_offs = tl.arange(0, PERM_BUF_SIZE)
            tl.store(perm_buf_ptr + buf_offs, tl.zeros([PERM_BUF_SIZE], dtype=tl.int32))

    offs = tl.arange(0, E)
    logits = tl.load(logits_ptr + tid * E + offs).to(tl.float32)
    bias = tl.load(bias_ptr + offs).to(tl.float32)

    s = tl.sigmoid(logits)
    s_wb = s + bias

    NEG_INF: tl.constexpr = -1e30
    group_id = offs // GRP_SIZE

    group_score_vec = tl.full([E], NEG_INF, dtype=tl.float32)
    for g in tl.static_range(N_GRP):
        g_mask = (group_id == g)
        g_vals = tl.where(g_mask, s_wb, NEG_INF)
        m1 = tl.max(g_vals)
        is_m1 = (g_vals == m1)
        cum1 = tl.cumsum(is_m1.to(tl.int32))
        first_m1 = is_m1 & (cum1 == 1)
        g_vals2 = tl.where(first_m1, NEG_INF, g_vals)
        m2 = tl.max(g_vals2)
        score = m1 + m2
        group_score_vec = tl.where(g_mask, score, group_score_vec)

    selected = tl.zeros([E], dtype=tl.int32)
    gs = group_score_vec + tl.zeros([E], dtype=tl.float32)
    for _ in tl.static_range(TOPK_GRP):
        best = tl.max(gs)
        is_best = (gs == best)
        best_gid = tl.min(tl.where(is_best, group_id, N_GRP))
        sel_mask = (group_id == best_gid)
        selected = tl.where(sel_mask, 1, selected)
        gs = tl.where(sel_mask, NEG_INF, gs)

    candidates = tl.where(selected == 1, s_wb, NEG_INF)

    out_base = tid * TOP_K_C
    weight_sum = 0.0
    for k in tl.static_range(TOP_K_C):
        best_val = tl.max(candidates)
        is_best = (candidates == best_val)
        best_idx = tl.min(tl.where(is_best, offs, E))
        best_s = tl.sum(tl.where(offs == best_idx, s, 0.0))
        tl.store(topk_idx_ptr + out_base + k, best_idx.to(tl.int32))
        tl.store(topk_weights_ptr + out_base + k, best_s)
        weight_sum += best_s
        candidates = tl.where(offs == best_idx, NEG_INF, candidates)

    if COUNT_LOCAL:
        k_range = tl.arange(0, TOP_K_C)
        eidx = tl.load(topk_idx_ptr + out_base + k_range)
        lid = eidx - offset
        local_mask = (lid >= 0) & (lid < num_local)
        tl.atomic_add(counts_ptr + tl.maximum(lid, 0), 1, mask=local_mask)

    inv_sum = routed_scaling_factor / (weight_sum + 1e-20)
    for k in tl.static_range(TOP_K_C):
        w = tl.load(topk_weights_ptr + out_base + k)
        tl.store(topk_weights_ptr + out_base + k, w * inv_sum)

    if USE_PDL or PDL_LAUNCH:
        tl.extra.cuda.gdc_launch_dependents()


# ============== FUSED ROUTING+PERM (single kernel, atomic barrier) ==============

@triton.jit
def _fused_routing_perm_kernel(
    logits_ptr, bias_ptr,
    topk_idx_ptr, topk_weights_ptr,
    expert_offsets_ptr, perm_ptr, sorted_tokens_ptr, inv_perm_ptr,
    sched_exp_ptr, sched_mstart_ptr, sched_count_ptr,
    counters_ptr, arrive_ptr,
    expert_counts_ptr,
    T, TK_actual, routed_scaling_factor,
    offset, num_local, max_sched, arrive_base,
    E: tl.constexpr, N_GRP: tl.constexpr,
    GRP_SIZE: tl.constexpr, TOP_K_C: tl.constexpr,
    TOPK_GRP: tl.constexpr,
    K: tl.constexpr, PERM_BLOCK: tl.constexpr, NUM_LOCAL: tl.constexpr,
    SCHED_BM: tl.constexpr,
    PDL_LAUNCH: tl.constexpr = False,
    PARALLEL_PERM: tl.constexpr = False,
):
    tid = tl.program_id(0)

    # ---- Phase 1: Routing (one token per block) ----
    NEG_INF: tl.constexpr = -1e30
    offs = tl.arange(0, E)
    logits = tl.load(logits_ptr + tid * E + offs).to(tl.float32)
    bias = tl.load(bias_ptr + offs).to(tl.float32)

    s = tl.sigmoid(logits)
    s_wb = s + bias

    group_id = offs // GRP_SIZE

    group_score_vec = tl.full([E], NEG_INF, dtype=tl.float32)
    for g in tl.static_range(N_GRP):
        g_mask = (group_id == g)
        g_vals = tl.where(g_mask, s_wb, NEG_INF)
        m1 = tl.max(g_vals)
        is_m1 = (g_vals == m1)
        cum1 = tl.cumsum(is_m1.to(tl.int32))
        first_m1 = is_m1 & (cum1 == 1)
        g_vals2 = tl.where(first_m1, NEG_INF, g_vals)
        m2 = tl.max(g_vals2)
        score = m1 + m2
        group_score_vec = tl.where(g_mask, score, group_score_vec)

    selected = tl.zeros([E], dtype=tl.int32)
    gs = group_score_vec + tl.zeros([E], dtype=tl.float32)
    for _ in tl.static_range(TOPK_GRP):
        best = tl.max(gs)
        is_best = (gs == best)
        best_gid = tl.min(tl.where(is_best, group_id, N_GRP))
        sel_mask = (group_id == best_gid)
        selected = tl.where(sel_mask, 1, selected)
        gs = tl.where(sel_mask, NEG_INF, gs)

    candidates = tl.where(selected == 1, s_wb, NEG_INF)

    out_base = tid * TOP_K_C
    weight_sum = 0.0
    for k in tl.static_range(TOP_K_C):
        best_val = tl.max(candidates)
        is_best = (candidates == best_val)
        best_idx = tl.min(tl.where(is_best, offs, E))
        best_s = tl.sum(tl.where(offs == best_idx, s, 0.0))
        tl.store(topk_idx_ptr + out_base + k, best_idx.to(tl.int32))
        tl.store(topk_weights_ptr + out_base + k, best_s)
        weight_sum += best_s
        candidates = tl.where(offs == best_idx, NEG_INF, candidates)

    inv_sum = routed_scaling_factor / (weight_sum + 1e-20)
    for k in tl.static_range(TOP_K_C):
        w = tl.load(topk_weights_ptr + out_base + k)
        tl.store(topk_weights_ptr + out_base + k, w * inv_sum)

    if PARALLEL_PERM:
        # ---- Phase 0: CTA 0 zeros expert_counts in-kernel (avoids host zero_ launch) ----
        if tid == 0:
            e_zero = tl.arange(0, NUM_LOCAL)
            tl.store(expert_counts_ptr + e_zero, tl.zeros([NUM_LOCAL], dtype=tl.int32))
            tl.debug_barrier()
            tl.atomic_add(arrive_ptr, 1)
        while tl.atomic_add(arrive_ptr, 0) < arrive_base + 1:
            pass

        # ---- Phase 1: all CTAs count, then last CTA does prefix sum ----
        k_range = tl.arange(0, TOP_K_C)
        eidx_vec = tl.load(topk_idx_ptr + out_base + k_range)
        lid_vec = eidx_vec - offset
        local_mask = (lid_vec >= 0) & (lid_vec < num_local)
        safe_lid = tl.maximum(lid_vec, 0)
        tl.atomic_add(expert_counts_ptr + safe_lid, 1, mask=local_mask)
        tl.store(inv_perm_ptr + out_base + k_range, tl.full([TOP_K_C], -1, dtype=tl.int32))

        arrived = tl.atomic_add(arrive_ptr, 1) + 1

        if arrived == arrive_base + 1 + T:
            cumulative = 0
            sched_idx = 0
            for e in tl.static_range(NUM_LOCAL):
                count_e = tl.load(expert_counts_ptr + e)
                start_e = cumulative
                tl.store(expert_offsets_ptr + e, start_e.to(tl.int32))
                cumulative = cumulative + count_e
                ntiles = (count_e + SCHED_BM - 1) // SCHED_BM
                for _t in range(ntiles):
                    if sched_idx < max_sched:
                        tl.store(sched_exp_ptr + sched_idx, e)
                        tl.store(sched_mstart_ptr + sched_idx, start_e + _t * SCHED_BM)
                    sched_idx += 1
            tl.store(expert_offsets_ptr + NUM_LOCAL, cumulative.to(tl.int32))
            tl.store(sched_count_ptr, sched_idx)
            e_range = tl.arange(0, NUM_LOCAL)
            tl.store(counters_ptr + e_range, tl.zeros([NUM_LOCAL], dtype=tl.int32))
            tl.debug_barrier()
            tl.atomic_add(arrive_ptr, 1)

        while tl.atomic_add(arrive_ptr, 0) < arrive_base + T + 2:
            pass

        base_vec = tl.load(expert_offsets_ptr + safe_lid, mask=local_mask, other=0)
        pos_vec = tl.atomic_add(counters_ptr + safe_lid, 1, mask=local_mask)
        out_pos = base_vec + pos_vec
        out_pos_safe = tl.maximum(out_pos, 0)
        perm_val = (tid * TOP_K_C + k_range).to(tl.int32)
        tl.store(perm_ptr + out_pos_safe, perm_val, mask=local_mask)
        tl.store(sorted_tokens_ptr + out_pos_safe, tl.full([TOP_K_C], tid, dtype=tl.int32), mask=local_mask)
        tl.store(inv_perm_ptr + out_base + k_range, out_pos.to(tl.int32), mask=local_mask)

        if PDL_LAUNCH:
            tl.extra.cuda.gdc_launch_dependents()
    else:
        # ---- Original serial perm: last block does all permutation ----
        arrived = tl.atomic_add(arrive_ptr, 1) + 1

        if arrived == arrive_base + T:
            p_offs = tl.arange(0, PERM_BLOCK)
            p_mask = p_offs < TK_actual

            expert_ids = tl.load(topk_idx_ptr + p_offs, mask=p_mask, other=-1)
            local_ids = expert_ids - offset
            valid = (local_ids >= 0) & (local_ids < num_local) & p_mask

            token_ids = (p_offs // K).to(tl.int32)
            slot_ids = (p_offs % K).to(tl.int32)
            perm_val = token_ids * K + slot_ids

            tl.store(inv_perm_ptr + p_offs, tl.full([PERM_BLOCK], -1, dtype=tl.int32), mask=p_mask)

            cumulative = 0
            sched_idx = 0
            for e in tl.static_range(NUM_LOCAL):
                is_e = (local_ids == e) & valid
                count_e = tl.sum(is_e.to(tl.int32))
                start_e = cumulative
                tl.store(expert_offsets_ptr + e, start_e.to(tl.int32))
                cumulative = cumulative + count_e
                ntiles = (count_e + SCHED_BM - 1) // SCHED_BM
                for _t in range(ntiles):
                    if sched_idx < max_sched:
                        tl.store(sched_exp_ptr + sched_idx, e)
                        tl.store(sched_mstart_ptr + sched_idx, start_e + _t * SCHED_BM)
                    sched_idx += 1

            tl.store(expert_offsets_ptr + NUM_LOCAL, cumulative.to(tl.int32))
            tl.store(sched_count_ptr, sched_idx)

            e_range = tl.arange(0, NUM_LOCAL)
            tl.store(counters_ptr + e_range, tl.zeros([NUM_LOCAL], dtype=tl.int32))
            tl.debug_barrier()

            base = tl.load(expert_offsets_ptr + local_ids, mask=valid, other=0)
            pos = tl.atomic_add(counters_ptr + local_ids, 1, mask=valid)
            out_pos = base + pos

            tl.store(perm_ptr + out_pos, perm_val, mask=valid)
            tl.store(sorted_tokens_ptr + out_pos, token_ids, mask=valid)
            tl.store(inv_perm_ptr + p_offs, out_pos.to(tl.int32), mask=valid)

            if PDL_LAUNCH:
                tl.extra.cuda.gdc_launch_dependents()


# ============== FUSED PERMUTATION (single block for small T) ==============

@triton.jit
def _fused_perm_small(
    topk_idx_ptr,
    expert_offsets_ptr, perm_ptr, sorted_tokens_ptr, inv_perm_ptr,
    sched_exp_ptr, sched_mstart_ptr, sched_count_ptr,
    counters_ptr,
    TK_actual, offset, num_local, max_sched,
    K: tl.constexpr, BLOCK: tl.constexpr, NUM_LOCAL: tl.constexpr,
    SCHED_BM: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    offs = tl.arange(0, BLOCK)
    mask = offs < TK_actual

    expert_ids = tl.load(topk_idx_ptr + offs, mask=mask, other=-1)
    local_ids = expert_ids - offset
    valid = (local_ids >= 0) & (local_ids < num_local) & mask

    token_ids = (offs // K).to(tl.int32)
    slot_ids = (offs % K).to(tl.int32)
    perm_val = token_ids * K + slot_ids

    tl.store(inv_perm_ptr + offs, tl.full([BLOCK], -1, dtype=tl.int32), mask=mask)

    cumulative = 0
    sched_idx = 0
    for e in tl.static_range(NUM_LOCAL):
        is_e = (local_ids == e) & valid
        count_e = tl.sum(is_e.to(tl.int32))
        start_e = cumulative
        tl.store(expert_offsets_ptr + e, start_e.to(tl.int32))
        cumulative = cumulative + count_e
        ntiles = (count_e + SCHED_BM - 1) // SCHED_BM
        for _t in range(ntiles):
            if sched_idx < max_sched:
                tl.store(sched_exp_ptr + sched_idx, e)
                tl.store(sched_mstart_ptr + sched_idx, start_e + _t * SCHED_BM)
            sched_idx += 1

    tl.store(expert_offsets_ptr + NUM_LOCAL, cumulative.to(tl.int32))
    tl.store(sched_count_ptr, sched_idx)

    e_range = tl.arange(0, NUM_LOCAL)
    tl.store(counters_ptr + e_range, tl.zeros([NUM_LOCAL], dtype=tl.int32))
    tl.debug_barrier()

    base = tl.load(expert_offsets_ptr + local_ids, mask=valid, other=0)
    pos = tl.atomic_add(counters_ptr + local_ids, 1, mask=valid)
    out_pos = base + pos

    tl.store(perm_ptr + out_pos, perm_val, mask=valid)
    tl.store(sorted_tokens_ptr + out_pos, token_ids, mask=valid)
    tl.store(inv_perm_ptr + offs, out_pos.to(tl.int32), mask=valid)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# ============== PERMUTATION KERNELS (for large T) ==============

@triton.jit
def _count_local_experts(
    topk_idx_ptr, counts_ptr,
    TK, offset, num_local,
    BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TK
    expert_ids = tl.load(topk_idx_ptr + offs, mask=mask, other=-1)
    local_ids = expert_ids - offset
    valid = (local_ids >= 0) & (local_ids < num_local) & mask
    tl.atomic_add(counts_ptr + local_ids, 1, mask=valid)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _scatter_permutation(
    topk_idx_ptr, offsets_ptr, counters_ptr,
    perm_ptr, sorted_tokens_ptr, inv_perm_ptr,
    T, K: tl.constexpr, offset, num_local,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    TK = T * K
    mask = offs < TK

    expert_ids = tl.load(topk_idx_ptr + offs, mask=mask, other=-1)
    local_ids = expert_ids - offset
    valid = (local_ids >= 0) & (local_ids < num_local) & mask

    base = tl.load(offsets_ptr + local_ids, mask=valid, other=0)
    pos = tl.atomic_add(counters_ptr + local_ids, 1, mask=valid)
    out_pos = base + pos

    token_ids = (offs // K).to(tl.int32)
    slot_ids = (offs % K).to(tl.int32)
    perm_val = token_ids * K + slot_ids

    tl.store(perm_ptr + out_pos, perm_val, mask=valid)
    tl.store(sorted_tokens_ptr + out_pos, token_ids, mask=valid)
    tl.store(inv_perm_ptr + offs, out_pos.to(tl.int32), mask=valid)
    tl.store(inv_perm_ptr + offs, -1, mask=~valid & mask)


@triton.jit
def _scatter_perm_fused(
    topk_idx_ptr, counts_ptr, counters_ptr,
    expert_offsets_ptr,
    perm_ptr, sorted_tokens_ptr, inv_perm_ptr,
    sched_exp_ptr, sched_mstart_ptr, sched_count_ptr,
    T, K: tl.constexpr, offset, num_local, max_sched,
    BLOCK: tl.constexpr, NUM_LOCAL: tl.constexpr,
    SCHED_BM: tl.constexpr = 64,
    EMIT_SCHED: tl.constexpr = False,
    USE_PDL: tl.constexpr = False,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    TK = T * K
    mask = offs < TK

    expert_ids = tl.load(topk_idx_ptr + offs, mask=mask, other=-1)
    local_ids = expert_ids - offset
    valid = (local_ids >= 0) & (local_ids < num_local) & mask

    base = tl.zeros([BLOCK], dtype=tl.int32)
    for e in tl.static_range(NUM_LOCAL):
        count_e = tl.load(counts_ptr + e)
        base += tl.where((local_ids > e) & valid, count_e, 0).to(tl.int32)

    if pid == 0:
        e_offs = tl.arange(0, NUM_LOCAL)
        counts_vec = tl.load(counts_ptr + e_offs)
        prefix = tl.cumsum(counts_vec) - counts_vec
        tl.store(expert_offsets_ptr + e_offs, prefix)
        tl.store(expert_offsets_ptr + NUM_LOCAL, tl.sum(counts_vec))
        if EMIT_SCHED:
            sched_idx = 0
            for _e in tl.static_range(NUM_LOCAL):
                start_e = tl.load(expert_offsets_ptr + _e)
                cnt_e = tl.load(counts_ptr + _e)
                ntiles = (cnt_e + SCHED_BM - 1) // SCHED_BM
                for _t in range(ntiles):
                    if sched_idx < max_sched:
                        tl.store(sched_exp_ptr + sched_idx, _e)
                        tl.store(sched_mstart_ptr + sched_idx, start_e + _t * SCHED_BM)
                    sched_idx += 1
            tl.store(sched_count_ptr, sched_idx)

    pos = tl.atomic_add(counters_ptr + local_ids, 1, mask=valid)
    out_pos = base + pos

    token_ids = (offs // K).to(tl.int32)
    slot_ids = (offs % K).to(tl.int32)
    perm_val = token_ids * K + slot_ids

    tl.store(perm_ptr + out_pos, perm_val, mask=valid)
    tl.store(sorted_tokens_ptr + out_pos, token_ids, mask=valid)
    tl.store(inv_perm_ptr + offs, out_pos.to(tl.int32), mask=valid)
    tl.store(inv_perm_ptr + offs, -1, mask=~valid & mask)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# ============== GATHER HIDDEN STATES ==============

@triton.jit
def _gather_hidden_kernel(
    h_fp8_ptr, h_scale_ptr, perm_ptr,
    a_fp8_ptr, a_scale_ptr,
    H, Kb, M_local,
    stride_h_t, stride_hs_k,
    stride_a_m, stride_as_k,
    BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    m = tl.program_id(0)
    h_pid = tl.program_id(1)
    if m >= M_local:
        return

    h_offs = h_pid * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    t = tl.load(perm_ptr + m).to(tl.int64)
    tl.store(a_fp8_ptr + m * stride_a_m + h_offs,
             tl.load(h_fp8_ptr + t * stride_h_t + h_offs, mask=h_mask),
             mask=h_mask)

    if h_pid == 0:
        k_offs = tl.arange(0, BLOCK_K)
        k_mask = k_offs < Kb
        tl.store(a_scale_ptr + k_offs * stride_as_k + m,
                 tl.load(h_scale_ptr + k_offs * stride_hs_k + t, mask=k_mask),
                 mask=k_mask)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _gather_hidden_gpu_bound(
    h_fp8_ptr, h_scale_ptr, perm_ptr,
    a_fp8_ptr, a_scale_ptr,
    expert_offsets_ptr,
    H, Kb,
    stride_h_t, stride_hs_k,
    stride_a_m, stride_as_k,
    num_local_experts,
    BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    m = tl.program_id(0)
    h_pid = tl.program_id(1)
    M_local = tl.load(expert_offsets_ptr + num_local_experts)
    if m >= M_local:
        return

    h_offs = h_pid * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    t = tl.load(perm_ptr + m).to(tl.int64)
    tl.store(a_fp8_ptr + m * stride_a_m + h_offs,
             tl.load(h_fp8_ptr + t * stride_h_t + h_offs, mask=h_mask),
             mask=h_mask)

    if h_pid == 0:
        k_offs = tl.arange(0, BLOCK_K)
        k_mask = k_offs < Kb
        tl.store(a_scale_ptr + k_offs * stride_as_k + m,
                 tl.load(h_scale_ptr + k_offs * stride_hs_k + t, mask=k_mask),
                 mask=k_mask)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# ============== SCATTER HIDDEN STATES (token-indexed) ==============

@triton.jit
def _scatter_hidden_kernel(
    h_fp8_ptr, h_scale_ptr,
    inv_perm_ptr,
    a_fp8_ptr, a_scale_ptr,
    H, Kb, T,
    stride_h_t, stride_hs_k,
    stride_a_m, stride_as_k,
    BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
    TOP_K: tl.constexpr,
    USE_PDL: tl.constexpr,
    WAIT_PDL: tl.constexpr = False,
):
    if WAIT_PDL:
        tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    h_pid = tl.program_id(1)
    if t >= T:
        return

    h_offs = h_pid * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    t_i64 = t.to(tl.int64)
    row = tl.load(h_fp8_ptr + t_i64 * stride_h_t + h_offs, mask=h_mask)

    for k in tl.static_range(TOP_K):
        pos = tl.load(inv_perm_ptr + t * TOP_K + k)
        is_local = (pos >= 0)
        pos_i64 = tl.maximum(pos, 0).to(tl.int64)
        tl.store(a_fp8_ptr + pos_i64 * stride_a_m + h_offs, row,
                 mask=h_mask & is_local)

    if h_pid == 0:
        k_offs = tl.arange(0, BLOCK_K)
        k_mask = k_offs < Kb
        scales = tl.load(h_scale_ptr + k_offs * stride_hs_k + t, mask=k_mask, other=0.0)
        for k in tl.static_range(TOP_K):
            pos = tl.load(inv_perm_ptr + t * TOP_K + k)
            is_local = (pos >= 0)
            pos_i64 = tl.maximum(pos, 0).to(tl.int64)
            tl.store(a_scale_ptr + k_offs * stride_as_k + pos_i64, scales,
                     mask=k_mask & is_local)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# ============== FUSED ROUTING+PERM+GEMM1 (single launch for small T) ==============

@triton.jit
def _fused_routing_perm_gemm1(
    # ---- Routing args ----
    logits_ptr, bias_ptr,
    topk_idx_ptr, topk_weights_ptr,
    expert_offsets_ptr, perm_ptr, sorted_tokens_ptr, inv_perm_ptr,
    sched_exp_ptr, sched_mstart_ptr, sched_count_ptr,
    counters_ptr, arrive_ptr,
    expert_counts_ptr,
    T, TK_actual, routed_scaling_factor,
    offset, num_local, max_sched, arrive_base,
    # ---- GEMM1 args ----
    H_ptr, H_scale_ptr,
    W1_ptr, C_fp8_ptr, C_scale_ptr,
    W1_scale_ptr,
    M_total, N_out, N_gemm,
    stride_w1_e, stride_h_scale_k,
    # ---- Constexprs ----
    E: tl.constexpr, N_GRP: tl.constexpr,
    GRP_SIZE: tl.constexpr, TOP_K_C: tl.constexpr,
    TOPK_GRP: tl.constexpr,
    PERM_BLOCK: tl.constexpr, NUM_LOCAL: tl.constexpr,
    SCHED_BM: tl.constexpr,
    PARALLEL_PERM: tl.constexpr,
    K_HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_BLOCKS: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    WARP_SPEC: tl.constexpr,
    USE_DOT_SCALED: tl.constexpr = False,
    PDL_LAUNCH: tl.constexpr = True,
):
    pid = tl.program_id(0)
    NEG_INF: tl.constexpr = -1e30

    # ===== Phase 1: Routing + Permutation (only pid < T) =====
    if pid < T:
        tid = pid
        offs = tl.arange(0, E)
        logits = tl.load(logits_ptr + tid * E + offs).to(tl.float32)
        bias = tl.load(bias_ptr + offs).to(tl.float32)
        s = tl.sigmoid(logits)
        s_wb = s + bias

        group_id = offs // GRP_SIZE
        group_score_vec = tl.full([E], NEG_INF, dtype=tl.float32)
        for g in tl.static_range(N_GRP):
            g_mask = (group_id == g)
            g_vals = tl.where(g_mask, s_wb, NEG_INF)
            m1 = tl.max(g_vals)
            is_m1 = (g_vals == m1)
            cum1 = tl.cumsum(is_m1.to(tl.int32))
            first_m1 = is_m1 & (cum1 == 1)
            g_vals2 = tl.where(first_m1, NEG_INF, g_vals)
            m2 = tl.max(g_vals2)
            score = m1 + m2
            group_score_vec = tl.where(g_mask, score, group_score_vec)

        selected = tl.zeros([E], dtype=tl.int32)
        gs = group_score_vec + tl.zeros([E], dtype=tl.float32)
        for _ in tl.static_range(TOPK_GRP):
            best = tl.max(gs)
            is_best = (gs == best)
            best_gid = tl.min(tl.where(is_best, group_id, N_GRP))
            sel_mask = (group_id == best_gid)
            selected = tl.where(sel_mask, 1, selected)
            gs = tl.where(sel_mask, NEG_INF, gs)

        candidates = tl.where(selected == 1, s_wb, NEG_INF)
        out_base = tid * TOP_K_C
        weight_sum = 0.0
        for k in tl.static_range(TOP_K_C):
            best_val = tl.max(candidates)
            is_best = (candidates == best_val)
            best_idx = tl.min(tl.where(is_best, offs, E))
            best_s = tl.sum(tl.where(offs == best_idx, s, 0.0))
            tl.store(topk_idx_ptr + out_base + k, best_idx.to(tl.int32))
            tl.store(topk_weights_ptr + out_base + k, best_s)
            weight_sum += best_s
            candidates = tl.where(offs == best_idx, NEG_INF, candidates)

        inv_sum = routed_scaling_factor / (weight_sum + 1e-20)
        for k in tl.static_range(TOP_K_C):
            w = tl.load(topk_weights_ptr + out_base + k)
            tl.store(topk_weights_ptr + out_base + k, w * inv_sum)

        if PARALLEL_PERM:
            if tid == 0:
                e_zero = tl.arange(0, NUM_LOCAL)
                tl.store(expert_counts_ptr + e_zero, tl.zeros([NUM_LOCAL], dtype=tl.int32))
                tl.debug_barrier()
                tl.atomic_add(arrive_ptr, 1)
            while tl.atomic_add(arrive_ptr, 0) < arrive_base + 1:
                pass

            k_range = tl.arange(0, TOP_K_C)
            eidx_vec = tl.load(topk_idx_ptr + out_base + k_range)
            lid_vec = eidx_vec - offset
            local_mask = (lid_vec >= 0) & (lid_vec < num_local)
            safe_lid = tl.maximum(lid_vec, 0)
            tl.atomic_add(expert_counts_ptr + safe_lid, 1, mask=local_mask)
            tl.store(inv_perm_ptr + out_base + k_range, tl.full([TOP_K_C], -1, dtype=tl.int32))

            arrived = tl.atomic_add(arrive_ptr, 1) + 1
            if arrived == arrive_base + 1 + T:
                cumulative = 0
                sched_idx = 0
                for e in tl.static_range(NUM_LOCAL):
                    count_e = tl.load(expert_counts_ptr + e)
                    start_e = cumulative
                    tl.store(expert_offsets_ptr + e, start_e.to(tl.int32))
                    cumulative = cumulative + count_e
                    ntiles = (count_e + SCHED_BM - 1) // SCHED_BM
                    for _t in range(ntiles):
                        if sched_idx < max_sched:
                            tl.store(sched_exp_ptr + sched_idx, e)
                            tl.store(sched_mstart_ptr + sched_idx, start_e + _t * SCHED_BM)
                        sched_idx += 1
                tl.store(expert_offsets_ptr + NUM_LOCAL, cumulative.to(tl.int32))
                tl.store(sched_count_ptr, sched_idx)
                e_range = tl.arange(0, NUM_LOCAL)
                tl.store(counters_ptr + e_range, tl.zeros([NUM_LOCAL], dtype=tl.int32))
                tl.debug_barrier()
                tl.atomic_add(arrive_ptr, 1)

            while tl.atomic_add(arrive_ptr, 0) < arrive_base + T + 2:
                pass

            base_vec = tl.load(expert_offsets_ptr + safe_lid, mask=local_mask, other=0)
            pos_vec = tl.atomic_add(counters_ptr + safe_lid, 1, mask=local_mask)
            out_pos = base_vec + pos_vec
            out_pos_safe = tl.maximum(out_pos, 0)
            perm_val = (tid * TOP_K_C + k_range).to(tl.int32)
            tl.store(perm_ptr + out_pos_safe, perm_val, mask=local_mask)
            tl.store(sorted_tokens_ptr + out_pos_safe, tl.full([TOP_K_C], tid, dtype=tl.int32), mask=local_mask)
            tl.store(inv_perm_ptr + out_base + k_range, out_pos.to(tl.int32), mask=local_mask)

            tl.debug_barrier()
            scatter_done = tl.atomic_add(arrive_ptr, 1) + 1
            if scatter_done == arrive_base + 2 * T + 2:
                tl.debug_barrier()
                tl.atomic_add(arrive_ptr, 1)
        else:
            arrived = tl.atomic_add(arrive_ptr, 1) + 1
            if arrived == arrive_base + T:
                p_offs = tl.arange(0, PERM_BLOCK)
                p_mask = p_offs < TK_actual
                expert_ids = tl.load(topk_idx_ptr + p_offs, mask=p_mask, other=-1)
                local_ids = expert_ids - offset
                valid = (local_ids >= 0) & (local_ids < num_local) & p_mask
                token_ids_p = (p_offs // TOP_K_C).to(tl.int32)
                slot_ids = (p_offs % TOP_K_C).to(tl.int32)
                perm_val_s = token_ids_p * TOP_K_C + slot_ids
                tl.store(inv_perm_ptr + p_offs, tl.full([PERM_BLOCK], -1, dtype=tl.int32), mask=p_mask)

                cumulative = 0
                sched_idx = 0
                for e in tl.static_range(NUM_LOCAL):
                    is_e = (local_ids == e) & valid
                    count_e = tl.sum(is_e.to(tl.int32))
                    start_e = cumulative
                    tl.store(expert_offsets_ptr + e, start_e.to(tl.int32))
                    cumulative = cumulative + count_e
                    ntiles = (count_e + SCHED_BM - 1) // SCHED_BM
                    for _t in range(ntiles):
                        if sched_idx < max_sched:
                            tl.store(sched_exp_ptr + sched_idx, e)
                            tl.store(sched_mstart_ptr + sched_idx, start_e + _t * SCHED_BM)
                        sched_idx += 1

                tl.store(expert_offsets_ptr + NUM_LOCAL, cumulative.to(tl.int32))
                tl.store(sched_count_ptr, sched_idx)
                e_range = tl.arange(0, NUM_LOCAL)
                tl.store(counters_ptr + e_range, tl.zeros([NUM_LOCAL], dtype=tl.int32))
                tl.debug_barrier()

                base_s = tl.load(expert_offsets_ptr + local_ids, mask=valid, other=0)
                pos_s = tl.atomic_add(counters_ptr + local_ids, 1, mask=valid)
                out_pos_s = base_s + pos_s
                tl.store(perm_ptr + out_pos_s, perm_val_s, mask=valid)
                tl.store(sorted_tokens_ptr + out_pos_s, token_ids_p, mask=valid)
                tl.store(inv_perm_ptr + p_offs, out_pos_s.to(tl.int32), mask=valid)

                tl.debug_barrier()
                tl.atomic_add(arrive_ptr, 1)

    # ===== Barrier: wait for routing+perm to complete =====
    if PARALLEL_PERM:
        _ARRIVE_DONE = arrive_base + 2 * T + 3
    else:
        _ARRIVE_DONE = arrive_base + T + 1
    while tl.atomic_add(arrive_ptr, 0) < _ARRIVE_DONE:
        pass

    # ===== Phase 2: GEMM1 (all CTAs, persistent) =====
    tl.assume(M_total >= 0)
    tl.assume(N_out > 0)
    tl.assume(N_gemm > 0)
    tl.assume(stride_w1_e >= 0)
    tl.assume(stride_h_scale_k > 0)
    tl.assume(num_local > 0)

    tile_idx = pid
    num_n_tiles = tl.cdiv(N_out, BLOCK_N)
    k_tiles = K_HIDDEN // BLOCK_K
    k_scale_blocks = K_HIDDEN // SCALE_BLOCK
    b_scale_expert_stride = (N_gemm // SCALE_BLOCK) * k_scale_blocks
    n_out_scale_blocks = N_out // SCALE_BLOCK
    MX_K: tl.constexpr = BLOCK_K // 32

    last_tile_end = 0

    for e in range(num_local):
        m_start = tl.load(expert_offsets_ptr + e)
        m_end = tl.load(expert_offsets_ptr + e + 1)
        M_e = m_end - m_start
        num_m_tiles = tl.cdiv(M_e, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles
        num_pid_in_group = GROUP_SIZE_M * num_n_tiles

        if tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
            b_w1_desc = tl.make_tensor_descriptor(
                W1_ptr + e * stride_w1_e,
                shape=[N_gemm, K_HIDDEN], strides=[K_HIDDEN, 1],
                block_shape=[BLOCK_N, BLOCK_K],
            )
            c_desc = tl.make_tensor_descriptor(
                C_fp8_ptr + m_start * N_out,
                shape=[M_e, N_out], strides=[N_out, 1],
                block_shape=[BLOCK_M, BLOCK_N],
            )

            while tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
                local_tile = tile_idx - last_tile_end
                group_id_g = local_tile // num_pid_in_group
                first_pid_m = group_id_g * GROUP_SIZE_M
                group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
                local_in_group = local_tile % num_pid_in_group
                tile_m_idx = first_pid_m + (local_in_group % group_size_m)
                tile_n_idx = local_in_group // group_size_m
                offs_m = tile_m_idx * BLOCK_M
                offs_n = tile_n_idx * BLOCK_N

                acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                offs_row = m_start + offs_m + tl.arange(0, BLOCK_M)
                row_mask = offs_row < m_end

                token_ids_g = tl.load(
                    sorted_tokens_ptr + offs_row,
                    mask=row_mask, other=0,
                )
                token_h_base = (token_ids_g * K_HIDDEN).to(tl.int64)
                token_s_base = H_scale_ptr + token_ids_g

                n_scale_up = offs_n // SCALE_BLOCK
                n_scale_gate = n_out_scale_blocks + n_scale_up
                b_scale_up_base = (W1_scale_ptr + e * b_scale_expert_stride
                                   + n_scale_up * k_scale_blocks)
                b_scale_gate_base = (W1_scale_ptr + e * b_scale_expert_stride
                                     + n_scale_gate * k_scale_blocks)

                for ki in tl.range(k_tiles, warp_specialize=WARP_SPEC and not USE_DOT_SCALED):
                    a_s = tl.load(
                        token_s_base + ki * stride_h_scale_k,
                        mask=row_mask, other=0.0,
                    )
                    bs_up = tl.load(b_scale_up_base + ki)
                    bs_gate = tl.load(b_scale_gate_base + ki)

                    a_tile = tl.load(
                        H_ptr + token_h_base[:, None] + (ki * BLOCK_K + tl.arange(0, BLOCK_K))[None, :],
                        mask=row_mask[:, None],
                        other=0.0,
                    )
                    b_up_tile = b_w1_desc.load([offs_n, ki * BLOCK_K])
                    b_gate_tile = b_w1_desc.load([N_out + offs_n, ki * BLOCK_K])

                    if USE_DOT_SCALED:
                        comb_up = a_s * bs_up
                        comb_gate = a_s * bs_gate
                        up_bits = comb_up.to(tl.int32, bitcast=True)
                        gate_bits = comb_gate.to(tl.int32, bitcast=True)
                        up_e8m0 = ((up_bits >> 23) & 0xFF).to(tl.uint8)
                        gate_e8m0 = ((gate_bits >> 23) & 0xFF).to(tl.uint8)
                        a_scale_up = tl.broadcast_to(tl.expand_dims(up_e8m0, 1), (BLOCK_M, MX_K))
                        a_scale_gate = tl.broadcast_to(tl.expand_dims(gate_e8m0, 1), (BLOCK_M, MX_K))
                        b_unit = tl.full((BLOCK_N, MX_K), 127, dtype=tl.uint8)
                        acc_up = tl.dot_scaled(a_tile, a_scale_up, "e4m3",
                                               b_up_tile.T, b_unit, "e4m3",
                                               acc=acc_up, fast_math=True)
                        acc_gate = tl.dot_scaled(a_tile, a_scale_gate, "e4m3",
                                                 b_gate_tile.T, b_unit, "e4m3",
                                                 acc=acc_gate, fast_math=True)
                    else:
                        dot_up = tl.dot(a_tile, b_up_tile.T)
                        dot_gate = tl.dot(a_tile, b_gate_tile.T)
                        acc_up += dot_up * (a_s * bs_up)[:, None]
                        acc_gate += dot_gate * (a_s * bs_gate)[:, None]

                silu_gate = acc_gate / (1.0 + tl.exp(-acc_gate))
                result = silu_gate * acc_up

                row_abs_max = tl.max(tl.abs(result), axis=1)
                scale = row_abs_max / 448.0
                scale = tl.maximum(scale, 1e-12)
                quantized = result / scale[:, None]
                c_desc.store([offs_m, offs_n], quantized.to(C_fp8_ptr.dtype.element_ty))

                k_block = tile_n_idx
                tl.store(
                    C_scale_ptr + k_block * M_total + offs_row,
                    scale, mask=row_mask,
                )
                tile_idx += NUM_BLOCKS

        last_tile_end = last_tile_end + num_tiles

    if PDL_LAUNCH:
        tl.extra.cuda.gdc_launch_dependents()


# ============== GEMM1 + SwiGLU + Fused FP8 Quantize (fused gather path) ==============

@triton.jit
def _grouped_gemm1_fused_gather(
    H_ptr, H_scale_ptr, sorted_token_ids_ptr,
    W1_ptr, C_fp8_ptr, C_scale_ptr,
    W1_scale_ptr, expert_offsets_ptr,
    M_total, N_out, N_gemm,
    stride_w1_e, stride_h_scale_k,
    num_experts,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_BLOCKS: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    PDL_LAUNCH: tl.constexpr,
    WARP_SPEC: tl.constexpr = True,
    USE_DOT_SCALED: tl.constexpr = False,
    EPILOGUE_SUBTILE: tl.constexpr = 1,
):
    tl.assume(M_total >= 0)
    tl.assume(N_out > 0)
    tl.assume(N_gemm > 0)
    tl.assume(stride_w1_e >= 0)
    tl.assume(stride_h_scale_k > 0)
    tl.assume(num_experts > 0)

    pid = tl.program_id(0)
    tile_idx = pid
    num_n_tiles = tl.cdiv(N_out, BLOCK_N)
    k_tiles = K // BLOCK_K
    k_scale_blocks = K // SCALE_BLOCK
    b_scale_expert_stride = (N_gemm // SCALE_BLOCK) * k_scale_blocks
    n_out_scale_blocks = N_out // SCALE_BLOCK
    MX_K: tl.constexpr = BLOCK_K // 32

    last_tile_end = 0

    if EPILOGUE_SUBTILE == 1:
        for e in range(num_experts):
            m_start = tl.load(expert_offsets_ptr + e)
            m_end = tl.load(expert_offsets_ptr + e + 1)
            M_e = m_end - m_start
            num_m_tiles = tl.cdiv(M_e, BLOCK_M)
            num_tiles = num_m_tiles * num_n_tiles
            num_pid_in_group = GROUP_SIZE_M * num_n_tiles

            if tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
                b_w1_desc = tl.make_tensor_descriptor(
                    W1_ptr + e * stride_w1_e,
                    shape=[N_gemm, K], strides=[K, 1],
                    block_shape=[BLOCK_N, BLOCK_K],
                )
                c_desc = tl.make_tensor_descriptor(
                    C_fp8_ptr + m_start * N_out,
                    shape=[M_e, N_out], strides=[N_out, 1],
                    block_shape=[BLOCK_M, BLOCK_N],
                )

                while tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
                    local_tile = tile_idx - last_tile_end
                    group_id = local_tile // num_pid_in_group
                    first_pid_m = group_id * GROUP_SIZE_M
                    group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
                    local_in_group = local_tile % num_pid_in_group
                    tile_m_idx = first_pid_m + (local_in_group % group_size_m)
                    tile_n_idx = local_in_group // group_size_m
                    offs_m = tile_m_idx * BLOCK_M
                    offs_n = tile_n_idx * BLOCK_N

                    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                    offs_row = m_start + offs_m + tl.arange(0, BLOCK_M)
                    row_mask = offs_row < m_end

                    token_ids = tl.load(
                        sorted_token_ids_ptr + offs_row,
                        mask=row_mask, other=0,
                    )
                    token_h_base = (token_ids * K).to(tl.int64)
                    token_s_base = H_scale_ptr + token_ids

                    n_scale_up = offs_n // SCALE_BLOCK
                    n_scale_gate = n_out_scale_blocks + n_scale_up
                    b_scale_up_base = (W1_scale_ptr + e * b_scale_expert_stride
                                       + n_scale_up * k_scale_blocks)
                    b_scale_gate_base = (W1_scale_ptr + e * b_scale_expert_stride
                                         + n_scale_gate * k_scale_blocks)

                    for ki in tl.range(k_tiles, warp_specialize=WARP_SPEC and not USE_DOT_SCALED):
                        a_s = tl.load(
                            token_s_base + ki * stride_h_scale_k,
                            mask=row_mask, other=0.0,
                        )
                        bs_up = tl.load(b_scale_up_base + ki)
                        bs_gate = tl.load(b_scale_gate_base + ki)

                        a_tile = tl.load(
                            H_ptr + token_h_base[:, None] + (ki * BLOCK_K + tl.arange(0, BLOCK_K))[None, :],
                            mask=row_mask[:, None],
                            other=0.0,
                        )
                        b_up_tile = b_w1_desc.load([offs_n, ki * BLOCK_K])
                        b_gate_tile = b_w1_desc.load([N_out + offs_n, ki * BLOCK_K])

                        if USE_DOT_SCALED:
                            comb_up = a_s * bs_up
                            comb_gate = a_s * bs_gate
                            up_bits = comb_up.to(tl.int32, bitcast=True)
                            gate_bits = comb_gate.to(tl.int32, bitcast=True)
                            up_e8m0 = ((up_bits >> 23) & 0xFF).to(tl.uint8)
                            gate_e8m0 = ((gate_bits >> 23) & 0xFF).to(tl.uint8)
                            a_scale_up = tl.broadcast_to(tl.expand_dims(up_e8m0, 1), (BLOCK_M, MX_K))
                            a_scale_gate = tl.broadcast_to(tl.expand_dims(gate_e8m0, 1), (BLOCK_M, MX_K))
                            b_unit = tl.full((BLOCK_N, MX_K), 127, dtype=tl.uint8)
                            acc_up = tl.dot_scaled(a_tile, a_scale_up, "e4m3",
                                                   b_up_tile.T, b_unit, "e4m3",
                                                   acc=acc_up, fast_math=True)
                            acc_gate = tl.dot_scaled(a_tile, a_scale_gate, "e4m3",
                                                     b_gate_tile.T, b_unit, "e4m3",
                                                     acc=acc_gate, fast_math=True)
                        else:
                            dot_up = tl.dot(a_tile, b_up_tile.T)
                            dot_gate = tl.dot(a_tile, b_gate_tile.T)
                            acc_up += dot_up * (a_s * bs_up)[:, None]
                            acc_gate += dot_gate * (a_s * bs_gate)[:, None]

                    silu_gate = acc_gate / (1.0 + tl.exp(-acc_gate))
                    result = silu_gate * acc_up

                    row_abs_max = tl.max(tl.abs(result), axis=1)
                    scale = row_abs_max / 448.0
                    scale = tl.maximum(scale, 1e-12)
                    quantized = result / scale[:, None]
                    c_desc.store([offs_m, offs_n], quantized.to(C_fp8_ptr.dtype.element_ty))

                    k_block = tile_n_idx
                    tl.store(
                        C_scale_ptr + k_block * M_total + offs_row,
                        scale, mask=row_mask,
                    )
                    tile_idx += NUM_BLOCKS

            last_tile_end = last_tile_end + num_tiles
    else:
        SUB_N: tl.constexpr = BLOCK_N // EPILOGUE_SUBTILE

        for e in range(num_experts):
            m_start = tl.load(expert_offsets_ptr + e)
            m_end = tl.load(expert_offsets_ptr + e + 1)
            M_e = m_end - m_start
            num_m_tiles = tl.cdiv(M_e, BLOCK_M)
            num_tiles = num_m_tiles * num_n_tiles
            num_pid_in_group = GROUP_SIZE_M * num_n_tiles

            if tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
                b_w1_desc_full = tl.make_tensor_descriptor(
                    W1_ptr + e * stride_w1_e,
                    shape=[N_gemm, K], strides=[K, 1],
                    block_shape=[BLOCK_N, BLOCK_K],
                )
                c_desc_full = tl.make_tensor_descriptor(
                    C_fp8_ptr + m_start * N_out,
                    shape=[M_e, N_out], strides=[N_out, 1],
                    block_shape=[BLOCK_M, BLOCK_N],
                )

                while tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
                    local_tile = tile_idx - last_tile_end
                    group_id = local_tile // num_pid_in_group
                    first_pid_m = group_id * GROUP_SIZE_M
                    group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
                    local_in_group = local_tile % num_pid_in_group
                    tile_m_idx = first_pid_m + (local_in_group % group_size_m)
                    tile_n_idx = local_in_group // group_size_m
                    offs_m = tile_m_idx * BLOCK_M
                    offs_n = tile_n_idx * BLOCK_N

                    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                    offs_row = m_start + offs_m + tl.arange(0, BLOCK_M)
                    row_mask = offs_row < m_end

                    token_ids = tl.load(
                        sorted_token_ids_ptr + offs_row,
                        mask=row_mask, other=0,
                    )
                    token_h_base = (token_ids * K).to(tl.int64)
                    token_s_base = H_scale_ptr + token_ids

                    n_scale_up = offs_n // SCALE_BLOCK
                    n_scale_gate = n_out_scale_blocks + n_scale_up
                    b_scale_up_base = (W1_scale_ptr + e * b_scale_expert_stride
                                       + n_scale_up * k_scale_blocks)
                    b_scale_gate_base = (W1_scale_ptr + e * b_scale_expert_stride
                                         + n_scale_gate * k_scale_blocks)

                    for ki in tl.range(k_tiles, warp_specialize=WARP_SPEC):
                        a_s = tl.load(
                            token_s_base + ki * stride_h_scale_k,
                            mask=row_mask, other=0.0,
                        )
                        bs_up = tl.load(b_scale_up_base + ki)
                        bs_gate = tl.load(b_scale_gate_base + ki)

                        a_tile = tl.load(
                            H_ptr + token_h_base[:, None] + (ki * BLOCK_K + tl.arange(0, BLOCK_K))[None, :],
                            mask=row_mask[:, None],
                            other=0.0,
                        )
                        b_up_tile = b_w1_desc_full.load([offs_n, ki * BLOCK_K])
                        b_gate_tile = b_w1_desc_full.load([N_out + offs_n, ki * BLOCK_K])

                        scale_up = (a_s * bs_up)[:, None]
                        scale_gate = (a_s * bs_gate)[:, None]

                        acc_up += tl.dot(a_tile, b_up_tile.T) * scale_up
                        acc_gate += tl.dot(a_tile, b_gate_tile.T) * scale_gate

                    silu_gate = acc_gate / (1.0 + tl.exp(-acc_gate))
                    result = silu_gate * acc_up

                    cols_sf = tl.arange(0, BLOCK_N)
                    result_abs = tl.abs(result)
                    scale_map = tl.full([BLOCK_M, BLOCK_N], 1e-12, dtype=tl.float32)
                    for _sub in tl.static_range(EPILOGUE_SUBTILE):
                        _lo = _sub * SUB_N
                        _hi = _lo + SUB_N
                        _smask = (cols_sf[None, :] >= _lo) & (cols_sf[None, :] < _hi)
                        _sabs = tl.where(_smask, result_abs, 0.0)
                        _smax = tl.max(_sabs, axis=1)
                        _sscale = tl.maximum(_smax / 448.0, 1e-12)
                        scale_map = tl.where(_smask, _sscale[:, None], scale_map)
                        tl.store(C_scale_ptr + (tile_n_idx * EPILOGUE_SUBTILE + _sub) * M_total + offs_row,
                                 _sscale, mask=row_mask)
                    quantized = result / scale_map
                    c_desc_full.store([offs_m, offs_n], quantized.to(C_fp8_ptr.dtype.element_ty))

                    tile_idx += NUM_BLOCKS

            last_tile_end = last_tile_end + num_tiles

    if PDL_LAUNCH:
        tl.extra.cuda.gdc_launch_dependents()


# ============== GEMM1 scheduled gather (flat 2D grid, sched_exp/sched_mstart) ==============

@triton.jit
def _grouped_gemm1_scheduled_gather(
    H_ptr, H_scale_ptr, sorted_token_ids_ptr,
    W1_ptr, C_fp8_ptr, C_scale_ptr,
    W1_scale_ptr,
    sched_exp_ptr, sched_mstart_ptr, sched_count_ptr,
    expert_offsets_ptr,
    M_total, N_out, N_gemm,
    stride_w1_e, stride_h_scale_k,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_KB: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    PDL_LAUNCH: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
):
    tl.assume(M_total >= 0)
    tl.assume(N_out > 0)
    tl.assume(N_gemm > 0)
    tl.assume(stride_w1_e >= 0)
    tl.assume(stride_h_scale_k > 0)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    total_m_tiles = tl.load(sched_count_ptr)
    if pid_m >= total_m_tiles:
        return

    e = tl.load(sched_exp_ptr + pid_m)
    offs_m_base = tl.load(sched_mstart_ptr + pid_m)
    m_start_e = tl.load(expert_offsets_ptr + e)
    m_end_e = tl.load(expert_offsets_ptr + e + 1)
    M_e = m_end_e - m_start_e

    b_w1_desc = tl.make_tensor_descriptor(
        W1_ptr + e * stride_w1_e,
        shape=[N_gemm, K], strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    c_desc = tl.make_tensor_descriptor(
        C_fp8_ptr + m_start_e * N_out,
        shape=[M_e, N_out], strides=[N_out, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    offs_n = pid_n * BLOCK_N
    offs_row = offs_m_base + tl.arange(0, BLOCK_M)
    row_mask = offs_row < m_end_e

    token_ids = tl.load(sorted_token_ids_ptr + offs_row, mask=row_mask, other=0)
    token_h_base = (token_ids * K).to(tl.int64)
    token_s_base = H_scale_ptr + token_ids

    k_scale_blocks = K // SCALE_BLOCK
    b_scale_expert_stride = (N_gemm // SCALE_BLOCK) * k_scale_blocks
    n_out_scale_blocks = N_out // SCALE_BLOCK

    n_scale_up = offs_n // SCALE_BLOCK
    n_scale_gate = n_out_scale_blocks + n_scale_up
    b_scale_up_base = W1_scale_ptr + e * b_scale_expert_stride + n_scale_up * k_scale_blocks
    b_scale_gate_base = W1_scale_ptr + e * b_scale_expert_stride + n_scale_gate * k_scale_blocks

    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for ki in tl.range(NUM_KB, warp_specialize=WARP_SPEC):
        a_s = tl.load(token_s_base + ki * stride_h_scale_k, mask=row_mask, other=0.0)
        bs_up = tl.load(b_scale_up_base + ki)
        bs_gate = tl.load(b_scale_gate_base + ki)

        a_tile = tl.load(
            H_ptr + token_h_base[:, None] + (ki * BLOCK_K + tl.arange(0, BLOCK_K))[None, :],
            mask=row_mask[:, None], other=0.0,
        )
        b_up_tile = b_w1_desc.load([offs_n, ki * BLOCK_K])
        b_gate_tile = b_w1_desc.load([N_out + offs_n, ki * BLOCK_K])

        acc_up += tl.dot(a_tile, b_up_tile.T) * (a_s * bs_up)[:, None]
        acc_gate += tl.dot(a_tile, b_gate_tile.T) * (a_s * bs_gate)[:, None]

    silu_gate = acc_gate / (1.0 + tl.exp(-acc_gate))
    result = silu_gate * acc_up

    row_abs_max = tl.max(tl.abs(result), axis=1)
    scale = row_abs_max / 448.0
    scale = tl.maximum(scale, 1e-12)
    quantized = result / scale[:, None]

    local_m = offs_m_base - m_start_e
    c_desc.store([local_m, offs_n], quantized.to(C_fp8_ptr.dtype.element_ty))

    tl.store(C_scale_ptr + pid_n * M_total + offs_row, scale, mask=row_mask)

    if PDL_LAUNCH:
        tl.extra.cuda.gdc_launch_dependents()


# ============== GEMM1 single projection (fused gather, BF16 output) ==============

@triton.jit
def _grouped_gemm1_single_proj_gather(
    H_ptr, H_scale_ptr, sorted_token_ids_ptr,
    W1_ptr, C_bf16_ptr,
    W1_scale_ptr, expert_offsets_ptr,
    M_total, N_out, N_gemm,
    stride_w1_e, stride_h_scale_k,
    stride_c, b_n_offset, c_col_offset,
    num_experts,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_BLOCKS: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    PDL_LAUNCH: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
):
    tl.assume(M_total >= 0)
    tl.assume(N_out > 0)
    tl.assume(N_gemm > 0)
    tl.assume(stride_w1_e >= 0)
    tl.assume(stride_h_scale_k > 0)
    tl.assume(num_experts > 0)

    pid = tl.program_id(0)
    tile_idx = pid
    num_n_tiles = tl.cdiv(N_out, BLOCK_N)
    k_tiles = K // BLOCK_K
    k_scale_blocks = K // SCALE_BLOCK
    b_scale_expert_stride = (N_gemm // SCALE_BLOCK) * k_scale_blocks

    last_tile_end = 0

    for e in range(num_experts):
        m_start = tl.load(expert_offsets_ptr + e)
        m_end = tl.load(expert_offsets_ptr + e + 1)
        M_e = m_end - m_start
        num_m_tiles = tl.cdiv(M_e, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles
        num_pid_in_group = GROUP_SIZE_M * num_n_tiles

        if tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
            b_w1_desc = tl.make_tensor_descriptor(
                W1_ptr + e * stride_w1_e,
                shape=[N_gemm, K], strides=[K, 1],
                block_shape=[BLOCK_N, BLOCK_K],
            )

            while tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
                local_tile = tile_idx - last_tile_end
                group_id = local_tile // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
                local_in_group = local_tile % num_pid_in_group
                tile_m_idx = first_pid_m + (local_in_group % group_size_m)
                tile_n_idx = local_in_group // group_size_m
                offs_m = tile_m_idx * BLOCK_M
                offs_n = tile_n_idx * BLOCK_N

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                offs_row = m_start + offs_m + tl.arange(0, BLOCK_M)
                row_mask = offs_row < m_end

                token_ids = tl.load(
                    sorted_token_ids_ptr + offs_row,
                    mask=row_mask, other=0,
                )
                token_h_base = (token_ids * K).to(tl.int64)
                token_s_base = H_scale_ptr + token_ids

                n_scale = (b_n_offset + offs_n) // SCALE_BLOCK
                b_scale_base = (W1_scale_ptr + e * b_scale_expert_stride
                                + n_scale * k_scale_blocks)

                for ki in tl.range(k_tiles, warp_specialize=WARP_SPEC):
                    a_s = tl.load(
                        token_s_base + ki * stride_h_scale_k,
                        mask=row_mask, other=0.0,
                    )
                    b_s = tl.load(b_scale_base + ki)

                    a_tile = tl.load(
                        H_ptr + token_h_base[:, None] + (ki * BLOCK_K + tl.arange(0, BLOCK_K))[None, :],
                        mask=row_mask[:, None],
                        other=0.0,
                    )
                    b_tile = b_w1_desc.load([b_n_offset + offs_n, ki * BLOCK_K])

                    acc += tl.dot(a_tile, b_tile.T) * (a_s * b_s)[:, None]

                offs_col = c_col_offset + offs_n + tl.arange(0, BLOCK_N)
                out_ptrs = (C_bf16_ptr
                            + offs_row[:, None].to(tl.int64) * stride_c
                            + offs_col[None, :].to(tl.int64))
                tl.store(out_ptrs, acc.to(tl.bfloat16), mask=row_mask[:, None])

                tile_idx += NUM_BLOCKS

        last_tile_end = last_tile_end + num_tiles

    if PDL_LAUNCH:
        tl.extra.cuda.gdc_launch_dependents()


# ============== GEMM1 + SwiGLU + FP8 Quantize V4 (per-expert, L2-resident B) ==============

@triton.jit
def _grouped_gemm1_swiglu_v4(
    A_ptr, W1_ptr, C_fp8_ptr, C_scale_ptr,
    A_scale_ptr, W1_scale_ptr, expert_offsets_ptr,
    M_total, N_out, K, N_gemm,
    stride_w1_e, stride_a_scale_k,
    num_experts,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
    WARP_SPEC: tl.constexpr = True,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    tl.assume(M_total >= 0)
    tl.assume(N_out > 0)
    tl.assume(K > 0)
    tl.assume(N_gemm > 0)
    tl.assume(stride_w1_e >= 0)
    tl.assume(stride_a_scale_k > 0)
    tl.assume(num_experts > 0)

    e = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = tl.load(expert_offsets_ptr + e)
    m_end = tl.load(expert_offsets_ptr + e + 1)
    cnt = m_end - m_start

    if cnt > 0:
        k_tiles = K // BLOCK_K
        k_scale_blocks = K // SCALE_BLOCK
        b_scale_expert_stride = (N_gemm // SCALE_BLOCK) * k_scale_blocks
        n_out_scale_blocks = N_out // SCALE_BLOCK

        offs_n = pid_n * BLOCK_N
        n_scale_up = offs_n // SCALE_BLOCK
        n_scale_gate = n_out_scale_blocks + n_scale_up
        b_scale_up_base = W1_scale_ptr + e * b_scale_expert_stride + n_scale_up * k_scale_blocks
        b_scale_gate_base = W1_scale_ptr + e * b_scale_expert_stride + n_scale_gate * k_scale_blocks

        a_desc = tl.make_tensor_descriptor(
            A_ptr + m_start * K,
            shape=[cnt, K], strides=[K, 1],
            block_shape=[BLOCK_M, BLOCK_K],
        )
        b_up_desc = tl.make_tensor_descriptor(
            W1_ptr + e * stride_w1_e,
            shape=[N_out, K], strides=[K, 1],
            block_shape=[BLOCK_N, BLOCK_K],
        )
        b_gate_desc = tl.make_tensor_descriptor(
            W1_ptr + e * stride_w1_e + N_out * K,
            shape=[N_out, K], strides=[K, 1],
            block_shape=[BLOCK_N, BLOCK_K],
        )
        c_desc = tl.make_tensor_descriptor(
            C_fp8_ptr + m_start * N_out,
            shape=[cnt, N_out], strides=[N_out, 1],
            block_shape=[BLOCK_M, BLOCK_N],
        )

        num_m_tiles = (cnt + BLOCK_M - 1) // BLOCK_M

        for m_idx in range(num_m_tiles):
            offs_m = m_idx * BLOCK_M
            offs_row = m_start + offs_m + tl.arange(0, BLOCK_M)
            row_mask = offs_row < m_end

            acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for ki in tl.range(k_tiles, warp_specialize=WARP_SPEC):
                a_s = tl.load(
                    A_scale_ptr + ki * stride_a_scale_k + offs_row,
                    mask=row_mask, other=0.0,


                )
                bs_up = tl.load(b_scale_up_base + ki)
                bs_gate = tl.load(b_scale_gate_base + ki)

                a_tile = a_desc.load([offs_m, ki * BLOCK_K])
                b_up_tile = b_up_desc.load([offs_n, ki * BLOCK_K])
                b_gate_tile = b_gate_desc.load([offs_n, ki * BLOCK_K])

                acc_up += tl.dot(a_tile, b_up_tile.T) * (a_s * bs_up)[:, None]
                acc_gate += tl.dot(a_tile, b_gate_tile.T) * (a_s * bs_gate)[:, None]

            silu_gate = acc_gate / (1.0 + tl.exp(-acc_gate))
            result = silu_gate * acc_up

            row_abs_max = tl.max(tl.abs(result), axis=1)
            scale = row_abs_max / 448.0
            scale = tl.maximum(scale, 1e-12)
            quantized = result / scale[:, None]
            c_desc.store([offs_m, offs_n], quantized.to(C_fp8_ptr.dtype.element_ty))

            tl.store(
                C_scale_ptr + pid_n * M_total + offs_row,
                scale, mask=row_mask,
            )

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# ============== GEMM1 + SwiGLU + Fused FP8 Quantize (TMA path) ==============

@triton.jit
def _grouped_gemm_fp8_swiglu_quant(
    A_ptr, W1_ptr, C_fp8_ptr, C_scale_ptr,
    A_scale_ptr, W1_scale_ptr, expert_offsets_ptr,
    M_total, N_out, K, N_gemm,
    stride_w1_e, stride_a_scale_k,
    num_experts,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_BLOCKS: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
    WARP_SPEC: tl.constexpr = True,
    EPILOGUE_SUBTILE: tl.constexpr = 1,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    tl.assume(M_total >= 0)
    tl.assume(N_out > 0)
    tl.assume(K > 0)
    tl.assume(N_gemm > 0)
    tl.assume(stride_w1_e >= 0)
    tl.assume(stride_a_scale_k > 0)
    tl.assume(num_experts > 0)

    pid = tl.program_id(0)
    tile_idx = pid
    num_n_tiles = tl.cdiv(N_out, BLOCK_N)
    k_tiles = K // BLOCK_K
    k_scale_blocks = K // SCALE_BLOCK
    b_scale_expert_stride = (N_gemm // SCALE_BLOCK) * k_scale_blocks
    n_out_scale_blocks = N_out // SCALE_BLOCK

    last_tile_end = 0

    if EPILOGUE_SUBTILE == 1:
        for e in range(num_experts):
            m_start = tl.load(expert_offsets_ptr + e)
            m_end = tl.load(expert_offsets_ptr + e + 1)
            M_e = m_end - m_start
            num_m_tiles = tl.cdiv(M_e, BLOCK_M)
            num_tiles = num_m_tiles * num_n_tiles
            num_pid_in_group = GROUP_SIZE_M * num_n_tiles

            if tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
                a_desc = tl.make_tensor_descriptor(
                    A_ptr + m_start * K,
                    shape=[M_e, K], strides=[K, 1],
                    block_shape=[BLOCK_M, BLOCK_K],
                )
                b_w1_desc = tl.make_tensor_descriptor(
                    W1_ptr + e * stride_w1_e,
                    shape=[N_gemm, K], strides=[K, 1],
                    block_shape=[BLOCK_N, BLOCK_K],
                )
                c_desc = tl.make_tensor_descriptor(
                    C_fp8_ptr + m_start * N_out,
                    shape=[M_e, N_out], strides=[N_out, 1],
                    block_shape=[BLOCK_M, BLOCK_N],
                )

                while tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
                    local_tile = tile_idx - last_tile_end
                    group_id = local_tile // num_pid_in_group
                    first_pid_m = group_id * GROUP_SIZE_M
                    group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
                    local_in_group = local_tile % num_pid_in_group
                    tile_m_idx = first_pid_m + (local_in_group % group_size_m)
                    tile_n_idx = local_in_group // group_size_m
                    offs_m = tile_m_idx * BLOCK_M
                    offs_n = tile_n_idx * BLOCK_N

                    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                    offs_row = m_start + offs_m + tl.arange(0, BLOCK_M)
                    row_mask = offs_row < m_end

                    n_scale_up = offs_n // SCALE_BLOCK
                    n_scale_gate = n_out_scale_blocks + n_scale_up
                    b_scale_up_base = (W1_scale_ptr + e * b_scale_expert_stride
                                       + n_scale_up * k_scale_blocks)
                    b_scale_gate_base = (W1_scale_ptr + e * b_scale_expert_stride
                                         + n_scale_gate * k_scale_blocks)

                    for ki in tl.range(k_tiles, warp_specialize=WARP_SPEC):
                        a_s = tl.load(
                            A_scale_ptr + ki * stride_a_scale_k + offs_row,
                            mask=row_mask, other=0.0,
                        )
                        bs_up = tl.load(b_scale_up_base + ki)
                        bs_gate = tl.load(b_scale_gate_base + ki)

                        a_tile = a_desc.load([offs_m, ki * BLOCK_K])
                        b_up_tile = b_w1_desc.load([offs_n, ki * BLOCK_K])
                        b_gate_tile = b_w1_desc.load([N_out + offs_n, ki * BLOCK_K])

                        dot_up = tl.dot(a_tile, b_up_tile.T)
                        dot_gate = tl.dot(a_tile, b_gate_tile.T)

                        acc_up += dot_up * (a_s * bs_up)[:, None]
                        acc_gate += dot_gate * (a_s * bs_gate)[:, None]

                    silu_gate = acc_gate / (1.0 + tl.exp(-acc_gate))
                    result = silu_gate * acc_up

                    row_abs_max = tl.max(tl.abs(result), axis=1)
                    scale = row_abs_max / 448.0
                    scale = tl.maximum(scale, 1e-12)
                    quantized = result / scale[:, None]
                    c_desc.store([offs_m, offs_n], quantized.to(C_fp8_ptr.dtype.element_ty))

                    k_block = tile_n_idx
                    tl.store(
                        C_scale_ptr + k_block * M_total + offs_row,
                        scale, mask=row_mask,
                    )
                    tile_idx += NUM_BLOCKS

            last_tile_end = last_tile_end + num_tiles
    else:
        SUB_N: tl.constexpr = BLOCK_N // EPILOGUE_SUBTILE

        for e in range(num_experts):
            m_start = tl.load(expert_offsets_ptr + e)
            m_end = tl.load(expert_offsets_ptr + e + 1)
            M_e = m_end - m_start
            num_m_tiles = tl.cdiv(M_e, BLOCK_M)
            num_tiles = num_m_tiles * num_n_tiles
            num_pid_in_group = GROUP_SIZE_M * num_n_tiles

            if tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
                a_desc = tl.make_tensor_descriptor(
                    A_ptr + m_start * K,
                    shape=[M_e, K], strides=[K, 1],
                    block_shape=[BLOCK_M, BLOCK_K],
                )
                b_w1_desc_full = tl.make_tensor_descriptor(
                    W1_ptr + e * stride_w1_e,
                    shape=[N_gemm, K], strides=[K, 1],
                    block_shape=[BLOCK_N, BLOCK_K],
                )
                c_desc_full = tl.make_tensor_descriptor(
                    C_fp8_ptr + m_start * N_out,
                    shape=[M_e, N_out], strides=[N_out, 1],
                    block_shape=[BLOCK_M, BLOCK_N],
                )

                while tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
                    local_tile = tile_idx - last_tile_end
                    group_id = local_tile // num_pid_in_group
                    first_pid_m = group_id * GROUP_SIZE_M
                    group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
                    local_in_group = local_tile % num_pid_in_group
                    tile_m_idx = first_pid_m + (local_in_group % group_size_m)
                    tile_n_idx = local_in_group // group_size_m
                    offs_m = tile_m_idx * BLOCK_M
                    offs_n = tile_n_idx * BLOCK_N

                    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                    offs_row = m_start + offs_m + tl.arange(0, BLOCK_M)
                    row_mask = offs_row < m_end

                    n_scale_up = offs_n // SCALE_BLOCK
                    n_scale_gate = n_out_scale_blocks + n_scale_up
                    b_scale_up_base = (W1_scale_ptr + e * b_scale_expert_stride
                                       + n_scale_up * k_scale_blocks)
                    b_scale_gate_base = (W1_scale_ptr + e * b_scale_expert_stride
                                         + n_scale_gate * k_scale_blocks)

                    for ki in tl.range(k_tiles, warp_specialize=WARP_SPEC):
                        a_s = tl.load(
                            A_scale_ptr + ki * stride_a_scale_k + offs_row,
                            mask=row_mask, other=0.0,
                        )
                        bs_up = tl.load(b_scale_up_base + ki)
                        bs_gate = tl.load(b_scale_gate_base + ki)

                        a_tile = a_desc.load([offs_m, ki * BLOCK_K])
                        b_up_tile = b_w1_desc_full.load([offs_n, ki * BLOCK_K])
                        b_gate_tile = b_w1_desc_full.load([N_out + offs_n, ki * BLOCK_K])

                        dot_up = tl.dot(a_tile, b_up_tile.T)
                        dot_gate = tl.dot(a_tile, b_gate_tile.T)

                        acc_up += dot_up * (a_s * bs_up)[:, None]
                        acc_gate += dot_gate * (a_s * bs_gate)[:, None]

                    silu_gate = acc_gate / (1.0 + tl.exp(-acc_gate))
                    result = silu_gate * acc_up

                    cols_sf = tl.arange(0, BLOCK_N)
                    result_abs = tl.abs(result)
                    scale_map = tl.full([BLOCK_M, BLOCK_N], 1e-12, dtype=tl.float32)
                    for _sub in tl.static_range(EPILOGUE_SUBTILE):
                        _lo = _sub * SUB_N
                        _hi = _lo + SUB_N
                        _smask = (cols_sf[None, :] >= _lo) & (cols_sf[None, :] < _hi)
                        _sabs = tl.where(_smask, result_abs, 0.0)
                        _smax = tl.max(_sabs, axis=1)
                        _sscale = tl.maximum(_smax / 448.0, 1e-12)
                        scale_map = tl.where(_smask, _sscale[:, None], scale_map)
                        tl.store(C_scale_ptr + (tile_n_idx * EPILOGUE_SUBTILE + _sub) * M_total + offs_row,
                                 _sscale, mask=row_mask)
                    quantized = result / scale_map
                    c_desc_full.store([offs_m, offs_n], quantized.to(C_fp8_ptr.dtype.element_ty))

                    tile_idx += NUM_BLOCKS

            last_tile_end = last_tile_end + num_tiles

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# ============== GEMM1 Serial Fused (serial accum, fused SwiGLU+FP8 quant) ==============

@triton.jit
def _grouped_gemm1_serial_fused(
    A_ptr, W1_ptr, C_fp8_ptr, C_scale_ptr,
    A_scale_ptr, W1_scale_ptr, expert_offsets_ptr,
    M_total, N_out, K, N_gemm,
    stride_w1_e, stride_a_scale_k,
    b_scale_stride_e,
    num_experts,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_BLOCKS: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
    WARP_SPEC: tl.constexpr = True,
    EPILOGUE_SUBTILE: tl.constexpr = 1,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    tl.assume(M_total >= 0)
    tl.assume(N_out > 0)
    tl.assume(K > 0)
    tl.assume(N_gemm > 0)
    tl.assume(stride_w1_e >= 0)
    tl.assume(stride_a_scale_k > 0)
    tl.assume(num_experts > 0)

    pid = tl.program_id(0)
    tile_idx = pid
    num_n_tiles = tl.cdiv(N_out, BLOCK_N)
    k_tiles = K // BLOCK_K
    k_scale_blocks = K // SCALE_BLOCK
    b_scale_expert_stride = b_scale_stride_e
    n_out_scale_blocks = N_out // SCALE_BLOCK

    last_tile_end = 0

    for e in range(num_experts):
        m_start = tl.load(expert_offsets_ptr + e)
        m_end = tl.load(expert_offsets_ptr + e + 1)
        M_e = m_end - m_start
        num_m_tiles = tl.cdiv(M_e, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles
        num_pid_in_group = GROUP_SIZE_M * num_n_tiles

        if tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
            a_desc = tl.make_tensor_descriptor(
                A_ptr + m_start * K,
                shape=[M_e, K], strides=[K, 1],
                block_shape=[BLOCK_M, BLOCK_K],
            )
            b_w1_desc = tl.make_tensor_descriptor(
                W1_ptr + e * stride_w1_e,
                shape=[N_gemm, K], strides=[K, 1],
                block_shape=[BLOCK_N, BLOCK_K],
            )
            c_desc = tl.make_tensor_descriptor(
                C_fp8_ptr + m_start * N_out,
                shape=[M_e, N_out], strides=[N_out, 1],
                block_shape=[BLOCK_M, BLOCK_N],
            )

            while tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
                local_tile = tile_idx - last_tile_end
                group_id = local_tile // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
                local_in_group = local_tile % num_pid_in_group
                tile_m_idx = first_pid_m + (local_in_group % group_size_m)
                tile_n_idx = local_in_group // group_size_m
                offs_m = tile_m_idx * BLOCK_M
                offs_n = tile_n_idx * BLOCK_N

                offs_row = m_start + offs_m + tl.arange(0, BLOCK_M)
                row_mask = offs_row < m_end

                n_scale_up = offs_n // SCALE_BLOCK
                n_scale_gate = n_out_scale_blocks + n_scale_up
                b_scale_up_base = (W1_scale_ptr + e * b_scale_expert_stride
                                   + n_scale_up * k_scale_blocks)
                b_scale_gate_base = (W1_scale_ptr + e * b_scale_expert_stride
                                     + n_scale_gate * k_scale_blocks)

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for ki in tl.range(k_tiles, warp_specialize=WARP_SPEC):
                    a_s = tl.load(
                        A_scale_ptr + ki * stride_a_scale_k + offs_row,
                        mask=row_mask, other=0.0,


                    )
                    bs_up = tl.load(b_scale_up_base + ki)
                    a_tile = a_desc.load([offs_m, ki * BLOCK_K])
                    b_up_tile = b_w1_desc.load([offs_n, ki * BLOCK_K])
                    acc += tl.dot(a_tile, b_up_tile.T) * (a_s * bs_up)[:, None]

                up_bf16 = acc.to(tl.bfloat16)

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for ki in tl.range(k_tiles, warp_specialize=WARP_SPEC):
                    a_s = tl.load(
                        A_scale_ptr + ki * stride_a_scale_k + offs_row,
                        mask=row_mask, other=0.0,


                    )
                    bs_gate = tl.load(b_scale_gate_base + ki)
                    a_tile = a_desc.load([offs_m, ki * BLOCK_K])
                    b_gate_tile = b_w1_desc.load([N_out + offs_n, ki * BLOCK_K])
                    acc += tl.dot(a_tile, b_gate_tile.T) * (a_s * bs_gate)[:, None]

                silu_gate = acc / (1.0 + tl.exp(-acc))
                result = silu_gate * up_bf16.to(tl.float32)

                if EPILOGUE_SUBTILE == 1:
                    row_abs_max = tl.max(tl.abs(result), axis=1)
                    scale = row_abs_max / 448.0
                    scale = tl.maximum(scale, 1e-12)
                    quantized = result / scale[:, None]
                    c_desc.store([offs_m, offs_n], quantized.to(C_fp8_ptr.dtype.element_ty))
                    k_block = tile_n_idx
                    tl.store(
                        C_scale_ptr + k_block * M_total + offs_row,
                        scale, mask=row_mask,
                    )
                else:
                    SUB_N_SF: tl.constexpr = BLOCK_N // EPILOGUE_SUBTILE
                    cols_sf = tl.arange(0, BLOCK_N)
                    result_abs = tl.abs(result)
                    scale_map = tl.full([BLOCK_M, BLOCK_N], 1e-12, dtype=tl.float32)
                    for _sub in tl.static_range(EPILOGUE_SUBTILE):
                        _lo = _sub * SUB_N_SF
                        _hi = _lo + SUB_N_SF
                        _smask = (cols_sf[None, :] >= _lo) & (cols_sf[None, :] < _hi)
                        _sabs = tl.where(_smask, result_abs, 0.0)
                        _smax = tl.max(_sabs, axis=1)
                        _sscale = tl.maximum(_smax / 448.0, 1e-12)
                        scale_map = tl.where(_smask, _sscale[:, None], scale_map)
                        tl.store(C_scale_ptr + (tile_n_idx * EPILOGUE_SUBTILE + _sub) * M_total + offs_row,
                                 _sscale, mask=row_mask)
                    quantized = result / scale_map
                    c_desc.store([offs_m, offs_n], quantized.to(C_fp8_ptr.dtype.element_ty))

                tile_idx += NUM_BLOCKS

        last_tile_end = last_tile_end + num_tiles

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# ============== GEMM1 Wide (single accumulator, BF16 output) ==============

@triton.jit
def _grouped_gemm1_wide_bf16(
    A_ptr, W1_ptr, C_bf16_ptr,
    A_scale_ptr, W1_scale_ptr, expert_offsets_ptr,
    M_total, N_out, K,
    stride_w1_e, stride_a_scale_k,
    b_scale_stride_e,
    num_experts,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_BLOCKS: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
    WARP_SPEC: tl.constexpr = True,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    tl.assume(M_total >= 0)
    tl.assume(N_out > 0)
    tl.assume(K > 0)
    tl.assume(stride_w1_e >= 0)
    tl.assume(stride_a_scale_k > 0)
    tl.assume(num_experts > 0)

    pid = tl.program_id(0)
    tile_idx = pid
    num_n_tiles = tl.cdiv(N_out, BLOCK_N)
    k_tiles = K // BLOCK_K
    k_scale_blocks = K // SCALE_BLOCK
    b_scale_expert_stride = b_scale_stride_e

    last_tile_end = 0

    for e in range(num_experts):
        m_start = tl.load(expert_offsets_ptr + e)
        m_end = tl.load(expert_offsets_ptr + e + 1)
        M_e = m_end - m_start
        num_m_tiles = tl.cdiv(M_e, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles
        num_pid_in_group = GROUP_SIZE_M * num_n_tiles

        if tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
            a_desc = tl.make_tensor_descriptor(
                A_ptr + m_start * K,
                shape=[M_e, K], strides=[K, 1],
                block_shape=[BLOCK_M, BLOCK_K],
            )
            b_desc = tl.make_tensor_descriptor(
                W1_ptr + e * stride_w1_e,
                shape=[N_out, K], strides=[K, 1],
                block_shape=[BLOCK_N, BLOCK_K],
            )
            c_desc = tl.make_tensor_descriptor(
                C_bf16_ptr + m_start * N_out,
                shape=[M_e, N_out], strides=[N_out, 1],
                block_shape=[BLOCK_M, BLOCK_N],
            )

            while tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
                local_tile = tile_idx - last_tile_end
                group_id = local_tile // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
                local_in_group = local_tile % num_pid_in_group
                tile_m_idx = first_pid_m + (local_in_group % group_size_m)
                tile_n_idx = local_in_group // group_size_m
                offs_m = tile_m_idx * BLOCK_M
                offs_n = tile_n_idx * BLOCK_N

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                offs_row = m_start + offs_m + tl.arange(0, BLOCK_M)
                row_mask = offs_row < m_end

                n_block = offs_n // SCALE_BLOCK
                b_scale_base = (W1_scale_ptr + e * b_scale_expert_stride
                                + n_block * k_scale_blocks)

                for ki in tl.range(k_tiles, warp_specialize=WARP_SPEC):
                    a_s = tl.load(
                        A_scale_ptr + ki * stride_a_scale_k + offs_row,
                        mask=row_mask, other=0.0,


                    )
                    b_s = tl.load(b_scale_base + ki)

                    a_tile = a_desc.load([offs_m, ki * BLOCK_K])
                    b_tile = b_desc.load([offs_n, ki * BLOCK_K])

                    partial = tl.dot(a_tile, b_tile.T)
                    acc += partial * (a_s * b_s)[:, None]

                c_desc.store([offs_m, offs_n], acc.to(C_bf16_ptr.dtype.element_ty))
                tile_idx += NUM_BLOCKS

        last_tile_end = last_tile_end + num_tiles

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# ============== SwiGLU + FP8 Quantize (separate kernel for wide GEMM1 path) ==============

@triton.jit
def _swiglu_fp8_quant_kernel(
    g1_ptr, c_ptr, c_scale_ptr,
    M_total, N_half,
    stride_g1_m,
    stride_c_m,
    stride_cs_block,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    for mi in tl.static_range(BLOCK_M):
        m = pid_m * BLOCK_M + mi
        if m < M_total:
            up = tl.load(g1_ptr + m * stride_g1_m + offs_n).to(tl.float32)
            gate = tl.load(g1_ptr + m * stride_g1_m + N_half + offs_n).to(tl.float32)

            silu_gate = gate / (1.0 + tl.exp(-gate))
            result = silu_gate * up

            abs_max = tl.max(tl.abs(result))
            scale = abs_max / 448.0
            scale = tl.maximum(scale, 1e-12)
            quantized = result / scale

            tl.store(c_ptr + m * stride_c_m + offs_n,
                     quantized.to(c_ptr.dtype.element_ty))
            tl.store(c_scale_ptr + pid_n * stride_cs_block + m, scale)


# ============== GEMM2 + Fused Scatter ==============

@triton.jit
def _grouped_gemm2_scatter(
    C_ptr, W2_ptr,
    output_ptr,
    C_scale_ptr, W2_scale_ptr,
    routing_weights_ptr, perm_ptr, expert_offsets_ptr,
    M_total, H, I, top_k, num_experts,
    stride_w2_e,
    stride_cs_block,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_BLOCKS: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    W_SCALE_BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
    WARP_SPEC: tl.constexpr = True,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    tl.assume(M_total >= 0)
    tl.assume(H > 0)
    tl.assume(I > 0)
    tl.assume(num_experts > 0)
    tl.assume(stride_w2_e >= 0)
    tl.assume(stride_cs_block > 0)

    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(H, BLOCK_N)
    k_tiles = I // BLOCK_K
    w_k_scale_blocks = I // W_SCALE_BLOCK
    n_scale_blocks = H // W_SCALE_BLOCK
    w2_scale_expert_stride = n_scale_blocks * w_k_scale_blocks
    W_SCALE_RATIO: tl.constexpr = W_SCALE_BLOCK // BLOCK_K

    tile_idx = pid
    last_tile_end = 0

    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[M_total, I],
        strides=[I, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    w2_desc = tl.make_tensor_descriptor(
        W2_ptr,
        shape=[num_experts * H, I],
        strides=[I, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )

    for _e in range(num_experts):
        m_start = tl.load(expert_offsets_ptr + _e)
        m_end = tl.load(expert_offsets_ptr + _e + 1)
        M_e = m_end - m_start
        num_m_tiles = tl.cdiv(M_e, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles
        num_pid_in_group = GROUP_SIZE_M * num_n_tiles

        while tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
            local_tile = tile_idx - last_tile_end
            group_id = local_tile // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
            local_in_group = local_tile % num_pid_in_group
            tile_m_idx = first_pid_m + (local_in_group % group_size_m)
            tile_n_idx = local_in_group // group_size_m

            offs_m = tile_m_idx * BLOCK_M
            offs_n = tile_n_idx * BLOCK_N

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            offs_row = m_start + offs_m + tl.arange(0, BLOCK_M)
            row_mask = offs_row < m_end

            n_block = offs_n // W_SCALE_BLOCK
            w2_scale_base = (W2_scale_ptr
                             + _e * w2_scale_expert_stride
                             + n_block * w_k_scale_blocks)

            for ki in tl.range(k_tiles, warp_specialize=WARP_SPEC):
                c_s = tl.load(
                    C_scale_ptr + ki * stride_cs_block + offs_row,
                    mask=row_mask, other=0.0,


                )
                w_s = tl.load(w2_scale_base + ki // W_SCALE_RATIO)

                c_tile = c_desc.load([m_start + offs_m, ki * BLOCK_K])
                w_tile = w2_desc.load([_e * H + offs_n, ki * BLOCK_K])

                partial = tl.dot(c_tile, w_tile.T)
                acc += partial * (c_s * w_s)[:, None]

            perm_vals = tl.load(perm_ptr + offs_row, mask=row_mask, other=0)
            token_idxs = perm_vals // top_k
            topk_idxs = perm_vals % top_k
            rw = tl.load(
                routing_weights_ptr + token_idxs * top_k + topk_idxs,
                mask=row_mask, other=0.0,
            )
            scaled_bf16 = (acc * rw.to(tl.float32)[:, None]).to(tl.bfloat16)
            scaled_i32 = scaled_bf16.to(tl.int16, bitcast=True).to(tl.int32)

            offs_col_n = offs_n + tl.arange(0, BLOCK_N)
            out_ptrs = (output_ptr
                        + token_idxs[:, None] * H
                        + offs_col_n[None, :])
            addr_int = out_ptrs.to(tl.uint64, bitcast=True)
            mask_2d = row_mask.to(tl.int32)[:, None] + tl.zeros([1, BLOCK_N], dtype=tl.int32)
            tl.inline_asm_elementwise(
                "{ .reg .pred p; .reg .b32 packed, lo, hi;"
                " setp.ne.s32 p, $3, 0;"
                " and.b32 lo, $5, 0xFFFF;"
                " shl.b32 hi, $6, 16;"
                " or.b32 packed, lo, hi;"
                " @p red.relaxed.gpu.global.add.noftz.bf16x2 [$1], packed;"
                " mov.b32 $0, 0; }",
                "=r, l, l, r, r, r, r",
                args=[addr_int, mask_2d, scaled_i32],
                dtype=tl.int16,
                is_pure=False,
                pack=2,
            )

            tile_idx += NUM_BLOCKS

        last_tile_end = last_tile_end + num_tiles


# ============== GEMM2 Persistent BF16 (coalesced writes, no atomics) ==============

@triton.jit
def _grouped_gemm2_persistent_bf16(
    C_ptr, W2_ptr,
    output_ptr,
    C_scale_ptr, W2_scale_ptr,
    expert_offsets_ptr,
    M_total, H, I, num_experts,
    stride_w2_e,
    stride_cs_block,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_BLOCKS: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    W_SCALE_BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
    WARP_SPEC: tl.constexpr = True,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    tl.assume(M_total >= 0)
    tl.assume(H > 0)
    tl.assume(I > 0)
    tl.assume(num_experts > 0)
    tl.assume(stride_w2_e >= 0)
    tl.assume(stride_cs_block > 0)

    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(H, BLOCK_N)
    k_tiles = I // BLOCK_K
    w_k_scale_blocks = I // W_SCALE_BLOCK
    n_scale_blocks = H // W_SCALE_BLOCK
    w2_scale_expert_stride = n_scale_blocks * w_k_scale_blocks
    W_SCALE_RATIO: tl.constexpr = W_SCALE_BLOCK // BLOCK_K

    tile_idx = pid
    last_tile_end = 0

    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[M_total, I],
        strides=[I, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    w2_desc = tl.make_tensor_descriptor(
        W2_ptr,
        shape=[num_experts * H, I],
        strides=[I, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )

    for _e in range(num_experts):
        m_start = tl.load(expert_offsets_ptr + _e)
        m_end = tl.load(expert_offsets_ptr + _e + 1)
        M_e = m_end - m_start
        num_m_tiles = tl.cdiv(M_e, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles
        num_pid_in_group = GROUP_SIZE_M * num_n_tiles

        while tile_idx >= last_tile_end and tile_idx < last_tile_end + num_tiles:
            local_tile = tile_idx - last_tile_end
            group_id = local_tile // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
            local_in_group = local_tile % num_pid_in_group
            tile_m_idx = first_pid_m + (local_in_group % group_size_m)
            tile_n_idx = local_in_group // group_size_m

            offs_m = tile_m_idx * BLOCK_M
            offs_n = tile_n_idx * BLOCK_N

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            offs_row = m_start + offs_m + tl.arange(0, BLOCK_M)
            row_mask = offs_row < m_end

            n_block = offs_n // W_SCALE_BLOCK
            w2_scale_base = (W2_scale_ptr
                             + _e * w2_scale_expert_stride
                             + n_block * w_k_scale_blocks)

            for ki in tl.range(k_tiles, warp_specialize=WARP_SPEC):
                c_s = tl.load(
                    C_scale_ptr + ki * stride_cs_block + offs_row,
                    mask=row_mask, other=0.0,


                )
                w_s = tl.load(w2_scale_base + ki // W_SCALE_RATIO)

                c_tile = c_desc.load([m_start + offs_m, ki * BLOCK_K])
                w_tile = w2_desc.load([_e * H + offs_n, ki * BLOCK_K])

                partial = tl.dot(c_tile, w_tile.T)
                acc += partial * (c_s * w_s)[:, None]

            offs_col_n = offs_n + tl.arange(0, BLOCK_N)
            out_ptrs = (output_ptr
                        + offs_row[:, None] * H
                        + offs_col_n[None, :])
            tl.store(out_ptrs, acc.to(tl.bfloat16), mask=row_mask[:, None])

            tile_idx += NUM_BLOCKS

        last_tile_end = last_tile_end + num_tiles

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# ============== GEMM2 V4 (per-expert schedule, TMA loads) ==============

@triton.jit
def _grouped_gemm2_bf16_v4(
    C_ptr, W2_ptr,
    output_ptr,
    C_scale_ptr, W2_scale_ptr,
    expert_offsets_ptr,
    M_total, H, I, num_experts,
    stride_w2_e,
    stride_cs_block,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    W_SCALE_BLOCK: tl.constexpr,
    NUM_KB: tl.constexpr,
    USE_PDL: tl.constexpr,
    C_SCALE_BLOCK: tl.constexpr = 128,
    WARP_SPEC: tl.constexpr = False,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    tl.assume(M_total >= 0)
    tl.assume(H > 0)
    tl.assume(I > 0)
    tl.assume(num_experts > 0)
    tl.assume(stride_w2_e >= 0)
    tl.assume(stride_cs_block > 0)

    e = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start_e = tl.load(expert_offsets_ptr + e)
    m_end_e = tl.load(expert_offsets_ptr + e + 1)
    cnt = m_end_e - m_start_e

    c_desc = tl.make_tensor_descriptor(
        C_ptr, shape=[M_total, I], strides=[I, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    w2_desc = tl.make_tensor_descriptor(
        W2_ptr, shape=[num_experts * H, I], strides=[I, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )

    if cnt > 0:
        W_SCALE_RATIO: tl.constexpr = W_SCALE_BLOCK // BLOCK_K
        C_SCALE_RATIO: tl.constexpr = C_SCALE_BLOCK // BLOCK_K
        w_k_scale_blocks = I // W_SCALE_BLOCK
        n_scale_blocks = H // W_SCALE_BLOCK
        w2_scale_expert_stride = n_scale_blocks * w_k_scale_blocks

        offs_n_start = pid_n * BLOCK_N
        n_block = offs_n_start // W_SCALE_BLOCK
        w2_scale_base = (W2_scale_ptr
                         + e * w2_scale_expert_stride
                         + n_block * w_k_scale_blocks)

        w2_n_offset = e * H + offs_n_start
        k_tiles = I // BLOCK_K

        num_m_tiles = (cnt + BLOCK_M - 1) // BLOCK_M

        for m_idx in range(num_m_tiles):
            offs_m_base = m_start_e + m_idx * BLOCK_M
            offs_row = offs_m_base + tl.arange(0, BLOCK_M)
            row_mask = offs_row < m_end_e

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for ki in tl.range(k_tiles, warp_specialize=WARP_SPEC):
                c_s = tl.load(C_scale_ptr + (ki // C_SCALE_RATIO) * stride_cs_block + offs_row,
                              mask=row_mask, other=0.0,
)
                w_s = tl.load(w2_scale_base + ki // W_SCALE_RATIO)

                c_tile = c_desc.load([offs_m_base, ki * BLOCK_K])
                w_tile = w2_desc.load([w2_n_offset, ki * BLOCK_K])

                partial = tl.dot(c_tile, w_tile.T)
                acc += partial * (c_s * w_s)[:, None]

            offs_n = offs_n_start + tl.arange(0, BLOCK_N)
            out_ptrs = output_ptr + offs_row[:, None] * H + offs_n[None, :]
            tl.store(out_ptrs, acc.to(tl.bfloat16), mask=row_mask[:, None])

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# ============== GEMM2 V4 SCATTER (per-expert schedule, TMA + atomic scatter) ==============

@triton.jit
def _grouped_gemm2_scatter_v4(
    C_ptr, W2_ptr,
    output_ptr,
    C_scale_ptr, W2_scale_ptr,
    routing_weights_ptr, perm_ptr, expert_offsets_ptr,
    M_total, H, I, top_k, num_experts,
    stride_w2_e,
    stride_cs_block,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    W_SCALE_BLOCK: tl.constexpr,
    NUM_KB: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    tl.assume(M_total >= 0)
    tl.assume(H > 0)
    tl.assume(I > 0)
    tl.assume(num_experts > 0)
    tl.assume(stride_w2_e >= 0)
    tl.assume(stride_cs_block > 0)

    e = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start_e = tl.load(expert_offsets_ptr + e)
    m_end_e = tl.load(expert_offsets_ptr + e + 1)
    cnt = m_end_e - m_start_e

    c_desc = tl.make_tensor_descriptor(
        C_ptr, shape=[M_total, I], strides=[I, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    w2_desc = tl.make_tensor_descriptor(
        W2_ptr, shape=[num_experts * H, I], strides=[I, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )

    if cnt > 0:
        W_SCALE_RATIO: tl.constexpr = W_SCALE_BLOCK // BLOCK_K
        w_k_scale_blocks = I // W_SCALE_BLOCK
        n_scale_blocks = H // W_SCALE_BLOCK
        w2_scale_expert_stride = n_scale_blocks * w_k_scale_blocks

        offs_n_start = pid_n * BLOCK_N
        offs_n = offs_n_start + tl.arange(0, BLOCK_N)
        n_block = offs_n_start // W_SCALE_BLOCK
        w2_scale_base = (W2_scale_ptr
                         + e * w2_scale_expert_stride
                         + n_block * w_k_scale_blocks)

        w2_n_offset = e * H + offs_n_start
        num_m_tiles = (cnt + BLOCK_M - 1) // BLOCK_M

        for m_idx in range(num_m_tiles):
            offs_m_base = m_start_e + m_idx * BLOCK_M
            offs_row = offs_m_base + tl.arange(0, BLOCK_M)
            row_mask = offs_row < m_end_e

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for ki in range(NUM_KB):
                c_s = tl.load(C_scale_ptr + ki * stride_cs_block + offs_row,
                              mask=row_mask, other=0.0,
)
                w_s = tl.load(w2_scale_base + ki // W_SCALE_RATIO)

                c_tile = c_desc.load([offs_m_base, ki * BLOCK_K])
                w_tile = w2_desc.load([w2_n_offset, ki * BLOCK_K])

                partial = tl.dot(c_tile, w_tile.T)
                acc += partial * (c_s * w_s)[:, None]

            perm_vals = tl.load(perm_ptr + offs_row, mask=row_mask, other=0)
            token_idxs = perm_vals // top_k
            topk_idxs = perm_vals % top_k
            rw = tl.load(
                routing_weights_ptr + token_idxs * top_k + topk_idxs,
                mask=row_mask, other=0.0,
            )
            scaled_bf16 = (acc * rw.to(tl.float32)[:, None]).to(tl.bfloat16)
            scaled_i32 = scaled_bf16.to(tl.int16, bitcast=True).to(tl.int32)

            out_ptrs = (output_ptr
                        + token_idxs[:, None] * H
                        + offs_n[None, :])
            addr_int = out_ptrs.to(tl.uint64, bitcast=True)
            mask_2d = row_mask.to(tl.int32)[:, None] + tl.zeros([1, BLOCK_N], dtype=tl.int32)
            tl.inline_asm_elementwise(
                "{ .reg .pred p; .reg .b32 packed, lo, hi;"
                " setp.ne.s32 p, $3, 0;"
                " and.b32 lo, $5, 0xFFFF;"
                " shl.b32 hi, $6, 16;"
                " or.b32 packed, lo, hi;"
                " @p red.relaxed.gpu.global.add.noftz.bf16x2 [$1], packed;"
                " mov.b32 $0, 0; }",
                "=r, l, l, r, r, r, r",
                args=[addr_int, mask_2d, scaled_i32],
                dtype=tl.int16,
                is_pure=False,
                pack=2,
            )

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# ============== GATHER-COMBINE (replaces scatter-combine atomics) ==============

@triton.jit
def _gather_combine_kernel(
    gemm2_out_ptr,
    routing_w_ptr,
    inv_perm_ptr,
    output_ptr,
    H, top_k,
    stride_g2_m,
    stride_rw_t,
    stride_out_t,
    T,
    BLOCK_H: tl.constexpr,
    TOP_K: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_t >= T:
        return

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for k in tl.static_range(TOP_K):
        m_idx = tl.load(inv_perm_ptr + pid_t * top_k + k)
        is_local = (m_idx >= 0)
        m_safe = tl.maximum(m_idx, 0).to(tl.int64)
        w = tl.load(routing_w_ptr + pid_t * stride_rw_t + k)
        vals = tl.load(
            gemm2_out_ptr + m_safe * stride_g2_m + h_offs,
            mask=h_mask & is_local,
            other=0.0,
        ).to(tl.float32)
        acc += tl.where(is_local, w, 0.0) * vals

    tl.store(output_ptr + pid_t * stride_out_t + h_offs,
             acc.to(tl.bfloat16), mask=h_mask)


# ============== GEMM2 PERSISTENT + FUSED COMBINE (for small T) ==============

@triton.jit
def _grouped_gemm2_persistent_combine(
    C_ptr, W2_ptr,
    gemm2_buf_ptr,
    final_output_ptr,
    C_scale_ptr, W2_scale_ptr,
    sched_exp_ptr, sched_mstart_ptr, sched_count_ptr,
    expert_offsets_ptr,
    routing_weights_ptr, inv_perm_ptr,
    done_counter_ptr,
    M_total, H, I, top_k, num_experts, T_tokens,
    stride_w2_e,
    stride_cs_block,
    barrier_base,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    W_SCALE_BLOCK: tl.constexpr,
    NUM_KB: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    C_SCALE_BLOCK: tl.constexpr = 128,
    TOP_K_C: tl.constexpr = 8,
    BLOCK_H_COMBINE: tl.constexpr = 1024,
    USE_PDL: tl.constexpr = False,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    tl.assume(M_total >= 0)
    tl.assume(H > 0)
    tl.assume(I > 0)
    tl.assume(num_experts > 0)
    tl.assume(stride_w2_e >= 0)
    tl.assume(stride_cs_block > 0)

    pid = tl.program_id(0)
    total_m_tiles = tl.load(sched_count_ptr)
    num_n_tiles = H // BLOCK_N
    total_gemm2_tiles = total_m_tiles * num_n_tiles

    c_desc = tl.make_tensor_descriptor(
        C_ptr, shape=[M_total, I], strides=[I, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    w2_desc = tl.make_tensor_descriptor(
        W2_ptr, shape=[num_experts * H, I], strides=[I, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )

    W_SCALE_RATIO: tl.constexpr = W_SCALE_BLOCK // BLOCK_K
    C_SCALE_RATIO: tl.constexpr = C_SCALE_BLOCK // BLOCK_K
    w_k_scale_blocks = I // W_SCALE_BLOCK
    n_scale_blocks = H // W_SCALE_BLOCK
    w2_scale_expert_stride = n_scale_blocks * w_k_scale_blocks

    tile_idx = pid
    while tile_idx < total_gemm2_tiles:
        m_tile_idx = tile_idx % total_m_tiles
        n_tile_idx = tile_idx // total_m_tiles

        e = tl.load(sched_exp_ptr + m_tile_idx)
        offs_m_base = tl.load(sched_mstart_ptr + m_tile_idx)
        m_end_e = tl.load(expert_offsets_ptr + e + 1)

        offs_n_start = n_tile_idx * BLOCK_N
        n_block = offs_n_start // W_SCALE_BLOCK
        w2_scale_base = (W2_scale_ptr + e * w2_scale_expert_stride + n_block * w_k_scale_blocks)
        w2_n_offset = e * H + offs_n_start

        offs_row = offs_m_base + tl.arange(0, BLOCK_M)
        row_mask = offs_row < m_end_e

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(NUM_KB):
            c_s = tl.load(C_scale_ptr + (ki // C_SCALE_RATIO) * stride_cs_block + offs_row,
                          mask=row_mask, other=0.0,
)
            w_s = tl.load(w2_scale_base + ki // W_SCALE_RATIO)
            c_tile = c_desc.load([offs_m_base, ki * BLOCK_K])
            w_tile = w2_desc.load([w2_n_offset, ki * BLOCK_K])
            partial = tl.dot(c_tile, w_tile.T)
            acc += partial * (c_s * w_s)[:, None]

        offs_n = offs_n_start + tl.arange(0, BLOCK_N)
        out_ptrs = gemm2_buf_ptr + offs_row[:, None] * H + offs_n[None, :]
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=row_mask[:, None])

        tile_idx += NUM_BLOCKS

    # Generation-counter barrier: all persistent CTAs increment, then spin
    tl.atomic_add(done_counter_ptr, 1)
    while tl.atomic_add(done_counter_ptr, 0) < barrier_base + NUM_BLOCKS:
        pass

    # Phase 2: Gather-combine (all CTAs cooperate)
    num_h_tiles = (H + BLOCK_H_COMBINE - 1) // BLOCK_H_COMBINE
    total_combine_tiles = T_tokens * num_h_tiles
    combine_tile = pid
    while combine_tile < total_combine_tiles:
        t_idx = combine_tile // num_h_tiles
        h_tile_idx = combine_tile % num_h_tiles

        h_offs = h_tile_idx * BLOCK_H_COMBINE + tl.arange(0, BLOCK_H_COMBINE)
        h_mask = h_offs < H

        acc_c = tl.zeros([BLOCK_H_COMBINE], dtype=tl.float32)
        for k in tl.static_range(TOP_K_C):
            m_idx = tl.load(inv_perm_ptr + t_idx * top_k + k)
            is_local = (m_idx >= 0)
            m_safe = tl.maximum(m_idx, 0).to(tl.int64)
            w = tl.load(routing_weights_ptr + t_idx * top_k + k)
            vals = tl.load(
                gemm2_buf_ptr + m_safe * H + h_offs,
                mask=h_mask & is_local,
                other=0.0,
            ).to(tl.float32)
            acc_c += tl.where(is_local, w, 0.0) * vals

        tl.store(final_output_ptr + t_idx * H + h_offs,
                 acc_c.to(tl.bfloat16), mask=h_mask)

        combine_tile += NUM_BLOCKS


# ============== GEMM2 FLAT SCHEDULE ==============

@triton.jit
def _build_gemm2_schedule(
    expert_offsets_ptr,
    sched_exp_ptr, sched_mstart_ptr, sched_count_ptr,
    max_tiles,
    BLOCK_M: tl.constexpr, NUM_LOCAL: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    idx = 0
    for e in tl.static_range(NUM_LOCAL):
        start_e = tl.load(expert_offsets_ptr + e)
        end_e = tl.load(expert_offsets_ptr + e + 1)
        cnt = end_e - start_e
        ntiles = (cnt + BLOCK_M - 1) // BLOCK_M
        for t in range(ntiles):
            if idx < max_tiles:
                tl.store(sched_exp_ptr + idx, e)
                tl.store(sched_mstart_ptr + idx, start_e + t * BLOCK_M)
            idx += 1
    tl.store(sched_count_ptr, idx)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _grouped_gemm2_bf16_flat(
    C_ptr, W2_ptr,
    output_ptr,
    C_scale_ptr, W2_scale_ptr,
    sched_exp_ptr, sched_mstart_ptr, sched_count_ptr,
    expert_offsets_ptr,
    routing_weights_ptr, perm_ptr,
    inv_perm_ptr,
    done_counter_ptr,
    final_output_ptr,
    M_total, H, I, top_k, num_experts,
    stride_w2_e,
    stride_cs_block,
    T_tokens,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    W_SCALE_BLOCK: tl.constexpr,
    NUM_KB: tl.constexpr,
    USE_PDL: tl.constexpr,
    SCATTER: tl.constexpr = False,
    C_SCALE_BLOCK: tl.constexpr = 128,
    WARP_SPEC: tl.constexpr = False,
    FUSE_COMBINE: tl.constexpr = False,
    TOP_K_C: tl.constexpr = 8,
    BLOCK_H_COMBINE: tl.constexpr = 1024,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    tl.assume(M_total >= 0)
    tl.assume(H > 0)
    tl.assume(I > 0)
    tl.assume(num_experts > 0)
    tl.assume(stride_w2_e >= 0)
    tl.assume(stride_cs_block > 0)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    total_tiles = tl.load(sched_count_ptr)
    if pid_m >= total_tiles:
        return

    e = tl.load(sched_exp_ptr + pid_m)
    offs_m_base = tl.load(sched_mstart_ptr + pid_m)
    m_end_e = tl.load(expert_offsets_ptr + e + 1)

    c_desc = tl.make_tensor_descriptor(
        C_ptr, shape=[M_total, I], strides=[I, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    w2_desc = tl.make_tensor_descriptor(
        W2_ptr, shape=[num_experts * H, I], strides=[I, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )

    W_SCALE_RATIO: tl.constexpr = W_SCALE_BLOCK // BLOCK_K
    C_SCALE_RATIO: tl.constexpr = C_SCALE_BLOCK // BLOCK_K
    w_k_scale_blocks = I // W_SCALE_BLOCK
    n_scale_blocks = H // W_SCALE_BLOCK
    w2_scale_expert_stride = n_scale_blocks * w_k_scale_blocks

    offs_n_start = pid_n * BLOCK_N
    n_block = offs_n_start // W_SCALE_BLOCK
    w2_scale_base = (W2_scale_ptr
                     + e * w2_scale_expert_stride
                     + n_block * w_k_scale_blocks)

    w2_n_offset = e * H + offs_n_start

    offs_row = offs_m_base + tl.arange(0, BLOCK_M)
    row_mask = offs_row < m_end_e

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for ki in tl.range(NUM_KB, warp_specialize=WARP_SPEC):
        c_s = tl.load(C_scale_ptr + (ki // C_SCALE_RATIO) * stride_cs_block + offs_row,
                      mask=row_mask, other=0.0,
)
        w_s = tl.load(w2_scale_base + ki // W_SCALE_RATIO)

        c_tile = c_desc.load([offs_m_base, ki * BLOCK_K])
        w_tile = w2_desc.load([w2_n_offset, ki * BLOCK_K])

        partial = tl.dot(c_tile, w_tile.T)
        acc += partial * (c_s * w_s)[:, None]

    offs_n = offs_n_start + tl.arange(0, BLOCK_N)

    if SCATTER:
        perm_vals = tl.load(perm_ptr + offs_row, mask=row_mask, other=0)
        token_idxs = perm_vals // top_k
        topk_idxs = perm_vals % top_k
        rw = tl.load(
            routing_weights_ptr + token_idxs * top_k + topk_idxs,
            mask=row_mask, other=0.0,
        )
        scaled_bf16 = (acc * rw.to(tl.float32)[:, None]).to(tl.bfloat16)
        scaled_i32 = scaled_bf16.to(tl.int16, bitcast=True).to(tl.int32)

        out_ptrs = (output_ptr
                    + token_idxs[:, None] * H
                    + offs_n[None, :])
        addr_int = out_ptrs.to(tl.uint64, bitcast=True)
        mask_2d = row_mask.to(tl.int32)[:, None] + tl.zeros([1, BLOCK_N], dtype=tl.int32)
        tl.inline_asm_elementwise(
            "{ .reg .pred p; .reg .b32 packed, lo, hi;"
            " setp.ne.s32 p, $3, 0;"
            " and.b32 lo, $5, 0xFFFF;"
            " shl.b32 hi, $6, 16;"
            " or.b32 packed, lo, hi;"
            " @p red.relaxed.gpu.global.add.noftz.bf16x2 [$1], packed;"
            " mov.b32 $0, 0; }",
            "=r, l, l, r, r, r, r",
            args=[addr_int, mask_2d, scaled_i32],
            dtype=tl.int16,
            is_pure=False,
            pack=2,
        )
    else:
        out_ptrs = output_ptr + offs_row[:, None] * H + offs_n[None, :]
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=row_mask[:, None])

    if FUSE_COMBINE:
        num_n_g2 = H // BLOCK_N
        total_active = total_tiles * num_n_g2
        prev = tl.atomic_add(done_counter_ptr, 1)
        if prev == total_active - 1:
            num_h_tiles = (H + BLOCK_H_COMBINE - 1) // BLOCK_H_COMBINE
            total_combine = T_tokens * num_h_tiles
            for ct in range(total_combine):
                t_idx = ct // num_h_tiles
                h_idx = ct % num_h_tiles
                h_offs = h_idx * BLOCK_H_COMBINE + tl.arange(0, BLOCK_H_COMBINE)
                h_mask = h_offs < H
                acc_c = tl.zeros([BLOCK_H_COMBINE], dtype=tl.float32)
                for k in tl.static_range(TOP_K_C):
                    m_idx = tl.load(inv_perm_ptr + t_idx * top_k + k)
                    is_local = (m_idx >= 0)
                    m_safe = tl.maximum(m_idx, 0).to(tl.int64)
                    w = tl.load(routing_weights_ptr + t_idx * top_k + k)
                    vals = tl.load(
                        output_ptr + m_safe * H + h_offs,
                        mask=h_mask & is_local,
                        other=0.0,
                    ).to(tl.float32)
                    acc_c += tl.where(is_local, w, 0.0) * vals
                tl.store(final_output_ptr + t_idx * H + h_offs,
                         acc_c.to(tl.bfloat16), mask=h_mask)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# ============== FUSED GEMM2+COMBINE (token-major, for small T) ==============

@triton.jit
def _fused_gemm2_combine_token_major(
    intermediate_fp8_ptr,
    W2_ptr,
    output_ptr,
    intermediate_scale_ptr,
    W2_scale_ptr,
    routing_weights_ptr,
    inv_perm_ptr,
    topk_idx_ptr,
    H, I, top_k,
    stride_cs_block,
    local_expert_offset,
    T,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    TOP_K_C: tl.constexpr,
    USE_PDL: tl.constexpr,
    C_SCALE_BLOCK: tl.constexpr = 128,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= T:
        return

    tl.assume(H > 0)
    tl.assume(I > 0)
    tl.assume(stride_cs_block > 0)

    offs_n = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)

    C_SCALE_RATIO: tl.constexpr = C_SCALE_BLOCK // BLOCK_K
    W_SCALE_RATIO: tl.constexpr = 128 // BLOCK_K
    w_k_sb = I // 128
    n_sb = H // 128
    w2_se_stride = n_sb * w_k_sb
    n_block_idx = (pid_h * BLOCK_N) // 128

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    inv_base = pid_t * top_k

    for k in tl.static_range(TOP_K_C):
        m_idx = tl.load(inv_perm_ptr + inv_base + k)
        if m_idx >= 0:
            w = tl.load(routing_weights_ptr + inv_base + k)
            expert_global = tl.load(topk_idx_ptr + inv_base + k)
            expert_local = expert_global - local_expert_offset

            m_i64 = m_idx.to(tl.int64)
            el_i64 = expert_local.to(tl.int64)

            w2_s_base = W2_scale_ptr + el_i64 * w2_se_stride + n_block_idx * w_k_sb
            a_base = intermediate_fp8_ptr + m_i64 * I

            w_row_indices = el_i64 * H + offs_n.to(tl.int64)
            w_base = W2_ptr + w_row_indices[:, None] * I

            partial = tl.zeros([BLOCK_N], dtype=tl.float32)

            for ki in range(NUM_K_BLOCKS):
                k_offs = ki * BLOCK_K + tl.arange(0, BLOCK_K)
                a_chunk = tl.load(a_base + k_offs).to(tl.float32)
                w_chunk = tl.load(w_base + k_offs[None, :].to(tl.int64)).to(tl.float32)

                c_s = tl.load(intermediate_scale_ptr + (ki // C_SCALE_RATIO) * stride_cs_block + m_idx)
                w_s = tl.load(w2_s_base + ki // W_SCALE_RATIO)

                partial += tl.sum(a_chunk[None, :] * w_chunk, axis=1) * (c_s * w_s)

            acc += w * partial

    out_offs = pid_t.to(tl.int64) * H + offs_n.to(tl.int64)
    tl.store(output_ptr + out_offs, acc.to(tl.bfloat16))


# ============== PIPELINE ==============

_num_sms = None
_ws_cache = {}

_SMALL_T_THRESH = 128
_PARALLEL_PERM_THRESH = 129
_TMA_GEMM1_THRESH = 128
_USE_PARALLEL_PERM = True

_GEMM1_FUSED_CFG = {
    'num_warps': 8,
    'num_stages': 4,
    'WARP_SPEC': False,
}

_GEMM1_FUSED_MAXNREG = None
_GEMM1_FUSED_GSM = 2

_TMA_GEMM1_CFG = {
    'num_warps': 8,
    'num_stages': 4,
    'WARP_SPEC': False,
}

_GEMM2_V4_CFG = {
    'BLOCK_M': 64,
    'BLOCK_N': 128,
    'BLOCK_K': _GEMM2_BK,
    'num_warps': 4,
    'num_stages': 3,
    'WARP_SPEC': False,
}

_GEMM2_UNFUSE_THRESH = 1024

_GEMM1_MAXNREG = None
_GEMM1_SQ_MAXNREG = None
_USE_SERIAL_GEMM1_MID = False
_USE_WIDE_GEMM1 = False
_WIDE_GEMM1_CFG = {
    'num_warps': 4,
    'num_stages': 3,
}
_GEMM2_MAXNREG = None
_GEMM2_PERSISTENT_MAXNREG = None

_GEMM1_MID_BM = 64
_GEMM1_MID_NW = 8
_GEMM1_MID_NS = 4
_GEMM1_MID_WS = False
_GEMM1_MID_MAXNREG = None
_GEMM2_MID_BM = 64
_USE_FUSED_GEMM2_COMBINE = False
_USE_MXSCALE_GEMM1 = False
_USE_SPLIT_GEMM1 = False
_SPLIT_GEMM1_THRESH = 52
_USE_FUSED_ROUTING_GEMM1 = False
_USE_SCHEDULED_GEMM1 = True

_GEMM2_PERSISTENT_CFG = {
    'BLOCK_M': 128,
    'BLOCK_N': 128,
    'GROUP_SIZE_M': 8,
    'num_warps': 8,
    'num_stages': 4,
    'WARP_SPEC': True,
}


class _Workspace:
    def __init__(self, T, device):
        TK = T * TOP_K
        self.T = T
        self.TK = TK
        self.topk_idx = torch.empty(T, TOP_K, dtype=torch.int32, device=device)
        self.topk_weights = torch.empty(T, TOP_K, dtype=torch.float32, device=device)
        self._perm_buf = torch.empty(2 * NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device)
        self.expert_counts = self._perm_buf[:NUM_LOCAL_EXPERTS]
        self.atomic_counters = self._perm_buf[NUM_LOCAL_EXPERTS:]
        self.expert_offsets = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
        self.perm = torch.empty(TK, dtype=torch.int32, device=device)
        self.sorted_token_ids = torch.zeros(TK, dtype=torch.int32, device=device)
        self.output = torch.empty(T, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        self.permuted_hs = torch.empty(TK, HIDDEN_SIZE, dtype=torch.float8_e4m3fn, device=device)
        self.permuted_hs_scale = torch.empty(HIDDEN_SIZE // SCALE_BLOCK, TK, dtype=torch.float32, device=device)
        self.intermediate_fp8 = torch.empty(TK, INTERMEDIATE_SIZE, dtype=torch.float8_e4m3fn, device=device)
        self.intermediate_scale = torch.empty(_INTER_SCALE_BLOCKS, TK, dtype=torch.float32, device=device)
        self.inv_perm = torch.full((TK,), -1, dtype=torch.int32, device=device)
        self.gemm1_bf16 = torch.empty(TK, 2 * INTERMEDIATE_SIZE, dtype=torch.bfloat16, device=device)
        self.gemm2_bf16 = torch.empty(TK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        _max_sched = NUM_LOCAL_EXPERTS * ((T + 63) // 64)
        self.sched_expert = torch.empty(_max_sched, dtype=torch.int32, device=device)
        self.sched_mstart = torch.empty(_max_sched, dtype=torch.int32, device=device)
        self.sched_count = torch.zeros(1, dtype=torch.int32, device=device)
        self.done_counter = torch.zeros(1, dtype=torch.int32, device=device)
        self._arrive_gen = 0
        self._done_gen = 0


def _get_workspace(T, device):
    global _ws_cache
    if T not in _ws_cache:
        _ws_cache[T] = _Workspace(T, device)
    return _ws_cache[T]


@torch.no_grad()
def moe_forward(
    routing_logits, routing_bias,
    hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor,
):
    global _num_sms
    _ensure_tma_allocator()

    if isinstance(local_expert_offset, torch.Tensor):
        local_expert_offset = int(local_expert_offset.item())
    if isinstance(routed_scaling_factor, torch.Tensor):
        routed_scaling_factor = float(routed_scaling_factor.item())

    T = routing_logits.shape[0]
    device = routing_logits.device
    TK = T * TOP_K

    if _num_sms is None:
        _num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    NUM_SMS = _num_sms

    ws = _get_workspace(T, device)

    # === Stage 1+2: Routing + Permutation (+ optional fused GEMM1) ===
    _split_gemm1 = _USE_SPLIT_GEMM1 and T >= _SPLIT_GEMM1_THRESH and T < _TMA_GEMM1_THRESH
    _g2_bm = _GEMM2_V4_CFG['BLOCK_M']
    _max_sched = NUM_LOCAL_EXPERTS * ((T + _g2_bm - 1) // _g2_bm)

    _use_fused_rg1 = _USE_FUSED_ROUTING_GEMM1 and T < _TMA_GEMM1_THRESH and not _split_gemm1

    if _use_fused_rg1:
        _use_pp = _USE_PARALLEL_PERM and T >= 33
        _arrive_base = ws._arrive_gen
        ws._arrive_gen += (2 * T + 3) if _use_pp else (T + 1)
        _g1f_nw = _GEMM1_FUSED_CFG['num_warps']
        _g1f_ns = _GEMM1_FUSED_CFG['num_stages']
        _g1f_ws = _GEMM1_FUSED_CFG['WARP_SPEC']
        _fused_routing_perm_gemm1[(NUM_SMS,)](
            routing_logits, routing_bias,
            ws.topk_idx, ws.topk_weights,
            ws.expert_offsets, ws.perm, ws.sorted_token_ids, ws.inv_perm,
            ws.sched_expert, ws.sched_mstart, ws.sched_count,
            ws.atomic_counters, ws.done_counter,
            ws.expert_counts,
            T, TK, routed_scaling_factor,
            local_expert_offset, NUM_LOCAL_EXPERTS, _max_sched, _arrive_base,
            hidden_states, hidden_states_scale,
            gemm1_weights, ws.intermediate_fp8, ws.intermediate_scale,
            gemm1_weights_scale,
            TK, INTERMEDIATE_SIZE, 2 * INTERMEDIATE_SIZE,
            gemm1_weights.stride(0), hidden_states_scale.stride(0),
            E=NUM_EXPERTS_GLOBAL, N_GRP=N_GROUP,
            GRP_SIZE=NUM_EXPERTS_GLOBAL // N_GROUP,
            TOP_K_C=TOP_K, TOPK_GRP=TOPK_GROUP,
            PERM_BLOCK=1024, NUM_LOCAL=NUM_LOCAL_EXPERTS,
            SCHED_BM=_g2_bm,
            PARALLEL_PERM=_use_pp,
            K_HIDDEN=HIDDEN_SIZE,
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=128,
            GROUP_SIZE_M=_GEMM1_FUSED_GSM, NUM_BLOCKS=NUM_SMS,
            SCALE_BLOCK=128,
            WARP_SPEC=_g1f_ws,
            USE_DOT_SCALED=_USE_MXSCALE_GEMM1,
            PDL_LAUNCH=True,
            num_warps=_g1f_nw, num_stages=_g1f_ns,
            launch_pdl=True,
        )
        _pdl_any = True
        _pdl_early = False
    elif T < _PARALLEL_PERM_THRESH:
        _pdl_early = (33 <= T <= _SMALL_T_THRESH) and not _split_gemm1
        _pdl_any = _pdl_early
        _use_pp = _USE_PARALLEL_PERM and T >= 33
        _arrive_base = ws._arrive_gen
        ws._arrive_gen += T + (2 if _use_pp else 0)
        _fused_routing_perm_kernel[(T,)](
            routing_logits, routing_bias,
            ws.topk_idx, ws.topk_weights,
            ws.expert_offsets, ws.perm, ws.sorted_token_ids, ws.inv_perm,
            ws.sched_expert, ws.sched_mstart, ws.sched_count,
            ws.atomic_counters, ws.done_counter,
            ws.expert_counts,
            T, TK, routed_scaling_factor,
            local_expert_offset, NUM_LOCAL_EXPERTS, _max_sched, _arrive_base,
            E=NUM_EXPERTS_GLOBAL, N_GRP=N_GROUP,
            GRP_SIZE=NUM_EXPERTS_GLOBAL // N_GROUP,
            TOP_K_C=TOP_K, TOPK_GRP=TOPK_GROUP,
            K=TOP_K, PERM_BLOCK=1024, NUM_LOCAL=NUM_LOCAL_EXPERTS,
            SCHED_BM=_g2_bm,
            PDL_LAUNCH=_pdl_early,
            PARALLEL_PERM=_use_pp,
            num_warps=4,
            launch_pdl=_pdl_early,
        )
    else:
        _pdl_early = False
        _count_in_routing = (T >= _PARALLEL_PERM_THRESH)
        _pdl_mid = _count_in_routing and (T < 2048) and not _split_gemm1
        _pdl_any = _pdl_mid
        if _count_in_routing:
            ws._perm_buf.zero_()
        _fused_routing_kernel[(T,)](
            routing_logits, routing_bias,
            ws.topk_idx, ws.topk_weights,
            ws._perm_buf, ws.expert_counts,
            T, routed_scaling_factor,
            local_expert_offset, NUM_LOCAL_EXPERTS,
            E=NUM_EXPERTS_GLOBAL, N_GRP=N_GROUP,
            GRP_SIZE=NUM_EXPERTS_GLOBAL // N_GROUP,
            TOP_K_C=TOP_K, TOPK_GRP=TOPK_GROUP,
            PERM_BUF_SIZE=2 * NUM_LOCAL_EXPERTS,
            USE_PDL=False,
            COUNT_LOCAL=_count_in_routing,
            PDL_LAUNCH=_pdl_mid,
            launch_pdl=_pdl_any,
        )

        topk_idx_flat = ws.topk_idx.view(-1)
        PERM_BLOCK = 1024
        grid_perm = (TK + PERM_BLOCK - 1) // PERM_BLOCK

        _scatter_perm_fused[(grid_perm,)](
            topk_idx_flat, ws.expert_counts, ws.atomic_counters,
            ws.expert_offsets,
            ws.perm, ws.sorted_token_ids, ws.inv_perm,
            ws.sched_expert, ws.sched_mstart, ws.sched_count,
            T, K=TOP_K, offset=local_expert_offset, num_local=NUM_LOCAL_EXPERTS,
            max_sched=_max_sched,
            BLOCK=PERM_BLOCK, NUM_LOCAL=NUM_LOCAL_EXPERTS,
            SCHED_BM=_g2_bm,
            EMIT_SCHED=True,
            USE_PDL=_pdl_mid,
            launch_pdl=_pdl_mid,
        )

    # === Stage 3+4: Gather + GEMM1 (TMA path — GPU-bound M_local) ===
    offsets = ws.expert_offsets
    M_total_param = TK
    Kb = HIDDEN_SIZE // SCALE_BLOCK

    _UNFUSE_GEMM1_THRESH = 2048
    _USE_PDL = (T < _UNFUSE_GEMM1_THRESH) and not _split_gemm1

    _GATHER_BH = 1024
    _GATHER_BK = 64
    num_h_tiles = (HIDDEN_SIZE + _GATHER_BH - 1) // _GATHER_BH

    if T > 1024:
        M_gather = int(offsets[NUM_LOCAL_EXPERTS].item())
        _gather_hidden_kernel[(M_gather, num_h_tiles)](
            hidden_states, hidden_states_scale, ws.sorted_token_ids,
            ws.permuted_hs, ws.permuted_hs_scale,
            HIDDEN_SIZE, Kb, M_gather,
            hidden_states.stride(0), hidden_states_scale.stride(0),
            ws.permuted_hs.stride(0), ws.permuted_hs_scale.stride(0),
            BLOCK_H=_GATHER_BH, BLOCK_K=_GATHER_BK,
            USE_PDL=_USE_PDL,
            num_warps=4,
            launch_pdl=_USE_PDL,
        )

    if T >= _UNFUSE_GEMM1_THRESH:
        _grouped_gemm1_serial_fused[(NUM_SMS,)](
            ws.permuted_hs, gemm1_weights, ws.intermediate_fp8, ws.intermediate_scale,
            ws.permuted_hs_scale, gemm1_weights_scale, offsets,
            M_total_param, INTERMEDIATE_SIZE, HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE,
            gemm1_weights.stride(0), M_total_param,
            gemm1_weights_scale.stride(0),
            NUM_LOCAL_EXPERTS,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=128,
            GROUP_SIZE_M=8, NUM_BLOCKS=NUM_SMS, SCALE_BLOCK=128,
            USE_PDL=False,
            WARP_SPEC=True,
            EPILOGUE_SUBTILE=_EPILOGUE_SUBTILE,
            num_warps=8, num_stages=4,
            **({'maxnreg': _GEMM1_MAXNREG} if _GEMM1_MAXNREG else {}),
        )
    elif T <= 1024:
        if T >= _TMA_GEMM1_THRESH:
            _scatter_hidden_kernel[(T, num_h_tiles)](
                hidden_states, hidden_states_scale,
                ws.inv_perm,
                ws.permuted_hs, ws.permuted_hs_scale,
                HIDDEN_SIZE, Kb, T,
                hidden_states.stride(0), hidden_states_scale.stride(0),
                ws.permuted_hs.stride(0), ws.permuted_hs_scale.stride(0),
                BLOCK_H=_GATHER_BH, BLOCK_K=_GATHER_BK,
                TOP_K=TOP_K,
                USE_PDL=_USE_PDL,
                WAIT_PDL=_pdl_any,
                num_warps=4,
                launch_pdl=_USE_PDL,
            )
            if _USE_SERIAL_GEMM1_MID:
                _grouped_gemm1_serial_fused[(NUM_SMS,)](
                    ws.permuted_hs, gemm1_weights, ws.intermediate_fp8, ws.intermediate_scale,
                    ws.permuted_hs_scale, gemm1_weights_scale, offsets,
                    M_total_param, INTERMEDIATE_SIZE, HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE,
                    gemm1_weights.stride(0), M_total_param,
                    gemm1_weights_scale.stride(0),
                    NUM_LOCAL_EXPERTS,
                    BLOCK_M=64, BLOCK_N=128, BLOCK_K=128,
                    GROUP_SIZE_M=8, NUM_BLOCKS=NUM_SMS, SCALE_BLOCK=128,
                    USE_PDL=_USE_PDL,
                    WARP_SPEC=False,
                    EPILOGUE_SUBTILE=_EPILOGUE_SUBTILE,
                    num_warps=8, num_stages=2,
                    launch_pdl=_USE_PDL,
                )
            elif _USE_WIDE_GEMM1:
                _grouped_gemm1_wide_bf16[(NUM_SMS,)](
                    ws.permuted_hs, gemm1_weights, ws.gemm1_bf16,
                    ws.permuted_hs_scale, gemm1_weights_scale, offsets,
                    M_total_param, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE,
                    gemm1_weights.stride(0), M_total_param,
                    gemm1_weights_scale.stride(0),
                    NUM_LOCAL_EXPERTS,
                    BLOCK_M=64, BLOCK_N=128, BLOCK_K=128,
                    GROUP_SIZE_M=8, NUM_BLOCKS=NUM_SMS, SCALE_BLOCK=128,
                    USE_PDL=_USE_PDL,
                    WARP_SPEC=False,
                    num_warps=_WIDE_GEMM1_CFG['num_warps'],
                    num_stages=_WIDE_GEMM1_CFG['num_stages'],
                    launch_pdl=_USE_PDL,
                )
                _sg_bm = 4
                _sg_bn = 128
                _swiglu_fp8_quant_kernel[(
                    (M_total_param + _sg_bm - 1) // _sg_bm,
                    INTERMEDIATE_SIZE // _sg_bn,
                )](
                    ws.gemm1_bf16,
                    ws.intermediate_fp8,
                    ws.intermediate_scale,
                    M_total_param,
                    INTERMEDIATE_SIZE,
                    ws.gemm1_bf16.stride(0),
                    ws.intermediate_fp8.stride(0),
                    ws.intermediate_scale.stride(0),
                    BLOCK_N=_sg_bn, BLOCK_M=_sg_bm,
                    num_warps=4,
                )
                _USE_PDL = False
            else:
                _g1_bm = _GEMM1_MID_BM
                _g1_nw = _GEMM1_MID_NW
                _g1_ns = _GEMM1_MID_NS
                _g1_ws = _GEMM1_MID_WS
                _g1_mnr = _GEMM1_MID_MAXNREG or _GEMM1_SQ_MAXNREG
                _grouped_gemm_fp8_swiglu_quant[(NUM_SMS,)](
                    ws.permuted_hs, gemm1_weights, ws.intermediate_fp8, ws.intermediate_scale,
                    ws.permuted_hs_scale, gemm1_weights_scale, offsets,
                    M_total_param, INTERMEDIATE_SIZE, HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE,
                    gemm1_weights.stride(0), M_total_param,
                    NUM_LOCAL_EXPERTS,
                    BLOCK_M=_g1_bm, BLOCK_N=128, BLOCK_K=128,
                    GROUP_SIZE_M=8, NUM_BLOCKS=NUM_SMS, SCALE_BLOCK=128,
                    USE_PDL=_USE_PDL,
                    WARP_SPEC=_g1_ws,
                    EPILOGUE_SUBTILE=_EPILOGUE_SUBTILE,
                    num_warps=_g1_nw, num_stages=_g1_ns,
                    launch_pdl=_USE_PDL,
                    **({'maxnreg': _g1_mnr} if _g1_mnr else {}),
                )
        else:
            if _split_gemm1:
                _g1_stride_c = ws.gemm1_bf16.stride(0)
                for _proj_idx in range(2):
                    _b_n_off = _proj_idx * INTERMEDIATE_SIZE
                    _c_col_off = _proj_idx * INTERMEDIATE_SIZE
                    _grouped_gemm1_single_proj_gather[(NUM_SMS,)](
                        hidden_states, hidden_states_scale, ws.sorted_token_ids,
                        gemm1_weights, ws.gemm1_bf16,
                        gemm1_weights_scale, offsets,
                        M_total_param, INTERMEDIATE_SIZE, 2 * INTERMEDIATE_SIZE,
                        gemm1_weights.stride(0), hidden_states_scale.stride(0),
                        _g1_stride_c, _b_n_off, _c_col_off,
                        NUM_LOCAL_EXPERTS,
                        K=HIDDEN_SIZE,
                        BLOCK_M=64, BLOCK_N=128, BLOCK_K=128,
                        GROUP_SIZE_M=8, NUM_BLOCKS=NUM_SMS, SCALE_BLOCK=128,
                        PDL_LAUNCH=False,
                        num_warps=8, num_stages=4,
                    )
                _sg_bm = 4
                _sg_bn = 128
                _swiglu_fp8_quant_kernel[(
                    (M_total_param + _sg_bm - 1) // _sg_bm,
                    INTERMEDIATE_SIZE // _sg_bn,
                )](
                    ws.gemm1_bf16,
                    ws.intermediate_fp8,
                    ws.intermediate_scale,
                    M_total_param,
                    INTERMEDIATE_SIZE,
                    ws.gemm1_bf16.stride(0),
                    ws.intermediate_fp8.stride(0),
                    ws.intermediate_scale.stride(0),
                    BLOCK_N=_sg_bn, BLOCK_M=_sg_bm,
                    num_warps=4,
                )
            else:
                if not _use_fused_rg1:
                    _g1f_nw = _GEMM1_FUSED_CFG['num_warps']
                    _g1f_ns = _GEMM1_FUSED_CFG['num_stages']
                    _g1f_ws = _GEMM1_FUSED_CFG['WARP_SPEC']
                    _g1f_maxnreg = _GEMM1_FUSED_MAXNREG
                    if _USE_SCHEDULED_GEMM1:
                        _NUM_N_TILES_G1 = INTERMEDIATE_SIZE // 128
                        _grouped_gemm1_scheduled_gather[(_max_sched, _NUM_N_TILES_G1)](
                            hidden_states, hidden_states_scale, ws.sorted_token_ids,
                            gemm1_weights, ws.intermediate_fp8, ws.intermediate_scale,
                            gemm1_weights_scale,
                            ws.sched_expert, ws.sched_mstart, ws.sched_count,
                            offsets,
                            M_total_param, INTERMEDIATE_SIZE, 2 * INTERMEDIATE_SIZE,
                            gemm1_weights.stride(0), hidden_states_scale.stride(0),
                            K=HIDDEN_SIZE,
                            BLOCK_M=64, BLOCK_N=128, BLOCK_K=128,
                            NUM_KB=HIDDEN_SIZE // 128,
                            SCALE_BLOCK=128,
                            PDL_LAUNCH=_USE_PDL,
                            WARP_SPEC=_g1f_ws,
                            num_warps=_g1f_nw, num_stages=_g1f_ns,
                            launch_pdl=_USE_PDL,
                            **({'maxnreg': _g1f_maxnreg} if _g1f_maxnreg else {}),
                        )
                    else:
                        _grouped_gemm1_fused_gather[(NUM_SMS,)](
                            hidden_states, hidden_states_scale, ws.sorted_token_ids,
                            gemm1_weights, ws.intermediate_fp8, ws.intermediate_scale,
                            gemm1_weights_scale, offsets,
                            M_total_param, INTERMEDIATE_SIZE, 2 * INTERMEDIATE_SIZE,
                            gemm1_weights.stride(0), hidden_states_scale.stride(0),
                            NUM_LOCAL_EXPERTS,
                            K=HIDDEN_SIZE,
                            BLOCK_M=64, BLOCK_N=128, BLOCK_K=128,
                            GROUP_SIZE_M=_GEMM1_FUSED_GSM, NUM_BLOCKS=NUM_SMS, SCALE_BLOCK=128,
                            PDL_LAUNCH=_USE_PDL,
                            WARP_SPEC=_g1f_ws,
                            USE_DOT_SCALED=_USE_MXSCALE_GEMM1,
                            EPILOGUE_SUBTILE=_EPILOGUE_SUBTILE,
                            num_warps=_g1f_nw, num_stages=_g1f_ns,
                            launch_pdl=_USE_PDL,
                            **({'maxnreg': _g1f_maxnreg} if _g1f_maxnreg else {}),
                        )
    else:
        _grouped_gemm_fp8_swiglu_quant[(NUM_SMS,)](
            ws.permuted_hs, gemm1_weights, ws.intermediate_fp8, ws.intermediate_scale,
            ws.permuted_hs_scale, gemm1_weights_scale, offsets,
            M_total_param, INTERMEDIATE_SIZE, HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE,
            gemm1_weights.stride(0), M_total_param,
            NUM_LOCAL_EXPERTS,
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=128,
            GROUP_SIZE_M=8, NUM_BLOCKS=NUM_SMS, SCALE_BLOCK=128,
            USE_PDL=_USE_PDL,
            WARP_SPEC=True,
            EPILOGUE_SUBTILE=_EPILOGUE_SUBTILE,
            num_warps=8, num_stages=3,
            launch_pdl=_USE_PDL,
            **({'maxnreg': _GEMM1_SQ_MAXNREG} if _GEMM1_SQ_MAXNREG else {}),
        )

    # === Stage 5: GEMM2 + output combine ===
    _UNFUSE_THRESH = _GEMM2_UNFUSE_THRESH
    _NUM_N_TILES_G2 = HIDDEN_SIZE // 128

    if T <= _UNFUSE_THRESH:
        _g2_bk = _GEMM2_V4_CFG['BLOCK_K']
        _g2_nw = _GEMM2_V4_CFG['num_warps']
        _g2_ns = _GEMM2_V4_CFG['num_stages']
        _g2_ws = _GEMM2_V4_CFG['WARP_SPEC']
        _g2_nkb = INTERMEDIATE_SIZE // _g2_bk
        if T <= _SMALL_T_THRESH:
            _grouped_gemm2_bf16_flat[(_max_sched, _NUM_N_TILES_G2)](
                ws.intermediate_fp8, gemm2_weights,
                ws.gemm2_bf16,
                ws.intermediate_scale, gemm2_weights_scale,
                ws.sched_expert, ws.sched_mstart, ws.sched_count,
                offsets,
                ws.topk_weights, ws.perm,
                ws.inv_perm,
                ws.done_counter,
                ws.output,
                M_total_param, HIDDEN_SIZE, INTERMEDIATE_SIZE, TOP_K, NUM_LOCAL_EXPERTS,
                gemm2_weights.stride(0),
                ws.intermediate_scale.stride(0),
                T,
                BLOCK_M=_g2_bm, BLOCK_N=128, BLOCK_K=_g2_bk,
                W_SCALE_BLOCK=128,
                NUM_KB=_g2_nkb,
                USE_PDL=_USE_PDL,
                SCATTER=False,
                C_SCALE_BLOCK=_GEMM1_BN,
                WARP_SPEC=_g2_ws,
                num_warps=_g2_nw, num_stages=_g2_ns,
                launch_pdl=_USE_PDL,
                **({'maxnreg': _GEMM2_MAXNREG} if _GEMM2_MAXNREG else {}),
            )
            _GATHER_BLOCK_H = 1024
            num_h_tiles_gc = (HIDDEN_SIZE + _GATHER_BLOCK_H - 1) // _GATHER_BLOCK_H
            _gather_combine_kernel[(T, num_h_tiles_gc)](
                ws.gemm2_bf16, ws.topk_weights, ws.inv_perm, ws.output,
                HIDDEN_SIZE, TOP_K,
                ws.gemm2_bf16.stride(0),
                ws.topk_weights.stride(0),
                ws.output.stride(0),
                T,
                BLOCK_H=_GATHER_BLOCK_H,
                TOP_K=TOP_K,
                USE_PDL=_USE_PDL,
                num_warps=4,
                launch_pdl=_USE_PDL,
            )
        else:
            _grouped_gemm2_bf16_v4[(NUM_LOCAL_EXPERTS, _NUM_N_TILES_G2)](
                ws.intermediate_fp8, gemm2_weights,
                ws.gemm2_bf16,
                ws.intermediate_scale, gemm2_weights_scale,
                offsets,
                M_total_param, HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_LOCAL_EXPERTS,
                gemm2_weights.stride(0),
                ws.intermediate_scale.stride(0),
                BLOCK_M=_g2_bm, BLOCK_N=128, BLOCK_K=_g2_bk,
                W_SCALE_BLOCK=128,
                NUM_KB=_g2_nkb,
                USE_PDL=_USE_PDL,
                C_SCALE_BLOCK=_GEMM1_BN,
                WARP_SPEC=_g2_ws,
                num_warps=_g2_nw, num_stages=_g2_ns,
                launch_pdl=_USE_PDL,
                **({'maxnreg': _GEMM2_MAXNREG} if _GEMM2_MAXNREG else {}),
            )

            _GATHER_BLOCK_H = 1024
            num_h_tiles_gc = (HIDDEN_SIZE + _GATHER_BLOCK_H - 1) // _GATHER_BLOCK_H
            _gather_combine_kernel[(T, num_h_tiles_gc)](
                ws.gemm2_bf16, ws.topk_weights, ws.inv_perm, ws.output,
                HIDDEN_SIZE, TOP_K,
                ws.gemm2_bf16.stride(0),
                ws.topk_weights.stride(0),
                ws.output.stride(0),
                T,
                BLOCK_H=_GATHER_BLOCK_H,
                TOP_K=TOP_K,
                USE_PDL=_USE_PDL,
                num_warps=4,
                launch_pdl=_USE_PDL,
            )
    else:
        _g2p = _GEMM2_PERSISTENT_CFG
        _grouped_gemm2_persistent_bf16[(NUM_SMS,)](
            ws.intermediate_fp8, gemm2_weights,
            ws.gemm2_bf16,
            ws.intermediate_scale, gemm2_weights_scale,
            offsets,
            M_total_param, HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_LOCAL_EXPERTS,
            gemm2_weights.stride(0),
            ws.intermediate_scale.stride(0),
            BLOCK_M=_g2p['BLOCK_M'], BLOCK_N=_g2p['BLOCK_N'], BLOCK_K=_GEMM2_BK,
            GROUP_SIZE_M=_g2p['GROUP_SIZE_M'], NUM_BLOCKS=NUM_SMS,
            SCALE_BLOCK=_GEMM1_BN,
            W_SCALE_BLOCK=128,
            USE_PDL=False,
            WARP_SPEC=_g2p['WARP_SPEC'],
            num_warps=_g2p['num_warps'], num_stages=_g2p['num_stages'],
            **({'maxnreg': _g2p['maxnreg']} if _g2p.get('maxnreg') else {}),
        )
        _GATHER_BLOCK_H_LARGE = 1024
        num_h_tiles_gc = (HIDDEN_SIZE + _GATHER_BLOCK_H_LARGE - 1) // _GATHER_BLOCK_H_LARGE
        _gather_combine_kernel[(T, num_h_tiles_gc)](
            ws.gemm2_bf16, ws.topk_weights, ws.inv_perm, ws.output,
            HIDDEN_SIZE, TOP_K,
            ws.gemm2_bf16.stride(0),
            ws.topk_weights.stride(0),
            ws.output.stride(0),
            T,
            BLOCK_H=_GATHER_BLOCK_H_LARGE,
            TOP_K=TOP_K,
            USE_PDL=False,
            num_warps=4,
        )

    return ws.output
