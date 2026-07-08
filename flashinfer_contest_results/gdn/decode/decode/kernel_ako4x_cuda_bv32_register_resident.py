# AKO4X 对照 kernel；CUDA-Agent ours 使用同题目录下的 decode/kernel.py。
# AKO4X repo source: reference/gdn-decode/variants/cuda_bv32_register_resident/kernel.py
# Variant: cuda_bv32_register_resident
# Source: ako4fib-run-gdn_decode_v3-2/solution/kernel.py (iter-13 final;
#         post variance-check + dead-code cleanup, 2026-04-25).
#
# ─── Identity ─────────────────────────────────────────────────────────
# **1.18× ± 0.00174× (CV 0.10%)** 3-run variance-check, Modal B200 CUDA
# 13.2, 2026-04-25, 54/54 PASS. Per-B (variance mean):
#   B=1  → 1.49×   B=4  → 1.33×   B=8  → 1.15×   B=16 → 1.12×
#   B=32 → 0.986×  B=48 → 1.03×   B=64 → 0.999×
# Pure CUDA via torch.utils.cpp_extension.load_inline (language=python).
# Required flags:
#   [benchmark] use_isolated_runner = true   (input-pointer-keyed graph
#     cache aliases across workloads in persistent runners when PyTorch
#     recycles addresses; same correctness req as the Triton anchors).
#   _NO_GRAPH env gate kept at module scope so NCU can profile via
#     `scripts/profile.sh --env NO_GRAPH=1` (graph replay is invisible
#     to NCU). No production effect.
#
# ─── Delta from prior anchor (triton_swap_grid_evict_lsr 1.13×) ──────
# Drift-free A/B vs the Triton anchor: **+5.0% mean** (stacked +4.43%
# from the structural BV=32 step + ~+0.6% from default-STG on new_state
# + ~+0.4% from `ld.global.nc` on q/k). Per-B drift-free Δ at B≥32:
# +5.3% / +7.2% / +5.3% (B=32/48/64). Architecture vs Triton: same
# single-pass register-resident dataflow, but the Python dispatcher
# uses **BV=32 at B≥32**, halving the CTA grid (2048→1024 at B=32,
# 4096→2048 at B=64) — which Triton couldn't do because its register
# allocator over-committed at BV=32 and spilled (per the prior anchor's
# own "Dead-ends" entry). CUDA pins occupancy with `__launch_bounds__
# (128, 4)`, fitting comfortably at ~88 regs/thread per NCU.
#
# ─── Lessons on this variant ─────────────────────────────────────────
#
# 1. **BV=32 register-resident at B≥32 — the CUDA register-budget lever.**
#    HOW:   Python dispatcher branches BV ∈ {8, 16, 32}; B≥32 → BV=32
#           (template uses `state_tile[V_PER_WARP=8][K_PER_LANE=4]` =
#           32 fp32 regs/thread + ~50 misc). Pinned via
#           `__launch_bounds__(128, 4)`.
#    WHY:   Halves CTA count at B≥32 (2048→1024 at B=32, etc.). At B=16
#           BV=16 already wins +7% drift-free vs Triton with 1024 CTAs,
#           so pulling B=32 down to the same CTA count via a larger tile
#           recovers that win. Triton's register allocator predicted spill
#           at BV=32 and refused; CUDA + an explicit `min_blocks_per_sm`
#           cap fits without spill (NCU: 88 regs/thread, 25% achieved
#           occupancy, 5 blocks/SM).
#    WHEN narrow:  gdn-decode this shape, B≥32, K=128 fp32 state RMW.
#                  At B≤16 BV=32 under-fills (see Dead-ends).
#    WHEN broad:   matvec-decode kernels where (a) per-CTA setup overhead
#                  × CTA count is non-trivial (memory-bound but not at
#                  HBM peak), AND (b) the Triton allocator is spilling
#                  on a larger-tile variant that NVCC fits comfortably
#                  under a chosen `__launch_bounds__` pin.
#
# 2. **`ld.global.nc` on q/k (read-only cache path).**
#    HOW:   `asm volatile("ld.global.nc.v2.b32 {%0,%1}, [%2];" ...)` for
#           the two 8-byte bf16x4 q/k loads. State stays on
#           `ld.global.L1::evict_first.v4.f32` — `nc` is q/k-only.
#    WHY:   q/k are 256 B each, strictly read-only this kernel, and
#           shared across 4 sibling CTAs under SWAP_GRID at BV=32. The
#           read-only cache is a separate hierarchy from L1 streaming
#           pressure — keeps q/k hot during state's 16 KB streaming
#           pass without paying L1 thrash cost.
#    WHEN narrow:  q/k-style tiny read-only tensors (≤ a few KB) on
#                  this kernel.
#    WHEN broad:   any tiny read-only tensor sharing the SM with a
#                  large streaming tensor on the L1 path. Strictly
#                  size-gated — see Dead-ends for what happens when
#                  it's applied to the 16 KB state tile.
#
# 3. **Default STG on new_state (no eviction hint on the streaming write).**
#    HOW:   `*reinterpret_cast<float4*>(dst) = out4;` (compiler emits
#           plain `st.global.v4.f32`). Earlier iters carried
#           `st.global.L1::evict_first` from the Triton-anchor pattern;
#           dropping it gave +0.6% drift-free.
#    WHY:   B200 stores go L1-write-combine → L2 → HBM. The
#           `evict_first` hint adds a redundant L1 invalidation on a
#           path that's already streaming; default `wb` matches the
#           HW's natural store buffer behavior on this shape.
#    WHEN narrow:  fp32 new_state streaming write that won't be
#                  re-read in this kernel, B200 / sm_100.
#    WHEN broad:   streaming-store paths on Blackwell where L1 hint
#                  adds invalidation traffic that the underlying
#                  write-combine path doesn't need.
#
# ─── Dead-ends tried on this variant (expectation priors) ────────────
# Each is re-verifiable; don't trust blindly if your toolchain shifted.
#
# - **SMEM-staged state at BV=32 to lift register-block-limit.** −16%
#   at B=32. NCU showed the kernel was register-occupancy-bound and
#   suggested up to 68% local speedup from raising occupancy; staging
#   the 32-reg state_tile into 16 KB SMEM raised theoretical occupancy
#   but the LDS+STS round-trip + `__syncthreads` barrier exceeded the
#   register-pressure savings. NCU's "Est. Local Speedup" estimate was
#   misleading on this shape — see ../TRAPS.md for the cross-variant
#   form of this gotcha.
# - **`num_warps=8` at BV=32** (256 threads/CTA, V_PER_WARP=4). Drift-
#   free A/B vs num_warps=4: +0.14% (noise). Doubles concurrent LDGs
#   per CTA but doesn't hide more memory latency than already in
#   flight. State per thread halves from 32 to 16 regs but other live
#   ranges fill the gap.
# - **`-maxrregcount=72` cflag.** Neutral. Compiler already pins ~88
#   regs/thread under `launch_bounds(128, 4)`; tightening the cap
#   doesn't free more occupancy (other constraints saturate).
# - **BV=32 at B=16.** −10% at B=16. 512 CTAs / 148 SMs ≈ 3.5 CTAs/SM,
#   under-fill — confirms 1024 CTAs as the lower edge of "well-filled"
#   on B200 for this shape (see ../TRAPS.md "BV / CTA-count sweet spot").
# - **Removing q/k `evict_last`** (load with no L1 eviction hint).
#   −2% at B=32. The hint helps even with reduced sibling sharing
#   (4 CTAs at BV=32 vs 8 at BV=16); L2 still benefits from
#   persistence under state pressure.
# - **Disabling SWAP_GRID at BV=32.** −3% at B≥32. DRAM row-buffer
#   coherence still real with 4 sibling CTAs sharing (b, h) — the
#   benefit shrinks with BV but doesn't vanish.
# - **`ld.global.nc` on the state load.** −1 to −2%. State is 16 KB
#   streaming per CTA; texture cache is smaller than L1 and gets
#   thrashed. Confirms `nc` is a size-gated optimization (works for
#   q/k 256 B, not 16 KB state).
# - **Parameter sweeps** around `{__launch_bounds__ minBlocks ∈
#   {2, 5, 6}, BV=32 at B=8, swap_grid disabled at all B}` — all
#   neutral or regressed. Retry only with new reasoning.
#
# ─── Open directions ─────────────────────────────────────────────────
# - **TMA + warp-spec via `cp.async.bulk.tensor` + `mbarrier`.** The
#   classic "producer warp issues bulk transfer, consumer warps run
#   gates/q/k" pattern. Estimated +0–5% with significant regression
#   risk: the SMEM-staged dead-end above already showed this kernel
#   pays a high SMEM round-trip cost; TMA only wins if warp-spec
#   genuinely hides that cost (unverified). Multi-hour rewrite for
#   marginal expected gain. NCU on iter-5 showed Memory Throughput
#   only 36% at B=48 — there IS bandwidth headroom, but unlocking it
#   has been blocked by the SMEM-stage cost.

import math
import os

_NO_GRAPH = bool(os.environ.get("NO_GRAPH"))

import torch
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#define HV 8
#define HQ 4
#define K_DIM 128
#define HQV_RATIO 2

// BV ∈ {8, 16, 32}. NUM_WARPS = 4 → 128 threads/CTA.
// Thread layout: (warp_id, lane_id) = (tid/32, tid%32).
// Each warp handles V_PER_WARP = BV / NUM_WARPS V-rows.
// Each lane handles K_PER_LANE = K_DIM / 32 = 4 K-cols.
// state_tile laid out as per-thread [V_PER_WARP][K_PER_LANE] fp32 regs.
// __launch_bounds__ caps regs so BV=32 at 8 V-rows/warp doesn't spill
// (Triton BV=32 regressed 2% due to spill; CUDA with explicit reg budget avoids).
template<int BV, bool SWAP_GRID, bool HAS_STATE>
__launch_bounds__(128, 4)
__global__ void gdn_decode_kernel(
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ k_ptr,
    const __nv_bfloat16* __restrict__ v_ptr,
    const float* __restrict__ state_ptr,
    const float* __restrict__ A_log_ptr,
    const __nv_bfloat16* __restrict__ a_ptr,
    const float* __restrict__ dt_bias_ptr,
    const __nv_bfloat16* __restrict__ b_in_ptr,
    __nv_bfloat16* __restrict__ out_ptr,
    float* __restrict__ new_state_ptr,
    float scale_f)
{
    constexpr int NUM_WARPS = 4;
    constexpr int V_PER_WARP = BV / NUM_WARPS;
    constexpr int K_PER_LANE = K_DIM / 32;  // 4

    int pid_v, pid_bh;
    if constexpr (SWAP_GRID) {
        pid_v  = blockIdx.x;
        pid_bh = blockIdx.y;
    } else {
        pid_bh = blockIdx.x;
        pid_v  = blockIdx.y;
    }
    int b    = pid_bh / HV;
    int h    = pid_bh % HV;
    int h_qk = h / HQV_RATIO;

    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int v_start_cta  = pid_v * BV;
    int v_start_warp = v_start_cta + warp_id * V_PER_WARP;
    int k_start_lane = lane_id * K_PER_LANE;

    long state_base = (long)b * HV * K_DIM * K_DIM + (long)h * K_DIM * K_DIM;

    // ── Issue state load FIRST so HBM request is in flight
    //    while gates / q / k / v compute (~300 cycles hide).
    //    evict_first: 64 KB/(b,h) streaming; don't pollute L2.
    float state_tile[V_PER_WARP][K_PER_LANE];
    if constexpr (HAS_STATE) {
        #pragma unroll
        for (int vr = 0; vr < V_PER_WARP; vr++) {
            int v_idx = v_start_warp + vr;
            const float* row = state_ptr + state_base + (long)v_idx * K_DIM + k_start_lane;
            float4 f4;
            asm volatile(
                "ld.global.L1::evict_first.v4.f32 {%0, %1, %2, %3}, [%4];"
                : "=f"(f4.x), "=f"(f4.y), "=f"(f4.z), "=f"(f4.w)
                : "l"(row)
            );
            state_tile[vr][0] = f4.x;
            state_tile[vr][1] = f4.y;
            state_tile[vr][2] = f4.z;
            state_tile[vr][3] = f4.w;
        }
    } else {
        #pragma unroll
        for (int vr = 0; vr < V_PER_WARP; vr++) {
            #pragma unroll
            for (int kc = 0; kc < K_PER_LANE; kc++) state_tile[vr][kc] = 0.0f;
        }
    }

    // ── Load q/k via read-only cache (ld.global.nc), per-lane bf16x4 8-byte load.
    //    q/k are strictly read-only and shared across sibling CTAs; the read-only
    //    cache path has a separate cache hierarchy from the state-streaming L1.
    int qk_base = b * (HQ * K_DIM) + h_qk * K_DIM;
    const __nv_bfloat16* q_row = q_ptr + qk_base + k_start_lane;
    const __nv_bfloat16* k_row = k_ptr + qk_base + k_start_lane;
    uint2 q_raw, k_raw;  // 2 × u32 = 4 bf16 each
    asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                 : "=r"(q_raw.x), "=r"(q_raw.y) : "l"(q_row));
    asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                 : "=r"(k_raw.x), "=r"(k_raw.y) : "l"(k_row));
    float q_lane[K_PER_LANE], k_lane[K_PER_LANE];
    q_lane[0] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(q_raw.x & 0xFFFFu)));
    q_lane[1] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(q_raw.x >> 16)));
    q_lane[2] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(q_raw.y & 0xFFFFu)));
    q_lane[3] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(q_raw.y >> 16)));
    k_lane[0] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(k_raw.x & 0xFFFFu)));
    k_lane[1] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(k_raw.x >> 16)));
    k_lane[2] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(k_raw.y & 0xFFFFu)));
    k_lane[3] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(k_raw.y >> 16)));

    // ── Load v (only V_PER_WARP values per warp, per warp's own V range)
    //    Each warp's lane `l` needs v[warp*V_PER_WARP + l] for l < V_PER_WARP.
    int v_base = b * (HV * K_DIM) + h * K_DIM + v_start_cta;
    float v_mine = 0.0f;
    if (lane_id < V_PER_WARP) {
        v_mine = __bfloat162float(v_ptr[v_base + warp_id * V_PER_WARP + lane_id]);
    }

    // ── Gates ──
    float A_log_h  = A_log_ptr[h];
    float a_h      = __bfloat162float(a_ptr[b * HV + h]);
    float dt_bias_h = dt_bias_ptr[h];
    float b_h      = __bfloat162float(b_in_ptr[b * HV + h]);
    float x        = a_h + dt_bias_h;
    float softplus_x = (x > 20.0f) ? x : __logf(1.0f + __expf(x));
    float g        = __expf(-__expf(A_log_h) * softplus_x);
    float beta     = 1.0f / (1.0f + __expf(-b_h));

    // ── Compute kdot_v[vr] = sum_k state[vr,k]*k_vec[k]
    //         qs[vr]     = sum_k state[vr,k]*q_vec[k]
    //         qk_dot     = sum_k q_vec[k]*k_vec[k]
    float kdot_v[V_PER_WARP], qs_v[V_PER_WARP];
    #pragma unroll
    for (int vr = 0; vr < V_PER_WARP; vr++) {
        float pk = 0.0f, pq = 0.0f;
        #pragma unroll
        for (int kc = 0; kc < K_PER_LANE; kc++) {
            pk += state_tile[vr][kc] * k_lane[kc];
            pq += state_tile[vr][kc] * q_lane[kc];
        }
        // Warp-all-reduce: every lane gets the sum
        #pragma unroll
        for (int offs = 16; offs > 0; offs >>= 1) {
            pk += __shfl_xor_sync(0xffffffffu, pk, offs);
            pq += __shfl_xor_sync(0xffffffffu, pq, offs);
        }
        kdot_v[vr] = pk;
        qs_v[vr]   = pq;
    }

    float qk_partial = 0.0f;
    #pragma unroll
    for (int kc = 0; kc < K_PER_LANE; kc++) qk_partial += q_lane[kc] * k_lane[kc];
    #pragma unroll
    for (int offs = 16; offs > 0; offs >>= 1)
        qk_partial += __shfl_xor_sync(0xffffffffu, qk_partial, offs);
    float qk_dot = qk_partial;

    // ── delta_v[vr] = beta * (v[vr] - g * kdot_v[vr])
    //    v for row vr is held in lane `vr`'s `v_mine` register; shuffle to broadcast.
    float delta_v[V_PER_WARP];
    #pragma unroll
    for (int vr = 0; vr < V_PER_WARP; vr++) {
        float v_val = __shfl_sync(0xffffffffu, v_mine, vr);
        delta_v[vr] = beta * (v_val - g * kdot_v[vr]);
    }

    // ── Write output (bf16), V_PER_WARP values per warp.
    //    All lanes have the reduced value; lanes 0..V_PER_WARP-1 each write one.
    if (lane_id < V_PER_WARP) {
        int vr = lane_id;
        float out_val = scale_f * (g * qs_v[vr] + qk_dot * delta_v[vr]);
        int v_idx = v_start_warp + vr;
        out_ptr[b * (HV * K_DIM) + h * K_DIM + v_idx] = __float2bfloat16(out_val);
    }

    // ── Write new_state: default STG.128 (no eviction hint).
    #pragma unroll
    for (int vr = 0; vr < V_PER_WARP; vr++) {
        int v_idx = v_start_warp + vr;
        float o0 = g * state_tile[vr][0] + delta_v[vr] * k_lane[0];
        float o1 = g * state_tile[vr][1] + delta_v[vr] * k_lane[1];
        float o2 = g * state_tile[vr][2] + delta_v[vr] * k_lane[2];
        float o3 = g * state_tile[vr][3] + delta_v[vr] * k_lane[3];
        float* dst = new_state_ptr + state_base + (long)v_idx * K_DIM + k_start_lane;
        float4 out4 = {o0, o1, o2, o3};
        *reinterpret_cast<float4*>(dst) = out4;
    }
}


// ── Host dispatcher ────────────────────────────────────────
void gdn_decode_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor state_arg,   // either real state or new_state buffer (dummy) if has_state=false
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b_in,
    torch::Tensor output,
    torch::Tensor new_state,
    double scale_f,
    bool has_state,
    int64_t BV,
    bool swap_grid)
{
    int B = q.size(0);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    const __nv_bfloat16* q_p    = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    const __nv_bfloat16* k_p    = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
    const __nv_bfloat16* v_p    = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
    const float*         state_p = reinterpret_cast<const float*>(state_arg.data_ptr());
    const float*         Alog_p  = reinterpret_cast<const float*>(A_log.data_ptr());
    const __nv_bfloat16* a_p    = reinterpret_cast<const __nv_bfloat16*>(a.data_ptr());
    const float*         dtb_p  = reinterpret_cast<const float*>(dt_bias.data_ptr());
    const __nv_bfloat16* bin_p  = reinterpret_cast<const __nv_bfloat16*>(b_in.data_ptr());
    __nv_bfloat16*       out_p  = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
    float*               ns_p   = reinterpret_cast<float*>(new_state.data_ptr());

    dim3 block(128);
    float scale = (float)scale_f;

    #define DISPATCH_LAUNCH(BV_V, SWAP_V, HAS_V) \
        gdn_decode_kernel<BV_V, SWAP_V, HAS_V><<<grid, block, 0, stream>>>( \
            q_p, k_p, v_p, state_p, Alog_p, a_p, dtb_p, bin_p, out_p, ns_p, scale)

    if (swap_grid) {
        dim3 grid(K_DIM / BV, B * HV);
        if (BV == 32) {
            if (has_state) DISPATCH_LAUNCH(32, true, true);
            else           DISPATCH_LAUNCH(32, true, false);
        } else if (BV == 16) {
            if (has_state) DISPATCH_LAUNCH(16, true, true);
            else           DISPATCH_LAUNCH(16, true, false);
        } else { // BV == 8
            if (has_state) DISPATCH_LAUNCH(8, true, true);
            else           DISPATCH_LAUNCH(8, true, false);
        }
    } else {
        dim3 grid(B * HV, K_DIM / BV);
        if (BV == 32) {
            if (has_state) DISPATCH_LAUNCH(32, false, true);
            else           DISPATCH_LAUNCH(32, false, false);
        } else if (BV == 16) {
            if (has_state) DISPATCH_LAUNCH(16, false, true);
            else           DISPATCH_LAUNCH(16, false, false);
        } else { // BV == 8
            if (has_state) DISPATCH_LAUNCH(8, false, true);
            else           DISPATCH_LAUNCH(8, false, false);
        }
    }
    #undef DISPATCH_LAUNCH
}
"""

CPP_SRC = r"""
void gdn_decode_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor state_arg,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b_in,
    torch::Tensor output,
    torch::Tensor new_state,
    double scale_f,
    bool has_state,
    int64_t BV,
    bool swap_grid);
"""


_module = load_inline(
    name="gdn_decode_qk4_v8_d128_k_last_cuda_bv32_register_resident",
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    functions=["gdn_decode_cuda"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-arch=sm_100a",
        "-std=c++17",
    ],
    verbose=False,
)


# ═══════════════════════════════════════════════════════════════
# CUDA graph cache — input-ptr-keyed, same pattern as Triton anchor.
# ═══════════════════════════════════════════════════════════════
_graph_cache = {}
_graph_cnt = {}
_static_out = {}
_static_new_state = {}

_last_key = None
_last_graph = None
_last_out = None
_last_new_state = None


def _launch(q, k, v, state_arg, A_log, a, dt_bias, b_in,
            output, new_state, scale_f, B, BV, has_state):
    swap_grid = B >= 32
    _module.gdn_decode_cuda(
        q, k, v, state_arg, A_log, a, dt_bias, b_in,
        output, new_state,
        scale_f, has_state, BV, swap_grid,
    )


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    global _last_key, _last_graph, _last_out, _last_new_state

    B = q.shape[0]
    has_state = state is not None

    state_ptr = state.data_ptr() if has_state else 0
    key = (q.data_ptr(), k.data_ptr(), v.data_ptr(), state_ptr,
           a.data_ptr(), b.data_ptr(), A_log.data_ptr(), dt_bias.data_ptr(), B)

    if key == _last_key and _last_graph is not None:
        _last_graph.replay()
        return _last_out, _last_new_state

    g = _graph_cache.get(key)
    if g is not None:
        _last_key = key
        _last_graph = g
        _last_out = _static_out[B]
        _last_new_state = _static_new_state[B]
        g.replay()
        return _last_out, _last_new_state

    if B not in _static_out:
        device = q.device
        _static_out[B] = torch.empty(B, 1, 8, 128, dtype=torch.bfloat16, device=device)
        _static_new_state[B] = torch.empty(B, 8, 128, 128, dtype=torch.float32, device=device)
    output = _static_out[B]
    new_state = _static_new_state[B]

    if scale is None or (isinstance(scale, float) and scale == 0.0):
        scale_f = 1.0 / math.sqrt(128)
    else:
        scale_f = float(scale)

    # BV dispatch:
    #   B<=8  → BV=8  (128 CTAs @ B=1; SM fill)
    #   B==16 → BV=16 (1024 CTAs; sweet spot, A/B shows +7% vs Triton)
    #   B>=32 → BV=32 (halves CTA count: 1024/1536/2048 vs 2048/3072/4096)
    if B <= 8:
        BV = 8
    elif B <= 16:
        BV = 16
    else:
        BV = 32
    state_arg = state if has_state else new_state

    cnt = _graph_cnt.get(key, 0) + 1
    _graph_cnt[key] = cnt

    def do_launch():
        _launch(q, k, v, state_arg, A_log, a, dt_bias, b,
                output, new_state, scale_f, B, BV, has_state)

    do_launch()

    if cnt >= 2 and not _NO_GRAPH:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            do_launch()
        _graph_cache[key] = graph
        _last_key = key
        _last_graph = graph
        _last_out = output
        _last_new_state = new_state

    return output, new_state
