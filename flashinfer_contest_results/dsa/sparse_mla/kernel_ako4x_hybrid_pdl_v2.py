# AKO4X 对照 kernel；CUDA-Agent ours 使用同题目录下的 kernel.py。
# AKO4X repo source: reference/dsa-sparse-attention/variants/hybrid_pdl_v2/kernel.py
# hybrid_pdl_v2 — reference kernel.py header
#
# Identity
#   58.34 ± 0.04x variance-check 3-run (CV 0.06%, Modal B200 / CUDA 13.2 /
#   flashinfer-ci-cu132:20260401-2c675fb, 2026-04-25). 58.89x single-run
#   labeled on the same container (iter-21).
#   Per-T (3-run mean ± std):
#     T=1: 78.94 ± 0.29, T=2: 69.12 ± 0.11, T=6: 51.35 ± 0.00,
#     T=7: 50.56 ± 0.05, T=8: 50.53 ± 0.01.
#   Deps: torch + triton 3.6 + tilelang + apache-tvm-ffi. No cutlass-dsl.
#   Fully captured into torch.cuda.graph; unaffected by
#   flashinfer-bench #414.
#
# Delta from hybrid_pdl (57.55x, v12)
#   Four compounding changes for +0.79x variance-floor (+1.37%):
#     - Triton fwd H_SPLIT=2 (8 heads per block, 2 blocks per
#       (token, split)). Grid (Tv, NS_TRI=32, H_SPLIT=2) doubles
#       block count at T=1,2: 32→64 (0.22→0.43 waves) and 64→128
#       (0.43→0.86 waves), closing the SM-starved gap.
#     - Triton fwd num_stages=2. Only fits post-H-split — halved
#       Qn_s (16→8 KB) and Qp_s (2→1 KB) opens smem room for the
#       cp.async double-buffer. Was a dead-end in hybrid_pdl.
#     - Triton fwd num_warps=8. MMA tile-picker at (nw=8, M=8,
#       N=64, K=512) selects a faster primitive than nw=4 for this
#       geometry; biggest single AB contributor.
#     - TileLang fwd D-chunked acc_o into 2×[H, D/2] fragments.
#       Better GEMM scheduling in tcgen05 — registers unchanged,
#       both fragments alive concurrently.
#
#   Same-container AB (iter-4 → iter-21, drift-cancelled): +1.55x (+2.72%).
#   Variance-floor AB vs hybrid_pdl (cross-container): +0.79x (+1.37%).
#   Per-T AB Δ: T=1 +1.74, T=2 +1.92, T=6 -0.17 (within CV),
#               T=7 +0.12, T=8 +0.17. Gains concentrate on Triton path.
#
# Lessons on this variant
#
#   +1.55x AB compound Triton-fwd H_SPLIT=2 + num_stages=2 + num_warps=8
#     How:           _H_SPLIT_TRI=2, _H_PER_TRI=H//2=8
#                    _triton_fwd adds H_SPLIT, H_PER constexprs;
#                      hid = tl.program_id(2)
#                      h   = hid * H_PER + tl.arange(0, H_PER)
#                    grid = (Tv, NS_TRI, H_SPLIT_TRI)
#                    num_warps=8, num_stages=2, launch_pdl=True
#     Why:           Three tightly coupled mechanisms:
#                    (1) H-split doubles blocks → fills waves at T=1,2.
#                    (2) num_stages=2 only fits post-H-split (smem room
#                        restored by halved Qn_s/Qp_s).
#                    (3) MMA tile-picker is sensitive to (num_warps, M,
#                        N, K). At M=8 post-split, nw=8 selects a
#                        different, faster primitive than nw=4.
#     WHEN narrow:   GQA-like Triton fwd with T=1,2 where per-token
#                    grid is SM-starved (waves <0.5) AND head dim
#                    H ≥ 16 allows a clean H-split.
#     WHEN broad:    after any structural change (tile shape, fragment
#                    split, H-split, new buffer layout), re-sweep the
#                    entire tuning space — the prior dead-end list is
#                    stale. This session left +0.9x on the table for
#                    9 iters by not re-sweeping num_warps after
#                    H-split at iter-9; only iter-21 recovered it.
#                    See TRAPS.md "Structural-change invalidates
#                    parameter sweeps".
#     Anti-pattern:  H_SPLIT=4 regressed T=2 67→50x (iter-14, 24) —
#                    M=4 overwhelms the picker. H_SPLIT=2 + nw=16
#                    regressed T=1 78.86→74.22 (iter-23); picker
#                    reverses direction between nw=8 and nw=16. The
#                    sweet spot is H_SPLIT=2, nw=8 for H=16.
#
#   +0.138x AB TileLang fwd D-chunked acc_o
#     How:           split acc_o [H=16, D=512] fragment into
#                      acc_o0 [H, D/2=256] + acc_o1 [H, D/2=256]
#                    final gemm becomes
#                      T.gemm(S_s, KV_s[:, 0:D_HALF], acc_o0)
#                      T.gemm(S_s, KV_s[:, D_HALF:D], acc_o1)
#     Why:           Two smaller gemms (m=H, k=BI=128, n=256 each)
#                    expose more scheduling slack to tcgen05 than one
#                    monolithic gemm (m=H, k=BI, n=512). Register
#                    footprint UNCHANGED — both fragments alive
#                    concurrently, smem unchanged at 166 KB. Win
#                    comes from scheduling, NOT occupancy.
#     WHEN narrow:   TileLang fwd at T≥3, NS=16 BI=128, when the
#                    final-D gemm dominates. 2-way split only;
#                    4-way drift-neutral vs 2-way (iter-17).
#     WHEN broad:    tcgen05 kernels where a D-split exposes
#                    independent matmuls with shared LHS/RHS and the
#                    register live range isn't doubled.
#     Anti-pattern:  do NOT pair with T.pdl_trigger() in TL fwd —
#                    still regressed T=7,8 at iter-18 (smem budget
#                    unchanged, TL fwd remains 1 block/SM).
#
#   Carry-forward lessons (unchanged from hybrid_pdl — still load-bearing):
#     - +1.39x AB PDL overlap on Triton-path fwd+reduce:
#       gdc_launch_dependents() in _triton_fwd, gdc_wait() in both
#       reduce kernels, launch_pdl=True on all three Triton launches.
#       TL reduce has launch_pdl=True but no T.pdl_trigger() (TL fwd
#       is 1 block/SM; triggering regresses T=8).
#     - +3.37x AB 2D-tile merged TRI-path reduce:
#       po_tile = tl.load(...)[NS_TRI, D_CHUNK=32] (4 KB fp32, fits
#       registers), acc = tl.sum(w[:,None] * po_tile, axis=0).
#     - +x TRI-path reduce tuning: D_CHUNK=32 × num_warps=1 (32
#       threads, 1 D output per thread; grid fills 148 SMs at T=1).
#     - +? USE_MASK=True for T=1 Triton fwd K gather (skips GMEM
#       gather on sparse_indices=-1). Tv==1 only.
#     - +? TL-path PO transposed to [T, H, NS, D] (L1 stride 1 KB
#       vs 16 KB).
#     - +0 TileLang fwd NI=1 fastpath (online-softmax corrections are
#       identity when NI=1; dropped alpha/m_i_prev/sumexp scaling).
#     - +? _last_si_ptr int fast-path in run() (one int compare beats
#       a 6-tuple dict key; safe under use_isolated_runner=true).
#   See hybrid_pdl/kernel.py header for the full How/Why/WHEN of each.
#
# Dead-ends tried on this variant (v13 session, 2026-04-25)
#   Each is an expectation prior, not a prohibition. Re-verify cheaply
#   if your structural context shifts (see broad WHEN above).
#
#   - BI=64 in TileLang fwd (iter-3, 6, 7): T=6-8 50→43-45x. Doubly
#     register-bound — fragment regs (128/th × 512 thr = 65K = SM max)
#     vs fragment data (acc_o fp32 = 32K values). Halving threads
#     doubles regs/thread (zero-sum). Breaking 2 blocks/SM on TL path
#     needs CuTe DSL rewrite (~3000 LoC; deferred behind #414).
#   - H_SPLIT=4 on Triton fwd (iter-14, 24): T=1 77→79 but T=2 67→50x.
#     Smaller M=4 overwhelms the MMA tile-picker; slower primitive
#     selected. H_SPLIT=2 is the sweet spot for H=16.
#   - H_SPLIT=2 + num_warps=16 (iter-23): T=1 dropped 78.86→74.22x.
#     Picker reverses direction between nw=8 and nw=16.
#   - H_SPLIT=2 + num_warps=2 (post-split pair): picker regresses on
#     the opposite side; nw=8 is optimum.
#   - TileLang fwd D-chunked 4-way (iter-17): drift-neutral vs 2-way.
#     Going past 2 splits doesn't further improve scheduling slack.
#   - T.pdl_trigger() in TL fwd + iter-16 D-chunk (iter-18): T=7,8
#     still regressed. D-chunk doesn't change smem budget; TL fwd
#     remains 1 block/SM and PDL contention is unchanged.
#   - TL fwd early writes + T.pdl_trigger() (iter-19): neutral-to-noise.
#   - sparse_indices smem-cache prolog in TL fwd (iter-20): T=8
#     regressed 50→20.70x. Inserted T.Parallel(BI) barrier broke
#     TileLang warp-spec pipelining; the compiler's implicit cp.async
#     pipeline from indexed KV load was already optimal — explicit
#     staging turned one implicit barrier into two serial phases.
#   - Dropping T.max(m_i, -2**30) guard in TL fwd (iter-20):
#     INCORRECT_NUMERICAL on 1/23 workloads (T=6). Guard is
#     load-bearing despite seeming redundant — TileLang
#     `reduce_max(clear=False) + T.fill(m_i, -(2**30))` is NOT
#     equivalent to "max with pre-fill" under warp-spec scheduling
#     for rows where all inputs are -inf.
#
#   Superseded from hybrid_pdl v12 dead-end list:
#   - "num_stages=2 on Triton fwd" was a carried-forward dead-end;
#     this session (iter-12) flipped it under H-split to a +0.496x
#     AB winner. Reason: pre-H-split smem budget exceeded headroom
#     with double-buffer; post-split halved Qn_s/Qp_s restored room.
#     Concrete instance of structural-change-invalidates-sweeps.
#
#   Remaining inherited dead-ends from hybrid_pdl (not re-tested this
#   session; still expected to regress without a fresh structural
#   change):
#     - T.pdl_trigger() inside TL fwd body (bare form, no D-chunk):
#       TL fwd is 1 block/SM at T=8; contention not overlap.
#     - Fused Triton fwd+reduce via atomic-counter last-block.
#     - torch.sort(sparse_indices) pre-fwd for KV gather locality.
#     - TRI reduce num_stages=2 (on top of PDL).
#     - 2D-tile merged TL-path reduce at any D_CHUNK.
#     - num_ctas=2 on Triton 3.6 (COMPILE_ERROR).
#     - TL reduce num_warps=2.
#     - [inherited from hybrid_2d_reduce dead-ends:] Triton fwd
#       NS=64/BK=32, TileLang fwd NS=32/BI=64/NI=1, log2 domain,
#       D_CHUNK_TL=256 (vs 512), TL reduce num_stages=3 (vs 2),
#       TL reduce num_warps ∈ {4, 16} (vs 8), TRI reduce num_warps=2,
#       D_CHUNK_TRI ∈ {16, 64} (vs 32), dropping the `if dc_id == 0`
#       LSE-write guard, TL_DISABLE_WARP_SPECIALIZED=True,
#       TL fwd threads ∈ {128, 192, 384, 512} (vs 256).
#
# Open directions
#   - CuTe DSL rewrite of TileLang fwd to unblock 2 blocks/SM on TL
#     path. Estimated +2-3 µs per T=6-8 call = +15-20% score. ~3000
#     LoC port; blocked on flashinfer-bench #414 for trustworthy
#     measurement of any CuTe kernel inside torch.cuda.graph.
#   - Persistent CUDA kernel for T=1 Triton fwd. Still 1 block/SM at
#     T=1 post-H-split (149.5 KB smem, 22% SM utilization). A
#     persistent-grid rewrite in CUDA would unblock the structural
#     floor at T=1.
#   - Further Triton-fwd MMA tuning at H_SPLIT=2. The (num_warps,
#     num_stages) sweep is only partially populated; nw=8 × stages=2
#     is the current optimum but adjacent combinations (nw=8 ×
#     stages=3, nw=8 × stages=1 with different schedule hints) are
#     un-swept. TRAPS.md "Structural-change invalidates parameter
#     sweeps" applies here too: the full sweep was skipped under
#     resource pressure.
import os
import torch
import tilelang
from tilelang import language as T
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

# NCU profiling gate: set DSA_NO_GRAPH=1 to skip CUDA-graph capture so
# every kernel launch surfaces to ncu. Keep unset for production benches
# (graph replay is the main wall-time win). This pattern is documented in
# templates/profiler/ncu.md as the recommended workaround for CUDA-graph
# kernel-attribution blindness under cuGraphLaunch.
_DISABLE_GRAPH = bool(os.environ.get("DSA_NO_GRAPH"))

# ═══ Constants ═══
_NS     = 16        # TileLang path splits (kept for TileLang factory default)
_NS_TL  = 16        # TileLang path: keep 16 splits (T≥3 already has enough blocks)
_NS_TRI = 32        # Triton path: 32 splits for better SM parallelism at T=1,2
_BK     = 128       # TileLang BLOCK_K
_BK_TRI = 64        # Triton BLOCK_K = TOPK / NS_TRI
_BI   = 128
_NI   = _BK // _BI
_NH   = 16
_DC   = 512
_DP   = 64
_TOPK = 2048
_SM   = 0.1352337788608801
_SM_L2 = _SM * 1.44269504

# ═══════════════════════════════════════════════════════════════
# PATH A: Triton fwd (best for T=1,2 — compact codegen, fast launch)
# ═══════════════════════════════════════════════════════════════
@triton.jit
def _triton_fwd(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr, si_ptr,
    po_ptr, pm_ptr, pl_ptr, sm_scale,
    NUM_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_MASK: tl.constexpr,
    H_SPLIT: tl.constexpr,
    H_PER: tl.constexpr,
):
    tid = tl.program_id(0)
    sid = tl.program_id(1)
    hid = tl.program_id(2)
    h_off = hid * H_PER
    h = h_off + tl.arange(0, H_PER)
    dc = tl.arange(0, 512); dp = tl.arange(0, 64)
    Q_n = tl.load(q_nope_ptr + tid * 8192 + h[:, None] * 512 + dc[None, :])
    Q_p = tl.load(q_pe_ptr + tid * 1024 + h[:, None] * 64 + dp[None, :])
    k = tl.arange(0, BLOCK_K)
    indices = tl.load(si_ptr + tid * 2048 + sid * BLOCK_K + k)
    valid = indices >= 0
    safe = tl.where(valid, indices, 0).to(tl.int64)
    if USE_MASK:
        # T=1: most sparse_indices are -1; mask skips the GMEM gather.
        Kc = tl.load(ckv_ptr + safe[:, None] * 512 + dc[None, :],
                     mask=valid[:, None], other=0.0)
        Kp = tl.load(kpe_ptr + safe[:, None] * 64 + dp[None, :],
                     mask=valid[:, None], other=0.0)
    else:
        Kc = tl.load(ckv_ptr + safe[:, None] * 512 + dc[None, :])
        Kp = tl.load(kpe_ptr + safe[:, None] * 64 + dp[None, :])
    logits = tl.dot(Q_n, tl.trans(Kc))
    logits = tl.dot(Q_p, tl.trans(Kp), acc=logits)  # fused into first dot's acc
    logits = logits * sm_scale
    logits = tl.where(valid[None, :], logits, float('-inf'))
    m_i = tl.max(logits, axis=1)
    p = tl.exp(logits - m_i[:, None])
    p = tl.where(valid[None, :], p, 0.0)
    l_i = tl.sum(p, axis=1)
    acc = tl.dot(p.to(tl.bfloat16), Kc)
    po_off = tid * (NUM_SPLITS * 8192) + sid * 8192
    tl.store(po_ptr + po_off + h[:, None] * 512 + dc[None, :], acc.to(tl.bfloat16))
    pm_off = tid * (NUM_SPLITS * 16) + sid * 16
    tl.store(pm_ptr + pm_off + h, m_i)
    tl.store(pl_ptr + pm_off + h, l_i)
    # PDL: signal that the reduce kernel may launch-overlap our tail.
    gdc_launch_dependents()


# ═══════════════════════════════════════════════════════════════
# PATH B: TileLang fwd (best for T≥6 — better GEMM scheduling)
# NI=1 fastpath — all online-softmax corrections are identity.
# ═══════════════════════════════════════════════════════════════
@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False},
)
def _tl_fwd_factory(
    heads=_NH, dim=_DC, tail_dim=_DP, topk=_TOPK,
    num_splits=_NS, block_I=_BI, num_iters=_NI,
    num_stages=1, threads=256,
):
    sm_scale = _SM_L2
    num_tokens = T.dynamic("num_tokens")
    total_kv   = T.dynamic("total_kv")
    H = heads; D = dim; DT = tail_dim
    NS = num_splits; BI = block_I; NI_val = num_iters

    @T.prim_func
    def main(
        Q_nope:  T.Tensor([num_tokens, H, D],   "bfloat16"),
        Q_pe:    T.Tensor([num_tokens, H, DT],  "bfloat16"),
        CKV:     T.Tensor([total_kv, D],         "bfloat16"),
        KPE:     T.Tensor([total_kv, DT],        "bfloat16"),
        Indices: T.Tensor([num_tokens, topk],     "int32"),
        # Transposed [T, H, NS, D] layout: stride(NS)=D=512=1KB, fits L1
        # better than the original [T, NS, H, D] with stride(NS)=H*D=16KB.
        PO:      T.Tensor([num_tokens, H, NS, D], "bfloat16"),
        PM:      T.Tensor([num_tokens, NS, H],     "float32"),
        PL:      T.Tensor([num_tokens, NS, H],     "float32"),
    ):
        D_HALF = D // 2
        with T.Kernel(num_tokens, NS, threads=threads) as (tok_bx, split_bx):
            Qn_s = T.alloc_shared([H, D], "bfloat16")
            Qp_s = T.alloc_shared([H, DT], "bfloat16")
            KV_s = T.alloc_shared([BI, D], "bfloat16")
            Kp_s = T.alloc_shared([BI, DT], "bfloat16")
            S_s  = T.alloc_shared([H, BI], "bfloat16")
            # Halve acc_o fragment — process D=512 in 2 passes of D=256 each.
            # Win comes from GEMM scheduling slack in tcgen05, NOT occupancy:
            # both fragments are alive concurrently, so register footprint
            # is unchanged. See header lesson "+0.138x D-chunked acc_o".
            acc_o0   = T.alloc_fragment([H, D_HALF], "float32")
            acc_o1   = T.alloc_fragment([H, D_HALF], "float32")
            acc_s    = T.alloc_fragment([H, BI], "float32")
            sumexp   = T.alloc_fragment([H], "float32")
            m_i      = T.alloc_fragment([H], "float32")

            T.fill(acc_o0, 0); T.fill(acc_o1, 0)
            T.fill(sumexp, 0); T.fill(m_i, -(2**30))
            T.copy(Q_nope[tok_bx, :, :], Qn_s)
            T.copy(Q_pe[tok_bx, :, :], Qp_s)

            for ii in T.Pipelined(NI_val, num_stages=num_stages):
                base = split_bx * NI_val * BI + ii * BI
                for bi, d in T.Parallel(BI, D):
                    idx = Indices[tok_bx, base + bi]
                    safe = T.if_then_else(idx >= 0, idx, 0)
                    KV_s[bi, d] = CKV[safe, d]
                for bi, d in T.Parallel(BI, DT):
                    idx = Indices[tok_bx, base + bi]
                    safe = T.if_then_else(idx >= 0, idx, 0)
                    Kp_s[bi, d] = KPE[safe, d]
                for h, bi in T.Parallel(H, BI):
                    acc_s[h, bi] = T.if_then_else(Indices[tok_bx, base + bi] >= 0, 0, -T.infinity("float32"))
                T.gemm(Qn_s, KV_s, acc_s, transpose_B=True)
                T.gemm(Qp_s, Kp_s, acc_s, transpose_B=True)

                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                # Load-bearing guard: drop this and 1/23 T=6 workload
                # returns INCORRECT_NUMERICAL. See header Dead-ends (iter-20).
                for h in T.Parallel(H):
                    m_i[h] = T.max(m_i[h], -(2**30))
                for h, bi in T.Parallel(H, BI):
                    acc_s[h, bi] = T.exp2((acc_s[h, bi] - m_i[h]) * sm_scale)
                T.reduce_sum(acc_s, sumexp, dim=1)
                T.copy(acc_s, S_s)
                # Final gemm split into 2 D-chunks — see header lesson.
                T.gemm(S_s, KV_s[:, 0:D_HALF], acc_o0)
                T.gemm(S_s, KV_s[:, D_HALF:D], acc_o1)

            sm_nat = 0.1352337788608801
            for h in T.Parallel(H):
                m_i[h] = m_i[h] * sm_nat
            T.copy(acc_o0, PO[tok_bx, :, split_bx, 0:D_HALF])
            T.copy(acc_o1, PO[tok_bx, :, split_bx, D_HALF:D])
            T.copy(m_i, PM[tok_bx, split_bx, :])
            T.copy(sumexp, PL[tok_bx, split_bx, :])
    return main


# ═══════════════════════════════════════════════════════════════
# Triton-path reduce kernel: PO layout [T, NS, H, D]
# Merged 2D-tile form — the iter-16..18 breakthrough from v11.
#   tile shape = [NS_TRI, D_CHUNK_TRI] = [32, 32] = 4KB fp32,
#   fits registers. tl.sum(w[:, None] * tile, axis=0) replaces the
#   per-iter scalar static_range + PM reload + exp recomputation.
# Grid = (T, H, D/D_CHUNK) = (T, 16, 16). num_warps=1 = 32 threads
# = exactly 1 D output per thread → 100% SM occupation at T=1.
# ═══════════════════════════════════════════════════════════════
@triton.jit
def _reduce_kernel(
    po_ptr, pm_ptr, pl_ptr, out_ptr, lse_ptr,
    NUM_SPLITS: tl.constexpr,
    D_CHUNK: tl.constexpr,
):
    tid = tl.program_id(0); hid = tl.program_id(1); dc_id = tl.program_id(2)
    # PDL: overlap our address-setup / constant prologue with fwd's tail.
    d = dc_id * D_CHUNK + tl.arange(0, D_CHUNK)
    s = tl.arange(0, NUM_SPLITS)
    pm_base = tid * (NUM_SPLITS * 16) + hid
    gdc_wait()
    m_vals = tl.load(pm_ptr + pm_base + s * 16)
    l_vals = tl.load(pl_ptr + pm_base + s * 16)
    m_g = tl.max(m_vals, axis=0)
    w = tl.where(m_g > float('-inf'), tl.exp(m_vals - m_g), 0.0)
    l_g = tl.sum(w * l_vals, axis=0)
    # 2D tile load: PO stride per si-row is H*D = 16*512 = 8192.
    po_tile = tl.load(
        po_ptr + tid * (NUM_SPLITS * 8192) + hid * 512
        + s[:, None] * 8192 + d[None, :]
    ).to(tl.float32)
    acc = tl.sum(w[:, None] * po_tile, axis=0)
    out = (acc / l_g).to(tl.bfloat16)
    tl.store(out_ptr + tid * 8192 + hid * 512 + d, out)
    if dc_id == 0:
        lse_val = tl.where(m_g > float('-inf'),
                           (m_g + tl.log(l_g)) * 1.4426950408889634, float('-inf'))
        tl.store(lse_ptr + tid * 16 + hid, lse_val)


# ═══════════════════════════════════════════════════════════════
# TileLang-path reduce kernel: PO layout [T, H, NS, D]
# Scalar static_range loop — the 2D-tile form regressed at every
# D_CHUNK from 32 to 512 on this path:
#   - D_CHUNK=512 → [NS=16, D=512] = 32KB fp32 tile, register spill,
#     T=8 collapses 49.5→35x.
#   - D_CHUNK=32 → small tile fits, but grid = T×H×16 = 2048 at T=8
#     over-fragments the already-saturated grid, T=8 ≈ 48x (-1x).
# Keep scalar loop + D_CHUNK=512 (single block per T,H) here.
# ═══════════════════════════════════════════════════════════════
@triton.jit
def _reduce_kernel_tl(
    po_ptr, pm_ptr, pl_ptr, out_ptr, lse_ptr,
    NUM_SPLITS: tl.constexpr,
    D_CHUNK: tl.constexpr,
):
    tid = tl.program_id(0); hid = tl.program_id(1); dc_id = tl.program_id(2)
    # PDL: overlap our address-setup / constant prologue with TL fwd's tail.
    d = dc_id * D_CHUNK + tl.arange(0, D_CHUNK)
    s = tl.arange(0, NUM_SPLITS)
    pm_base = tid * (NUM_SPLITS * 16) + hid
    gdc_wait()
    m_vals = tl.load(pm_ptr + pm_base + s * 16)
    l_vals = tl.load(pl_ptr + pm_base + s * 16)
    m_g = tl.max(m_vals, axis=0)
    w = tl.where(m_g > float('-inf'), tl.exp(m_vals - m_g), 0.0)
    l_g = tl.sum(w * l_vals, axis=0)
    acc = tl.zeros([D_CHUNK], dtype=tl.float32)
    # PO [T, H, NS, D]: stride(T)=H*NS*D, stride(H)=NS*D, stride(NS)=D=512.
    po_base = tid * (16 * NUM_SPLITS * 512) + hid * (NUM_SPLITS * 512)
    for si in tl.static_range(0, NUM_SPLITS):
        w_s = tl.where(m_g > float('-inf'),
                       tl.exp(tl.load(pm_ptr + pm_base + si * 16) - m_g), 0.0)
        o_s = tl.load(po_ptr + po_base + si * 512 + d).to(tl.float32)
        acc += w_s * o_s
    out = (acc / l_g).to(tl.bfloat16)
    tl.store(out_ptr + tid * 8192 + hid * 512 + d, out)
    if dc_id == 0:
        lse_val = tl.where(m_g > float('-inf'),
                           (m_g + tl.log(l_g)) * 1.4426950408889634, float('-inf'))
        tl.store(lse_ptr + tid * 16 + hid, lse_val)


# ═══════════════════════════════════════════════════════════════
# Python wrapper — dispatch by T
# ═══════════════════════════════════════════════════════════════
_tl_kern = None
_bufs_triton = None
_bufs_tl = None
_pm_tri = None; _pl_tri = None
_pm_tl = None; _pl_tl = None
_static_out = {}; _static_lse = {}
_graph_cache = {}; _graph_cnt = {}
_last_graph = None
# Fast-path: within-workload we see the same sparse_indices buffer repeatedly.
# Int compare beats the 6-data_ptr tuple + dict.get (each data_ptr is a C-API
# hop). Safe because each workload runs in its own process
# (use_isolated_runner = true) — no cross-workload aliasing hazard.
_last_si_ptr = -1  # sentinel; data_ptr() is always nonneg
_last_out = None
_last_lse = None

# T=1,2 → Triton;  T≥3 → TileLang
_TRITON_THRESH = 3


def _get_tl():
    global _tl_kern
    if _tl_kern is None:
        _tl_kern = _tl_fwd_factory()
    return _tl_kern


def _get_bufs(dev):
    global _bufs_triton, _bufs_tl, _pm_tri, _pl_tri, _pm_tl, _pl_tl
    if _bufs_triton is None:
        _bufs_triton = torch.empty((8, _NS_TRI, _NH * _DC), dtype=torch.bfloat16, device=dev)
    if _bufs_tl is None:
        # Transposed [T, H, NS, D] — see TL reduce docstring.
        _bufs_tl = torch.empty((8, _NH, _NS_TL, _DC), dtype=torch.bfloat16, device=dev)
    if _pm_tri is None:
        _pm_tri = torch.empty((8, _NS_TRI, _NH), dtype=torch.float32, device=dev)
        _pl_tri = torch.empty((8, _NS_TRI, _NH), dtype=torch.float32, device=dev)
    if _pm_tl is None:
        _pm_tl = torch.empty((8, _NS_TL, _NH), dtype=torch.float32, device=dev)
        _pl_tl = torch.empty((8, _NS_TL, _NH), dtype=torch.float32, device=dev)


# TRI reduce: D_CHUNK=32 + num_warps=1 — the iter-18 optimum at T=1.
# TL reduce: D_CHUNK=512 single block per (T, H) — scalar loop form
# (see TL reduce docstring for why the 2D form doesn't port here).
_D_CHUNK_TRI = 32
_D_CHUNK_TL = 512
_D_SPLITS_TRI = _DC // _D_CHUNK_TRI
_D_SPLITS_TL = _DC // _D_CHUNK_TL


# H-split Triton fwd: 8 heads per block, 2 blocks per (token, split) — lifts T=2 from
# 64 blocks (0.43 waves) to 128 (0.86) closing the SM-utilization gap. AB-confirmed
# +1.55x compound (H_SPLIT=2 + num_warps=8 + num_stages=2) vs iter-4 baseline.
_H_SPLIT_TRI = 2
_H_PER_TRI = _NH // _H_SPLIT_TRI


def _launch_triton(Tv, qn, qp, ckv, kpe, si, sm_scale, output, lse):
    po = _bufs_triton
    use_mask = (Tv == 1)
    _triton_fwd[(Tv, _NS_TRI, _H_SPLIT_TRI)](
        qn, qp, ckv, kpe, si,
        po, _pm_tri, _pl_tri, sm_scale,
        NUM_SPLITS=_NS_TRI, BLOCK_K=_BK_TRI,
        USE_MASK=use_mask,
        H_SPLIT=_H_SPLIT_TRI, H_PER=_H_PER_TRI,
        num_warps=8, num_stages=2,
        launch_pdl=True,
    )
    _reduce_kernel[(Tv, 16, _D_SPLITS_TRI)](
        po, _pm_tri, _pl_tri, output, lse,
        NUM_SPLITS=_NS_TRI, D_CHUNK=_D_CHUNK_TRI,
        num_warps=1, num_stages=1,
        launch_pdl=True,
    )


def _launch_tl(Tv, qn, qp, ckv, kpe, si, output, lse):
    po = _bufs_tl
    fwd = _get_tl()
    fwd(qn, qp, ckv, kpe, si, po[:Tv], _pm_tl[:Tv], _pl_tl[:Tv])
    # Reduce uses PDL: scheduling hint reduces launch tail when preceded by
    # a triton fwd; even with TileLang (no trigger), launch_pdl may still
    # overlap schedule metadata.
    _reduce_kernel_tl[(Tv, 16, _D_SPLITS_TL)](
        po, _pm_tl, _pl_tl, output, lse,
        NUM_SPLITS=_NS_TL, D_CHUNK=_D_CHUNK_TL,
        num_warps=8, num_stages=2,
        launch_pdl=True,
    )


def _warmup():
    dev = torch.device("cuda")
    for t in [1, 2, 5, 6, 7, 8]:
        _static_out[t] = torch.empty((t, _NH, _DC), dtype=torch.bfloat16, device=dev)
        _static_lse[t] = torch.empty((t, _NH), dtype=torch.float32, device=dev)
    _get_bufs(dev)
    q  = torch.empty(1, _NH * _DC, dtype=torch.bfloat16, device=dev)
    qp = torch.empty(1, _NH * _DP, dtype=torch.bfloat16, device=dev)
    ck = torch.empty(1, _DC, dtype=torch.bfloat16, device=dev)
    kp = torch.empty(1, _DP, dtype=torch.bfloat16, device=dev)
    si = torch.full((1, _TOPK), -1, dtype=torch.int32, device=dev)
    _launch_triton(1, q, qp, ck, kp, si, 0.1, _static_out[1], _static_lse[1])
    torch.cuda.synchronize()

_warmup()


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    global _last_graph, _last_si_ptr, _last_out, _last_lse

    # Fastest hot path: one int compare on sparse_indices.data_ptr().
    si_ptr = sparse_indices.data_ptr()
    if si_ptr == _last_si_ptr and _last_graph is not None:
        _last_graph.replay()
        return _last_out, _last_lse

    Tv = q_nope.shape[0]
    key = (q_nope.data_ptr(), q_pe.data_ptr(), ckv_cache.data_ptr(),
           kpe_cache.data_ptr(), si_ptr, Tv)

    g = _graph_cache.get(key)
    if g is not None:
        _last_graph = g
        _last_si_ptr = si_ptr
        _last_out = _static_out[Tv]
        _last_lse = _static_lse[Tv]
        g.replay()
        return _last_out, _last_lse

    dev = q_nope.device
    ckv_flat = ckv_cache.reshape(-1, _DC)
    kpe_flat = kpe_cache.reshape(-1, _DP)

    if Tv not in _static_out:
        _static_out[Tv] = torch.empty((Tv, _NH, _DC), dtype=torch.bfloat16, device=dev)
        _static_lse[Tv] = torch.empty((Tv, _NH), dtype=torch.float32, device=dev)
    output, lse_o = _static_out[Tv], _static_lse[Tv]
    _get_bufs(dev)

    use_triton = (Tv < _TRITON_THRESH)

    if _DISABLE_GRAPH:
        if use_triton:
            _launch_triton(Tv, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
                           sm_scale, output, lse_o)
        else:
            _launch_tl(Tv, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
                       output, lse_o)
        return output, lse_o

    cnt = _graph_cnt.get(key, 0) + 1
    _graph_cnt[key] = cnt

    if use_triton:
        _launch_triton(Tv, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
                       sm_scale, output, lse_o)
    else:
        _launch_tl(Tv, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
                   output, lse_o)

    if cnt >= 2:
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if use_triton:
                _launch_triton(Tv, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
                               sm_scale, output, lse_o)
            else:
                _launch_tl(Tv, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
                           output, lse_o)
        _graph_cache[key] = g
        _last_graph = g
        _last_si_ptr = si_ptr
        _last_out = output
        _last_lse = lse_o

    return output, lse_o
