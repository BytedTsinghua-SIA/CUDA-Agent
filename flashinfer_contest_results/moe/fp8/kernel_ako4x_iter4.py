# AKO4X 对照 kernel；CUDA-Agent ours 使用同题目录下的 kernel.py。
# AKO4X repo source: reference/moe-fp8-block-scale-ds-routing-topk8-ng8-kg4-e32-h7168-i2048/variants/iter4-autotune-tactic-sweep-gated/kernel.py
"""
iter4-autotune-tactic-sweep — round-4 anchor candidate for
moe-fp8-block-scale-ds-routing-topk8-ng8-kg4-e32-h7168-i2048 (B200 / Modal).

═══════════════════════════════════════════════════════════════════════════
1) Identity
─────────────────────────────────────────────────────────────────────────
Round-4 lever: **per-shape autotune sweep of trtllm's `[tile_N, config]`
tactic space**, layered on top of iter3c's CUDA-graph capture+replay.

Inherited from iter3c: `torch.cuda.graph()` capture of the full 6-kernel
`moe_op.trtllm_fp8_block_scale_moe` pipeline; `_GRAPH_MIN_T=2` T=1 eager
fallback (matched_ratio tolerance gate); `data_ptr()` trial-boundary
detection → recapture.

New: before the first capture for a (T, hidden_size, intermediate_size,
local_num_experts) shape, enumerate valid tactics via
`moe_op.trtllm_get_valid_moe_configs(...)`, time each on the actual
workload tensors with `cudaEvent` start/end pairs (3 timed iters, take
median), and pick the fastest `[tile_N, config]` pair. The cached
fallback `[-1, -1]` is included in the sweep — if vendor's
`selectDefaultTileN` heuristic is already optimal, autotune is a no-op
relative to iter3c. The chosen tactic is then baked into the captured
graph (and into the T=1 eager call).

What the lever targets:
- The T≥901 ceiling at 1.01-1.16x in iter3c. These workloads are at
  vendor GEMM throughput — Python overhead and inter-kernel gaps are
  already stripped. The remaining lift requires picking a better cubin
  kernel for the (M=tile_N, N=intermediate_size, K=hidden_size) shape.
- `[-1, -1]` triggers vendor's `selectDefaultTileN` heuristic:
  `nextPowerOfTwo(num_tokens * top_k / num_local_experts)`. For our
  num_local_experts=32, top_k=8: T=14107 → tile_N=4096, T=901 → 256,
  T=80 → 32. The heuristic clamps to supported_tile_nums and picks the
  smallest matching candidate; nearby tiles ±1 are NOT explored. The
  autotune sweep includes the full vendor-validated `getValidConfigs`
  list — likely on the order of dozens of (tile_N, config) pairs per
  shape, covering different tile_N values AND different cubin configs
  per tile.

Capture/timing budget: bench harness does 1 (initial-warmup) + 5
(estimate-kernel-time) + 3 (dry_run_iters) = 9 untimed calls before the
100 CUPTI-timed iters per trial. The autotune sweep lives within these
9 untimed calls of the FIRST trial (which is also when the first
capture happens). Per shape, sweep cost ≈ N_tactics × 3 timed-iter ×
per-call-time. Even at T=14107 (~2.3ms/call) × 50 candidates × 3 ≈
0.35 s — far below the 900s `timeout_seconds` budget. Subsequent
trials' recapture-on-data_ptr-mismatch path uses the cached tactic
(no re-sweep) — recapture stays cheap.

═══════════════════════════════════════════════════════════════════════════
2) Delta from prior anchor
─────────────────────────────────────────────────────────────────────────
Parent: `iter3c-cuda-graph-eager-T1-fallback` (1.32x, 19/19 PASS, range
1.016x-1.755x). Parent passes `tactic=[-1, -1]` ("vendor default") on
every binding call.

This iter adds a sweep over `Array<Array<int64_t>>
moe_op.trtllm_get_valid_moe_configs(...)` and uses the best per-shape
tactic in BOTH the eager (T=1) and captured (T≥2) paths.

Same correctness contract as iter3c: `do_finalize=True`, in-place write
to the pre-allocated `(T, H, device)` output buffer cache.

═══════════════════════════════════════════════════════════════════════════
3) Lessons on this variant
─────────────────────────────────────────────────────────────────────────
- **Autotune-by-T gate (`_AUTOTUNE_MIN_T=1000`) is load-bearing.**
  Round-4 evidence: drift-cancelled A/B vs iter3c (same-container) showed
  the lever pays off ONLY at the two largest workloads. Numbers:
  ```
  T=11948: +0.047x   T=14107: +0.068x      (vendor floor moved)
  T=901:   ≈0        T=1..80: ≤|0.012|x    (autotune ties vendor)
  T=1:     -0.265x   T=15:    -0.048x      (timing-noise regression)
  ```
  At T=1 each call is ~70µs and cudaEvent timing noise is order-10µs;
  median-of-3 tactic ranking becomes unreliable and de-ranks the
  fallback. The fix is to skip autotune below T=1000 — preserves the
  vendor-default eager/captured behavior on the 17 workloads where it
  was already optimal, while letting autotune land its real lift on the
  two large-T workloads where the vendor heuristic clamps to
  tile_N=4096 with config=-1 default.
- **Tactic schema is `[tile_N, config]`** — not `[gemm1_tactic,
  gemm2_tactic]` as the parent kernel's open-direction note implied.
  Source: `trtllm_fused_moe_kernel_launcher.cu:143-148`
  ("Python side convention: tactic is [tile_N, config]"). The
  `gemm1/gemm2` story is for cutlass_fused_moe, a different code path.
- **`trtllm_get_valid_moe_configs` is a pure host-side function** — it
  takes (dtype_act, dtype_weights, fp8_quant_type, top_k, hidden_size,
  intermediate_size, num_local_experts, act_type, use_shuffled_weight,
  weight_layout, num_tokens) and returns the per-shape valid candidates.
  No GPU work; safe to call any time. Source:
  `trtllm_fused_moe_kernel_launcher.cu:2249-2313`.
- **Autotune timing during the harness's untimed prelude** — per-trial
  bench accounting:
  ```
  bench_gpu_time_with_cupti:
    1 initial-warmup call
    5 estimate-kernel-time calls
    3 dry_run_iters (warmup_runs=3 from config.toml)
    100 timed CUPTI iters
  ```
  All ~9 prelude calls land in the FIRST `run()` invocation per trial.
  The autotune sweep happens within the first of those — overhead is
  invisible to CUPTI. Re-captures on trial-boundary `data_ptr()` change
  also reuse the cached tactic; only the very first call per shape pays
  autotune cost.
- **Each subprocess autotunes ONE shape** — `use_isolated_runner=true`
  in config.toml means each workload runs in its own process. So the
  cached `_TACTIC_BY_KEY` table only ever has one entry. Cross-env
  portability (the blocker on a Modal volume-based persistent cache
  the parent flagged) is sidestepped: no on-disk state.
- **Tactic baked into captured graph** — graph capture happens AFTER
  the autotune sweep. The captured graph's kernel launches encode the
  chosen `[tile_N, config]` via the cubin's `prepare_moe_common`
  workspace selection + kernel dispatch. Replay just re-issues those
  exact launches. So autotune compositionality with capture is clean.
- **Fallback path preserved** — if `trtllm_get_valid_moe_configs`
  throws or returns no candidates, or if every candidate throws during
  timing, we cache `_FALLBACK_TACTIC = [-1, -1]` and proceed as iter3c.
  This guarantees we never regress below iter3c's behavior.
- **`_FALLBACK_TACTIC` is also a candidate** — the sweep includes
  `[-1, -1]` (= vendor heuristic). If the heuristic is already optimal
  for a shape, autotune is a no-op for that shape relative to iter3c.

═══════════════════════════════════════════════════════════════════════════
4) Dead-ends tried (this iter)
─────────────────────────────────────────────────────────────────────────
None yet — first iter on this lever.

Inherits iter3c's dead-ends + open question: graph capture at T=1
remains gated off (`_GRAPH_MIN_T=2`).

═══════════════════════════════════════════════════════════════════════════
5) Open directions (if this lever closes)
─────────────────────────────────────────────────────────────────────────
1. **Weight pre-conditioning** (`use_shuffled_weight=True` +
   `reorder_rows_for_gated_act_gemm`): currently both `False`. Reorder
   cost amortizes over 100 captured replays per trial. Composes with
   per-shape autotune (separate tactic set per weight layout — the
   sweep would auto-pick the best layout-aware tactic).
2. **T=1 graph-capture revisit** with a stabilization that survives
   matched_ratio gate (currently 1.48x eager — small headroom).
3. **Whole-DSL alternative** (CUTLASS / CuTe-DSL grouped-GEMM /
   DeepGEMM as a leaner reference). Many-iter commitment; not a
   round-4 lever.

═══════════════════════════════════════════════════════════════════════════
"""

import os

import torch

from flashinfer.fused_moe.core import (
    DtypeTrtllmGen,
    get_trtllm_moe_sm100_module,
)
from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
from flashinfer.tllm_enums import (
    ActivationType,
    Fp8QuantizationType,
    RoutingMethodType,
    WeightLayout,
)
from flashinfer.utils import device_support_pdl


# Operator-fixed constants (per docs/definition.json / operator name).
_TOP_K = 8
_N_GROUP = 8
_TOPK_GROUP = 4
_NUM_EXPERTS = 256
_INTERMEDIATE_SIZE = 2048
_HIDDEN_SIZE = 7168

# Vendor-default fallback: tile_N=-1 → selectDefaultTileN; config=-1 → default cfg.
_FALLBACK_TACTIC = [-1, -1]

# Hard-coded enum / flag values (per parent anchor's config).
_ROUTING_METHOD = RoutingMethodType.DeepSeekV3.value
_WEIGHT_LAYOUT = WeightLayout.MajorK.value
_FP8_QUANT = Fp8QuantizationType.DeepSeekFp8
_ACTIVATION = ActivationType.Swiglu.value
_USE_SHUFFLED_WEIGHT = False
_DO_FINALIZE = True
_NORM_TOPK_PROB = True

# Dtype enums (FP8 E4m3 for activation + weights, matching the operator spec).
_DTYPE_ACT = DtypeTrtllmGen.E4m3
_DTYPE_WEIGHTS = DtypeTrtllmGen.E4m3

# Env-gated escape hatches.
_NO_GRAPH = bool(int(os.environ.get("NO_GRAPH", "0")))
_VALIDATE_CAPTURE = bool(int(os.environ.get("VALIDATE_CAPTURE", "0")))
_NO_AUTOTUNE = bool(int(os.environ.get("NO_AUTOTUNE", "0")))
# If T < _GRAPH_MIN_T, skip capture and run eager. Default 2 (skip T=1 only).
# See parent kernel's section-3 rationale for the matched_ratio tolerance
# interaction with capture-mode FP8 perturbations at T=1.
_GRAPH_MIN_T = int(os.environ.get("GRAPH_MIN_T", "2"))

# Per-tactic timing budget. 3 iters × N tactics is the autotune cost per
# shape (subprocess). Lower = noisier; higher = more prelude time but
# untimed so it doesn't affect CUPTI latency.
_AUTOTUNE_ITERS = int(os.environ.get("AUTOTUNE_ITERS", "3"))
_AUTOTUNE_WARMUP = int(os.environ.get("AUTOTUNE_WARMUP", "1"))

# Autotune only runs for T >= _AUTOTUNE_MIN_T. Below that, the vendor
# heuristic [-1, -1] is the proven winner. Rationale (round-4 ab-compare
# evidence): drift-cancelled per-workload delta vs iter3c showed
# +0.047x at T=11948, +0.068x at T=14107, ≈0 for T=1..901, and a severe
# -0.265x regression at T=1 (timing-noise floor at sub-100µs per call
# breaks the median-of-3 tactic ranking — fallback gets de-ranked). Gate
# at T>=1000 captures the two real wins and preserves vendor-default
# behavior everywhere else.
_AUTOTUNE_MIN_T = int(os.environ.get("AUTOTUNE_MIN_T", "1000"))

# Shared graph mempool — all captures within a process share the same
# private pool so re-captures don't fragment the allocator.
_GRAPH_POOL = None

# Lazy-initialized module + per-process caches.
_MOE_OP = None
_PDL_BY_DEVICE: dict = {}
_EMPTY_BY_DTYPE_DEVICE: dict = {}
_EMPTY_INT32_BY_DEVICE: dict = {}
_OUTPUT_BY_SHAPE_DEVICE: dict = {}

# Graph state — keyed by (T, device.index).
_GRAPH_BY_KEY: dict = {}

# Tactic cache — keyed by (T, hidden_size, intermediate_size,
# local_num_experts, device.index). Value is a 2-list [tile_N, config].
_TACTIC_BY_KEY: dict = {}


def _init_moe_op():
    """Force module build, cubin-loader setup, and grab the raw C++ binding."""
    global _MOE_OP
    if _MOE_OP is not None:
        return _MOE_OP
    _ = get_trtllm_moe_sm100_module()
    module = gen_trtllm_gen_fused_moe_sm100_module()
    _MOE_OP = module.build_and_load()
    return _MOE_OP


def _get_output_buffer(num_tokens: int, hidden_size: int, device: torch.device) -> torch.Tensor:
    key = (num_tokens, hidden_size, device.index)
    buf = _OUTPUT_BY_SHAPE_DEVICE.get(key)
    if buf is None:
        buf = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        _OUTPUT_BY_SHAPE_DEVICE[key] = buf
    return buf


def _get_empty(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (dtype, device.index)
    t = _EMPTY_BY_DTYPE_DEVICE.get(key)
    if t is None:
        t = torch.empty(0, dtype=dtype, device=device)
        _EMPTY_BY_DTYPE_DEVICE[key] = t
    return t


def _get_empty_int32(device: torch.device) -> torch.Tensor:
    key = device.index
    t = _EMPTY_INT32_BY_DEVICE.get(key)
    if t is None:
        t = torch.empty(0, dtype=torch.int32, device=device)
        _EMPTY_INT32_BY_DEVICE[key] = t
    return t


def _get_pdl(device: torch.device) -> bool:
    key = device.index
    p = _PDL_BY_DEVICE.get(key)
    if p is None:
        p = device_support_pdl(device)
        _PDL_BY_DEVICE[key] = p
    return p


def _eager_call(
    moe_op,
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    output,
    topk_ids,
    expert_weights,
    enable_pdl,
    local_expert_offset,
    local_num_experts,
    routed_scaling_factor,
    tactic,
):
    """The single binding invocation — used in eager, capture, and autotune paths."""
    moe_op.trtllm_fp8_block_scale_moe(
        routing_logits,
        topk_ids,
        expert_weights,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output,
        _NUM_EXPERTS,
        _TOP_K,
        _N_GROUP,
        _TOPK_GROUP,
        _INTERMEDIATE_SIZE,
        int(local_expert_offset),
        local_num_experts,
        float(routed_scaling_factor),
        _ROUTING_METHOD,
        _USE_SHUFFLED_WEIGHT,
        _WEIGHT_LAYOUT,
        _DO_FINALIZE,
        enable_pdl,
        tactic,
        _FP8_QUANT,
        _ACTIVATION,
        _NORM_TOPK_PROB,
        None,  # routing_replay_out
    )


def _autotune_tactic(
    moe_op,
    num_tokens,
    local_num_experts,
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    output,
    topk_ids,
    expert_weights,
    enable_pdl,
    local_expert_offset,
    routed_scaling_factor,
):
    """Sweep all valid [tile_N, config] tactics; return the fastest.

    Returns _FALLBACK_TACTIC on any enumeration/timing failure.
    """
    # Enumerate candidates. Signature matches MoERunner.get_valid_tactics
    # at flashinfer/fused_moe/core.py:1005-1027.
    try:
        candidates = moe_op.trtllm_get_valid_moe_configs(
            _DTYPE_ACT,
            _DTYPE_WEIGHTS,
            _FP8_QUANT,
            _TOP_K,
            _HIDDEN_SIZE,
            _INTERMEDIATE_SIZE,
            local_num_experts,
            _ACTIVATION,
            _USE_SHUFFLED_WEIGHT,
            _WEIGHT_LAYOUT,
            num_tokens,
        )
    except Exception:
        return _FALLBACK_TACTIC

    # Normalize cubin's Array<Array<int64_t>> to Python list-of-2-lists.
    explicit = []
    for c in candidates:
        try:
            tac = [int(c[0]), int(c[1])]
            explicit.append(tac)
        except Exception:
            continue

    # Always include the vendor fallback in the sweep.
    sweep = explicit + [list(_FALLBACK_TACTIC)]

    # Initial warmup so first-tactic measurement is not contaminated by
    # cubin cold-load / cudaMalloc-for-workspace.
    for _ in range(_AUTOTUNE_WARMUP):
        try:
            _eager_call(
                moe_op, routing_logits, routing_bias, hidden_states,
                hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                gemm2_weights, gemm2_weights_scale, output, topk_ids,
                expert_weights, enable_pdl, local_expert_offset,
                local_num_experts, routed_scaling_factor,
                list(_FALLBACK_TACTIC),
            )
        except Exception:
            pass
    torch.cuda.synchronize()

    best_tactic = list(_FALLBACK_TACTIC)
    best_ms = float("inf")

    start_evt = [torch.cuda.Event(enable_timing=True) for _ in range(_AUTOTUNE_ITERS)]
    end_evt = [torch.cuda.Event(enable_timing=True) for _ in range(_AUTOTUNE_ITERS)]

    for tactic in sweep:
        try:
            # Per-tactic warmup (1 call) — handles tile-switch workspace
            # alloc & cubin variant load.
            _eager_call(
                moe_op, routing_logits, routing_bias, hidden_states,
                hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                gemm2_weights, gemm2_weights_scale, output, topk_ids,
                expert_weights, enable_pdl, local_expert_offset,
                local_num_experts, routed_scaling_factor, tactic,
            )
            for i in range(_AUTOTUNE_ITERS):
                start_evt[i].record()
                _eager_call(
                    moe_op, routing_logits, routing_bias, hidden_states,
                    hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                    gemm2_weights, gemm2_weights_scale, output, topk_ids,
                    expert_weights, enable_pdl, local_expert_offset,
                    local_num_experts, routed_scaling_factor, tactic,
                )
                end_evt[i].record()
            torch.cuda.synchronize()
            elapsed = sorted(start_evt[i].elapsed_time(end_evt[i]) for i in range(_AUTOTUNE_ITERS))
            median = elapsed[_AUTOTUNE_ITERS // 2]
            if median < best_ms:
                best_ms = median
                best_tactic = tactic
        except Exception:
            # Tactic invalid for this shape / cubin path; skip.
            continue

    return best_tactic


def _capture_graph(
    moe_op,
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    output,
    topk_ids,
    expert_weights,
    enable_pdl,
    local_expert_offset,
    local_num_experts,
    routed_scaling_factor,
    tactic,
):
    """Capture the binding call into a CUDA graph and return it.

    Three warmup calls on a side stream precede capture (the canonical
    torch.cuda.graph pattern); these settle the binding's one-time state
    and let PyTorch's allocator's first-time cudaMalloc paths fire
    outside the capture window.
    """
    global _GRAPH_POOL

    torch.cuda.synchronize()
    s = torch.cuda.Stream(device=output.device)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _eager_call(
                moe_op, routing_logits, routing_bias, hidden_states,
                hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                gemm2_weights, gemm2_weights_scale, output, topk_ids,
                expert_weights, enable_pdl, local_expert_offset,
                local_num_experts, routed_scaling_factor, tactic,
            )
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    if _GRAPH_POOL is None:
        _GRAPH_POOL = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=_GRAPH_POOL):
        _eager_call(
            moe_op, routing_logits, routing_bias, hidden_states,
            hidden_states_scale, gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale, output, topk_ids,
            expert_weights, enable_pdl, local_expert_offset,
            local_num_experts, routed_scaling_factor, tactic,
        )
    return g


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    moe_op = _MOE_OP if _MOE_OP is not None else _init_moe_op()

    device = hidden_states.device
    num_tokens = hidden_states.shape[0]
    output = _get_output_buffer(num_tokens, _HIDDEN_SIZE, device)
    topk_ids = _get_empty_int32(device)
    expert_weights = _get_empty(routing_logits.dtype, device)
    enable_pdl = _get_pdl(device)
    local_num_experts = gemm1_weights.shape[0]

    # Resolve per-shape tactic (autotune once per subprocess; cache hit
    # on subsequent calls within the same shape).
    tactic_key = (num_tokens, _HIDDEN_SIZE, _INTERMEDIATE_SIZE, local_num_experts, device.index)
    tactic = _TACTIC_BY_KEY.get(tactic_key)
    if tactic is None:
        if _NO_AUTOTUNE or num_tokens < _AUTOTUNE_MIN_T:
            tactic = list(_FALLBACK_TACTIC)
        else:
            tactic = _autotune_tactic(
                moe_op, num_tokens, local_num_experts,
                routing_logits, routing_bias, hidden_states,
                hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                gemm2_weights, gemm2_weights_scale, output, topk_ids,
                expert_weights, enable_pdl, local_expert_offset,
                routed_scaling_factor,
            )
        _TACTIC_BY_KEY[tactic_key] = tactic

    if _NO_GRAPH or num_tokens < _GRAPH_MIN_T:
        _eager_call(
            moe_op, routing_logits, routing_bias, hidden_states,
            hidden_states_scale, gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale, output, topk_ids,
            expert_weights, enable_pdl, local_expert_offset,
            local_num_experts, routed_scaling_factor, tactic,
        )
        return output

    key = (num_tokens, device.index)
    entry = _GRAPH_BY_KEY.get(key)
    hs_ptr = hidden_states.data_ptr()

    if entry is None or entry["hs_ptr"] != hs_ptr:
        if entry is not None:
            del _GRAPH_BY_KEY[key]
            entry = None
        g = _capture_graph(
            moe_op, routing_logits, routing_bias, hidden_states,
            hidden_states_scale, gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale, output, topk_ids,
            expert_weights, enable_pdl, local_expert_offset,
            local_num_experts, routed_scaling_factor, tactic,
        )
        entry = {"graph": g, "hs_ptr": hs_ptr, "output": output}
        _GRAPH_BY_KEY[key] = entry

    if _VALIDATE_CAPTURE:
        output.zero_()

    entry["graph"].replay()

    if _VALIDATE_CAPTURE:
        torch.cuda.synchronize()
        assert output.abs().sum().item() > 0, (
            f"VALIDATE_CAPTURE: output is all-zero after replay at T={num_tokens}"
        )

    return output
