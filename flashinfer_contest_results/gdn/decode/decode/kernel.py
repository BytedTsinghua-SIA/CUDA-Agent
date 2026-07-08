"""GDN decode 的自包含 Triton kernel。

功能：
  实现 run(q, k, v, state, A_log, a, dt_bias, b, scale) -> (output, new_state)。

参数：
  q/k/v/state/A_log/a/dt_bias/b/scale：FlashInfer GDN decode benchmark 输入。

示例：
  from kernel import run
  output, new_state = run(q, k, v, state, A_log, a, dt_bias, b, scale)
"""
import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _gdn_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, o_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr, scale,
    HAS_STATE: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
):
    cta_per_head: tl.constexpr = 128 // BV
    cta_per_batch: tl.constexpr = cta_per_head * 8
    pid = tl.program_id(0)
    b_idx = pid // cta_per_batch
    rem = pid % cta_per_batch
    hv = rem // cta_per_head
    ci = rem % cta_per_head
    ik = tl.arange(0, NK)
    iv = ci * BV + tl.arange(0, BV)
    qh = hv >> 1
    qk_off = b_idx * (4 * NK) + qh * NK

    b_q = tl.load(q_ptr + qk_off + ik).to(tl.float32) * scale
    b_k = tl.load(k_ptr + qk_off + ik).to(tl.float32)
    k_dot_q = tl.sum(b_k * b_q)

    ab_off = b_idx * 8 + hv
    A = tl.load(A_log_ptr + hv).to(tl.float32)
    av = tl.load(a_ptr + ab_off).to(tl.float32)
    dv = tl.load(dt_bias_ptr + hv).to(tl.float32)
    bv = tl.load(b_ptr + ab_off).to(tl.float32)
    x = av + dv
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = tl.exp(-tl.exp(A) * sp)
    beta = tl.sigmoid(bv)

    s_base = state_ptr + b_idx * (8 * 128 * NK) + hv * (128 * NK)
    s_ptrs = s_base + iv[:, None] * NK + ik[None, :]
    vo_base = b_idx * (8 * NK) + hv * NK
    if HAS_STATE:
        h = tl.load(s_ptrs, eviction_policy='evict_last')
    else:
        h = tl.zeros([BV, NK], dtype=tl.float32)
    b_v = tl.load(v_ptr + vo_base + iv, eviction_policy='evict_last').to(tl.float32)

    h = h * g
    old_v = tl.sum(h * b_k[None, :], axis=1)
    out_decayed = tl.sum(h * b_q[None, :], axis=1)
    v_new = beta * (b_v - old_v)
    h = h + b_k[None, :] * v_new[:, None]
    out_vals = out_decayed + v_new * k_dot_q

    tl.store(o_ptr + vo_base + iv, out_vals.to(tl.bfloat16))
    tl.store(s_ptrs, h, eviction_policy='evict_first')


_NK = 128

# Per-batch-size dispatch table: (BV, num_warps, num_stages)
# Tuned via exhaustive CUPTI sweep on B200 SM100.
_CONFIGS = {
    1:  (2, 1, 2),
    4:  (4, 1, 1),
    8:  (4, 1, 2),
    16: (8, 1, 2),
    32: (8, 1, 1),
    48: (8, 1, 2),
    64: (8, 1, 1),
}


def _select_config(B):
    for b_max in sorted(_CONFIGS):
        if B <= b_max:
            return _CONFIGS[b_max]
    return _CONFIGS[64]


def run(q, k, v, state: Optional[torch.Tensor], A_log, a, dt_bias, b, scale=None,
        output=None, new_state=None):
    B = q.shape[0]
    if output is None:
        output = torch.empty(B, 1, 8, 128, dtype=torch.bfloat16, device=q.device)
    if scale is None:
        scale = 0.08838834764831843

    has_state = state is not None
    if has_state:
        if new_state is None or new_state.data_ptr() == state.data_ptr():
            ns = state
        else:
            new_state.copy_(state)
            ns = new_state
    else:
        ns = new_state if new_state is not None else torch.empty(
            B, 8, 128, 128, dtype=torch.float32, device=q.device)

    BV, nw, ns_stages = _select_config(B)
    cpb = (128 // BV) * 8
    _gdn_kernel[(B * cpb,)](
        q, k, v, ns, output, A_log, a, dt_bias, b, scale,
        HAS_STATE=has_state, BV=BV, NK=_NK,
        num_warps=nw, num_stages=ns_stages,
    )
    return output, ns
