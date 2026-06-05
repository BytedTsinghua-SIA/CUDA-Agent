"""Reference fast-path for triangular matrix multiplication (Triton, tensor-core).

For upper/lower-triangular `A` and `B`, the product `A @ B` is triangular on the same side,
so the dense GEMM that `torch` / `torch.compile` runs (then masks) does ~2-6x more work than
necessary. This kernel exploits the structure exactly:

  UPPER  C = triu(A @ B):  skip output tiles fully BELOW the diagonal; restrict each tile's
                           k-loop to [tile_row_start, tile_col_end).
  LOWER  C = tril(A @ B):  skip output tiles fully ABOVE the diagonal; restrict each tile's
                           k-loop to [tile_col_start, tile_row_end).

It is exact (the skipped work is provably zero for same-side-triangular inputs), runs on tensor
cores via `tl.dot`, and supports 2D and batched inputs. Benchmarked at MATCHED precision (TF32)
against `torch.compile(dense + mask)` on an A100-80GB (KernelBench tasks 14/15 are the 2D cases):

    upper 2D 4096:        2.14x      lower 2D 4096:        3.19x
    upper batched 16x1024: 3.69x     lower batched 16x1024: 3.31x
    upper batched 8x2048:  3.27x     lower batched 8x2048:  3.43x

All outputs match the dense reference at atol=rtol=1e-2. See discussion in issue #16.

Usage:
    from utils.tri_matmul import tri_matmul
    C = tri_matmul(A, B, lower=False)          # 2D or batched, A/B same-side triangular
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _tri_bmm_kernel(A, B, C, M, N, K,
                    sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                    LOWER: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    pb = tl.program_id(2)
    row0 = pm * BM
    col0 = pn * BN
    if LOWER:
        if (row0 + BM) <= col0:          # tile fully above the diagonal -> all zero
            return
        k_lo, k_hi = col0, row0 + BM
    else:
        if (col0 + BN) <= row0:          # tile fully below the diagonal -> all zero
            return
        k_lo, k_hi = row0, col0 + BN
    offs_m = row0 + tl.arange(0, BM)
    offs_n = col0 + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptr = A + pb * sab
    b_ptr = B + pb * sbb
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k = (k_lo // BK) * BK
    while k < k_hi:
        kk = k + offs_k
        a = tl.load(a_ptr + offs_m[:, None] * sam + kk[None, :] * sak,
                    mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + kk[:, None] * sbk + offs_n[None, :] * sbn,
                    mask=(kk[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        k += BK
    c_ptr = C + pb * scb
    tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def tri_matmul(A: torch.Tensor, B: torch.Tensor, lower: bool = False,
               block_m: int = 64, block_n: int = 64, block_k: int = 64) -> torch.Tensor:
    """Triangular matmul. `A`, `B` must be same-side (upper or lower) triangular, 2D or batched.

    Returns ``triu(A @ B)`` (``lower=False``) or ``tril(A @ B)`` (``lower=True``), computed
    without the dense work the structure makes redundant.
    """
    batched = A.dim() == 3
    if not batched:
        A, B = A[None], B[None]
    bsz, M, K = A.shape
    N = B.shape[2]
    C = torch.zeros((bsz, M, N), device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n), bsz)
    _tri_bmm_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        LOWER=lower, BM=block_m, BN=block_n, BK=block_k,
    )
    return C if batched else C[0]


def _selftest():
    torch.manual_seed(0)
    dev = "cuda"
    for lower in (False, True):
        for shape in ((512, 512), (3, 256, 256)):
            X = torch.rand(shape, device=dev)
            Y = torch.rand(shape, device=dev)
            A = torch.tril(X) if lower else torch.triu(X)
            B = torch.tril(Y) if lower else torch.triu(Y)
            mm = torch.bmm(A, B) if A.dim() == 3 else torch.matmul(A, B)
            ref = torch.tril(mm) if lower else torch.triu(mm)
            out = tri_matmul(A, B, lower=lower)
            assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2), (lower, shape)
            print(f"ok lower={lower} shape={shape} max_err={(out - ref).abs().max().item():.4f}")
    print("tri_matmul self-test passed")


if __name__ == "__main__":
    _selftest()
