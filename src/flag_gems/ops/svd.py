import logging

import torch

logger = logging.getLogger(__name__)


def svd(inp, some=True, compute_uv=True):
    """Singular Value Decomposition: A = U @ diag(S) @ V^T.

    Wraps torch.linalg.svd to match the torch.svd API convention where
    V is the right singular vectors (transpose of Vh from linalg.svd).

    Args:
        inp: Input tensor of shape (..., m, n). Must be at least 2-D.
        some: If True (default), compute the reduced/compact SVD with
              U of shape (..., m, k) and V of shape (..., n, k) where
              k = min(m, n). If False, compute full SVD with U of shape
              (..., m, m) and V of shape (..., n, n).
        compute_uv: If True (default), compute U, S, V. If False, only
                    compute S (singular values). U and V are returned as
                    empty tensors.

    Returns:
        (U, S, V) tuple of tensors:
        - U: Left singular vectors, shape (..., m, k) or (..., m, m)
        - S: Singular values in descending order, shape (..., k)
        - V: Right singular vectors, shape (..., n, k) or (..., n, n)
        When compute_uv=False, U and V are empty tensors.

    Notes:
        - float16/bfloat16 inputs are promoted to float32 for computation,
          then cast back to the original dtype.
        - Singular values are always non-negative and in descending order.
        - The decomposition satisfies A ≈ U @ diag(S) @ V.T (up to
          floating-point precision).
        - NaN/Inf in input propagates to outputs (matching torch.svd behavior).
    """
    logger.debug("GEMS SVD")

    if inp.ndim < 2:
        raise ValueError("svd: input must be at least 2-D")

    dtype = inp.dtype
    need_cast = dtype not in (torch.float32, torch.float64)
    work = inp.to(torch.float32) if need_cast else inp

    if compute_uv:
        # torch.linalg.svd returns (U, S, Vh) where A = U @ diag(S) @ Vh
        # torch.svd returns (U, S, V) where A = U @ diag(S) @ V.T
        # So V = Vh.mT (lazy transpose, no data copy)
        U, S, Vh = torch.linalg.svd(work, full_matrices=not some)
        V = Vh.mT
        if need_cast:
            U = U.to(dtype)
            V = V.to(dtype)
    else:
        S = torch.linalg.svdvals(work)
        U = torch.empty(0, device=inp.device, dtype=dtype)
        V = torch.empty(0, device=inp.device, dtype=dtype)

    if need_cast:
        S = S.to(dtype)

    return U, S, V


def svd_out(inp, some=True, compute_uv=True, *, U_out, S_out, V_out):
    """SVD with pre-allocated output tensors (out variant).

    Computes SVD and copies results into the provided output tensors,
    avoiding allocation overhead when outputs are pre-allocated.

    Args:
        inp: Input tensor (see svd() for details).
        some: Reduced vs full SVD (see svd()).
        compute_uv: Whether to compute U/V (see svd()).
        U_out: Pre-allocated tensor for U output.
        S_out: Pre-allocated tensor for S output.
        V_out: Pre-allocated tensor for V output.

    Returns:
        (U_out, S_out, V_out) - the same tensors passed in, now filled
        with SVD results.
    """
    logger.debug("GEMS SVD_OUT")
    U, S, V = svd(inp, some=some, compute_uv=compute_uv)
    U_out.copy_(U)
    S_out.copy_(S)
    V_out.copy_(V)
    return U_out, S_out, V_out
