# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import logging

import torch

from flag_gems import runtime
from flag_gems.fused.FLA.utils import SUPPRESS_LEVEL

logger = logging.getLogger(__name__)

# Detect vendor for backend-specific dispatch
_VENDOR = getattr(runtime.device, "vendor_name", "")
_IS_ILUVATAR = _VENDOR in ("iluvatar",)

# Always import native implementation for fallback (large K, Iluvatar)
from flag_gems.fused.FLA.chunk_gated_delta_rule_native import (
    chunk_gated_delta_rule_native_fwd as _native_fwd,
)

# Import Triton kernels only when not on Iluvatar
if not _IS_ILUVATAR:
    from flag_gems.fused.FLA.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from flag_gems.fused.FLA.chunk_o import chunk_fwd_o
    from flag_gems.fused.FLA.fused_cumsum_kkt_solve_tril import (
        chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril,
    )
    from flag_gems.fused.FLA.wy_fast import recompute_w_u_fwd


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
):
    logger.debug("GEMS CHUNK GATED DELTA RULE FWD")

    K = k.shape[-1]

    # Use native implementation on Iluvatar or when K > 128 (shared memory limit)
    if _IS_ILUVATAR or K > 128:
        return _native_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

    # Use Triton path for NVIDIA with K <= 128
    g, A = chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
        g=g, k=k, beta=beta, cu_seqlens=cu_seqlens, chunk_size=64, output_dtype=k.dtype
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new
