# Benchmark for chunk_gated_delta_rule using FlagGems benchmark framework.
#
# Baseline: PyTorch-native chunk implementation (vectorized, no Python loop over T).
# This is a fair comparison since both implementations use the same chunk-based
# algorithm with batched matrix operations.

import pytest
import torch
import torch.nn.functional as F

import flag_gems
from benchmark.base import Benchmark
from benchmark.conftest import Config
from flag_gems.fused.FLA.chunk_gated_delta_rule_native import (
    chunk_gated_delta_rule_native_fwd,
)


def _torch_op_wrapper(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
):
    """Baseline: PyTorch-native chunk implementation (same algorithm, no Triton)."""
    return chunk_gated_delta_rule_native_fwd(
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


class ChunkGatedDeltaRuleBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.float32, torch.float16]
    DEFAULT_SHAPES = [
        (64,),
        (128,),
        (256,),
        (512,),
        (1024,),
        (2048,),
    ]

    def init_user_config(self):
        """Override to skip YAML shape loading — use custom shapes only."""
        self.mode = Config.mode
        self.set_dtypes(Config.user_desired_dtypes)
        self.set_metrics(Config.user_desired_metrics)
        self.shapes = self.DEFAULT_SHAPES

    def set_more_shapes(self):
        return []

    def get_input_iter(self, cur_dtype):
        for (T,) in self.shapes:
            yield self._build_inputs(T, cur_dtype)

    def _build_inputs(self, T: int, dtype: torch.dtype):
        device = flag_gems.device
        B, H, K, V = 1, 8, 64, 64
        Hg = H

        q = torch.randn(B, T, Hg, K, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, T, Hg, K, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1
        g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype)) * 0.1
        beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
        scale = K**-0.5

        return (
            q,
            k,
            v,
            g,
            beta,
            scale,
            None,  # initial_state
            False,  # output_final_state
            None,  # cu_seqlens
        )


@pytest.mark.skipif(flag_gems.device != "cuda", reason="benchmark requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_perf_chunk_gated_delta_rule():
    bench = ChunkGatedDeltaRuleBenchmark(
        op_name="chunk_gated_delta_rule",
        torch_op=_torch_op_wrapper,
    )
    bench.set_gems(flag_gems.chunk_gated_delta_rule_fwd)
    bench.run()
