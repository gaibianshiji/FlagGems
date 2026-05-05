"""
Benchmark for CTC loss operator.

Uses FlagGems benchmark framework to measure device-side performance
against PyTorch native implementation.
"""

import pytest
import torch

import flag_gems

from . import base, consts


class CtcLossBenchmark(base.GenericBenchmark):
    """Custom benchmark for CTC loss that preserves custom shapes."""

    def init_user_config(self):
        # Don't call super().init_user_config() to avoid loading from YAML
        self.mode = base.Config.mode
        self.set_dtypes(base.Config.user_desired_dtypes)
        self.set_metrics(base.Config.user_desired_metrics)


def ctc_loss_ref(log_probs, targets, input_lengths, target_lengths, **kwargs):
    """Reference CTC loss that works with float16/bfloat16 by casting to float32."""
    return torch.nn.functional.ctc_loss(log_probs.float(), targets, input_lengths, target_lengths, **kwargs)


def ctc_loss_gems(log_probs, targets, input_lengths, target_lengths, **kwargs):
    """FlagGems CTC loss with reduction string-to-int conversion."""
    reduction_map = {"none": 0, "mean": 1, "sum": 2}
    reduction = kwargs.pop("reduction", "mean")
    kwargs["reduction"] = reduction_map.get(reduction, 1)
    return flag_gems.ctc_loss(log_probs, targets, input_lengths, target_lengths, **kwargs)


def ctc_loss_input_fn(shape, dtype, device):
    """Generate inputs for CTC loss benchmark."""
    T, B, C = shape[0], shape[1], shape[2]
    log_probs = torch.randn(T, B, C, dtype=dtype, device=device).log_softmax(2)
    S = min(C - 1, 10)
    targets = torch.randint(1, C, (B, S), device=device)
    input_lengths = torch.full((B,), T, device=device, dtype=torch.long)
    target_lengths = torch.full((B,), S, device=device, dtype=torch.long)
    yield log_probs, targets, input_lengths, target_lengths

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield log_probs, targets, input_lengths, target_lengths, {"reduction": "mean"}
        yield log_probs, targets, input_lengths, target_lengths, {"reduction": "sum"}
        yield log_probs, targets, input_lengths, target_lengths, {"reduction": "none"}


@pytest.mark.ctc_loss
def test_ctc_loss():
    bench = CtcLossBenchmark(
        op_name="ctc_loss",
        input_fn=ctc_loss_input_fn,
        torch_op=ctc_loss_ref,
        gems_op=ctc_loss_gems,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(5, 1, 3), (50, 16, 26), (100, 32, 50), (200, 64, 50)]
    bench.shape_desc = "T, B, C"
    bench.run()
