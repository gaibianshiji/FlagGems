import pytest
import torch

from . import base, consts, utils


class SVDBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        return [
            (32, 32),
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
            (128, 64),
            (256, 128),
            (512, 256),
            (1024, 512),
            (64, 128),
            (128, 256),
            (256, 512),
            (512, 1024),
        ]


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"some": True},


@pytest.mark.svd
def test_perf_svd():
    bench = SVDBenchmark(
        input_fn=_input_fn,
        op_name="svd",
        torch_op=torch.svd,
        dtypes=[torch.float32],
    )

    bench.run()
