import numpy as np
import pytest

from cellxpredict.synthetic import create_synthetic_trajectory


@pytest.mark.parametrize("init_encoding", [np.random.randn(x, 2) for x in [1, 10, 100]])
@pytest.mark.parametrize("length", [1, 10, 100])
def test_synthetic(init_encoding, length):
    """Test synthetic data generation."""
    synth = create_synthetic_trajectory(init_encoding, length=length)
    dims = init_encoding.shape[0]
    assert synth.shape == (length, dims, 2)
