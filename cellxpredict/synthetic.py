import numpy as np


def create_synthetic_trajectory(
    init_encoding: np.ndarray, length: int = 128, noise: float = 0.1
):
    """Create a synthetic trajectory based on an intial encoding.

    Parameters
    ----------
    init_encoding : array
        An initial encoding to initialize the trajectory.
    length : int
        The length of the desired trajectory.
    noise : float
        The amplitude of the noise in the random walk (stdev).

    Returns
    -------
    synthetic : array
        The synthetic trajectory.
    """

    dims = init_encoding.shape[0]
    synthetic = np.zeros((length, dims, 2), dtype=np.float32)
    synthetic[..., 1] = 1.0  # assume variance of 1
    synthetic[0, ...] = init_encoding
    synthetic[1:, ..., 0] = np.random.randn(length - 1, dims) * noise
    synthetic[..., 0] = np.cumsum(synthetic[..., 0], axis=0)
    return synthetic
