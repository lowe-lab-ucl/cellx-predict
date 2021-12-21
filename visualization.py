import numpy as np
from skimage.filters import median as median_filter
from skimage.morphology import disk


def norm_channel(
    input: np.array,
    sigma: float = 2.0,
    threshold: float = 5.0,
    low_percentile: float = 5.0,
    high_percentile: float = 99.0,
) -> np.ndarray:

    copy = np.copy(input, dtype=np.float32)

    # remove outliers
    med = median_filter(copy, disk(sigma))
    diff = np.abs(copy - med)
    mask = diff > threshold
    copy[mask] = med[mask]

    # norm by histogram
    p_lo = np.percentile(copy, low_percentile)
    p_hi = np.percentile(copy, high_percentile)

    norm_copy = np.clip((copy - p_lo) / (p_hi - p_lo), 0.0, 1.0)

    return norm_copy


def to_rgb(input: np.array, axis: int = -1):
    _rgb_ch_0 = norm_channel(np.take_along_axis(input, 1, axis))
    _rgb_ch_1 = norm_channel(np.take_along_axis(input, 0, axis))
    _rgb_ch_2 = norm_channel(np.take_along_axis(input, 1, axis))
    _rgb = np.stack([_rgb_ch_0, _rgb_ch_1, _rgb_ch_2], axis=axis)
    return _rgb
