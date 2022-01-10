import numpy as np
from skimage.filters import median as median_filter
from skimage.morphology import disk

from typing import List

def float_to_uint8(input: np.ndarray[np.float32]) -> np.ndarray[np.uint8]:
    '''
    Transforms an np.float32 image into a np.uint8 image

    Parameters
    ----------
    input: np.array[np.float32]
        - Input image of shape (width,height,channels)
    
    Returns
    -------
    output: np.array[np.uint8]
        - Output image of shape (width,height,channels)
    '''
    mn = input.min()
    mx = input.max()
    mx -= mn
    output = ((input - mn)/mx) * 255
    return output.astype(np.uint8)

def norm_channel(
    input: np.ndarray,
    sigma: float = 2.0,
    threshold: float = 5.0,
    low_percentile: float = 5.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    '''
    Normalizes a single image channel for visualization purposes.

    Parameters
    ----------
    input: np.ndarray
        - Single-channel image
    sigma: float
        - Scales the size of the median filter
    threshold: float
        - Value for deviation-from-median outside of which to remove outliers
    low_percentile:
        - Pixel-intensity percentile below which to clip the image
    high_percentile:width
        - Normalized single-channel image
    '''
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

def _to_rgb_single_image(input: np.ndarray,
                         axis: int = -1,
                         to_uint8: bool = True,
                         ordered_channels: List[int] = [1,0,1]) -> np.ndarray:
    _rgb_ch_0 = norm_channel(np.take_along_axis(input, ordered_channels[0], axis))
    _rgb_ch_1 = norm_channel(np.take_along_axis(input, ordered_channels[1], axis))
    _rgb_ch_2 = norm_channel(np.take_along_axis(input, ordered_channels[2], axis))
    _rgb_channels = [_rgb_ch_0, _rgb_ch_1, _rgb_ch_2]
    if to_uint8:
        _rgb_channels = list(map(float_to_uint8, _rgb_channels))
    output = np.stack(_rgb_channels, axis=axis)
    return output


def to_rgb(input: np.ndarray,
           axis: int = -1,
           to_uint8: bool = True,
           ordered_channels: List[int] = [1,0,1]) -> np.ndarray:
    '''
    Transforms a multichannel image (of series of images) into RGB.

    Parameters
    ----------
    input: np.ndarray
        - Image(s) to transform into rgb, of shape (...,width,height,channels)
    axis: int
        - Axis on which to take the channels
    to_uint8: bool
        - Whether to transform the output into uint8 from float32
    ordered_channels: List[int]
        - The input channels corresponding to the rgb output channels

    Returns
    -------
    output: np.ndarray
        - RGB-transformed output image(s), of shape (...,width,height,3)
    '''
    if len(ordered_channels) != 3 or not all([type(ch)==int for ch in ordered_channels]):
        raise ValueError('''Parameter 'ordered_channels' must contain three integers specifying the input channels 
                         that will map on to the red, green and blue channels in the output.''')

    to_rgb_single_image = lambda x: _to_rgb_single_image(x,
                                                         axis=axis, 
                                                         to_uint8=to_uint8,
                                                         ordered_channels=ordered_channels)
    output = np.apply_over_axes(to_rgb_single_image, input, [-3,-2,-1])
    return output

