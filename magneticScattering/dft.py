import numpy as np
import logging


def dft(data: np.ndarray, ft_pix: int = None, offset: float = None, upsample: float = 1) -> np.ndarray:
    """One dimensional discrete Fourier transform (DFT).

    :param data:        Input array of data to transform.
    :param ft_pix:      Number of Fourier transform points.
    :param offset:      Frequency offset to adjust the frequency grid.
    :param upsample:    Factor by which to upsample the frequency grid.

    :returns:           Array of the computed Fourier transform.
    """
    n_pix = data.shape[0]
    ft_pix = n_pix if ft_pix is None else ft_pix
    offset = 0 if offset is None else offset
    kernel = np.einsum('k,n->kn',
                       np.fft.fftfreq(ft_pix, upsample) - offset / ft_pix, np.arange(n_pix))
    return np.sum(data * np.exp(-2j * np.pi * kernel), axis=1)


def dftn_axes(data, ft_pix_vec=None, offset_vec=None, upsample_vec=None, axes=None) -> np.ndarray:
    """Performs n-dimensional discrete Fourier transform along specified axes.

    :param data:            Input array of data to transform.
    :param ft_pix_vec:      Number of Fourier transform points.
    :param offset_vec:      Frequency offset to adjust the frequency grid.
    :param upsample_vec:    Factor by which to upsample the frequency grid.
    :param axes:            Axes along which to compute Fourier transform.
    :returns:               Array of the computed Fourier transform.
    """
    n_pix_vec, ndim = data.shape, data.ndim
    if axes is None:
        axes = tuple(range(data.ndim))
    axes = np.array(axes)
    axes = np.where(axes < 0, ndim + axes, axes)

    ft_pix_vec, offset_vec, upsample_vec = _dft_properties(ft_pix_vec, n_pix_vec, ndim, offset_vec, upsample_vec)

    for dim, n_pix, ft_pix, offset, upsample in zip(range(data.ndim), data.shape, ft_pix_vec, offset_vec,
                                                    upsample_vec):
        # Here, we take the fft of the first dimension, send it to the back and iterate:
        # >>> abc..xyz -> bc..xyzA -> c...xyzAB -> ... -> zABC...XY -> ABC...XYZ
        # capital letters denote fourier domain
        if dim in axes:
            kernel = np.einsum('k,n->kn',
                               np.fft.fftfreq(ft_pix, upsample) - offset / ft_pix, np.arange(n_pix))
            data = np.einsum('kn,n...->...k', np.exp(-2j * np.pi * kernel), data)
        else:
            data = np.rollaxis(data, data.ndim - 1)
    return data


def _dft_properties(ft_pix_vec, n_pix_vec, ndim, offset_vec, upsample_vec):
    """Validates the dft properties for the desired dimension.

    :param ft_pix_vec:      Vector of Fourier transform points.
    :param n_pix_vec:       Shape of in initial image.
    :param ndim:            Number of spatial dimensions.
    :param offset_vec:      Vector of frequency offsets.
    :param upsample_vec:    Vector of upsampling factors.
    :returns:               Appropriate array for Fourier transform points, offset and upsampling.
    """

    ft_pix_vec = n_pix_vec if ft_pix_vec is None else ft_pix_vec
    offset_vec = 0 if offset_vec is None else offset_vec
    upsample_vec = 1 if upsample_vec is None else upsample_vec
    ft_pix_vec = (ft_pix_vec,) * ndim if isinstance(ft_pix_vec, (int, float)) else ft_pix_vec
    offset_vec = (offset_vec,) * ndim if isinstance(offset_vec, (int, float)) else offset_vec
    upsample_vec = (upsample_vec,) * ndim if isinstance(upsample_vec, (int, float)) else upsample_vec
    return ft_pix_vec, offset_vec, upsample_vec


def calc_params_from_roi(current: tuple[float, float, float, float], roi: tuple[float, float, float, float],
                         datashape: tuple[int,...]) -> tuple[tuple, tuple]:
    """Given the current extent of the image and the desired roi to upsample, the appropriate offset and upsample values
    are provided.

    :param current:         Current extent of the image.
    :param roi:             Desired roi.
    :param datashape:       Desired shape of the data.
    :returns:               Corresponding offset and upsample values.
    """
    shape = np.array(datashape)
    current = np.array(current)
    roi = np.array(roi)

    range_ext = current[1] - current[0], current[3] - current[2]
    range_roi = roi[1] - roi[0], roi[3] - roi[2]
    pix_size = range_ext / shape

    logging.debug("range_ext: %s, range_roi: %s", range_ext, range_roi)

    center_ext = np.array([(current[0] + current[1]), (current[2] + current[3])])
    center_roi = np.array([(roi[0] + roi[1]), (roi[2] + roi[3])])
    logging.debug("center_ext: %s, center_roi: %s", center_ext, center_roi)

    upsample = range_ext[0] / range_roi[0], range_ext[1] / range_roi[1]
    logging.debug("upsample: %s", upsample)
    logging.debug("pix_size: %s", pix_size)
    logging.debug("pix_size*shape (should be extent no?): %s", pix_size * shape)

    offset = tuple((center_ext - center_roi) / 2 / pix_size)
    logging.debug("offset: %s", offset)

    return offset, upsample
