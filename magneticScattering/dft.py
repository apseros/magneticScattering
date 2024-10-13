import numpy as np
import logging


def dft(data, ft_pix=None, offset=None, upsample=1):
    n_pix = data.shape[0]
    ft_pix = n_pix if ft_pix is None else ft_pix
    offset = 0 if offset is None else offset

    kernel = np.einsum('k,n->kn', np.fft.fftfreq(ft_pix, upsample) - offset / ft_pix, np.arange(n_pix))
    return np.sum(data * np.exp(-2j * np.pi * kernel), axis=1)


def dftn(data, ft_pix_vec=None, offset_vec=None, upsample_vec=None):
    # offset behaves like a shift
    # upsample behaves like a zoom
    n_pix_vec, ndim = data.shape, data.ndim
    ft_pix_vec, offset_vec, upsample_vec = dft_properties(ft_pix_vec, n_pix_vec, ndim, offset_vec, upsample_vec)

    for n_pix, ft_pix, offset, upsample in zip(data.shape, ft_pix_vec, offset_vec, upsample_vec):
        # Here, we take the fft of the first dimension, send it to the back iteratively:
        # >>> abc..xyz -> bc..xyzA -> c...xyzAB -> ... -> zABC...XY -> ABC...XYZ
        # capital letters denote fourier domain
        kernel = np.einsum('k,n->kn', np.fft.fftfreq(ft_pix, upsample) - offset / ft_pix, np.arange(n_pix))
        data = np.einsum('kn,n...->...k', np.exp(-2j * np.pi * kernel), data)
    return data


def dftn_axes(data, ft_pix_vec=None, offset_vec=None, upsample_vec=None, axes=None):
    n_pix_vec, ndim = data.shape, data.ndim
    if axes is None:
        axes = tuple(range(data.ndim))
    axes = np.array(axes)
    axes = np.where(axes < 0, ndim + axes, axes)

    ft_pix_vec, offset_vec, upsample_vec = dft_properties(ft_pix_vec, n_pix_vec, ndim, offset_vec, upsample_vec)

    for dim, n_pix, ft_pix, offset, upsample in zip(range(data.ndim), data.shape, ft_pix_vec, offset_vec,
                                                    upsample_vec):
        # Here, we take the fft of the first dimension, send it to the back iteratively:
        # >>> abc..xyz -> bc..xyzA -> c...xyzAB -> ... -> zABC...XY -> ABC...XYZ
        # capital letters denote fourier domain
        if dim in axes:
            kernel = np.einsum('k,n->kn', np.fft.fftfreq(ft_pix, upsample) - offset / ft_pix, np.arange(n_pix))
            data = np.einsum('kn,n...->...k', np.exp(-2j * np.pi * kernel), data)
        else:
            data = np.rollaxis(data, data.ndim - 1)
    return data


def dft_properties(ft_pix_vec, n_pix_vec, ndim, offset_vec, upsample_vec):
    ft_pix_vec = n_pix_vec if ft_pix_vec is None else ft_pix_vec
    offset_vec = 0 if offset_vec is None else offset_vec
    upsample_vec = 1 if upsample_vec is None else upsample_vec
    ft_pix_vec = (ft_pix_vec,) * ndim if isinstance(ft_pix_vec, (int, float)) else ft_pix_vec
    offset_vec = (offset_vec,) * ndim if isinstance(offset_vec, (int, float)) else offset_vec
    upsample_vec = (upsample_vec,) * ndim if isinstance(upsample_vec, (int, float)) else upsample_vec
    return ft_pix_vec, offset_vec, upsample_vec


def calc_params_from_roi(current, roi, datashape):
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
