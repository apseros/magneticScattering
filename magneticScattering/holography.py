import magpack.image_utils
from magneticScattering.scatter import Sample, Scatter
import numpy as np


def holography_reference(sample, hole_size=1, axis='y'):
    """Creates a holography reference hole for the given sample.

    The new sample has a fourfold increased size along the specified axis to allow for recovery.

    Parameters
    ----------
    sample : np.ndarray | Sample
        The sample to add holography reference holes.
    hole_size : int (optional)
        The size of reference hole in voxels.
    axis : {'x', 'y', 'xy', 'yx'} (optional)
        Axis along which to add the reference.

    Returns
    -------
    np.ndarray (optional)
        The holography sample or None if directly applied to a Sample class.
    """
    if isinstance(sample, Sample):
        initial_structure = np.array(sample.structure)
    else:
        initial_structure = sample

    if hole_size <= 0:
        hole_size = 1

    if axis in ['xy', 'yx']:
        if isinstance(sample, Sample):
            holography_reference(sample, hole_size, 'x')
            holography_reference(sample, hole_size, 'y')
            return None
        else:
            step2 = holography_reference(holography_reference(initial_structure, hole_size, axis='x'), hole_size,
                                         axis='y')
            return step2
    elif axis == 'x':
        initial_structure = initial_structure.transpose(0, 2, 1)
    else:
        axis = 'y'

    components, init_x, init_y = initial_structure.shape

    # Create a zero array that is 4x4 times the size of the tile
    grid = np.zeros((components, init_x, init_y * 4))

    # Place the tile in the center
    center_x, center_y = init_x // 2, 2 * init_y  # True center
    grid[:, center_x - init_x // 2:center_x + init_x // 2,
    center_y - init_y // 2:center_y + init_y // 2] = initial_structure

    ref_x_center, ref_y_center = int(center_x), int(init_y * 10.5 / 3)
    # structure placed in the lower right corner
    grid[0, ref_x_center - hole_size // 2:ref_x_center + hole_size // 2 + 1,
    ref_y_center - hole_size // 2:ref_y_center + hole_size // 2 + 1] = 1

    if axis == 'x':
        grid = grid.transpose(0, 2, 1)

    if isinstance(sample, Sample):
        sample.sample_length = sample.sample_length * [1, 4] if axis == 'y' else sample.sample_length * [4, 1]
        sample.structure = grid

    else:
        return grid


def invert_holography(scatter_a, scatter_b=None):
    """Inverts the scattering pattern assuming Fourier transform holography was performed.

    Parameters
    ----------
    scatter_a : Scatter
        The scatter pattern to invert.
    scatter_b : Scatter (optional)
        The scatter pattern to invert for taking the difference.

    Returns
    -------
    inv : np.ndarray
        The inverted scatter pattern.
    roi : tuple of float
        The corresponding real space dimensions.
    """

    intensity1, len1 = scatter_a.intensity, scatter_a.sample.sample_length / 2
    if scatter_b is not None:
        intensity2, len2 = scatter_b.intensity, scatter_b.sample.sample_length / 2
    else:
        intensity2, len2 = None, None

    roi1 = [-len1[0], len1[0], -len1[1], len1[1]]

    if intensity2 is not None and intensity1.shape != intensity2.shape:
        raise ValueError("Scattering intensities must have the same shape.")
    if len1 is not None and len2 is not None and np.any(len1 != len2):
        raise ValueError("Scattering regions must be similar.")

    inv1 = np.abs(magpack.image_utils.fft(intensity1))
    inv2 = np.abs(magpack.image_utils.fft(intensity2)) if intensity2 is not None else None

    if inv2 is not None:
        return inv1 - inv2, roi1
    else:
        return inv1, roi1
