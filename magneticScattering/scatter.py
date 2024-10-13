import logging
import numpy as np
import copy
from typing import Union
import dft
import structures


def pauli():
    p_matrix = np.zeros([4, 2, 2], dtype=np.complex128)
    p_matrix[0] = np.eye(2)
    p_matrix[1] = np.array([[1, 0], [0, -1]])
    p_matrix[2] = np.array([[0, 1], [1, 0]])
    p_matrix[3] = np.array([[0, -1j], [1j, 0]])
    return p_matrix


def en2wave(energy):
    """Converts energy in eV to wavelength in m."""
    return 1.24e-6 / energy


def vec2np(vec: Union[float, list, complex, int, np.ndarray], size=2, dtype=np.float64):
    if isinstance(vec, list):
        if len(vec) == 1:
            return np.array(vec * size, dtype=dtype)
        elif len(vec) == size:
            return np.array(vec, dtype=dtype)
        else:
            logging.error('Too many or not enough elements in list, expected {} got {}'.format(size, len(vec)))
            raise ValueError

    elif isinstance(vec, (int, float, complex)):
        return np.array([vec] * size, dtype=dtype)

    elif isinstance(vec, np.ndarray):
        assert vec.shape == (size,), 'Wrong shape of array, expected {} got {}'.format((size,), vec.shape)
        return vec


class Beam:
    """Describes the properties of the beam.

    Attributes:
        -wavelength :class:`float`:     The wavelength of the beam [m].
        -fwhm :class:`np.ndarray`:      The full width at half maximum of the beam in x and y.
        -pol :class:`list[float]`:      The polarization of the beam in Stokes parameters, 4 component list.
    """

    def __init__(self, wavelength: float, fwhm: Union[np.ndarray, list[float], float],
                 pol: Union[np.ndarray, list[float]]):
        self._wavelength = wavelength
        self._fwhm = vec2np(fwhm)
        self._beam_sigma = self._fwhm / np.sqrt(8 * np.log(2))
        self._pol = vec2np(pol, 4)
        self._density_matrix, self._degree_of_pol = self.calc_density_matrix()
        self._change_tracker = [lambda x: None]

    def calc_density_matrix(self):
        """Converts the Stokes polarization vector to a density matrix

        The expression for converting the Strokes polarization "vector" to a density matrix is:
        rho = 1 / 2 * (P . sigma)
        where P is the four-dimensional vector and sigma is the vector containing the Pauli spin matrices and the
        identity matrix (sigma_0 = I).
        """
        pol_intensity = np.sum(self._pol[-3:] ** 2)
        pol_norm = self._pol[0] ** 2
        if pol_intensity > pol_norm:
            self._pol[-3:] = self._pol[-3:] / np.sqrt(pol_intensity)
            logging.warning("The intensity of the polarization vector is larger than the norm.")

        if pol_norm > 1:
            self._pol = self._pol / np.sqrt(pol_norm)
            logging.warning("The norm of the polarization vector is larger than 1.")

        density_matrix = np.einsum('i,ijk->jk', self._pol, pauli()) / 2
        degree_of_pol = np.einsum('aij,ji->a', pauli(), density_matrix)[-3:].real ** 2 * 100

        return density_matrix, degree_of_pol

    @property
    def tracker(self):
        return self._change_tracker

    @tracker.setter
    def tracker(self, parent_tracker_function):
        self._change_tracker.append(parent_tracker_function)

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength: float):
        self._wavelength = wavelength
        [f('wavelength') for f in self._change_tracker]

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, fwhm: Union[np.ndarray, list[float], float]):
        self._fwhm = vec2np(fwhm)
        self._beam_sigma = self._fwhm / np.sqrt(8 * np.log(2))
        [f('fwhm') for f in self._change_tracker]

    @property
    def sigma(self):
        return self._beam_sigma

    @property
    def pol(self):
        return self._pol

    @pol.setter
    def pol(self, pol: Union[np.ndarray, list[float]]):
        self._pol = vec2np(pol, 4)
        self._density_matrix, self._degree_of_pol = self.calc_density_matrix()
        [f('pol') for f in self._change_tracker]

    @property
    def beam_sigma(self):
        return self._fwhm / np.sqrt(8 * np.log(2))

    @property
    def density_matrix(self):
        return self._density_matrix

    @property
    def degree_of_pol(self):
        return self._degree_of_pol


class Sample:
    """Describes the sample properties.

    Attributes:
        -sample_length :class:`np.ndarray`:          The dimensions of the sample [m].
        -scattering_factors :class:`list[complex]`:  List of the three complex scattering factors.
        -structure :class:`np.ndarray`:              The magnetic configuration of the sample, (4, X, Y) numpy array.
        -reference_structure :class:`np.ndarray`:    Backup of the original structure for when the structure is rotated.
    """

    def __init__(self, sample_length: Union[np.ndarray, list[float], float],
                 scattering_factors: Union[list[complex], np.ndarray], structure: np.ndarray):
        self._sample_length = vec2np(sample_length)
        self._scattering_factors = vec2np(scattering_factors, 3, dtype=np.complex128)

        if structure.ndim != 3 or structure.shape[0] != 4:
            logging.error("Structure has wrong shape, must be (4, x, y).")
            raise ValueError

        self._structure = copy.deepcopy(structure)
        self._pix_size = self.calc_pix_size()
        self._change_tracker = [lambda x: None]

    @property
    def tracker(self):
        return self._change_tracker

    @tracker.setter
    def tracker(self, parent_tracker_function):
        self._change_tracker.append(parent_tracker_function)

    def calc_pix_size(self):
        return np.divide(self._sample_length, self.shape)

    @property
    def sample_length(self):
        return self._sample_length

    @sample_length.setter
    def sample_length(self, v: Union[np.ndarray, list[float], float]):
        self._sample_length = vec2np(v, 2)
        self._pix_size = self.calc_pix_size()
        [f('sample_length') for f in self._change_tracker]

    @property
    def scattering_factors(self):
        return self._scattering_factors

    @scattering_factors.setter
    def scattering_factors(self, v: Union[list[complex], np.ndarray]):
        self._scattering_factors = vec2np(v, 3, dtype=np.complex128)
        [f('scattering_factors') for f in self._change_tracker]

    @property
    def shape(self):
        return self._structure.shape[-2:]

    @property
    def pix_size(self):
        return self._pix_size

    @pix_size.setter
    def pix_size(self, v: Union[np.ndarray, list[float], float]):
        self._pix_size = vec2np(v, 2)
        self._sample_length = np.multiply(self._pix_size, self.shape)
        [f('pix_size') for f in self._change_tracker]

    @property
    def structure(self) -> np.ndarray:
        return self._structure

    @structure.setter
    def structure(self, v: np.ndarray):
        if v.ndim != 3 or v.shape[0] != 4:
            logging.error("Structure has wrong shape, must be (4, x, y).")
            raise ValueError
        self._structure = copy.deepcopy(v)
        self._pix_size = self.calc_pix_size()
        [f('structure') for f in self._change_tracker]

    def get_extent(self):
        extent_x = self.sample_length[0] * np.array([-0.5, 0.5])
        extent_y = self.sample_length[1] * np.array([-0.5, 0.5])
        return np.hstack([extent_x, extent_y])

    def get_coordinates(self):
        nx, ny = self.shape
        extent = self.get_extent()
        xx, yy = np.linspace(extent[0], extent[1], nx), np.linspace(extent[2], extent[3], ny)
        return np.meshgrid(xx, yy, indexing='ij')


class Geometry:
    """Defines the geometry of the experiment.

    Attributes:
        angle :class:`float`:               The angle of incidence (and scattering) of the beam [degrees].
        detector_distance :class:`float`:   The distance between the sample and the detector [m].
    """

    def __init__(self, angle: float, detector_distance: float):
        self._angle = angle
        self._detector_distance = detector_distance
        self._change_tracker = [lambda x: None]

    @property
    def tracker(self):
        return self._change_tracker

    @tracker.setter
    def tracker(self, parent_tracker_function):
        self._change_tracker = parent_tracker_function

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, v: float):
        self._angle = v
        [f('angle') for f in self._change_tracker]

    @property
    def detector_distance(self):
        return self._detector_distance

    @detector_distance.setter
    def detector_distance(self, v: float):
        self._detector_distance = v
        [f('detector_distance') for f in self._change_tracker]


class Scatter:
    def __init__(self, beam: Beam, sample: Sample, geometry: Geometry):
        """Initializes the Scatter class.

        :param beam:        The beam properties form the Beam class.
        :param sample:      The sample properties from the Sample class.
        :param geometry:    The geometry of the experiment from the Geometry class.
        """

        self.beam = beam
        self.beam.tracker = self.change_tracker
        self.sample = sample
        self.sample.tracker = self.change_tracker
        self.geometry = geometry
        self.geometry.tracker = self.change_tracker
        self._extent = self._get_extent()
        self._roi = self._extent
        self._roi_shape = self.sample.shape
        self.run()

    @property
    def roi_shape(self):
        return self._roi_shape

    @property
    def extent(self):
        return self._extent

    @property
    def angular_extent(self):
        return np.rad2deg(self._extent / self.geometry.detector_distance)

    @property
    def roi(self):
        return self._roi

    @roi.setter
    def roi(self, v: list[float]):
        v = vec2np(v, 4)
        self._roi = v
        self.run()

    @property
    def roi_angular_extent(self):
        # angular extent in degrees around the center
        return np.rad2deg(self._roi / self.geometry.detector_distance)

    def change_tracker(self, name):
        logging.debug("Detected change in subclass attribute: {}.".format(name))
        if name in ['wavelength', 'sample_length', 'pix_size', 'detector_distance']:
            self._get_extent()
        elif name in ['pol', 'fwhm', 'scattering_factors', 'angle', 'structure']:
            self.run()
        else:
            logging.error("Unknown attribute change detected: {}.".format(name))

    def _get_extent(self):
        angular_space = 1 / (2 * self.sample.pix_size) * self.beam.wavelength
        real_space = angular_space * self.geometry.detector_distance
        return np.array([-real_space[0], real_space[0], -real_space[1], real_space[1]])

    def run(self):
        conv_structure = self.sample.structure * structures.gaussian_2d(self.sample.get_coordinates(),
                                                                        self.beam.sigma)
        structure_sf = self._calc_scattering_factor(conv_structure)
        fft_structure_sf = self._calc_ft(structure_sf)
        self._calc_intensity(fft_structure_sf)

    def _calc_scattering_factor(self, m):
        f_0, f_1, f_2 = self.sample.scattering_factors
        theta = np.deg2rad(self.geometry.angle)

        s = np.sin(theta)
        c = np.cos(theta)
        s2 = np.sin(2 * theta)
        c2 = np.cos(2 * theta)

        nx, ny = self.sample.shape
        f = np.zeros((2, 2, nx, ny), dtype=np.complex128)
        f[0, 0, :, :] = f_0 * m[0] + f_2 * m[1] ** 2
        f[0, 1, :, :] = -1j * f_1 * (m[3] * c - m[2] * s) + f_2 * m[1] * (m[2] * c + m[3] * s)
        f[1, 0, :, :] = -1j * f_1 * (m[2] * s - m[3] * c) + f_2 * m[1] * (m[2] * c - m[3] * s)
        f[1, 1, :, :] = f_0 * m[0] * c2 - 1j * f_1 * m[1] * s2 + f_2 * ((m[2] * c) ** 2 - (m[3] * s) ** 2)
        return f

    def _calc_ft(self, structure_sf):
        if np.allclose(self.extent, self.roi):
            fft_structure_sf = np.fft.fftshift(np.fft.fftn(structure_sf, axes=(-2, -1)), axes=(-2, -1))
        else:
            roi_shape = self._roi_shape
            offset, sampling = dft.calc_params_from_roi(self.extent, self.roi, roi_shape)
            fft_structure_sf = dft.dftn_axes(structure_sf, ft_pix_vec=(0, 0) + roi_shape, offset_vec=(0, 0) + offset,
                                             upsample_vec=(0, 0) + sampling, axes=(-2, -1))
            fft_structure_sf = np.fft.fftshift(fft_structure_sf, axes=(-2, -1))
        return fft_structure_sf

    def _calc_intensity(self, fft_sf):
        mu_prime = np.einsum('ijab,jk,lkab->ilab', fft_sf, self.beam.density_matrix, fft_sf.conjugate())
        self.intensity = np.abs(np.einsum('iiab->ab', mu_prime))
        self.pol_out = np.abs(np.einsum('ijk,kjab->iab', pauli(), mu_prime))

