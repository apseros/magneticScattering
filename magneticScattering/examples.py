import numpy as np
from magneticScattering import scatter, plot
import matplotlib.pyplot as plt
from magneticScattering.holography import holography_reference
from importlib import resources
import magneticScattering.data
import gzip

pol_dict = {'LH': [1, 1, 0, 0], 'LV': [1, -1, 0, 0],
            'CL': [1, 0, 0, 1], 'CR': [1, 0, 0, -1]}


def dichroism():
    """Simulated the circular magnetic dichroism from a labyrinthine pattern.

    See :ref:`example` for a more detailed description."""
    energy = 706  # energy of the beam in eV
    fwhm = 20e-6  # full width at half maximum of the beam in meters
    angle = 0  # angle of incidence
    detector_distance = 50e-2  # sample-detector distance
    sample_length = 10e-6  # sample size
    scattering_factors = [1j + 1, 1j + 1, 1 + 1j]  # scattering factors
    reference_hole_size = 3 # size of reference hole in pixels

    # load magnetic configuration
    inp_file = resources.files(magneticScattering.data) / 'labyrinthine.npy.gz'

    # Open it using gzip
    with inp_file.open("rb") as raw_f:
        with gzip.GzipFile(fileobj=raw_f, mode="rb") as gz_f:
            mag_config = np.load(gz_f)

    # create the sample
    sample = scatter.Sample(sample_length, scattering_factors, mag_config)
    # characterise the two beams
    beam_cp = scatter.Beam(scatter.en2wave(energy), [fwhm, fwhm], pol_dict['CL'])
    beam_cl = scatter.Beam(scatter.en2wave(energy), [fwhm, fwhm], pol_dict['CR'])
    # describe the geometry
    geometry = scatter.Geometry(angle, detector_distance)
    s_cp = scatter.Scatter(beam_cp, sample, geometry)
    s_cl = scatter.Scatter(beam_cl, sample, geometry)

    # see the magnetic configuration of the sample
    plot.structure(sample)
    plt.show()

    # interactive plot for looking at features more closely
    returned_roi = plot.intensity_interactive(s_cp, log=True)
    s_cl.roi = returned_roi
    s_cp.roi = returned_roi
    plot.difference(s_cl, s_cp, log=True)
    plt.show()

    # add a reference hole and perform holography
    holography_reference(sample, reference_hole_size, 'xy')
    plot.structure(sample)

    # invert holography to recover structure
    plot.holography(s_cp, s_cl, log=True, recons_only=False)  # entire pattern
    plot.holography(s_cp, s_cl)  # only sample reconstruction
    plt.show()


if __name__ == '__main__':
    dichroism()
