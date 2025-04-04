magneticScattering
==================

This package can be used to simulate magnetic scattering and Fourier transform holography from a structure.

Installation
------------
magneticScattering can be installed through pip:

.. code-block:: console

   (.venv)$ pip install magneticScattering

Documentation
-------------
Comprehensive documentation with examples is available online at
`readthedocs <https://magneticScattering.readthedocs.io/en/latest/index.html>`_.


Quickstart
----------
To calculate the observed magnetic scattering from a system, three classes describing the experimental setup must
first be specified, namely the:

1. Sample
2. Beam
3. Geometry

These classes are then passed to the Scatter class in order to perform the calculation. In the
following sections, these parameters are described in more detail and examples are provided.

Sample
^^^^^^

The Sample is initialized with the following parameters:

.. code-block:: python

    Sample(sample_length, scattering_factors, magnetic_configuration)

- ``sample_length``: size of the sample in meters as a scalar or a two-component vector in meters

- ``scattering_factors``: scattering factors list ``[f0, f1, f2]``. These are the complex pre-factors corresponding to the
  charge, magnetic circular and magnetic linear scattering. The real part is the refractive index (or phase) and the
  imaginary part is the dissipation (or absorption)

- ``structure``: the magnetic configuration of the sample as a numpy array with shape ``(3, nx, ny)`` (charge component
  will be inferred) or ``(4, nx, ny)`` with the charge component at index ``0``

Beam
^^^^

The Beam is initialized with the following parameters:

.. code-block:: python

    Beam(wavelength, beam_fwhm, polarization)

- ``wavelength``: wavelength of incident radiation in meters.

- ``beam_fwhm``: full width at half maximum of the beam as a scalar or a two-component vector in meters.

- ``polarization``: four-component polarization in the form of a Stokes vector.


Geometry
^^^^^^^^

The Geometry is initialized with the following parameters:

.. code-block:: python

    Geometry(angle, detector_distance)

- ``angle``: The angle of incidence between the beam and the sample in degrees.

- ``detector_distance``: The distance between the sample and the detector in meters.

The geometry use here is such that the beam travels in the positive *z*-directions when ``angle_d = 0`` and along the
negative *y*-direction when ``angle_d = 90``.

The sample plane is the *x-y* plane, such that the :math:`m_x` and :math:`m_y` components are in-plane, and :math:`m_z`
is out of plane.

Scatter
^^^^^^^

To compute the scattering pattern, call the Scatter class with the three classes from before as arguments:

.. code-block:: python

    Scatter(Sample, Beam, Geometry)

The intensity of the scattering can be obtained from Scatter.intensity or plotted directly using functions in the
`plot` submodule. For example, for a labyrinthine structure

.. image:: doc/source/_static/images/structure.png
    :width: 50%
    :alt: Magnetization components of a labyrinthine magnetic domain structure

the full-view scattering pattern (left) can be used to isolate a region of interest (left, red rectangle) to calculate
a higher resolution scattering. The difference between the scattering obtained from the two circular polarizations is
shown on the right

|img1| |img2|

.. |img1| image:: doc/source/_static/images/full_scattering.png
    :width: 25%
    :alt: Full view of the scattering pattern, with a red rectangle representing the desired region of interest



.. |img2| image:: doc/source/_static/images/scattering_roi.png
    :width: 25%
    :alt: Higher resolution scattering pattern of the region of interest

References
----------

van der Laan, G., "Theory from Soft X-ray resonant magnetic scattering of magnetic nano structures,"
https://doi.org/10.1016/j.crhy.2007.06.004


