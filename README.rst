Magnetic Diffraction
====================

This package can be used to simulate scattering from a magnetic structure. To to this, three classes describing the
experimental setup must first be specified:

1. Sample
2. Beam
3. Geometry


These classes are then passed to the Scatter class in order to calculate the resulting scattering pattern. In the
following sections, these parameters are described in more detail and examples are provided.

Sample
------

The Sample is initialized with the following parameters:

.. code-block:: python

    Sample(sample_length, scattering_factors, magnetic_configuration)

-``sample_length``: size of the sample in meters as a scalar or a two-component vector in meters

-``scattering_factors``: scattering factors list ``[f0, f1, f2]``. These are the complex pre-factors corresponding to the
charge, magnetic circular and magnetic linear scattering. The real part is the refractive index (or phase) and the
imaginary part is the dissipation (absorption).

-``structure``: the magnetic configuration of the sample as a numpy array with shape ``(3, nx, ny)`` (charge component
will be inferred) or ``(4, nx, ny)`` with the charge component at index ``0`` (see :ref:`struct_label`.)

Beam
----

The Beam is initialized with the following parameters:

.. code-block:: python

    Beam(wavelength, beam_fwhm, polarization)

- ``wavelength``: wavelength of incident radiation in meters.

- ``beam_fwhm``: full width at half maximum of the beam as a scalar or a two-component vector in meters.

- ``polarization``: four-component polarization in the form of a Stokes vector (see :ref:`stokes_label`.).


Geometry
--------

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
-------

To compute the scattering pattern, call the Scatter class with the three classes from before:

.. code-block:: python

    Scatter(Sample, Beam, Geometry)

The intensity of the scattering can be obtained from Scatter.intensity or plotted directly using functions in the
`plot` submodule. Some examples are:

.. code-block:: python

    plot.structure(Sample, quiver=True)             # plot the components of the magnetic structure
    plot.intensity(Scatter, log=True)               # plot the intensity of the scattering
    plot.difference(Scatter_a, Scatter_b, log=True) # plot the difference between two scattering patterns

.. _stokes_label:

Stokes Parameters
-----------------

The Stokes parameters are four components that define the polarization state of light.
For convenience, they are combined to form a vector :math:`(S_0,S_1,S_2,S_3)` defined as follows:

-:math:`S_0`: Intensity of the light, conventionally normalized to unity.

-:math:`S_1`: Component of light that is linearly polarized. :math:`+1` corresponds to purely
linear horizontal polarization and :math:`-1` to purely linear vertical.

-:math:`S_2`: Component of light that is linearly polarized along the diagonals. :math:`+1` corresponds to
purely :math:`+45^\circ` polarization, :math:`-1` to purely :math:`-45^\circ` polarization.

-:math:`S_3`: Component of light that is circularly polarized. :math:`+1` corresponds to purely right-handed circular
polarization, :math:`-1` to purely left-handed circular polarization.

.. _struct_label:

Structure
---------

The magnetic vector field has three components, :math:`(m_x,m_y,m_z)` and extends is two-dimensional in space.
Therefore, it is represented here as a numpy array of shape ``(3, nx, ny)``. The charge component can also be specified
using a ``(4, nx, ny)`` shaped numpy array, where the index 0 corresponds to the electron density of the sample.

To create such an array this array given its individual, 2D scalar components, :math:`m_x,\, m_y,\, m_z`, one can use:

.. code-block:: python

    structure = np.array([mx, my, mz])

:math:`m_x,\, m_y,\, m_z` must all be the same size and should be two-dimensional, e.g. ``(nx, ny)`` and should have
the same physical properties (e.g., lateral dimension)

Structures can be made or imported using the structures header. Some examples are:

when the beam is perpendicular to the sample (``angle = 0``) the magnetization components :math:`m_x,\,m_y` are in the
plane of the sample, while the magnetization component :math:`m_z` is out of the plane (and parallel to the beam).

References
----------

van der Laan, G., "Theory from Soft X-ray resonant magnetic scattering of magnetic nano structures,"
https://doi.org/10.1016/j.crhy.2007.06.004


Documentation
-------------
Comprehensive documentation is available online at [readthedocs](https://magneticScattering.readthedocs.io/en/latest/index.html).
