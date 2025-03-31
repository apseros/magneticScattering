from typing import Tuple
from magneticScattering.holography import invert_holography
from magneticScattering.scatter import *
from matplotlib import colors
from matplotlib.widgets import RectangleSelector
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

units = {'k': 1e3, '': 1, 'm': 1e-3, 'µ': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15}


def _lin_thresh_pow(data):
    """Calculates a linear threshold value for logarithmic plotting."""
    return 10 ** np.ceil(np.log10(np.cbrt(np.abs(data).max())))


def _choose_scale(roi: Tuple[float, float, float, float]) -> Tuple[str, Tuple[float, float, float, float]]:
    """Choose appropriate units for plotting based on the range of values."""

    # Compute the log10 scale difference in x and y directions
    dx, dy = np.log10([roi[1] - roi[0], roi[3] - roi[2]])
    log_scales = {np.log10(value): prefix for prefix, value in units.items()}

    # Find the closest order of magnitude (rounded to a multiple of 3)
    x_closest, y_closest = 3 * round(dx / 3), 3 * round(dy / 3)

    overall_closest = min(x_closest, y_closest)
    prefix = log_scales.get(overall_closest, '')

    # Scale the ROI using the chosen unit
    scaled_roi = np.array(roi) / units[prefix]

    return prefix, cast(tuple[float, float, float, float], scaled_roi)


def structure(struct: Sample, **kwargs) -> None:
    """Plot the components of magnetisation structure.

    :param struct:  Sample class.
    :param kwargs:  Keyword arguments for matplotlib imshow.
    :return:        None.
    """
    extent = struct.get_extent()
    prefix, scaled_roi = _choose_scale(extent)

    fig, ax = plt.subplots(1, 4)
    fig.suptitle("Structure")
    title_index = ["Charge", "$m_x$", "$m_y$", "$m_z$"]
    fig.supxlabel("$x$ / " + prefix + "m")
    fig.supylabel("$y$ / " + prefix + "m")

    for i, (ax_i, colormap) in enumerate(zip(ax.flatten(), ['gray', 'PiYG', 'PuOr', 'RdBu'])):
        cbar_val = np.max(np.abs([struct.structure[i].min(), struct.structure[i].max()])) if i in [1, 2, 3] else None
        vmin = -cbar_val if cbar_val is not None else 0
        color = ax_i.imshow(struct.structure[i].T, origin='lower', extent=scaled_roi, cmap=colormap,
                            vmin=vmin, vmax=cbar_val, **kwargs)
        plt.colorbar(color, ax=ax_i, location='bottom')
        ax_i.set_aspect('equal')
        ax_i.set_title(title_index[i])


def pol(scatter: Scatter, log=True, **kwargs) -> None:
    """Plot the polarization states of the scattered light.

    :param scatter:     Scatter class.
    :param log:         Logarithmic scale
    :return:            None.
    """
    roi = scatter.roi
    prefix, scaled_roi = _choose_scale(roi)
    vmin, vmax = scatter.pol_out[-3:].min(), scatter.pol_out[-3:].max()
    norm = colors.SymLogNorm(vmin, vmin=vmin, vmax=vmax) if log else None

    fig, ax = plt.subplots(1, 3)
    fig.suptitle("Relative Polarization States")
    title_index = ["Horizontal", "Diagonal", "Circular"]
    fig.supxlabel("Detector Position $x$ / " + prefix + "m")
    fig.supylabel("Detector Position $y$ / " + prefix + "m")

    im = None
    for i, ax_i in enumerate(ax.flatten()):
        im = ax_i.imshow(scatter.pol_out[i + 1].T, origin='lower', extent=scaled_roi, norm=norm, **kwargs)
        ax_i.set_aspect('equal')
        ax_i.set_title(title_index[i])
    cbar_ax = fig.add_axes((0.15, 0.15, 0.7, 0.05))
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')


def difference(scatter_a: Scatter, scatter_b: Scatter, log: bool = False, cmap: str = 'Spectral', **kwargs) -> None:
    """Plot the difference between two scattering patterns.

    :param scatter_a:   Scatter class.
    :param scatter_b:   Scatter class.
    :param log:         Boolean choice to plot in log scale.
    :param cmap:        Color map to use.
    :return:            None.
    """
    if np.any(scatter_a.roi != scatter_b.roi):
        raise ValueError("Diffraction geometries have different parameters.")
    extent = scatter_a.roi
    diff = scatter_a.intensity - scatter_b.intensity
    if np.all(diff == 0):
        raise ValueError("No dichroism.")

    prefix, scaled_roi = _choose_scale(extent)
    norm = colors.SymLogNorm(_lin_thresh_pow(diff)) if log else None

    fig, ax = plt.subplots(1, 1)
    colorscale = ax.imshow(diff.T, origin='lower', extent=scaled_roi, norm=norm, cmap=cmap, **kwargs)
    fig.colorbar(colorscale)
    ax.set_title("Intensity Difference")
    ax.set_xlabel("Detector Position $x_0$ / " + prefix + "m")
    ax.set_ylabel("Detector Position $y_0$ / " + prefix + "m")


def intensity_interactive(scatter: Scatter, log: bool = False, **kwargs) -> tuple[float, float, float, float]:
    """Interactive plot of the intensity difference that can be used for selecting regions to view in higher resolution.
    :param scatter: Scatter class.
    :param log:     Boolean choice to plot in log scale.
    :param kwargs:  Keyword arguments passed to matplotlib imshow
    :return:        Coordinates of the selected region edges.
    """
    selected_regions = []

    def select_callback(click, release):
        x_start, y_start = click.xdata, click.ydata
        x_end, y_end = release.xdata, release.ydata
        selected_regions.append([x_start, x_end, y_start, y_end])

    roi = scatter.roi
    selected_regions.append(roi)
    norm = colors.SymLogNorm(1) if log else None
    prefix, scaled_roi = _choose_scale(roi)
    logging.info(scaled_roi)
    intensity_array = copy.deepcopy(scatter.intensity)

    fig, ax = plt.subplots(1, 1)
    selector = RectangleSelector(
        ax, select_callback,
        useblit=True,
        minspanx=0, minspany=0,
        spancoords='data',
        interactive=True,
        props=dict(facecolor='None', edgecolor='red'))
    fig.suptitle("Select region to view in higher resolution.")
    _intensity_image(ax, fig, intensity_array, norm, scaled_roi, prefix, prefix, **kwargs)
    fig.canvas.mpl_connect('key_press_event', selector)
    plt.show()

    x1, x2, y1, y2 = selected_regions[-1]

    return x1 * units[prefix], x2 * units[prefix], y1 * units[prefix], y2 * units[prefix]


def _intensity_image(ax: Axes, fig, intensity_array, norm, scaled_roi, x_prefix, y_prefix, **kwargs):
    """Adds appropriate axes labels to the intensity image."""
    colorscale = ax.imshow(intensity_array.T, origin='lower', extent=scaled_roi, norm=norm, **kwargs)
    fig.colorbar(colorscale)
    ax.set_title("Intensity")
    ax.set_xlabel("Detector Position $x_0$ / " + x_prefix + "m")
    ax.set_ylabel("Detector Position $y_0$ / " + y_prefix + "m")
    ax.axis('scaled')


def intensity(scatter: Scatter, log: bool = False, **kwargs) -> None:
    """Plot the intensity of the scattered light.

    :param scatter:     Scatter class.
    :param log:         Boolean choice to plot in log scale.
    :param kwargs:      Keyword arguments passed to matplotlib imshow function.
    :return:            None.
    """
    roi = scatter.roi
    norm = colors.SymLogNorm(1) if log else None
    prefix, scaled_roi = _choose_scale(roi)
    intensity_array = copy.deepcopy(scatter.intensity)
    fig, ax = plt.subplots(1, 1)
    _intensity_image(ax, fig, intensity_array, norm, scaled_roi, prefix, prefix, **kwargs)


def holography(scatter_a: Scatter, scatter_b=None, log: bool = False, recons_only=True,
               cmap='Spectral', **kwargs) -> None:
    """Plot the result of inverting the holography pattern.

    :param scatter_a:   First scattering pattern.
    :param scatter_b:   Second scattering pattern (optional).
    :param log:         Boolean choice to plot in log scale.
    :param recons_only: Focus on the reconstruction on the sample and not the entire inverse scattering.
    :param cmap:        Colormap for the image.
    :param kwargs:      Keyword arguments passed to matplotlib imshow function.
    :return:            None.
    """
    inverse, roi = invert_holography(scatter_a, scatter_b)
    if recons_only:
        sx, sy = inverse.shape
        if sx > 3 * sy:
            inverse = inverse[0:sx // 4, :]  # reference hole along x
            roi[0], roi[1] = roi[0] / 4, roi[1] / 4

        elif sy > 3 * sx:
            inverse = inverse[:, 0:sy // 4]  # reference hole along y
            roi[2], roi[3] = roi[2] / 4, roi[3] / 4

        else:
            inverse = inverse[0:sx // 4, sy // 2 - sy // 8: sy // 2 + sy // 8]  # reference hole along x and y
            roi = [i / 4 for i in roi]
    prefix, scaled_roi = _choose_scale(roi)
    norm = colors.SymLogNorm(_lin_thresh_pow(inverse)) if log else None

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Reconstructed FTH")
    ax.set_xlabel("Relative Real Size $x_0$ / " + prefix + "m")
    ax.set_ylabel("Relative Real Size $y_0$ / " + prefix + "m")
    colorscale = ax.imshow(inverse.T, origin='lower', extent=scaled_roi, norm=norm, cmap=cmap, **kwargs)
    fig.colorbar(colorscale)

def scale_data(data: np.ndarray, scale: list[float]) -> np.ndarray:
    """Scale the values in the array so that the minimum is scale[0] and maximum is scale[1].

    :param data:    Array to be scaled.
    :param scale:   Scaling range as a list: [lower, upper].
    :return:        Scaled array.
    """
    data = data - np.min(data)  # 0 to max+min
    data = data / np.max(data)  # 0 to 1
    data = data * (scale[1] - scale[0]) + scale[0]  # scale[0] to scale[1]
    return data
