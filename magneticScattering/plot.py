from scatter import *
from matplotlib import colors
from matplotlib.widgets import RectangleSelector
from matplotlib import pyplot as plt
from typing import cast

units = {'k': 1e3, '': 1, 'm': 1e-3, 'µ': 1e-6, 'n': 1e-9}


def lin_thresh_pow(data):
    return 10 ** np.ceil(np.log10(np.cbrt(np.abs(data).max())))


def choose_scale(roi):
    dx, dy = np.log10([roi[1] - roi[0], roi[3] - roi[2]])
    scales = {k: np.log10(v) for k, v in units.items()}

    x_prefix, y_prefix = min(scales, key=lambda k: abs(scales[k] - dx)), min(scales, key=lambda k: abs(scales[k] - dy))
    scaled_roi = [roi[i] / units[p] for i, p in zip(range(4), [x_prefix, x_prefix, y_prefix, y_prefix])]
    return x_prefix, y_prefix, tuple(scaled_roi)


def structure(struct: Sample, **kwargs) -> None:
    """Plot the components of magnetisation structure.

    :param struct:   Sample class or numpy array of shape (4, nx, ny).
    :return:         None.
    """
    extent = struct.get_extent()
    x_prefix, y_prefix, scaled_roi = choose_scale(extent)

    fig, ax = plt.subplots(1, 4)
    fig.suptitle("Structure")
    title_index = ["Charge", "$m_x$", "$m_y$", "$m_z$"]
    fig.supxlabel("$x$ / " + x_prefix + "m")
    fig.supylabel("$y$ / " + y_prefix + "m")

    for i, (ax_i, colormap) in enumerate(zip(ax.flatten(), [None, 'Reds', 'Greens', 'Blues'])):
        ax_i.imshow(struct.structure[i].T, origin='lower', extent=scaled_roi, cmap=colormap, **kwargs)
        ax_i.set_aspect('equal')
        ax_i.set_title(title_index[i])


def pol(scatter: Scatter, log=True, **kwargs) -> None:
    """Plot the polarization states of the scattered light.

    :param scatter:     Scatter class.
    :param log:         Logarithmic scale
    :return:            None.
    """
    roi = scatter.roi
    x_prefix, y_prefix, scaled_roi = choose_scale(roi)
    vmin, vmax = scatter.pol_out[-3:].min(), scatter.pol_out[-3:].max()
    norm = colors.SymLogNorm(vmax, vmin=vmin, vmax=vmax) if log else None

    fig, ax = plt.subplots(1, 3)
    fig.suptitle("Relative Polarization States")
    title_index = ["Horizontal", "Diagonal", "Circular"]
    fig.supxlabel("Detector Position $x$ / " + x_prefix + "m")
    fig.supylabel("Detector Position $y$ / " + y_prefix + "m")

    im = None
    for i, ax_i in enumerate(ax.flatten()):
        im = ax_i.imshow(scatter.pol_out[i + 1].T, origin='lower', extent=scaled_roi, norm=norm, **kwargs)
        ax_i.set_aspect('equal')
        ax_i.set_title(title_index[i])
    cbar_ax = fig.add_axes([0.15, 0.15, 0.7, 0.05])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')


def pol_color(scatter: Scatter) -> None:
    # find location of max intensity
    intensity_img = np.copy(scatter.intensity)
    roi = scatter.roi
    modulate = scale_data(np.log10(intensity_img), [0, 1])

    final_image = modulate * (scatter.pol_out[-3:] / intensity_img) ** 2
    final_image = np.swapaxes(final_image, 0, 2)  # this also transposes the image

    x_prefix, y_prefix, scaled_roi = choose_scale(roi)
    fig, ax = plt.subplots(1, 1)
    fig.suptitle("Relative Polarization States")
    scaled_roi = cast(tuple[float, float, float, float], scaled_roi)  # to remove typing warning
    plt.imshow(final_image, origin='lower', extent=scaled_roi)


def difference(scatter_a: Scatter, scatter_b: Scatter, log: bool = False, **kwargs) -> None:
    """Plot the difference between two scattering patterns.

    :param scatter_a:   Scatter class.
    :param scatter_b:   Scatter class.
    :param log:         Boolean choice to plot in log scale.
    :return:            None.
    """
    if np.any(scatter_a.roi != scatter_b.roi):
        raise ValueError("Diffraction geometries have different parameters.")
    extent = scatter_a.roi
    diff = scatter_a.intensity - scatter_b.intensity
    if np.all(diff == 0):
        raise ValueError("No dichroism.")

    x_prefix, y_prefix, scaled_roi = choose_scale(extent)
    norm = colors.SymLogNorm(lin_thresh_pow(diff)) if log else None

    fig, ax = plt.subplots(1, 1)
    colorscale = ax.imshow(diff.T, origin='lower', extent=scaled_roi, norm=norm, **kwargs)
    fig.colorbar(colorscale)
    ax.set_title("Intensity Difference")
    ax.set_xlabel("Detector Position $x_0$ / " + x_prefix + "m")
    ax.set_ylabel("Detector Position $y_0$ / " + y_prefix + "m")


def intensity_interactive(scatter: Scatter, log: bool = False, **kwargs):
    selected_regions = []

    def select_callback(click, release):
        x_start, y_start = click.xdata, click.ydata
        x_end, y_end = release.xdata, release.ydata
        selected_regions.append([x_start, x_end, y_start, y_end])

    roi = scatter.roi
    selected_regions.append(roi)
    norm = colors.SymLogNorm(1) if log else None
    x_prefix, y_prefix, scaled_roi = choose_scale(roi)
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

    intensity_image(ax, fig, intensity_array, kwargs, norm, scaled_roi, x_prefix, y_prefix)
    fig.canvas.mpl_connect('key_press_event', selector)
    plt.show()

    x1, x2, y1, y2 = selected_regions[-1]

    return [x1 * units[x_prefix], x2 * units[x_prefix], y1 * units[y_prefix], y2 * units[y_prefix]]


def intensity_image(ax, fig, intensity_array, kwargs, norm, scaled_roi, x_prefix, y_prefix):
    colorscale = ax.imshow(intensity_array.T, origin='lower', extent=scaled_roi, norm=norm, **kwargs)
    fig.colorbar(colorscale)
    ax.set_title("Intensity")
    ax.set_xlabel("Detector Position $x_0$ / " + x_prefix + "m")
    ax.set_ylabel("Detector Position $y_0$ / " + y_prefix + "m")
    ax.axis('scaled')


def intensity(scatter: Scatter, log: bool = False, **kwargs):
    """Plot the intensity of the scattered light.

    :param scatter:     Scatter class.
    :param log:         Boolean choice to plot in log scale.
    :return:            None.
    """
    roi = scatter.roi
    norm = colors.SymLogNorm(1) if log else None
    x_prefix, y_prefix, scaled_roi = choose_scale(roi)
    intensity_array = copy.deepcopy(scatter.intensity)
    fig, ax = plt.subplots(1, 1)
    intensity_image(ax, fig, intensity_array, kwargs, norm, scaled_roi, x_prefix, y_prefix)


def scale_data(data, scale):
    """Scale the values in the arrary so that the minimum is scale[0] and maximum is scale[1]."""
    data = data - np.min(data)  # 0 to max+min
    data = data / np.max(data)  # 0 to 1
    data = data * (scale[1] - scale[0]) + scale[0]  # scale[0] to scale[1]
    return data
