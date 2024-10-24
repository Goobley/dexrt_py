from typing import Callable, Tuple
import matplotlib.axes
import matplotlib.collections
import matplotlib.colors
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import yaml
from scipy.ndimage import rotate
from dexrt.utils import centres_to_edges


def compute_min_max_param(
    ray_data: xr.Dataset, param: str = "I"
) -> Tuple[float, float]:
    """Compute the min and max values of a spectral parameter across the
    different ray viewing angles.

    Parameters
    ----------
    ray_data : xr.Dataset
        The dexrt_ray output data
    param : str, optional
        The parameter to compute the min and max of. Default: I

    Returns
    -------
    min, max: tuple of 2 floats
        The min and max values of the parameter
    """
    min_I = np.inf
    max_I = -np.inf
    for m in range(ray_data.mu.shape[0]):
        param_m = ray_data[f"{param}_{m}"]
        min_I = min(param_m.min(), min_I)
        max_I = max(param_m.max(), max_I)

    return min_I, max_I


def plot_ray_output(
    ax: matplotlib.axes.Axes,
    synth_data: xr.Dataset,
    atmos_data: xr.Dataset,
    ray_data: xr.Dataset,
    mu_idx: int,
    lambda0: float,
    delta_lambda: float,
    snap_lambda0: bool = True,
    label_axes: bool = True,
    label_delta_lambda: bool = True,
    add_colorbar: bool = True,
    param: str = "I",
    cmap: str | matplotlib.colors.Colormap = "magma",
    min_val=None,
    max_val=None,
    norm: matplotlib.colors.Normalize | None = None,
    rasterized: bool = True,
) -> matplotlib.collections.QuadMesh:
    """Plot the synthetic spectrum produced by dexrt_ray.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes to plot into.
    synth_data : xr.Dataset
        The dexrt output data
    atmos_data : xr.Dataset
        The dexrt input atmosphere
    ray_data : xr.Dataset
        The dexrt_ray output data
    mu_idx : int
        The index of the angle (in ray_data.mu) to plot
    lambda0 : float
        The central wavelength (nm)
    delta_lambda : float
        The half-wavelength of the window (i.e. -delta_lambda, delta_lambda)
    snap_lambda0 : bool, optional
        Whether to "snap" lambda0 to the closest atomic rest wavelength.
        Default: True.
    label_axes : bool, optional
        Whether to label the axes. Default: True
    label_delta_lambda : bool, optional
        Whether to label the wavelength axis in offset from lambda0 or in
        absolute wavelength [nm]. Default: True
    add_colorbar : bool, optional
        Whether to add a colorbar to this plot (by stealing size from this
        axis). Default: True
    param : str, optional
        The spectral parameter to plot. Can be either I or tau. Default: I
    cmap : str or colormap, optional
        The colormap to use when plotting the data. Default: magma
    min_val : float, optional
        The minimum value for the colorbar. Default: from the data.
    max_val : float, optional
        The maximum value for the colorbar. Default: from the data.
    norm : matplotlib Normalize, optional
        A normalization to apply, e.g. matplotlib.colors.LogNorm. Default: None,
        i.e. linear.
    rasterized : bool, optional
        Whether to use matplotlib's rasterization (can make PDF figures much
        lighter with detailed plots). Default: True

    Returns
    -------
    mappable : QuadMesh
        The matplotlib quadmesh plotted, which can be used in combination with
        fig.colorbar to add your own colorbar.
    """

    if snap_lambda0:
        lambda0s = synth_data.lambda0
        idx = np.abs(lambda0s - lambda0).argmin()
        lambda0 = lambda0s[idx]

    wavelength = ray_data.wavelength.values
    start_la = max(np.searchsorted(wavelength, lambda0 - delta_lambda) - 1, 0)
    end_la = min(
        np.searchsorted(wavelength, lambda0 + delta_lambda) + 1, wavelength.shape[0]
    )
    if end_la <= start_la:
        raise ValueError("Requested wavelengths made grid collapse.")

    voxel_scale = atmos_data.voxel_scale.values / 1e6

    if mu_idx >= ray_data.mu.shape[0]:
        raise ValueError(
            f"mu_idx ({mu_idx}) requested out of bounds (max: {ray_data.mu.shape[0]})"
        )

    spectral_data = ray_data[f"{param}_{mu_idx}"][start_la:end_la].values.T
    ray_starts = ray_data[f"ray_start_{mu_idx}"].values
    vox_slit_pos = np.sqrt(np.sum((ray_starts - ray_starts[0]) ** 2, axis=1)) - 0.5
    slit_pos = vox_slit_pos * voxel_scale

    wave_grid = np.copy(wavelength[start_la:end_la])
    if label_delta_lambda:
        wave_grid -= lambda0

    vmin, vmax = None, None
    if min_val is not None:
        vmin = min_val
    if max_val is not None:
        vmax = max_val
    mappable = ax.pcolormesh(
        centres_to_edges(wave_grid),
        centres_to_edges(slit_pos),
        spectral_data,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        rasterized=rasterized,
        norm=norm,
    )

    lower_bound = -delta_lambda
    upper_bound = delta_lambda
    if not label_delta_lambda:
        lower_bound += lambda0
        upper_bound += lambda0
    ax.set_xlim(lower_bound, upper_bound)
    ax.tick_params("both", direction="in")
    if label_axes:
        xlabel = r"$\Delta\lambda$ [nm]" if label_delta_lambda else r"$\lambda$ [nm]"
        ax.set_xlabel(xlabel)
        ax.tick_params("x", labelrotation=30.0)
        ax.set_ylabel("Slit Position [Mm]")

    if add_colorbar:
        ax.get_figure().colorbar(mappable, ax=ax)

    return mappable


def plot_rotated_field(
    ax: matplotlib.axes.Axes,
    synth_data: xr.Dataset,
    atmos_data: xr.Dataset,
    ray_data: xr.Dataset,
    mu_idx: int,
    param: str = "temperature",
    leading_index: int | None = None,
    label_axes: bool = True,
    add_colorbar: bool = True,
    cmap: matplotlib.colors.Colormap | str = "magma",
    norm: matplotlib.colors.Normalize | None = None,
    rasterized: bool = True,
):
    """Plot an atmospheric field rotated to match the viewing direction.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes to plot into.
    synth_data : xr.Dataset
        The dexrt output data
    atmos_data : xr.Dataset
        The dexrt input atmosphere
    ray_data : xr.Dataset
        The dexrt_ray output data
    mu_idx : int
        The index of the angle (in ray_data.mu) to plot
    param : str, optional
        The atmospheric parameter to plot. This is looked up first on
        synth_data, then atmos_data, in case it is modified during the non-LTE
        solve (e.g. ne). Default: temperature
    leading_index : int, optional
        Whether to index the first axis of the field to produce a 2d slice. Default
        None, i.e. assume field is 2d.
    label_axes : bool, optional
        Whether to label the axes. Default: True
    add_colorbar : bool, optional
        Whether to add a colorbar to this plot (by stealing size from this
        axis). Default: True
    cmap : str or colormap, optional
        The colormap to use when plotting the data. Default: magma
    norm : matplotlib Normalize, optional
        A normalization to apply, e.g. matplotlib.colors.LogNorm. Default: LogNorm.
    rasterized : bool, optional
        Whether to use matplotlib's rasterization (can make PDF figures much
        lighter with detailed plots). Default: True

    Returns
    -------
    mappable : QuadMesh
        The matplotlib quadmesh plotted, which can be used in combination with
        fig.colorbar to add your own colorbar.
    """
    if norm is None:
        norm = matplotlib.colors.LogNorm()

    voxel_scale = atmos_data.voxel_scale.values / 1e6
    angle = 1.5 * np.pi - np.sign(ray_data.mu[mu_idx, 0]) * np.arccos(
        ray_data.mu[mu_idx, 2]
    )
    if param in synth_data:
        field_data = synth_data[param]
        if field_data.dims[-1] == "ks":
            if leading_index is not None:
                field = synth_data.dexrt.rehydrated.plane(param, leading_index)
            else:
                field = synth_data.dexrt.rehydrated[param]
        else:
            field = synth_data[param]
    else:
        field = atmos_data[param]
    if leading_index is not None:
        field = field[leading_index]
    if not isinstance(field, np.ndarray):
        field = field.values

    rot_field = rotate(field, angle=np.rad2deg(angle), reshape=True, mode="nearest")

    mappable = ax.imshow(
        rot_field,
        norm=norm,
        origin="lower",
        extent=[
            0.0,
            rot_field.shape[1] * voxel_scale,
            0.0,
            rot_field.shape[0] * voxel_scale,
        ],
        aspect="auto",
        cmap=cmap,
        rasterized=rasterized,
    )
    ax.tick_params("both", direction="in")
    if label_axes:
        ax.set_xlabel("Inclined pos [Mm]")
        ax.set_ylabel("Slit pos [Mm]")

    if add_colorbar:
        ax.get_figure().colorbar(mappable, ax=ax)

    return mappable


def plot_slit_view_ray(
    ax: matplotlib.axes.Axes,
    synth_data: xr.Dataset,
    atmos_data: xr.Dataset,
    ray_data: xr.Dataset,
    mu_idx: int,
    param: str = "temperature",
    leading_index: int | None = None,
    label_axes: bool = True,
    image_plane_color="C0",
    view_ray_color="C1",
    head_width: float = 0.2,
    view_ray_frac: float = 0.4,
    cmap: matplotlib.colors.Colormap | str = "magma",
    norm: matplotlib.colors.Normalize | None = None,
    rasterized: bool = True,
) -> matplotlib.collections.QuadMesh:
    """Plot an atmospheric field with imaging slit and viewing rays
    superimposed.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes to plot into.
    synth_data : xr.Dataset
        The dexrt output data
    atmos_data : xr.Dataset
        The dexrt input atmosphere
    ray_data : xr.Dataset
        The dexrt_ray output data
    mu_idx : int
        The index of the angle (in ray_data.mu) to plot
    param : str, optional
        The atmospheric parameter to plot. This is looked up first on
        synth_data, then atmos_data, in case it is modified during the non-LTE
        solve (e.g. ne). Default: temperature
    leading_index : int, optional
        Whether to index the first axis of the field to produce a 2d slice.
        Default None, i.e. assume field is 2d.
    label_axes : bool, optional
        Whether to label the axes. Default: True
    image_plane_color : str or matplotlib compatible color, optional
        The color to draw the image plane arrow. Default: first entry in current
        colour cycle.
    view_ray_color : str or matplotlib compatible color, optional
        The color to draw the view ray arrow. Default: second entry in current
        colour cycle.
    head_width : float, optional
        The width of the arrowheads. Default: 0.2
    view_ray_frac : float, optional
        The length fraction of image_plane to make the view_ray arrow. Default:
        0.4
    cmap : str or colormap, optional
        The colormap to use when plotting the data. Default: magma
    norm : matplotlib Normalize, optional
        A normalization to apply, e.g. matplotlib.colors.LogNorm. Default:
        LogNorm.
    rasterized : bool, optional
        Whether to use matplotlib's rasterization (can make PDF figures much
        lighter with detailed plots). Default: True

    Returns
    -------
    mappable : QuadMesh
        The matplotlib quadmesh plotted, which can be used in combination with
        fig.colorbar to add your own colorbar.
    """
    if norm is None:
        norm = matplotlib.colors.LogNorm()

    min_x, max_x, min_z, max_z = np.inf, -np.inf, np.inf, -np.inf
    for m in range(ray_data.mu.shape[0]):
        starts = ray_data[f"ray_start_{m}"]
        min_x = min(starts[:, 0].min(), min_x)
        max_x = max(starts[:, 0].max(), max_x)
        min_z = min(starts[:, 1].min(), min_z)
        max_z = max(starts[:, 1].max(), max_z)

    if param in synth_data:
        field_data = synth_data[param]
        if field_data.dims[-1] == "ks":
            field = synth_data.dexrt.rehydrated[param]
        else:
            field = synth_data[param].values
    else:
        field = atmos_data[param].values
    if leading_index is not None:
        field = field[leading_index]

    voxel_scale = atmos_data.voxel_scale.values / 1e6
    offset_x = atmos_data.offset_x.values / 1e6
    offset_z = atmos_data.offset_z.values / 1e6

    mappable = ax.imshow(
        field,
        norm=norm,
        origin="lower",
        extent=[
            offset_x,
            field.shape[1] * voxel_scale + offset_x,
            offset_z,
            field.shape[0] * voxel_scale + offset_z,
        ],
        aspect=1,
        cmap=cmap,
        rasterized=rasterized,
    )
    ray_starts = ray_data[f"ray_start_{mu_idx}"].values
    ax.set_xlim(min_x * voxel_scale + offset_x, max_x * voxel_scale + offset_x)
    ax.set_ylim(min_z * voxel_scale + offset_z, max_z * voxel_scale + offset_z)
    arrow_x = ray_starts[0, 0] * voxel_scale + offset_x
    arrow_y = ray_starts[0, 1] * voxel_scale + offset_z
    arrow_dx = (ray_starts[-1, 0] - ray_starts[0, 0]) * voxel_scale
    arrow_dy = (ray_starts[-1, 1] - ray_starts[0, 1]) * voxel_scale
    ax.arrow(
        arrow_x,
        arrow_y,
        arrow_dx,
        arrow_dy,
        length_includes_head=True,
        head_width=head_width,
        color=image_plane_color,
    )
    view_arrow_length = view_ray_frac * np.sqrt(arrow_dx**2 + arrow_dy**2)
    ax.arrow(
        arrow_x + 0.5 * arrow_dx,
        arrow_y + 0.5 * arrow_dy,
        -view_arrow_length * ray_data.mu[mu_idx, 0],
        -view_arrow_length * ray_data.mu[mu_idx, 2],
        length_includes_head=True,
        head_width=head_width,
        color=view_ray_color,
    )

    if label_axes:
        ax.set_xlabel("x [Mm]")
        ax.set_ylabel("z [Mm]")

    return mappable
