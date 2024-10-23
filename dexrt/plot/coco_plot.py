import numpy as np


# NOTE(cmo): This tonemapping function takes colour channel as the first axis,
# so the output needs swapping before display in matplotlib
def tonemap(c, mode="aces", gamma=2.2, bias=None):
    """Tonemap (nonlinearly reduce the dynamic range to 0-1) of a tristimulus (rgb) array.
    Parameters
    ----------
    c : ndarray [c, h, w]
        The array to tonemap. The first axis is expected to be 3 for the colour channels.
    mode : str
        The tonemapping function to apply. Options are reinhard, uncharted2,
        aces or linear. linear represents a simple Gamma correction. Default: aces
    gamma : float
        The gamma index to use (ignored by aces).
    bias : float, optional
        A constant factor to scale the output by before linear to srgb
        conversion. Used by uncharted2 and aces.
    """
    # http://filmicworlds.com/blog/filmic-tonemapping-operators/
    if mode == "reinhard":
        c = c / (1.0 + c)
        return c ** (1.0 / gamma)
    elif mode == "uncharted2":
        A = 0.15
        B = 0.50
        C = 0.10
        D = 0.20
        E = 0.02
        F = 0.30
        W = 11.2
        if bias is None:
            bias = 2.0
        mapper = (
            lambda c: ((c * (A * c + C * B) + D * E) / (c * (A * c + B) + D * F))
            - E / F
        )
        curr = mapper(c * bias)
        whiteScale = 1.0 / mapper(W)
        return (curr * whiteScale) ** (1.0 / gamma)
    elif mode == "aces":
        # https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
        # https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
        # https://therealmjp.github.io/posts/sg-series-part-6-step-into-the-baking-lab/
        acesIn = np.ascontiguousarray(
            np.array(
                [
                    [0.59719, 0.35458, 0.04823],
                    [0.07600, 0.90834, 0.01566],
                    [0.02840, 0.13383, 0.83777],
                ]
            ).T
        )
        acesOut = np.ascontiguousarray(
            np.array(
                [
                    [1.60475, -0.53108, -0.07367],
                    [-0.10208, 1.10813, -0.00605],
                    [-0.00327, -0.07276, 1.07602],
                ]
            ).T
        )
        if bias is None:
            bias = 0.8

        def RRTAndODTFit(v):
            # RRT: Reference rendering transform
            # ODT: Output display transform
            a = v * (v + 0.0245786) - 0.000090537
            b = v * (0.983729 * v + 0.4329510) + 0.238081
            return a / b

        def LinearTosRGB(color):
            x = color * 12.92
            y = 1.055 * np.clip(color, 0.0, 1.0) ** (1.0 / 2.4) - 0.055

            clr = color
            clr[0] = np.where(color[0] < 0.0031308, x[0], y[0])
            clr[1] = np.where(color[1] < 0.0031308, x[1], y[1])
            clr[2] = np.where(color[2] < 0.0031308, x[2], y[2])
            return clr

        color = np.tensordot(acesIn, c, axes=(0, 0))
        color = RRTAndODTFit(color)
        color = np.tensordot(acesOut, color, axes=(0, 0))
        color = np.clip(color, 0.0, 1.0)
        color = LinearTosRGB(color * bias)
        color = np.clip(color, 0.0, 1.0)
        return color
    elif mode == "linear":
        c = np.clip(c, 0.0, 1.0)
        return c ** (1.0 / gamma)
    else:
        raise NotImplementedError()


def coco_plot(
    ax,
    arr,
    filt,
    thresh=None,
    log=False,
    max_pre_tonemap=4.0,
    edges=None,
    normalise_channels_individually=True,
    **kwargs,
):
    """Plot a tonemapped COCOplot of arr into ax.
    Parameters
    ----------
    ax : matplotlib.Axes
        matplotlib axes to plot into
    arr : ndarray [c, w, h]
        The array to colour-collapse.
    filter : list (length 3) of array
        The 3 filters to apply to compute the 3 channels from the full first dimension of arr.
    thresh : float, optional
        A minimum threshold to clamp the data to, default: None
    log : bool
        Whether to log positive data (before tonemapping), default: False.
    max_pre_tonemap : float, optional
        Rescale the input to the tonemapper to have this as maximal value. Default: 4.0
    edges : iterable of array, optional
        Edges to apply when spatially plotting the data (length+1 of each data
        axis). If None, imshow is used, other pcolormesh is employed.
    normalise_channels_individually : bool, optional
        Whether to normalise each of the 3 rgb channels separately, or to the
        same maximum.
    kwargs
        Extra arguments passed to tonemap function.

    Returns
    -------
    The tonemapped COCO array of the data plotted into ax.
    """
    cococfn = np.tensordot(filt, arr, axes=(0, 0))
    if thresh is not None:
        cococfn[cococfn < thresh] = thresh

    if log:
        cococfn[cococfn > 0.0] = np.log10(cococfn[cococfn > 0.0])

    for chan in range(cococfn.shape[0]):
        if normalise_channels_individually:
            max_val = np.nanmax(cococfn[chan])
            min_val = np.nanmin(cococfn[chan])
        else:
            max_val = np.nanmax(cococfn)
            min_val = np.nanmin(cococfn)

        cococfn[chan] = (
            (cococfn[chan] - min_val) / (max_val - min_val) * max_pre_tonemap
        )

    if not "bias" in kwargs:
        kwargs["bias"] = 1.0
    cococfn_tm = tonemap(cococfn, **kwargs)
    if edges is None:
        ax.imshow(np.moveaxis(cococfn_tm, 0, 2), rasterized=True, aspect="auto")
    else:
        ax.pcolormesh(*edges, np.moveaxis(cococfn_tm, 0, 2), rasterized=True)

    return np.moveaxis(cococfn_tm, 0, 2)
