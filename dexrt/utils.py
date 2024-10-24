import numpy as np


def centres_to_edges(x: np.ndarray) -> np.ndarray:
    """Map array centres to edges

    Parameters
    ----------
    x : array
        The array to convert
    Returns
    -------
    edges : array
        The edges of the array, has length x.shape[0] + 1
    """

    centres = 0.5 * (x[1:] + x[:-1])
    return np.concatenate(
        [[x[0] - (centres[0] - x[0])], centres, [x[-1] + (x[-1] - centres[-1])]]
    )
