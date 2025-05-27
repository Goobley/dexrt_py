import numpy as np
import xarray as xr
from dexrt.sparse_data.morton import decode_morton_2, decode_morton_3
from typing import Tuple


def get_tile(field: np.ndarray, block_size: int, x: int, z: int) -> np.ndarray:
    """Extract the tile from field with side length block_size and (tile)
    coordinate given by x and z. This will be a copy of the field data.
    """
    if field.ndim == 2:
        return field[
            z * block_size : (z + 1) * block_size, x * block_size : (x + 1) * block_size
        ]
    else:
        return field[
            :,
            z * block_size : (z + 1) * block_size,
            x * block_size : (x + 1) * block_size,
        ]

def get_tile_3d(field: np.ndarray, block_size: int, x: int, y, z: int) -> np.ndarray:
    """Extract the tile from field with side length block_size and (tile)
    coordinate given by x, y, and z. This will be a copy of the field data.
    """
    if field.ndim == 3:
        return field[
            z * block_size : (z + 1) * block_size,
            y * block_size : (y + 1) * block_size,
            x * block_size : (x + 1) * block_size
        ]
    else:
        return field[
            :,
            z * block_size : (z + 1) * block_size,
            y * block_size : (y + 1) * block_size,
            x * block_size : (x + 1) * block_size,
        ]


def set_tile(field: np.ndarray, block_size: int, x: int, z: int, data: np.ndarray):
    """Set the tile in field with side length block_size and tile coordinate
    given by x and z to the data contained in data (which will be reshaped)."""
    if field.ndim == 2:
        field[
            z * block_size : (z + 1) * block_size, x * block_size : (x + 1) * block_size
        ] = data.reshape(block_size, block_size)
    else:
        field[
            :,
            z * block_size : (z + 1) * block_size,
            x * block_size : (x + 1) * block_size,
        ] = data.reshape(-1, block_size, block_size)

def set_tile_3d(field: np.ndarray, block_size: int, x: int, y: int, z: int, data: np.ndarray):
    """Set the tile in field with side length block_size and tile coordinate
    given by x, y, and z to the data contained in data (which will be reshaped)."""
    if field.ndim == 3:
        field[
            z * block_size : (z + 1) * block_size,
            y * block_size : (y + 1) * block_size,
            x * block_size : (x + 1) * block_size
        ] = data.reshape(block_size, block_size, block_size)
    else:
        field[
            :,
            z * block_size : (z + 1) * block_size,
            y * block_size : (y + 1) * block_size,
            x * block_size : (x + 1) * block_size,
        ] = data.reshape(-1, block_size, block_size, block_size)


def rehydrate_quantity(ds: xr.Dataset, qty: str | xr.DataArray | np.ndarray) -> np.ndarray:
    """Rehydrates a sparse quantity (one spatial dimension) from a dataset into
    a dense quantity (two spatial dimensions). Raises errors if a non-sparse
    quantity is requested.

    Parameters
    ----------
    ds : xarray Dataset
        The dataset to load from.
    qty : str | array
        The name of the quantity (string) in `ds` or an array, where the
        attributes of `ds` will be used to rehydrate.

    Returns
    -------
    rehydrated : array
        The rehydrated array
    """
    if "program" not in ds.attrs or ds.program != "dexrt (2d)":
        raise ValueError("Provided dataset not written by dexrt 2d")
    if ds.output_format != "sparse":
        raise ValueError("Data is not sparse and does not need to be rehydrated")

    block_size = ds.block_size
    block_entries = block_size * block_size
    if isinstance(qty, str):
        qty = ds[qty].values
    if isinstance(qty, xr.DataArray):
        qty = qty.values

    leading_dim = 1
    if qty.ndim > 1:
        leading_dim = qty.shape[0]
    else:
        qty = qty.reshape(1, -1)

    result_shape = (
        leading_dim,
        ds.num_z_blocks * block_size,
        ds.num_x_blocks * block_size,
    )
    result = np.empty(result_shape, dtype=qty.dtype)

    morton_tiles = ds.morton_tiles.values
    for flat_tile_idx, morton_code in enumerate(morton_tiles):
        x, z = decode_morton_2(morton_code)
        flat_tile = qty[
            :, flat_tile_idx * block_entries : (flat_tile_idx + 1) * block_entries
        ]
        set_tile(result, block_size, x, z, flat_tile)
    return result.squeeze()

def rehydrate_quantity_3d(ds: xr.Dataset, qty: str | xr.DataArray | np.ndarray) -> np.ndarray:
    """Rehydrates a sparse quantity (one spatial dimension) from a dataset into
    a dense quantity (three spatial dimensions). Raises errors if a non-sparse
    quantity is requested.

    Parameters
    ----------
    ds : xarray Dataset
        The dataset to load from.
    qty : str | array
        The name of the quantity (string) in `ds` or an array, where the
        attributes of `ds` will be used to rehydrate.
    plane : int | None
        If the sparse array is 2d, the plane to use.

    Returns
    -------
    rehydrated : array
        The rehydrated array
    """
    if "program" not in ds.attrs or ds.program != "dexrt (3d)":
        raise ValueError("Provided dataset not written by dexrt 3d")
    if ds.output_format != "sparse":
        raise ValueError("Data is not sparse and does not need to be rehydrated")

    block_size = ds.block_size
    block_entries = block_size**3
    if isinstance(qty, str):
        qty = ds[qty].values
    if isinstance(qty, xr.DataArray):
        qty = qty.values

    leading_dim = 1
    if qty.ndim > 1:
        leading_dim = qty.shape[0]
    else:
        qty = qty.reshape(1, -1)

    result_shape = (
        leading_dim,
        ds.num_z_blocks * block_size,
        ds.num_y_blocks * block_size,
        ds.num_x_blocks * block_size,
    )
    result = np.empty(result_shape, dtype=qty.dtype)

    morton_tiles = ds.morton_tiles.values
    for flat_tile_idx, morton_code in enumerate(morton_tiles):
        x, y, z = decode_morton_3(morton_code)
        flat_tile = qty[
            :, flat_tile_idx * block_entries : (flat_tile_idx + 1) * block_entries
        ]
        set_tile_3d(result, block_size, x, y, z, flat_tile)
    return result.squeeze()
