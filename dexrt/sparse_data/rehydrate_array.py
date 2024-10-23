import numpy as np
import xarray as xr
from dexrt.sparse_data.morton import decode_morton_2


def get_tile(field: np.ndarray, block_size: int, x: int, z: int) -> np.ndarray:
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


def set_tile(field: np.ndarray, block_size: int, x: int, z: int, data: np.ndarray):
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


def rehydrate_quantity(ds: xr.Dataset, qty: str | np.ndarray) -> np.ndarray:
    if "program" not in ds.attrs or ds.program != "dexrt (2d)":
        raise ValueError("Provided dataset not written by dexrt 2d")
    if ds.output_format != "sparse":
        raise ValueError("Data is not sparse and does not need to be rehydrated")

    block_size = ds.block_size
    block_entries = block_size * block_size
    if isinstance(qty, str):
        qty = ds[qty].values

    leading_dim = 1
    if qty.ndim > 1:
        leading_dim = qty.shape[0]

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
    return result
