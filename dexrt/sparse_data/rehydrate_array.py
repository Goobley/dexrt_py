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
    if "sparse_layout" in ds.attrs:
        if ds.attrs.get("output_format") != "sparse":
            raise ValueError("Data is not sparse and does not need to be rehydrated")
        return rehydrate_sc_quantity(ds, qty)
    if ds.attrs.get("program") not in ("dexrt (2d)", "dexrt_sc (2d)"):
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
    result = np.zeros(result_shape, dtype=qty.dtype)

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
    result = np.zeros(result_shape, dtype=qty.dtype)

    morton_tiles = ds.morton_tiles.values
    for flat_tile_idx, morton_code in enumerate(morton_tiles):
        x, y, z = decode_morton_3(morton_code)
        flat_tile = qty[
            :, flat_tile_idx * block_entries : (flat_tile_idx + 1) * block_entries
        ]
        set_tile_3d(result, block_size, x, y, z, flat_tile)
    return result.squeeze()


def _sc_indices(ds: xr.Dataset) -> np.ndarray:
    """Global flat cell indices of the SC compact storage (row-major active cells)."""
    if ds.attrs.get("sparse_layout") != "row_major":
        raise ValueError("Unsupported SC sparse layout")
    mask = np.asarray(ds["active"].values)
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
        raise ValueError("SC activity must be a binary [z, x] mask")
    indices = np.flatnonzero(mask)
    if len(indices) != ds.sizes.get("ks", 0):
        raise ValueError("SC activity does not match the compact arrays")
    return indices


def rehydrate_sc_quantity(ds: xr.Dataset, qty: str | xr.DataArray | np.ndarray) -> np.ndarray:
    """Rehydrate row-major SC compact output."""
    indices = _sc_indices(ds)
    if isinstance(qty, str):
        qty = ds[qty]
    if isinstance(qty, xr.DataArray):
        if not qty.dims or qty.dims[-1] != "ks":
            raise ValueError("Requested SC quantity is not compact")
        qty = qty.values
    qty = np.asarray(qty)
    if qty.ndim < 1 or qty.shape[-1] != len(indices):
        raise ValueError("SC quantity's compact dimension does not match the activity")
    shape = ds["active"].shape
    result = np.zeros(qty.shape[:-1] + (int(np.prod(shape)),), dtype=qty.dtype)
    result[..., indices] = qty
    return result.reshape(qty.shape[:-1] + shape).squeeze()


def sc_to_morton(ds: xr.Dataset, block_size: int = 16) -> xr.Dataset:
    """Convert tile-equivalent SC compact fields to legacy 2D Morton storage.

    Per-cell masks and partial outer tiles are rejected. The returned Dataset
    owns its reordered arrays; the input is not changed.
    """
    from dexrt.sparse_data.morton import encode_morton_2

    indices = _sc_indices(ds)
    mask = np.asarray(ds["active"].values, dtype=bool)
    nz, nx = mask.shape
    if (
        not isinstance(block_size, (int, np.integer))
        or block_size < 1
        or nx % block_size
        or nz % block_size
    ):
        raise ValueError("Morton conversion requires complete outer tiles")
    if max(nx // block_size, nz // block_size) > 65536:
        raise ValueError("Tile coordinates exceed the legacy Morton encoding")
    tiles = []
    for z in range(0, nz, block_size):
        for x in range(0, nx, block_size):
            tile = mask[z : z + block_size, x : x + block_size]
            if tile.any() != tile.all():
                raise ValueError("Morton conversion requires tile-equivalent activity")
            if tile.all():
                tiles.append((int(encode_morton_2(x // block_size, z // block_size)), x, z))
    tiles.sort()
    inverse = np.full(nx * nz, -1, dtype=np.int64)
    inverse[indices] = np.arange(len(indices))
    order = []
    for _, x, z in tiles:
        for dz in range(block_size):
            order.extend(inverse[(z + dz) * nx + x : (z + dz) * nx + x + block_size])
    out = ds.copy(deep=True)
    order = np.asarray(order, dtype=np.int64)
    for name, var in ds.data_vars.items():
        if var.dims and var.dims[-1] == "ks":
            out[name] = (var.dims, np.asarray(var.values)[..., order], var.attrs)
    out["morton_tiles"] = (
        "num_active_tiles",
        np.asarray([code for code, _, _ in tiles], dtype=np.uint32),
    )
    out.attrs.pop("sparse_layout", None)
    out.attrs.update(
        program="dexrt (2d)",
        output_format="sparse",
        block_size=block_size,
        num_x_blocks=nx // block_size,
        num_z_blocks=nz // block_size,
    )
    return out
