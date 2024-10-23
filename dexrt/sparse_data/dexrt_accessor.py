import xarray as xr
from dexrt.sparse_data.rehydrate_array import rehydrate_quantity

"""Simple accessor extensions for xarray.
Enables:
```
ds = xr.load_dataset(...)
full_pops = ds.dexrt.rehydrated["pops"]
# or
full_pops = ds.dexrt.rehydrated.pops
```
"""


class SparseAccessor:
    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    def __getattr__(self, attr):
        return self[attr]

    def __getitem__(self, name):
        if name not in self.ds:
            raise ValueError(f"Quantity {name} not present in dataset.")

        return rehydrate_quantity(self.ds, name)


@xr.register_dataset_accessor("dexrt")
class DexrtAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self.ds = xarray_obj
        self.rehydrated = SparseAccessor(self.ds)
