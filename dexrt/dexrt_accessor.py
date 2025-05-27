from pathlib import Path
from matplotlib import pyplot as plt
import xarray as xr
import yaml
from dexrt.sparse_data.rehydrate_array import rehydrate_quantity, rehydrate_quantity_3d
import numpy as np
import os.path as path

from dexrt.plot.ray_output import plot_ray_output, plot_slit_view_ray

"""Simple accessor extensions for xarray.
Enables:
```
ds = xr.load_dataset(...)
full_pops = ds.dexrt.rehydrated["pops"]
# or
full_pops = ds.dexrt.rehydrated.pops
```
Also connects the different outputs via their config files:
If a dexrt_ray output is loaded then ds.ray, ds.dex, and ds.atmos point to the
ray_data, synth_data and atmos_data, respectively. For a core dexrt output,
ds.dex and ds.atmos are populated.

The dexrt access also enables quick look for ray output:
```
ds.dexrt.plot_ray(lambda0=854, theta=48) # theta in degres or muz as muz=1
```
"""


class SparseAccessor:
    def __init__(self, ds: xr.Dataset, dimensionality: int=2):
        self.ds = ds
        self.dimensionality = dimensionality
        self.rehydrate = rehydrate_quantity if dimensionality == 2 else rehydrate_quantity_3d

    def __getattr__(self, attr: str) -> np.ndarray:
        return self[attr]

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self.ds:
            raise ValueError(f"Quantity {name} not present in dataset.")

        return self.rehydrate(self.ds, name)

    def plane(self, name: str, entries: int | slice) -> np.ndarray:
        if name not in self.ds:
            raise ValueError(f"Quantity {name} not present in dataset.")

        return self.rehydrate(self.ds, self.ds[name][entries])


@xr.register_dataset_accessor("dexrt")
class DexrtAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self.ds = xarray_obj
        self._config = None
        self.config_prefix = None

        if "config_path" in self.ds.attrs:
            self._infer_config_locations(self.ds.config_path)

        self._ray = None
        self._dex = None
        self._atmos = None
        self._fig = None

        # NOTE(cmo): dexrt (3d) has always had a conformant program field
        dimensionality = 2
        if "program" in self.ds.attrs:
            if self.ds.program.startswith("dexrt ("):
                self._dex = self.ds
                self._attempt_load_atmos_data()
            elif self.ds.program.startswith("dexrt_ray ("):
                self._ray = self.ds
                self._attempt_load_synth_and_atmos_data()

            program_split = self.ds.program.split('(')
            if len(program_split) == 2:
                dimensionality = int(program_split[1][0])

        self.rehydrated = SparseAccessor(self.ds, dimensionality)

    @property
    def ray(self):
        if self._ray is not None:
            return self._ray
        raise AttributeError("Cannot find associated ray data")

    @property
    def dex(self):
        if self._dex is None:
            self._attempt_load_synth_and_atmos_data()
        if self._dex is not None:
            return self._dex
        raise AttributeError("Cannot find associated dex data")

    @property
    def atmos(self):
        if self._atmos is None:
            self._attempt_load_synth_and_atmos_data()
            self._attempt_load_atmos_data()
        if self._atmos is not None:
            return self._atmos
        raise AttributeError("Cannot find associated atmos data")

    def _attempt_load_synth_and_atmos_data(self):
        # NOTE(cmo): Will only do anything if self.ds is a ray file
        if self.config_prefix is not None and "dexrt_config_path" in self._config:
            dexrt_config_path = self.config_prefix / self._config["dexrt_config_path"]
            with open(dexrt_config_path, "r") as f:
                dexrt_config = yaml.load(f, Loader=yaml.Loader)

            atmos_data = xr.open_dataset(
                self.config_prefix / dexrt_config["atmos_path"]
            )
            synth_data = xr.open_dataset(
                self.config_prefix / dexrt_config["output_path"]
            )
            self._atmos = atmos_data
            self._dex = synth_data

    def _attempt_load_atmos_data(self):
        # NOTE(cmo): Will only do anything if self.ds is a main dex file
        if self.config_prefix is not None and "atmos_path" in self._config:
            atmos_data = xr.open_dataset(
                self.config_prefix / self._config["atmos_path"]
            )
            self._atmos = atmos_data

    def _infer_config_locations(self, config_path: str, raise_err: bool = False):
        ds_path = self.ds[list(self.ds.data_vars.keys())[0]].encoding["source"]
        ds_dir = path.split(ds_path)[0]

        config_path_prefix = Path(ds_dir)
        config_path = config_path_prefix / path.basename(config_path)
        if path.isfile(config_path):
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.Loader)
            self._config = config
            self.config_prefix = config_path_prefix
        elif raise_err:
            raise ValueError(
                f'Unable to infer location of config path. Tried: "{config_path}". Consider setting config_path in the args.'
            )

    def plot_ray(
        self,
        lambda0: float,
        delta_lambda: float = 0.12,
        muz: float | None = None,
        theta: float | None = None,
        config_path: str | None = None,
        reuse_fig: bool = True
    ):
        """Quicklook for dexrt_ray data. Must provide one of muz or theta.

        Parameters
        ----------
        lambda0 : float
            The central wavelength to plot around. Will be "snapped" to closest
            line core.
        delta_lambda : float, optional
            The half-width of the spectral plotting window (nm). Default: 0.12
        muz : float, optional
            The muz of the viewing ray
        theta : float, optional
            The angle to z of the viewing ray in degrees.
        config_path : str, optional
            The path to the config file to use if this cannot be inferred from
            the output (e.g. output is very old).
        reuse_fig : bool, optional
            Whether to reuse the most recent figure used by the dexrt accessor.
            Default: True.
        """
        if self.ds.program != "dexrt_ray (2d)":
            raise ValueError(
                "Calling plot_ray on data that is not output from dexrt_ray (2d)"
            )

        need_config = self._config is None or self.config_prefix is None
        if need_config:
            if config_path is None:
                raise ValueError(
                    "config_path not present in file attributes, maybe this was generated with an old version of dexrt_ray?"
                )
            self._infer_config_locations(config_path, raise_err=True)

        atmos_data = self.atmos
        synth_data = self.dex

        if (muz is None and theta is None) or (muz is not None and theta is not None):
            raise ValueError("Must specify one of the muz or theta (but not both)")

        if theta is not None:
            muz = np.cos(np.deg2rad(theta))
        mu_idx = np.abs(self.ds.mu[:, 2].values - muz).argmin()

        figsize = (10, 6)
        if reuse_fig and self._fig is not None:
            fig = self._fig
            fig.clear()
            fig.set_size_inches(figsize)
            ax = fig.subplots(1, 2)
        else:
            fig, ax = plt.subplots(1, 2, figsize=figsize)
        plot_ray_output(
            ax[0],
            synth_data=synth_data,
            atmos_data=atmos_data,
            ray_data=self.ds,
            mu_idx=mu_idx,
            lambda0=lambda0,
            delta_lambda=delta_lambda,
        )
        plot_slit_view_ray(
            ax[1],
            synth_data=synth_data,
            atmos_data=atmos_data,
            ray_data=self.ds,
            mu_idx=mu_idx,
        )
        print_angle = np.rad2deg(
            np.sign(self.ds.mu[mu_idx, 0]) * np.arccos(self.ds.mu[mu_idx, 2])
        )
        snapped_lambda0 = synth_data.lambda0[
            np.abs(synth_data.lambda0 - lambda0).argmin()
        ]
        fig.suptitle(f"{snapped_lambda0:.2f} nm @ {print_angle:.1f} °")

        self._fig = fig
        return fig
