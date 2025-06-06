from typing import Dict, Literal, Union, List
from pydantic import BaseModel, Field


class AtomicModelConfig(BaseModel):
    path: str
    treatment: Union[Literal["Detailed"], Literal["Golding"], Literal["Active"]] = (
        "Active"
    )
    initial_populations: Union[Literal["Lte"], Literal["ZeroRadiation"]] = (
        "Lte"
    )


class DexrtSystemConfig(BaseModel):
    mem_pool_gb: float = 4.0


class DexrtOutputConfig(BaseModel):
    sparse: bool = False
    wavelength: bool = True
    J: bool = True
    pops: bool = True
    lte_pops: bool = True
    ne: bool = True
    nh_tot: bool = True
    max_mip_level: bool = True
    alo: bool = False
    cascades: List[int] = Field(default_factory=list)


class DexrtMipConfig(BaseModel):
    mip_levels: Union[int, List[int]] = Field(
        default_factory=lambda: [0, 0, 1, 2, 3, 3]
    )
    opacity_threshold: float = 0.25
    log_chi_mip_variance: float = 1.0
    log_eta_mip_variance: float = 1.0


class DexrtConfig(BaseModel):
    system: DexrtSystemConfig = Field(default_factory=DexrtSystemConfig)
    atmos_path: str = "dexrt_atmos.nc"
    output_path: str = "dexrt.nc"
    mode: Union[Literal["Lte"], Literal["NonLte"], Literal["GivenFs"]] = "NonLte"
    store_J_on_cpu: bool = True
    output: DexrtOutputConfig = Field(default_factory=DexrtOutputConfig)
    max_cascade: int = 5
    mip_config: DexrtMipConfig = Field(default_factory=DexrtMipConfig)


class DexrtLteConfig(DexrtConfig):
    mode: Literal["Lte"] = "Lte"
    sparse_calculation: bool = False
    threshold_temperature: float = 250e3
    atoms: Dict[str, AtomicModelConfig]
    boundary_type: Union[Literal["Zero"], Literal["Promweaver"]]
    initial_pops_path: str = ""


class DexrtNgConfig(BaseModel):
    enable: bool = True
    threshold: float = 5e-2
    lower_threshold: float = 2e-4


class DexrtNonLteConfig(DexrtLteConfig):
    mode: Literal["NonLte"] = "NonLte"
    max_iter: int = 200
    pop_tol: float = 1e-3
    conserve_charge: bool = False
    conserve_pressure: bool = False
    snapshot_frequency: int = 0
    initial_lambda_iterations: int = 2
    final_dense_fs: bool = True
    ng_config: DexrtNgConfig = Field(default_factory=DexrtNgConfig)


class DexrtGivenFsConfig(DexrtConfig):
    mode: Literal["GivenFs"] = "GivenFs"
