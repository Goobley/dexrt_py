from typing import Dict, Literal, Union, List, Optional
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
    rad_loss: bool = True
    pops: bool = True
    lte_pops: bool = True
    ne: bool = True
    nh_tot: bool = True
    max_mip_level: bool = True
    psi_star: bool = False
    cascades: List[int] = Field(default_factory=list)


class DexrtMipConfig(BaseModel):
    mip_levels: Union[int, List[int]] = Field(
        default_factory=lambda: [0, 0, 1, 2, 3, 3]
    )
    opacity_threshold: float = 0.25
    log_chi_mip_variance: float = 1.0
    log_eta_mip_variance: float = 1.0


class DexrtScConfig(BaseModel):
    """Options for the short characteristics solver (dexrt_sc); ignored by the
    radiance cascade executables."""
    quadrature: Union[Literal["Stepan"], Literal["StepanMirrored"], Literal["Healpix"]] = "Stepan"
    # None: 7 for the Stepan quadratures, 1 for Healpix (nside = 2**order)
    quadrature_order: Optional[int] = None
    formal_solver: Union[Literal["PiecewiseConstant"], Literal["PiecewiseLinear"], Literal["PiecewiseLinearEmissivity"]] = "PiecewiseLinearEmissivity"
    interpolation: Union[Literal["Linear"], Literal["VanLeer"], Literal["MonotonizedCentral"]] = "MonotonizedCentral"
    x_boundary: Union[Literal["Fixed"], Literal["Periodic"]] = "Fixed"
    characteristics: Union[Literal["Short"], Literal["LongerCharacteristics"]] = "Short"
    # Promweaver boundary: average each direction's incident intensity over its
    # equal-area cone with boundary_cone_mu Gauss-Legendre nodes x
    # boundary_cone_phi azimuths (0: off).
    boundary_cone_mu: int = 4
    boundary_cone_phi: int = 8
    # PRD-Voigt lines: Hybrid (comoving-frame angle-averaged redistribution,
    # impact-limit coherence) or Off (CRD). Sub-iterations per iteration, their
    # tolerance on the relative change of rho, and the first iteration with PRD.
    prd: Union[Literal["Off"], Literal["Hybrid"]] = "Off"
    prd_subiterations: int = 3
    prd_tolerance: float = 1e-3
    prd_start_iter: int = 0


class DexrtConfig(BaseModel):
    system: DexrtSystemConfig = Field(default_factory=DexrtSystemConfig)
    atmos_path: str = "dexrt_atmos.nc"
    output_path: str = "dexrt.nc"
    mode: Union[Literal["Lte"], Literal["NonLte"], Literal["GivenFs"]] = "NonLte"
    store_J_on_cpu: bool = True
    output: DexrtOutputConfig = Field(default_factory=DexrtOutputConfig)
    max_cascade: int = 5
    mip_config: DexrtMipConfig = Field(default_factory=DexrtMipConfig)
    sc: DexrtScConfig = Field(default_factory=DexrtScConfig)


class DexrtLteConfig(DexrtConfig):
    mode: Literal["Lte"] = "Lte"
    sparse_calculation: bool = False
    threshold_temperature: float = 250e3
    # Background opacity tables from dexrt.background_tables (empty for none).
    background_path: str = ""
    atoms: Dict[str, AtomicModelConfig]
    boundary_type: Union[Literal["Zero"], Literal["Promweaver"]]
    initial_pops_path: str = ""
    rad_loss: str  = Union[Literal["None"], Literal["PerWavelength"], Literal["Integrated"]]


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
    snapshot_frequency: int = 10
    initial_lambda_iterations: int = 2
    final_dense_fs: bool = True
    ng_config: DexrtNgConfig = Field(default_factory=DexrtNgConfig)


class DexrtGivenFsConfig(DexrtConfig):
    mode: Literal["GivenFs"] = "GivenFs"
