"""Background (continuum) opacity tables for dexrt, generated with
Lightweaver's BasicBackground (RH's treatment) on a grid of temperature and
electron density.

Each table is a function of (wavelength, log10 T, log10 n_e) and is stored per
unit of the density it scales with, so dexrt can combine them at run time:

- ``<El>_bf``: LTE bound-free opacity of element El, per hydrogen nucleus
  [m2] (one table per element, so elements treated as atoms in dexrt can be
  left out).
- ``He_rayleigh``: He I Rayleigh scattering, per hydrogen nucleus [m2].
- ``Hminus``: H- bound-free + free-free, per ground state neutral hydrogen
  atom [m2].
- ``H2plus``: H2+ bound-free + free-free, per (ground state neutral hydrogen
  atom x proton) [m5].
- ``H_ff``: hydrogen free-free, per proton [m2].
- ``H_rayleigh``: H I Rayleigh scattering, per ground state neutral hydrogen
  atom [m2] (wavelength only).

The opacities include stimulated emission and the emissivity of the
absorption terms is chi B_lambda(T). Thomson scattering is left to dexrt
(sigma_T n_e). Values are stored as log10 (floored at LOG_FLOOR).

Usage: python -m dexrt.background_tables OUT.nc
"""
import argparse
import sys
import numpy as np
import netCDF4

LOG_FLOOR = -99.0
DEFAULT_ELEMENTS = ["He", "C", "N", "O", "Na", "Mg", "Al", "Si", "S", "Ca", "Fe"]


def _disable_debye():
    # NOTE: dexrt's LTE populations don't include Debye shielding.
    import lightweaver.atomic_set as lw_atomic_set
    lte_pops = lw_atomic_set.lte_pops

    def lte_pops_no_debye(*args, **kwargs):
        kwargs["debye"] = False
        return lte_pops(*args, **kwargs)

    lw_atomic_set.lte_pops = lte_pops_no_debye
    for mod in list(sys.modules.values()):
        if getattr(mod, "lte_pops", None) is lte_pops:
            mod.lte_pops = lte_pops_no_debye


def _atom_factories():
    from lightweaver import rh_atoms
    return {
        "H": rh_atoms.H_6_atom,
        "He": rh_atoms.He_9_atom,
        "C": rh_atoms.C_atom,
        "N": rh_atoms.N_atom,
        "O": rh_atoms.O_atom,
        "Na": rh_atoms.Na_atom,
        "Mg": rh_atoms.MgII_atom,
        "Al": rh_atoms.Al_atom,
        "Si": rh_atoms.Si_atom,
        "S": rh_atoms.S_atom,
        "Ca": rh_atoms.CaII_atom,
        "Fe": rh_atoms.Fe_atom,
    }


def h_ff_per_proton(wavelength, temperature, ne):
    """Hydrogen free-free opacity per proton [m2] (Mihalas 1978, p. 101, as
    Lightweaver's hydrogen_ff). wavelength [nm] broadcasts against
    temperature [K] and ne [m-3]."""
    import lightweaver.constants as C
    C0 = C.QElectron**2 / (4.0 * np.pi * C.Epsilon0) / np.sqrt(C.MElectron)
    sigma = 4.0 / 3.0 * np.sqrt(2.0 * np.pi / (3.0 * C.KBoltzmann)) * C0**3 / C.HC
    nu3 = (wavelength * C.NM_TO_M / C.CLight)**3
    hc_kla = C.HC / (C.KBoltzmann * C.NM_TO_M * wavelength)
    stim = 1.0 - np.exp(-hc_kla / temperature)
    x = (C.HC / (wavelength * C.NM_TO_M)) / C.ERydberg
    x3 = x**(1.0 / 3.0)
    y = 2.0 * wavelength * C.NM_TO_M * C.KBoltzmann * temperature / C.HC
    gff = 1.0 + 0.1728 * x3 * (1.0 + y) - 0.0496 * x3**2 * (1.0 + (1.0 + y) / 3.0 * y)
    gff = np.maximum(gff, 1.0)
    return sigma / np.sqrt(temperature) * nu3 * ne * stim * gff


def _background(atoms, elements, wavelength, temperature, ne, nh_tot, threads):
    """Background (absorption, scattering) [m-1] on the (T, ne) points for
    the RadiativeSet of the given elements (H active, the rest passive), at
    the requested wavelengths. Also returns the LTE H populations."""
    import lightweaver as lw
    n = temperature.shape[0]
    atmos = lw.Atmosphere.make_1d(
        scale=lw.ScaleType.Geometric,
        depth_scale=np.linspace(1e6, 0.0, n),
        temperature=temperature,
        vlos=np.zeros(n),
        vturb=np.zeros(n),
        ne=ne,
        nh_tot=nh_tot,
    )
    atmos.quadrature(1)
    rad_set = lw.RadiativeSet([atoms[e]() for e in ["H"] + elements])
    rad_set.set_active("H")
    spect = rad_set.compute_wavelength_grid(extra_wavelengths=wavelength)
    eq_pops = rad_set.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eq_pops, Nthreads=threads)
    idx = np.searchsorted(spect.wavelength, wavelength)
    assert np.allclose(spect.wavelength[idx], wavelength)
    chi = np.asarray(ctx.background.chi)[idx]
    sca = np.asarray(ctx.background.sca)[idx]
    return chi - sca, sca, np.asarray(eq_pops["H"])


def _edges(atoms, elements):
    edges = [1643.91]  # H- photodetachment threshold [nm]
    for e in ["H"] + elements:
        for c in atoms[e]().continua:
            edges.append(c.lambdaEdge)
    return np.array(edges)


def make_tables(
    path,
    elements=DEFAULT_ELEMENTS,
    lambda_min=50.0,
    lambda_max=5000.0,
    num_lambda=1500,
    log_t=(3.31, 5.0, 139),
    log_ne=(12.0, 23.0, 45),
    nh_tot=1e20,
    threads=16,
):
    _disable_debye()
    import lightweaver.constants as C
    atoms = _atom_factories()

    # NOTE: Log spaced, with points either side of every continuum edge.
    wavelength = np.geomspace(lambda_min, lambda_max, num_lambda)
    edges = _edges(atoms, elements)
    edges = edges[(edges > lambda_min) & (edges < lambda_max)]
    wavelength = np.unique(np.concatenate([wavelength, edges * (1.0 - 1e-5), edges * (1.0 + 1e-5)]))

    lt = np.linspace(*log_t)
    lne = np.linspace(*log_ne)
    LT, LNE = np.meshgrid(lt, lne, indexing="ij")
    temperature = 10.0**LT.ravel()
    ne = 10.0**LNE.ravel()
    nh = np.full_like(temperature, nh_tot)
    shape = (wavelength.shape[0], lt.shape[0], lne.shape[0])

    def log_table(a):
        with np.errstate(divide="ignore"):
            out = np.log10(np.maximum(a, 0.0))
        return np.maximum(out, LOG_FLOOR).reshape(shape).astype(np.float32)

    tables = {}
    base_abs, base_sca, h_pops = _background(atoms, [], wavelength, temperature, ne, nh, threads)
    n1s = h_pops[0]
    n_p = h_pops[-1]
    h_ff = h_ff_per_proton(wavelength[:, None], temperature[None, :], ne[None, :])
    thomson = 8.0 * np.pi / 3.0 * (C.QElectron / (np.sqrt(4.0 * np.pi * C.Epsilon0) * np.sqrt(C.MElectron) * C.CLight))**4 * ne
    # NOTE: The remaining H terms are H- (prop. n1s, at fixed T, ne) and H2+
    # (prop. n1s n_p): separate them with a second hydrogen density.
    abs2, _, h_pops2 = _background(atoms, [], wavelength, temperature, ne, 10.0 * nh, threads)
    y1 = (base_abs - h_ff * n_p) / n1s
    y2 = (abs2 - h_ff * h_pops2[-1]) / h_pops2[0]
    h2plus = (y2 - y1) / (h_pops2[-1] - n_p)
    tables["Hminus"] = log_table(y1 - h2plus * n_p)
    tables["H2plus"] = log_table(h2plus)
    tables["H_ff"] = log_table(h_ff)
    h_ray = (base_sca - thomson[None, :]) / n1s[None, :]
    h_rayleigh = np.median(h_ray, axis=1)

    for e in elements:
        abs_e, sca_e, _ = _background(atoms, [e], wavelength, temperature, ne, nh, threads)
        tables[f"{e}_bf"] = log_table((abs_e - base_abs) / nh)
        if e == "He":
            tables["He_rayleigh"] = log_table((sca_e - base_sca) / nh)
        print(f"{e} done", flush=True)

    with netCDF4.Dataset(path, "w") as f:
        f.createDimension("wavelength", wavelength.shape[0])
        f.createDimension("log_temperature", lt.shape[0])
        f.createDimension("log_ne", lne.shape[0])
        f.createVariable("wavelength", "f8", ("wavelength",))[...] = wavelength
        f.createVariable("log_temperature", "f8", ("log_temperature",))[...] = lt
        f.createVariable("log_ne", "f8", ("log_ne",))[...] = lne
        for k, v in tables.items():
            f.createVariable(k, "f4", ("wavelength", "log_temperature", "log_ne"), zlib=True)[...] = v
        with np.errstate(divide="ignore"):
            f.createVariable("H_rayleigh", "f4", ("wavelength",))[...] = np.maximum(np.log10(np.maximum(h_rayleigh, 0.0)), LOG_FLOOR)
        f.elements = " ".join(elements)
        f.log_floor = LOG_FLOOR
        f.description = "log10 background opacities per unit density [m2], see dexrt.background_tables"
        f.lightweaver_version = __import__("lightweaver").__version__


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dexrt background opacity tables with Lightweaver.")
    parser.add_argument("output")
    parser.add_argument("--num-lambda", type=int, default=1500)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    make_tables(args.output, num_lambda=args.num_lambda, threads=args.threads)
