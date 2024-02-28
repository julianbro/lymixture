"""
Helper functions for the scripts.
"""
from pathlib import Path
from typing import Generator
import argparse
from emcee.backends import HDFBackend

import h5py
from lyscripts.plot.utils import COLORS as USZ
from matplotlib.colors import to_rgba


OROPHARYNX_ICDS = ["C01", "C09", "C10"]
SUBSITE = ("tumor", "1", "subsite")
SIMPLE_SUBSITE = ("tumor", "1", "simple_subsite")
LOCATION = ("tumor", "1", "location")
T_STAGE = ("tumor", "1", "t_stage")
LNLS = ["I", "II", "III", "IV"]
LNL_I = ("max_llh", "ipsi", "I")
LNL_II = ("max_llh", "ipsi", "II")
LNL_III = ("max_llh", "ipsi", "III")
LNL_IV = ("max_llh", "ipsi", "IV")


def simplify_subsite(icd_code: str) -> str:
    """Only use the part of the ICD code before the decimal point."""
    return icd_code.split(".")[0]


def generate_location_colors(
    icd_codes: list[str],
    delta_alpha: float = 0.15,
) -> Generator[tuple[float, float, float, float], None, None]:
    """Make a list of colors for each location."""
    oropharynx_alpha = 1.0
    oral_cavity_alpha = 1.0
    colors, alphas = [], []
    for icd_code in icd_codes:
        if icd_code in OROPHARYNX_ICDS:
            colors.append(USZ["orange"])
            alphas.append(oropharynx_alpha)
            oropharynx_alpha -= delta_alpha
            yield to_rgba(USZ["orange"], oropharynx_alpha)
        else:
            colors.append(USZ["blue"])
            alphas.append(oral_cavity_alpha)
            oral_cavity_alpha -= delta_alpha
            yield to_rgba(USZ["blue"], oral_cavity_alpha)


def str2bool(v: str) -> bool | None:
    """Transform a string to a boolean or ``None``."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't'):
        return True
    elif v.lower() in ('no', 'false', 'f'):
        return False
    elif v.lower() in ('none', 'null', 'n'):
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean or None value expected.')


def get_location(for_subsite):
    """Return the location of the given subsite."""
    if for_subsite in OROPHARYNX_ICDS:
        return "oropharynx"
    else:
        return "oral cavity"


def get_indie_chain(from_dir: Path, for_location: str, thin_by: int):
    """Get the independent model's chain from the given directory."""
    if for_location == "oropharynx":
        indie_chain = HDFBackend(
            from_dir / "oropharynx.hdf5",
            read_only=True,
        ).get_chain(flat=True, thin=thin_by)
    else:
        indie_chain = HDFBackend(
            from_dir / "oral_cavity.hdf5",
            read_only=True,
        ).get_chain(flat=True, thin=thin_by)

    return indie_chain


def get_mixture_components(
    from_dir: Path,
    for_subsite: str,
    icd_code_map: dict,
) -> tuple[float, float]:
    """Get the mixture model's components from the given directory."""
    with h5py.File(from_dir / "mixture.hdf5", mode="r") as h5_file:
        component_assignments = h5_file["em/cluster_components"][...]

    idx = list(icd_code_map.keys()).index(for_subsite)
    comp_A_prob = component_assignments[idx]
    comp_B_prob = 1 - comp_A_prob

    return comp_A_prob, comp_B_prob


def get_prevalence_pattern(for_lnl: str) -> dict[str, dict[str, bool | None]]:
    """Create a pattern correpsonding to the prevalence of one LNL's involvement."""
    pattern = {"ipsi": {}}
    for lnl in LNLS:
        pattern["ipsi"][lnl] = None
        if lnl == for_lnl:
            pattern["ipsi"][lnl] = True
    return pattern
