"""
Fixtures and helpers for the unit tests.
"""
from pathlib import Path

import pandas as pd


SIMPLE_SUBSITE = ("tumor", "1", "simple_subsite")
SUBSITE = ("tumor", "1", "subsite")


def get_graph(size: str = "large") -> dict[tuple[str, str], list[str]]:
    """Return either a ``"small"``, a ``"medium"`` or a ``"large"`` graph."""
    if size == "small":
        return {
            ("tumor", "T"): ["II", "III"],
            ("lnl", "II"): ["III"],
            ("lnl", "III"): [],
        }

    if size == "medium":
        return {
            ("tumor", "T"): ["I", "II", "III", "IV"],
            ("lnl", "I"): ["II"],
            ("lnl", "II"): ["III"],
            ("lnl", "III"): ["IV"],
            ("lnl", "IV"): [],
        }

    if size == "large":
        return {
            ("tumor", "T"): ["I", "II", "III", "IV", "V", "VII"],
            ("lnl", "I"): [],
            ("lnl", "II"): ["I", "III", "V"],
            ("lnl", "III"): ["IV", "V"],
            ("lnl", "IV"): [],
            ("lnl", "V"): [],
            ("lnl", "VII"): [],
        }

    raise ValueError(f"Unknown graph size: {size}")


def simplify_subsite(icd_code: str) -> str:
    """Only use the part of the ICD code before the decimal point."""
    return icd_code.split(".")[0]


def get_patient_data(do_simplify_subsite: bool = True) -> pd.DataFrame:
    """Load the patient data for the tests and simplify the ICD codes."""
    patient_data = pd.read_csv(
        Path(__file__).parent / "data" / "patients.csv",
        header=[0,1,2],
    )

    if do_simplify_subsite:
        patient_data[SIMPLE_SUBSITE] = patient_data[SUBSITE].apply(simplify_subsite)

    return patient_data
