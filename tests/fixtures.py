"""
Fixtures and helpers for the unit tests.
"""
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from lymixture import LymphMixtureModel
from lymixture.utils import map_to_simplex

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


class MixtureModelFixture:
    """Fixture for the mixture model tests."""

    def setup_rng(self, seed: int = 42):
        """Initialize random number generator with ``seed``."""
        self.rng = np.random.default_rng(seed)

    def setup_mixture_model(
        self,
        model_cls: type,
        num_components: int,
        graph_size: Literal["small", "medium", "large"] = "small",
        load_data: bool = True,
    ):
        """Initialize the fixture."""
        self.num_components = num_components
        self.model_cls = model_cls

        self.mixture_model = LymphMixtureModel(
            model_cls=self.model_cls,
            model_kwargs={"graph_dict": get_graph(graph_size)},
            num_components=self.num_components,
        )
        if load_data:
            self.patient_data = get_patient_data()
            self.mixture_model.load_patient_data(
                self.patient_data,
                split_by=SIMPLE_SUBSITE,
            )


    def setup_responsibilities(self):
        """Initialize a set of responsibilities for the mixture model."""
        unit_cube = self.rng.uniform(
            size=(len(self.patient_data), len(self.mixture_model.components) - 1),
        )
        self.resp = np.array([map_to_simplex(line) for line in unit_cube])
