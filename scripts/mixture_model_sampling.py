"""
Perform an EM-like sampling round to infer parameters and mixture components of a
n-component mixture of lymphatic progression models for k different tumor subsites.
"""
import argparse
from pathlib import Path
import numpy as np
import yaml

import matplotlib.pyplot as plt
import h5py
from lyscripts.utils import (
    load_patient_data,
    create_model_from_config,
)

from helpers import simplify_subsite, SUBSITE, SIMPLE_SUBSITE
from lymixture.mixture_model import LymphMixtureModel


SUBSITE = ("tumor", "1", "subsite")
def create_parser() -> argparse.ArgumentParser:
    """Assemble the parser for the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input", type=Path, default="data/enhanced.csv",
        help="Path to the patient data file.",
    )
    parser.add_argument(
        "-m", "--model", type=Path, default="models/mixture.hdf5",
        help="Path for sampled mixture model.",
    )
    parser.add_argument(
        "-f", "--figure", type=Path, default="figures/mixture_history.png",
        help="Path for figure over EM history.",
    )
    parser.add_argument(
        "-p", "--params", type=Path, default="_variables.yml",
        help="Path to the parameter file. Looks for a key `barplot_kwargs`.",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
        help="Seed for the random number generator.",
    )
    return parser


def main():
    """Execute the main routine."""
    args = create_parser().parse_args()

    with open(args.params, mode="r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
        params["mixture_model"]["em_config"]["m_step"].update({
            "imputation_function": lambda x: int(10 / 6 * x**1.1 +1)
        })

    patient_data = load_patient_data(args.input)
    patient_data[SIMPLE_SUBSITE] = patient_data[SUBSITE].apply(simplify_subsite)
    num_subsites = len(patient_data[SIMPLE_SUBSITE].unique())
    lymph_model = create_model_from_config(params)
    lymph_model.modalities = {"max_llh": [1., 1.]}

    mixture_model = LymphMixtureModel(
        lymph_model=lymph_model,
        n_clusters=params["mixture_model"]["num_clusters"],
        n_subpopulation=num_subsites,
        hdf5_output=args.model,
    )
    mixture_model.load_data(
        patient_data=patient_data,
        split_by=SIMPLE_SUBSITE,
        mapping=lambda x: "all",
    )

    np.random.seed(args.seed)
    _chain, cluster_components, history = mixture_model.fit(
        em_config=params["mixture_model"]["em_config"],
        mcmc_config=params["mixture_model"]["mcmc_config"],
    )
    with h5py.File(args.model, mode="a") as h5_file:
        if "em/cluster_components" in h5_file:
            del h5_file["em/cluster_components"]
        h5_file.create_dataset("em/cluster_components", data=cluster_components)
    history.plot_history(
        mixture_model.subpopulation_labels,
        list(lymph_model.get_params(as_dict=True).keys()),
        mixture_model.n_clusters,
        None,
    )
    plt.savefig(args.figure)


if __name__ == "__main__":
    main()
