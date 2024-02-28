"""
Plot the prevalence for involvement, as predicted by the trained mixture model, and
compare it to the prevalence of involvement in the data.
"""
import argparse
from pathlib import Path
import yaml
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from tueplots import figsizes, fontsizes
from emcee.backends import HDFBackend
from lyscripts.utils import load_patient_data, create_model_from_config
from lyscripts.predict.prevalences import (
    compute_observed_prevalence,
    generate_predicted_prevalences,
)
from lyscripts.plot.utils import COLORS as USZ

from helpers import (
    SUBSITE,
    T_STAGE,
    LNLS,
    get_indie_chain,
    get_location,
    get_mixture_components,
    get_prevalence_pattern,
)


def create_parser() -> argparse.ArgumentParser:
    """Assemble the parser for the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d", "--data", type=Path, default="data/enhanced.csv",
        help="Path to the data file.",
    )
    parser.add_argument(
        "--mixture-model", type=Path, default="models/mixture.hdf5",
        help=(
            "Path to the mixture model HDF5 file. Needs to contain a dataset called "
            "``em/cluster_components``. Parent directory is assumed to also contain "
            "the HDF5 file of the independent model (either `oropharynx.hdf5` or "
            "`oral_cavity.hdf5`)."
        )
    )
    parser.add_argument(
        "-o", "--output", type=Path, default="figures/prevalence_comparison.png",
        help="Path to the output file.",
    )
    parser.add_argument(
        "-p", "--params", type=Path, default="_variables.yml",
        help="Path to the parameter file..",
    )
    parser.add_argument(
        "-l", "--lnl", type=str, choices=LNLS, default="II",
        help="LNL to plot involvement prevalence for.",
    )
    parser.add_argument(
        "-s", "--subsite", type=str, default="C01",
        help="ICD code of the subsite to plot the prevalences for.",
    )
    parser.add_argument(
        "--thin-indie", type=int, default=2,
        help="Thinning factor for the independent model's chain.",
    )
    parser.add_argument(
        "--thin-mixture", type=int, default=200,
        help="Thinning factor for the mixture model's chain.",
    )
    return parser


def main():
    args = create_parser().parse_args()

    with open(args.params) as file:
        params = yaml.safe_load(file)

    lymph_model = create_model_from_config(params)
    lymph_model.modalities = {"max_llh": [1., 1.]}

    pattern = get_prevalence_pattern(for_lnl=args.lnl)
    location = get_location(for_subsite=args.subsite)

    patient_data = load_patient_data(args.data)
    patient_data[T_STAGE] = ["all"] * len(patient_data)

    indie_chain = get_indie_chain(
        from_dir=args.mixture_model.parent,
        for_location=location,
        thin_by=args.thin_indie,
    )
    mixture_chain = HDFBackend(
        args.mixture_model, read_only=True,
    ).get_chain(flat=True, thin=args.thin_mixture, discard=1000)

    comp_A_prob, comp_B_prob = get_mixture_components(
        from_dir=args.mixture_model.parent,
        for_subsite=args.subsite,
        icd_code_map=params["icd_code_map"],
    )

    matching, total = compute_observed_prevalence(
        pattern=pattern,
        data=patient_data.loc[patient_data[SUBSITE].str.contains(args.subsite)],
        lnls=LNLS,
        t_stage="all",
        modality="max_llh",
        invert=False,
    )
    independent_prevs = np.array(list(generate_predicted_prevalences(
        pattern=pattern,
        model=lymph_model,
        samples=indie_chain,
        t_stage="all",
        modality_spsn=[1., 1.],
    )))
    comp_A_prevs = np.array(list(generate_predicted_prevalences(
        pattern=pattern,
        model=lymph_model,
        samples=mixture_chain[:,:7],
        t_stage="all",
        modality_spsn=[1., 1.],
    )))
    comp_B_prevs = np.array(list(generate_predicted_prevalences(
        pattern=pattern,
        model=lymph_model,
        samples=mixture_chain[:,7:],
        t_stage="all",
        modality_spsn=[1., 1.],
    )))
    mixture_model_prevs = (
        comp_A_prob * comp_A_prevs
        + comp_B_prob * comp_B_prevs
    )

    total_min = math.floor(np.min([
        independent_prevs.min(),
        mixture_model_prevs.min(),
        matching / total,
    ]) * 100) / 100
    total_max = math.ceil(np.max([
        independent_prevs.max(),
        mixture_model_prevs.max(),
        matching / total,
    ]) * 100) / 100

    plt.rcParams.update(figsizes.icml2022_half())
    plt.rcParams.update(fontsizes.icml2022())
    fig, ax = plt.subplots()

    _, bins, _ = ax.hist(
        independent_prevs,
        bins=20,
        range=(total_min, total_max),
        density=True,
        histtype="stepfilled",
        label=f"independent {location} model",
        color=USZ["blue"],
        alpha=0.8,
    )
    ax.hist(
        mixture_model_prevs,
        bins=bins,
        density=True,
        histtype="stepfilled",
        label=f"mixture model for subsite {args.subsite}",
        color=USZ["orange"],
        alpha=0.8,
    )
    ax.axvline(
        matching / total,
        color=USZ["red"],
        linestyle="--",
        label=f"observed ({matching} of {total})",
    )
    ax.legend()
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0%}"))
    ax.set_yticks([])

    plt.savefig(args.output, bbox_inches="tight", dpi=300)

    print(f"Observed prevalence:    {matching / total:.2%}")
    print(f"Independent prevalence: {independent_prevs.mean():.2%}")
    print(f"Mixture prevalence:     {mixture_model_prevs.mean():.2%}")


if __name__ == "__main__":
    main()
