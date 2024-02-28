"""
Plot selected prevalence predictions into one, nice figure.
"""
import argparse
from pathlib import Path
import yaml
from cycler import cycler

import numpy as np
import matplotlib.pyplot as plt
from tueplots import figsizes, fontsizes
from emcee.backends import HDFBackend
from lyscripts.utils import load_patient_data, create_model_from_config
from lyscripts.predict.prevalences import (
    compute_observed_prevalence,
    generate_predicted_prevalences,
)
from lyscripts.plot.utils import COLORS as USZ
from lyscripts.plot.utils import Histogram, draw

from helpers import (
    SUBSITE,
    T_STAGE,
    LNLS,
    get_indie_chain,
    get_location,
    get_mixture_components,
    get_prevalence_pattern,
)


COLOR_CYCLER = cycler(color=[USZ["green"], USZ["blue"], USZ["orange"], USZ["red"]])


def create_parser() -> argparse.ArgumentParser:
    """Assemble the parser for the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d", "--data", type=Path, default="data/enhanced.csv",
        help="Path to the data file.",
    )
    parser.add_argument(
        "--models", type=Path, default="models",
        help=(
            "Path to the directory containing the HDF5 files of the trained models. "
            "The directory is assumed to contain `mixture.hdf5`, `oropharynx.hdf5`, "
            "and `oral_cavity.hdf5`."
        )
    )
    parser.add_argument(
        "-o", "--output", type=Path, default="figures/prevalence_superplot.png",
        help="Path to the output file.",
    )
    parser.add_argument(
        "-p", "--params", type=Path, default="_variables.yml",
        help="Path to the parameter file..",
    )
    parser.add_argument(
        "-l", "--lnls", nargs="+", type=str, default=["I", "II", "III"],
        help="LNLs to plot involvement prevalences for.",
    )
    parser.add_argument(
        "-s", "--subsites", nargs="+", type=str, default=["C01", "C09", "C05"],
        help=(
            "ICD codes of the subsites to plot the prevalences for. Each subsite will "
            "be plotted in a separate row."
        ),
    )
    parser.add_argument(
        "--thin-indie", type=int, default=2,
        help="Thinning factor for the independent model's chain.",
    )
    parser.add_argument(
        "--thin-mixture", type=int, default=100,
        help="Thinning factor for the mixture model's chain.",
    )
    return parser


def main():
    args = create_parser().parse_args()

    with open(args.params) as file:
        params = yaml.safe_load(file)

    lymph_model = create_model_from_config(params)
    lymph_model.modalities = {"max_llh": [1., 1.]}

    patient_data = load_patient_data(args.data)
    patient_data[T_STAGE] = ["all"] * len(patient_data)

    mixture_chain = HDFBackend(
        args.models / "mixture.hdf5",
        read_only=True,
    ).get_chain(
        flat=True,
        thin=args.thin_mixture,
        discard=1000,
    )

    plt.rcParams.update(figsizes.icml2022_full(
        nrows=len(args.subsites),
        ncols=len(args.lnls)
    ))
    plt.rcParams.update(fontsizes.icml2022())
    _fig, axes = plt.subplots(
        nrows=len(args.subsites),
        ncols=len(args.lnls),
    )

    for row, subsite in zip(axes, args.subsites):
        location = get_location(for_subsite=subsite)
        indie_chain = get_indie_chain(
            from_dir=args.models,
            for_location=location,
            thin_by=args.thin_indie,
        )
        row[0].set_yticks([])
        row[0].set_ylabel(f"{location}: {subsite}")
        for ax, cycler, lnl in zip(row, COLOR_CYCLER, args.lnls):
            histograms = []
            pattern = get_prevalence_pattern(for_lnl=lnl)
            matching, total = compute_observed_prevalence(
                pattern=pattern,
                data=patient_data.loc[patient_data[SUBSITE].str.contains(subsite)],
                lnls=LNLS,
                t_stage="all",
                modality="max_llh",
                invert=False,
            )
            indie_prevs = np.array(list(generate_predicted_prevalences(
                pattern=pattern,
                model=lymph_model,
                samples=indie_chain,
                t_stage="all",
                modality_spsn=[1., 1.],
            )))
            _A_prevs = np.array(list(generate_predicted_prevalences(
                pattern=pattern,
                model=lymph_model,
                samples=mixture_chain[:,:7],
                t_stage="all",
                modality_spsn=[1., 1.],
            )))
            _B_prevs = np.array(list(generate_predicted_prevalences(
                pattern=pattern,
                model=lymph_model,
                samples=mixture_chain[:,7:],
                t_stage="all",
                modality_spsn=[1., 1.],
            )))
            comp_A_prob, comp_B_prob = get_mixture_components(
                from_dir=args.models,
                for_subsite=subsite,
                icd_code_map=params["icd_code_map"],
            )
            mixture_prevs = (
                comp_A_prob * _A_prevs
                + comp_B_prob * _B_prevs
            )
            histograms.append(Histogram(
                values=indie_prevs,
                kwargs={
                    "color": cycler["color"],
                    "label": "independent model",
                    "histtype": "step",
                    "linewidth": 2,
                }
            ))
            histograms.append(Histogram(
                values=mixture_prevs,
                kwargs={
                    "color": cycler["color"],
                    "label": "mixture model",
                    "histtype": "stepfilled",
                    "alpha": 0.5,
                }
            ))
            ax.axvline(
                100 * matching / total,
                color=cycler["color"],
                linestyle="--",
                linewidth=2.5,
                label=f"observed ({matching} / {total})",
            )
            draw(
                axes=ax,
                contents=histograms,
                percent_lims=(0, 0),
                hist_kwargs={"nbins": 20},
            )
            ax.set_yticks([])
            y_lim = ax.get_ylim()
            ax.set_ylim(0, 1.2 * y_lim[1])
            ax.legend(fontsize="x-small", loc="upper right", labelspacing=0.2)

    for ax, lnl in zip(axes[0], args.lnls):
        ax.set_title(f"ipsi LNL {lnl}")

    for ax in axes[-1]:
        ax.set_xlabel("prevalence [%]")

    plt.savefig(args.output, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
