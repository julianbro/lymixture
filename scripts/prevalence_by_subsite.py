"""
Make a bar plot to visualize the prevalence of involvement in each of the four
considered lymph node levels (LNLs), stratified by subsite (different bras) and by
location (different colors).
"""
import argparse
from pathlib import Path
import yaml

import matplotlib.pyplot as plt
from tueplots import figsizes, fontsizes
from lyscripts.utils import load_patient_data

from helpers import (
    simplify_subsite,
    generate_location_colors,
    SUBSITE,
    LNL_I,
    LNL_II,
    LNL_III,
    LNL_IV,
)


def create_parser() -> argparse.ArgumentParser:
    """Assemble the parser for the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input", type=Path, default="data/enhanced.csv",
        help="Path to the patient data file.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default="figures/prevalence_by_subsite.png",
        help="Path to the output file.",
    )
    parser.add_argument(
        "-p", "--params", type=Path, default="_variables.yml",
        help="Path to the parameter file. Looks for a key `barplot_kwargs`.",
    )
    return parser


def main():
    """Load data and create bar plot from it."""
    args = create_parser().parse_args()
    barplot_kwargs = {}
    if args.params.exists():
        with open(args.params) as f:
            params = yaml.safe_load(f)
        barplot_kwargs.update(params.get("barplot_kwargs", {}))
        icd_code_map = params.get("icd_code_map", {})
        icd_code_map = {k: f"{v} ({k})" for k, v in icd_code_map.items()}

    patient_data = load_patient_data(args.input)
    patient_data[SUBSITE] = patient_data[SUBSITE].apply(simplify_subsite)
    pivot_table = patient_data.pivot_table(
        index=[SUBSITE],
        values=[LNL_I, LNL_II, LNL_III, LNL_IV]
    ).sort_values(by=LNL_II)
    colors = list(generate_location_colors(pivot_table.index))
    pivot_table.index = pivot_table.index.map(icd_code_map)

    plt.rcParams.update(figsizes.icml2022_half())
    plt.rcParams.update(fontsizes.icml2022())
    fig, ax = plt.subplots()
    ax = (100 * pivot_table.T).plot.bar(
        ax=ax,
        color=colors,
        **barplot_kwargs,
    )

    ax.grid(axis="y")
    ax.set_xlabel("Lymph node level (ipsilateral)")
    ax.set_xticklabels(["I", "II", "III", "IV"], rotation=0)
    ax.set_ylabel("Prevalence [%]")
    ax.legend(fontsize="x-small", labelspacing=0.3)
    ax.set_xlim(-0.46, 3.46)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
