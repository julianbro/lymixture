"""
Visualize the component assignments of the trained mixture model.
"""
import argparse
from pathlib import Path
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import yaml

import h5py
import matplotlib.pyplot as plt
from tueplots import figsizes, fontsizes
from lyscripts.plot.utils import COLORS as USZ

from helpers import generate_location_colors


def create_parser() -> argparse.ArgumentParser:
    """Assemble the parser for the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m", "--model", type=Path, default="models/mixture.hdf5",
        help=(
            "Path to the model HDF5 file. Needs to contain a dataset called "
            "``em/cluster_assignments``."
        )
    )
    parser.add_argument(
        "-o", "--output", type=Path, default="figures/cluster_assignments.png",
        help="Path to the output file.",
    )
    parser.add_argument(
        "-p", "--params", type=Path, default="_variables.yml",
        help="Path to the parameter file..",
    )
    return parser


def main():
    """Execute the main routine."""
    args = create_parser().parse_args()
    with open(args.params) as file:
        params = yaml.safe_load(file)
        num_patients = params["num_patients"]
        num_patients.pop("total")

    with h5py.File(args.model, mode="r") as h5_file:
        cluster_components = h5_file["em/cluster_components"][...]

    plt.rcParams.update(figsizes.icml2022_half())
    plt.rcParams.update(fontsizes.icml2022())

    _, bottom_ax = plt.subplots()
    cluster_x = [cluster_components[i] for i, _ in enumerate(num_patients.keys())]
    cluster_y = [0. for _ in num_patients.keys()]
    annotations = [f"{label}\n({num})" for label, num in num_patients.items()]
    bottom_ax.scatter(
        cluster_x, cluster_y,
        s=[num for num in num_patients.values()],
        c=list(generate_location_colors(num_patients.keys())),
        alpha=0.7,
        linewidths=0.,
        zorder=10,
    )

    sorted_idx = cluster_components.argsort()
    sorted_x = cluster_components[sorted_idx]
    sorted_annotations = [annotations[i] for i in sorted_idx]
    sorted_num = [list(num_patients.values())[i] for i in sorted_idx]
    for i, (x, num, annotation) in enumerate(zip(sorted_x, sorted_num, sorted_annotations)):
        bottom_ax.annotate(
            annotation,
            # sqrt, because marker's area grows linearly with patient num, not radius
            xy=(x, np.sqrt(0.0000003 * num) * (- 1)**i),
            xytext=(x, 0.025 * (- 1)**i),
            ha="center",
            va="bottom" if i % 2 == 0 else "top",
            fontsize="small",
            arrowprops={
                "arrowstyle": "-",
                "color": USZ["gray"],
                "linewidth": 1.,
            }
        )

    bottom_ax.set_xlabel("assignment to component A")
    bottom_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0%}"))
    top_ax = bottom_ax.secondary_xaxis(
        location="top",
        functions=(lambda x: 1. - x, lambda x: 1. - x),
    )
    top_ax.set_xlabel("assignment to component B")
    top_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0%}"))
    bottom_ax.set_yticks([])
    bottom_ax.grid(axis="x", alpha=0.5, color=USZ["gray"], linestyle=":")
    plt.savefig(args.output, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
