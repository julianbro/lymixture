"""
Plot how the accuracy develops over the course of a thermodynamic integration run.

This can also be used to compare how the accuracy of different models develops during
thermdynamic integration.
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

from lyscripts.plot.utils import (
    COLORS,
    get_size,
    save_figure,
    use_mpl_stylesheet,
)

logger = logging.getLogger(__name__)
LINE_CYCLER = cycler(linestyle=["-", "--"]) * cycler(color=list(COLORS.values()))


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """
    Add an `ArgumentParser` to the subparsers action.
    """
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """
    Add arguments needed to run this script to a `subparsers` instance
    and run the respective main function when chosen.
    """
    parser.add_argument(
        "inputs", type=Path, nargs="+",
        help="Paths to the CSV files containing the stored TI runs"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-o", "--output", type=Path,
        help="Path to where the plot should be stored (PNG and SVG)"
    )
    group.add_argument(
        "--show", action="store_true",
        help="Show the plot instead of saving it"
    )

    parser.add_argument(
        "--title", default=None,
        help="Title of the plot"
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=[],
        help="Labels for the individual data series"
    )
    parser.add_argument(
        "--power", default=5., type=float,
        help="Scale the x-axis with this power"
    )
    parser.add_argument(
        "--mplstyle", default="./.mplstyle", type=Path,
        help="Path to the MPL stylesheet"
    )

    parser.set_defaults(run_main=main)


def main(args: argparse.Namespace):
    """
    Load the CSV files where the sampling processes stored the accuracies during
    the thermodynamic integration [^1] processes and plot them against the inverse
    temparature to get a visual idea of how the evidence developed.

    The function's help page shows this:

    ```
    USAGE: lyscripts plot thermo_int [-h] [-o OUTPUT | --show] [--title TITLE]
                                     [--labels LABELS [LABELS ...]] [--power POWER]
                                     [--mplstyle MPLSTYLE]
                                     inputs [inputs ...]

    Plot how the accuracy develops over the course of a thermodynamic integration run.

    This can also be used to compare how the accuracy of different models develops
    during thermdynamic integration.

    POSITIONAL ARGUMENTS:
        inputs                Paths to the CSV files containing the stored TI runs

    OPTIONAL ARGUMENTS:
        -h, --help            show this help message and exit
        -o, --output OUTPUT   Path to where the plot should be stored (PNG and SVG)
                              (default: None)
        --show                Show the plot instead of saving it (default: False)
        --title TITLE         Title of the plot (default: None)
        --labels LABELS [LABELS ...]
                              Labels for the individual data series (default: [])
        --power POWER         Scale the x-axis with this power (default: 5.0)
        --mplstyle MPLSTYLE   Path to the MPL stylesheet (default: ./.mplstyle)
    ```

    [^1]: https://doi.org/10.1007/s11571-021-09696-9
    """
    use_mpl_stylesheet(args.mplstyle)

    accuracy_series = []
    min_acc = np.inf
    max_acc = -np.inf
    for input in args.inputs:
        tmp = pd.read_csv(input)
        min_acc = np.min([min_acc, *tmp["accuracy"]])
        max_acc = np.max([max_acc, *tmp["accuracy"]])
        accuracy_series.append(tmp)
        logger.info(f"+ read in {input}")
    logger.info("Loaded CSV file(s)")


    fig, ax = plt.subplots(figsize=get_size())
    if args.title is not None:
        fig.suptitle(args.title)

    ax.set_xlabel("inverse temperature $\\beta$")
    xticks = np.linspace(0., 1., 7)
    xticklabels = [f"{x**args.power:.2g}" for x in xticks]
    ax.set_xticks(ticks=xticks, labels=xticklabels)
    ax.set_xlim(left=0., right=1.)

    ax.set_ylabel("accuracy $\\mathcal{A}(\\beta)$")
    ax.set_yscale("symlog")
    ax.get_yaxis().set_major_locator(matplotlib.ticker.MultipleLocator(800))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.ticklabel_format(axis="y", style="sci", scilimits=(2,2))
    logger.info("Prepared figure")


    for i, series in enumerate(accuracy_series):
        last_acc = series['accuracy'].values[-1]
        try:
            label = args.labels[i] + " $\\mathcal{A}(1)$ = " + f"{last_acc:g}"
        except IndexError:
            label = None

        if "stddev" in series:
            ax.errorbar(
                series["β"]**(1./args.power),
                series["accuracy"],
                yerr=series["stddev"],
                label=label,
            )
        else:
            ax.plot(
                series["β"]**(1./args.power),
                series["accuracy"],
                label=label,
            )

    if len(args.labels) > 0:
        ax.legend()
    logger.info("Plotted series")


    if args.show:
        plt.show()
        logger.info("Showed the plot")
    else:
        save_figure(fig, args.output, formats=["png", "svg"], logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
