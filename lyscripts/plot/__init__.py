"""
Provide various plotting utilities for displaying results of e.g. the inference
or prediction process. At the moment, three subcommands are grouped under
`lyscripts.plot`:

1. `lyscripts.plot.corner`, which simply outputs a corner plot with nice labels for
a set of drawn samples.
2. The module `lyscripts.plot.histograms` can be used to draw histograms, e.g. the ones
over risks and prevalences as computed by the `lyscripts.predict` module.
3. Module `lyscripts.plot.thermo_int` allows comparing rounds of thermodynamic
integration for different models.
"""
import argparse
from pathlib import Path

from . import corner, histograms, thermo_int


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """
    Add an `ArgumentParser` to the subparsers action and then add more subparsers.
    """
    parser = subparsers.add_parser(
        Path(__file__).parent.name,
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    subparsers = parser.add_subparsers()
    corner._add_parser(subparsers, help_formatter=parser.formatter_class)
    histograms._add_parser(subparsers, help_formatter=parser.formatter_class)
    thermo_int._add_parser(subparsers, help_formatter=parser.formatter_class)
