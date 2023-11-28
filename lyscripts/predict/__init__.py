"""
This module provides functions and scripts to predict the risk of hidden involvement,
given observed diagnoses, and prevalences of patterns for diagnostic modalities.

The submodules for prediction are currently:

1. The `lyscripts.predict.prevalences` module for computing prevalences of certain
involvement patterns that may also be compared to observed prevalences.
2. A module `lyscripts.predict.risks` for predicting the risk of any specific pattern
of involvement given any particular diagnosis.
"""
import argparse
from pathlib import Path

from . import prevalences, risks


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
    prevalences._add_parser(subparsers, help_formatter=parser.formatter_class)
    risks._add_parser(subparsers, help_formatter=parser.formatter_class)
