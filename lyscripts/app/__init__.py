"""
Module containing scripts to run different `streamlit` applications.
"""
import argparse
from pathlib import Path

from . import prevalence


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
    prevalence._add_parser(subparsers, help_formatter=parser.formatter_class)
