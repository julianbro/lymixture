"""
Split the full dataset into cross-validation folds according to the
content of the params.yaml file.
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lyscripts.utils import load_yaml_params

warnings.simplefilter(action="ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


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
        "input", type=Path,
        help="The path to the full dataset to split."
    )
    parser.add_argument(
        "output", type=Path,
        help="Folder to store the split CSV files in."
    )

    parser.add_argument(
        "-p", "--params", default="./params.yaml", type=Path,
        help="Path to parameter YAML file."
    )

    parser.set_defaults(run_main=main)


def main(args: argparse.Namespace):
    """
    The output from `lyscripts split --help` looks like this:

    ```
    USAGE: lyscripts data split [-h] [-p PARAMS] input output

    Split the full dataset into cross-validation folds according to the content of the
    params.yaml file.

    POSITIONAL ARGUMENTS:
      input                 The path to the full dataset to split.
      output                Folder to store the split CSV files in.

    OPTIONAL ARGUMENTS:
      -h, --help            show this help message and exit
      -p, --params PARAMS   Path to parameter YAML file. (default: ./params.yaml)
    ```
    """
    params = load_yaml_params(args.params)

    header = [0,1] if params["model"]["class"] == "Unilateral" else [0,1,2]
    concatenated_df = pd.read_csv(args.input, header=header)
    logger.info(f"Read in concatenated CSV file from {args.input}")

    args.output.mkdir(exist_ok=True)
    shuffled_df = concatenated_df.sample(
        frac=1.,
        replace=False,
        random_state=params["cross-validation"]["seed"]
    ).reset_index(drop=True)

    split_dfs = np.array_split(
        shuffled_df,
        len(params["cross-validation"]["folds"])
    )
    for fold_id, split_pattern in params["cross-validation"]["folds"].items():
        # Concatenate training and evaluation DataFrames from the split DataFrames
        # using the split pattern defined in the params file.
        eval_df = pd.concat(
            [split_dfs[k] for k,sym in enumerate(split_pattern) if sym == "e"],
            ignore_index=True
        )
        train_df = pd.concat(
            [split_dfs[k] for k,sym in enumerate(split_pattern) if sym == "t"],
            ignore_index=True
        )

        eval_df.to_csv(args.output / f"{fold_id}_eval.csv", index=None)
        train_df.to_csv(args.output / f"{fold_id}_train.csv", index=None)
        logger.info(f"+ split data into train & eval sets for round {fold_id}")

    logger.info(
        "Split data into train & eval sets for all "
        f"{len(params['cross-validation']['folds'])} folds"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
