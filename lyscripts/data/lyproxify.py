"""
Consumes raw data and transforms it into a CSV of the format that [LyProX] understands.

To do so, it needs a dictionary that defines a mapping from raw columns to the LyProX
style data format. See the documentation of the `transform_to_lyprox` function for
more information.

[LyProX]: https://lyprox.org
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import importlib.util
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from lyscripts.data.utils import load_csv_table, save_table_to_csv
from lyscripts.decorators import log_state
from lyscripts.utils import delete_private_keys, flatten

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
        "-i", "--input", type=Path, required=True,
        help="Location of raw CSV data."
    )
    parser.add_argument(
        "-r", "--header-rows", nargs="+", default=[0], type=int,
        help="List with header row indices of raw file."
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Location to store the lyproxified CSV file."
    )
    parser.add_argument(
        "-m", "--mapping", type=Path, required=True,
        help=(
            "Location of the Python file that contains column mapping instructions. "
            "This must contain a dictionary with the name 'column_map'."
        )
    )
    parser.add_argument(
        "--drop-rows", nargs="+", type=int, default=[],
        help=(
            "Delete rows of specified indices. Counting of rows start at 0 _after_ "
            "the `header-rows`."
        )
    )
    parser.add_argument(
        "--drop-cols", nargs="+", type=int, default=[],
        help="Delete columns of specified indices.",
    )
    parser.add_argument(
        "--add-index", action="store_true",
        help="If the data doesn't contain an index, add one by enumerating the patients"
    )

    parser.set_defaults(run_main=main)


class ParsingError(Exception):
    """Error while parsing the CSV file."""


def clean_header(
    table: pd.DataFrame,
    num_cols: int,
    num_header_rows: int,
) -> pd.DataFrame:
    """Rename the header cells in the `table`."""
    for col in range(num_cols):
        for row in range(num_header_rows):
            table.rename(
                columns={f"Unnamed: {col}_level_{row}": f"{col}_lvl_{row}"},
                inplace=True,
            )
    return table


def get_instruction_depth(nested_column_map: Dict[Tuple, Dict[str, Any]]) -> int:
    """
    Get the depth at which the column mapping instructions are nested.

    Instructions are a dictionary that contains either a 'func' or 'default' key.

    Example:
    >>> nested_column_map = {"patient": {"age": {"func": int}}}
    >>> get_instruction_depth(nested_column_map)
    2
    >>> flat_column_map = flatten(nested_column_map, max_depth=2)
    >>> get_instruction_depth(flat_column_map)
    1
    >>> nested_column_map = {"patient": {"__doc__": "some patient info", "age": 61}}
    >>> get_instruction_depth(nested_column_map)
    Traceback (most recent call last):
        ...
    ValueError: Leaf of column map must be a dictionary with 'func' or 'default' key.
    """
    for _, value in nested_column_map.items():
        if isinstance(value, dict):
            if "func" in value or "default" in value:
                return 1

            return 1 + get_instruction_depth(value)

        raise ValueError(
            "Leaf of column map must be a dictionary with 'func' or 'default' key."
        )


def generate_markdown_docs(
    nested_column_map: Dict[Tuple, Dict[str, Any]],
    depth: int = 0,
    indent_len: int = 4,
) -> str:
    """
    Generate a markdown nested, ordered list as documentation for the column map.

    A key in the doctionary is supposed to be documented, when its value is a dictionary
    containing a `"__doc__"` key.

    Example:
    >>> nested_column_map = {
    ...     "patient": {
    ...         "__doc__": "some patient info",
    ...         "age": {
    ...             "__doc__": "age of the patient",
    ...             "func": int,
    ...             "columns": ["age"],
    ...         },
    ...     },
    ... }
    >>> generate_markdown_docs(nested_column_map)
    '1. **`patient:`** some patient info\\n    1. **`age:`** age of the patient\\n'
    """
    md_docs = ""
    indent = " " * indent_len * depth
    i = 1
    for key, value in nested_column_map.items():
        if isinstance(value, dict):
            if "__doc__" in value:
                md_docs += f"{indent}{i}. **`{key}:`** {value['__doc__']}\n"
                i += 1

            md_docs += generate_markdown_docs(value, depth + 1, indent_len)

    return md_docs


@log_state(
    success_msg="Transformed raw data to LyProX style table",
    logger=logger,
)
def transform_to_lyprox(
    raw: pd.DataFrame,
    column_map: Dict[Tuple, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Transform `raw` data frame into table that can be uploaded directly to [LyProX].

    To do so, it uses instructions in the `colum_map` dictionary, that needs to have
    a particular structure:

    For each column in the final 'lyproxified' `pd.DataFrame`, one entry must exist in
    the `column_map` dctionary. E.g., for the column corresponding to a patient's age,
    the dictionary should contain a key-value pair of this shape:

    ```python
    column_map = {
        ("patient", "#", "age"): {
            "func": compute_age_from_raw,
            "kwargs": {"randomize": False},
            "columns": ["birthday", "date of diagnosis"]
        },
    }
    ```

    In this example, the function `compute_age_from_raw` is called with the values of
    the columns `birthday` and `date of diagnosis` as positional arguments, and the
    keyword argument `randomize` is set to `False`. The function then returns the
    patient's age, which is subsequently stored in the column `("patient", "#", "age")`.

    Note that the `column_map` dictionary must have either a `default` key or `func`
    along with `columns` and `kwargs`, depending on the function definition. If the
    function does not take any arguments, `columns` can be omitted. If it also does
    not take any keyword arguments, `kwargs` can be omitted, too.

    [LyProX]: https://lyprox.org
    """
    column_map = delete_private_keys(column_map)

    if (instruction_depth := get_instruction_depth(column_map)) > 1:
        column_map = flatten(column_map, max_depth=instruction_depth)

    multi_idx = pd.MultiIndex.from_tuples(column_map.keys())
    processed = pd.DataFrame(columns=multi_idx)

    for multi_idx_col, instruction in column_map.items():
        if instruction != "":
            if "default" in instruction:
                processed[multi_idx_col] = [instruction["default"]] * len(raw)
            elif "func" in instruction:
                cols = instruction.get("columns", [])
                kwargs = instruction.get("kwargs", {})
                func = instruction["func"]

                try:
                    processed[multi_idx_col] = [
                        func(*vals, **kwargs) for vals in raw[cols].values
                    ]
                except Exception as exc:
                    raise ParsingError(
                        f"Exception encountered while parsing column {multi_idx_col}"
                    ) from exc
            else:
                raise ParsingError(
                    f"Column {multi_idx_col} has neither a `default` value nor `func` "
                    "describing how to fill this column."
                )
    return processed


@log_state(
    success_msg="Transformed absolute side reporting to tumor-relative",
    logger=logger,
)
def leftright_to_ipsicontra(data: pd.DataFrame):
    """
    Change absolute side reporting to tumor-relative.

    Transform reporting of LNL involvement by absolute side (right & left) to a
    reporting relative to the tumor (ipsi- & contralateral). The table `data` should
    already be in the format LyProX requires, except for the side-reporting of LNL
    involvement.
    """
    len_before = len(data)
    left_data = data.loc[
                data["tumor", "1", "side"] != "right"
            ]
    right_data = data.loc[
                data["tumor", "1", "side"] == "right"
            ]

    left_data = left_data.rename(columns={"left": "ipsi"}, level=1)
    left_data = left_data.rename(columns={"right": "contra"}, level=1)
    right_data = right_data.rename(columns={"left": "contra"}, level=1)
    right_data = right_data.rename(columns={"right": "ipsi"}, level=1)

    data = pd.concat(
                [left_data, right_data], ignore_index=True
            )
    assert len_before == len(data), "Number of patients changed"
    return data


@log_state(
    success_msg="Excluded patients based on provided criteria",
    logger=logger,
)
def exclude_patients(raw: pd.DataFrame, exclude: List[Tuple[str, Any]]):
    """
    Exclude patients in the `raw` data based on a list of what to `exclude`. This
    list contains tuples `(column, check)`. The `check` function will then exclude
    any patients from the cohort where `check(raw[column])` evaluates to `True`.

    Example:
    >>> exclude = [("age", lambda s: s > 50)]
    >>> table = pd.DataFrame({
    ...     "age":        [43, 82, 18, 67],
    ...     "T-category": [ 3,  4,  2,  1],
    ... })
    >>> exclude_patients(table, exclude)
       age  T-category
    0   43           3
    2   18           2
    """
    for column, check in exclude:
        exclude = check(raw[column])
        raw = raw.loc[~exclude]
    return raw


def main(args: argparse.Namespace):
    """
    The main entry point for the CLI of this command. Upon requesting `lyscripts
    data lyproxify --help`, this is the help output:

    ```
    USAGE: lyscripts data lyproxify [-h] -i INPUT [-r HEADER_ROWS [HEADER_ROWS ...]]
                                    -o OUTPUT -m MAPPING
                                    [--drop-rows DROP_ROWS [DROP_ROWS ...]]
                                    [--drop-cols DROP_COLS [DROP_COLS ...]]
                                    [--add-index]

    Consumes raw data and transforms it into a CSV of the format that LyProX can
    understand.

    To do so, it needs a dictionary that defines a mapping from raw columns to the
    LyProX style data format. See the documentation of the `transform_to_lyprox`
    function for more information.

    OPTIONAL ARGUMENTS:
      -h, --help            show this help message and exit
      -i, --input INPUT     Location of raw CSV data. (default: None)
      -r, --header-rows HEADER_ROWS [HEADER_ROWS ...]
                            List with header row indices of raw file. (default: [0])
      -o, --output OUTPUT   Location to store the lyproxified CSV file. (default:
                            None)
      -m, --mapping MAPPING
                            Location of the Python file that contains column mapping
                            instructions. This must contain a dictionary with the name
                            'column_map'. (default: None)
      --drop-rows DROP_ROWS [DROP_ROWS ...]
                            Delete rows of specified indices. Counting of rows start
                            at 0 _after_ the `header-rows`. (default: [])
      --drop-cols DROP_COLS [DROP_COLS ...]
                            Delete columns of specified indices. (default: [])
      --add-index           If the data doesn't contain an index, add one by
                            enumerating the patients (default: False)
    ```
    """
    raw: pd.DataFrame = load_csv_table(args.input, header_row=args.header_rows, logger=logger)
    raw = clean_header(raw, num_cols=raw.shape[1], num_header_rows=len(args.header_rows))

    cols_to_drop = raw.columns[args.drop_cols]
    trimmed = raw.drop(cols_to_drop, axis="columns")
    trimmed = trimmed.drop(index=args.drop_rows)
    trimmed = trimmed.dropna(axis="index", how="all")
    logger.info(f"Dropped rows {args.drop_rows} and columns {cols_to_drop}.")

    spec = importlib.util.spec_from_file_location("map_module", args.mapping)
    mapping = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapping)
    logger.info(f"Imported mapping instructions from {args.mapping}")

    reduced = exclude_patients(trimmed, mapping.EXCLUDE)

    if args.add_index:
        reduced.insert(0, ("patient", "#", "id"), list(range(len(reduced))))
        logger.info("Added index column to data.")

    processed = transform_to_lyprox(reduced, mapping.COLUMN_MAP)

    if ("tumor", "1", "side") in processed.columns:
        processed = leftright_to_ipsicontra(processed)

    save_table_to_csv(args.output, processed, logger=logger)
