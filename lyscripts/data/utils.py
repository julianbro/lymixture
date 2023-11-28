"""
Utilities related to the commands for data cleaning and processing.
"""
from pathlib import Path
from typing import List

import pandas as pd

from lyscripts.decorators import (
    check_input_file_exists,
    check_output_dir_exists,
    log_state,
)


@log_state(success_msg="Saved processed CSV file")
@check_output_dir_exists
def save_table_to_csv(output_path: Path, table: pd.DataFrame):
    """Save a `pd.DataFrame` to `output_path`."""
    table.to_csv(output_path, index=None)


@log_state(success_msg="Loaded input CSV file")
@check_input_file_exists
def load_csv_table(input_path: Path, header_row: List[int]) -> pd.DataFrame:
    """
    Load a CSV table from `input_path` into a `pd.DataFrame` where the list `header`
    defines which rows make up the column names.
    """
    return pd.read_csv(input_path, header=header_row)
