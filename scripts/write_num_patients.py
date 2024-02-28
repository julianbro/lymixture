"""
Write the number of patients in the data CSV file into the `_variables.yml` file.
"""
import argparse
from pathlib import Path
import yaml

from lyscripts.utils import load_patient_data

from helpers import SUBSITE


def create_parser() -> argparse.ArgumentParser:
    """Assemble the parser for the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input", type=Path, default="data/enhanced.csv",
        help="Path to the patient data file.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default="_variables.yml",
        help="Path to the parameter file.",
    )
    return parser


def main():
    """Load data and create bar plot from it."""
    args = create_parser().parse_args()
    with open(args.output, "r") as file:
        variables = yaml.safe_load(file)

    patient_data = load_patient_data(args.input)
    icd_code_map = variables.get("icd_code_map", {})
    patient_info = variables.get("num_patients", {})
    patient_info["total"] = len(patient_data)

    for icd_code in icd_code_map.keys():
        num_icd_code = int(patient_data[SUBSITE].str.contains(icd_code).sum())
        patient_info[icd_code] = num_icd_code

    variables.update({"num_patients": patient_info})

    with open(args.output, "w") as file:
        yaml.dump(variables, file)


if __name__ == "__main__":
    main()
