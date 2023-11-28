"""
Evaluate the performance of the trained model by computing quantities like the
Bayesian information criterion (BIC) or (if thermodynamic integration was performed)
the actual evidence (with error) of the model.
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import emcee
import h5py
import lymph
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from lyscripts.utils import (
    create_model_from_config,
    load_data_for_model,
    load_yaml_params,
)

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
        "data", type=Path,
        help="Path to the tables of patient data (CSV)."
    )
    parser.add_argument(
        "model", type=Path,
        help="Path to model output files (HDF5)."
    )

    parser.add_argument(
        "-p", "--params", default="./params.yaml", type=Path,
        help="Path to parameter file"
    )
    parser.add_argument(
        "--plots", default="./plots", type=Path,
        help="Directory for storing plots"
    )
    parser.add_argument(
        "--metrics", default="./metrics.json", type=Path,
        help="Path to metrics file"
    )

    parser.set_defaults(run_main=main)


def comp_bic(
    log_probs: np.ndarray,
    num_params: int,
    num_data: int
) -> float:
    """
    Compute the negative one half of the Bayesian Information Criterion (BIC).

    The BIC is defined as [^1]
    $$ BIC = k \\ln{n} - 2 \\ln{\\hat{L}} $$
    where $k$ is the number of parameters `num_params`, $n$ the number of datapoints
    `num_data` and $\\hat{L}$ the maximum likelihood estimate of the `log_prob`.
    It is constructed such that the following is an
    approximation of the model evidence:
    $$ p(D \\mid m) \\approx \\exp{\\left( - BIC / 2 \\right)} $$
    which is why this function returns the negative one half of it.

    [^1]: https://en.wikipedia.org/wiki/Bayesian_information_criterion
    """
    return np.max(log_probs) - num_params * np.log(num_data) / 2.

def compute_evidence(
    temp_schedule: np.ndarray,
    log_probs: np.ndarray,
    num: int = 1000,
) -> Tuple[float, float]:
    """Compute the evidene and its standard deviation.

    Given a `temp_schedule` of inverse temperatures and corresponding sets of
    `log_probs`, draw `num` "paths" of log-probabilities and compute the evidence for
    each using trapezoidal integration.

    The evidence is then the mean of those `num` integrations, while the error is their
    standard deviation.
    """
    integrals = np.zeros(shape=num)
    for i in range(num):
        rand_idx = np.random.choice(log_probs.shape[1], size=log_probs.shape[0])
        drawn_accuracy = log_probs[np.arange(log_probs.shape[0]),rand_idx].copy()
        integrals[i] = trapezoid(y=drawn_accuracy, x=temp_schedule)
    return np.mean(integrals), np.std(integrals)


def compute_ti_results(
    metrics: dict,
    params: dict,
    ndim: int,
    h5_file: Path,
    model: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the results in case of a thermodynamic integration run."""
    temp_schedule = params["sampling"]["temp_schedule"]
    num_temps = len(temp_schedule)

    if num_temps != len(h5_file["ti"]):
        raise RuntimeError(
            f"Parameters suggest temp schedule of length {num_temps}, "
            f"but stored are {len(h5_file['ti'])}"
        )

    nwalker = ndim * params["sampling"]["walkers_per_dim"]
    nsteps = params["sampling"]["nsteps"]
    ti_log_probs = np.zeros(shape=(num_temps, nsteps * nwalker))

    for i, round in enumerate(h5_file["ti"]):
        reader = emcee.backends.HDFBackend(model, name=f"ti/{round}", read_only=True)
        ti_log_probs[i] = reader.get_blobs(flat=True)

    evidence, evidence_std = compute_evidence(temp_schedule, ti_log_probs)
    metrics["evidence"] = evidence
    metrics["evidence_std"] = evidence_std

    return temp_schedule, ti_log_probs


def main(args: argparse.Namespace):
    """
    To evaluate the performance of a sampling round, this program follows these steps:

    1. Read in the paramter file `params.yaml` and the data that was used for inference
    2. Recreate an instance of the model that was used during the training stage
    3. If thermodynamic integration (TI) [^2] was performed, compute the accuracy for
    every TI step and integrate over it to obtain the evidence. This is computed for
    a number of samples of the computed accuracies to also obtain an error on the
    evidence.
    4. Use the recreated model and data to compute quantities like the mean and maximum
    likelihood, as well as the Bayesian information criterion (BIC).

    Everything computed is then stored in a `metrics.json` file and for TI, a CSV is
    created that shows how the accuracy developed during the runs.

    The output of running `lyscripts evaluate --help` shows this:

    ```
    usage: lyscripts evaluate [-h] [-p PARAMS] [--plots PLOTS] [--metrics METRICS]
                            data model

    Evaluate the performance of the trained model by computing quantities like the
    Bayesian information criterion (BIC) or (if thermodynamic integration was performed)
    the actual evidence (with error) of the model.


    POSITIONAL ARGUMENTS
    data                 Path to the tables of patient data (CSV).
    model                Path to model output files (HDF5).

    OPTIONAL ARGUMENTS
    -h, --help           show this help message and exit
    -p, --params PARAMS  Path to parameter file (default: ./params.yaml)
    --plots PLOTS        Directory for storing plots (default: ./plots)
    --metrics METRICS    Path to metrics file (default: ./metrics.json)
    ```

    [^2]: https://doi.org/10.1007/s11571-021-09696-9
    """
    metrics = {}

    params = load_yaml_params(args.params, logger=logger)
    model = create_model_from_config(params, logger=logger)
    ndim = len(model.spread_probs) + model.diag_time_dists.num_parametric
    is_uni = isinstance(model, lymph.models.Unilateral)

    data = load_data_for_model(args.data, header_rows=[0,1] if is_uni else [0,1,2])
    h5_file = h5py.File(args.model, mode="r")

    # if TI has been performed, compute the accuracy for every step
    if "ti" in h5_file:
        temp_schedule, ti_log_probs = compute_ti_results(
            metrics=metrics,
            params=params,
            ndim=ndim,
            h5_file=h5_file,
            model=args.model,
        )
        logger.info(
            "Computed results of thermodynamic integration with "
            f"{len(temp_schedule)} steps"
        )

        # store inverse temperatures and log-probs in CSV file
        args.plots.parent.mkdir(exist_ok=True)

        beta_vs_accuracy = pd.DataFrame(
            np.array([
                temp_schedule,
                np.mean(ti_log_probs, axis=1),
                np.std(ti_log_probs, axis=1)
            ]).T,
            columns=["β", "accuracy", "std"],
        )
        beta_vs_accuracy.to_csv(args.plots, index=False)
        logger.info(f"Plotted β vs accuracy at {args.plots}")


    # use blobs, because also for TI, this is the unscaled log-prob
    backend = emcee.backends.HDFBackend(args.model, read_only=True, name="mcmc")
    final_log_probs = backend.get_blobs()
    logger.info(f"Opened samples from emcee backend from {args.model}")

    # store metrics in JSON file
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.touch(exist_ok=True)

    metrics["BIC"] = comp_bic(
        final_log_probs, ndim, len(data),
    )
    metrics["max_llh"] = np.max(final_log_probs)
    metrics["mean_llh"] = np.mean(final_log_probs)

    with open(args.metrics, mode="w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file)

    logger.info(f"Wrote out metrics to {args.metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
