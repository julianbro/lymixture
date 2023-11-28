"""
Learn the spread probabilities of the HMM for lymphatic tumor progression using
the preprocessed data as input and MCMC as sampling method.

This is the central script performing for our project on modelling lymphatic spread
in head & neck cancer. We use it for model comparison via the thermodynamic
integration functionality and use the sampled parameter estimates for risk
predictions. This risk estimate may in turn some day guide clinicians to make more
objective decisions with respect to defining the *elective clinical target volume*
(CTV-N) in radiotherapy.
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import logging
import os
import warnings
from multiprocessing import Pool
import multiprocess as mp
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import emcee
import h5py
import numpy as np
import pandas as pd

from lyscripts.utils import (
    CustomProgress,
    get_modalities_subset,
    load_yaml_params,
    model_from_config,
    report,
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
        "input", type=Path,
        help="Path to training data files"
    )
    parser.add_argument(
        "output", type=Path,
        help="Path to the HDF5 file to store the results in"
    )

    parser.add_argument(
        "--params", default="./params.yaml", type=Path,
        help="Path to parameter file"
    )
    parser.add_argument(
        "--modalities", nargs="+",
        default=["max_llh"],
        help="List of modalities for inference. Must be defined in `params.yaml`"
    )
    parser.add_argument(
        "--plots", default="./plots", type=Path,
        help="Directory to store plot of acor times",
    )
    parser.add_argument(
        "--ti", action="store_true",
        help="Perform thermodynamic integration"
    )
    parser.add_argument(
        "--pools", type=int, required=False,
        help=(
            "Number of parallel worker pools (CPU cores) to use. If not provided, it "
            "will use all cores. If set to zero, multiprocessing will not be used."
        )
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed value to reproduce the same sampling round."
    )

    parser.set_defaults(run_main=main)


class DummyPool:
    """
    Dummy class returning `None` instead of a `Pool` instance when the user chose not
    to use multiprocessing.
    """
    _processes = "no parallel"

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class ConvenienceSampler(emcee.EnsembleSampler):
    """Class that adds some useful defaults to the `emcee.EnsembleSampler`."""
    def __init__(
        self,
        nwalkers,
        ndim,
        log_prob_fn,
        pool=None,
        backend=None,
    ):
        """Initialize sampler with sane defaults."""
        moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
        super().__init__(nwalkers, ndim, log_prob_fn, pool, moves, backend=backend)


    def run_sampling(
        self,
        min_steps: int = 0,
        max_steps: int = 10000,
        thin_by: int = 1,
        initial_state: Optional[Union[emcee.State, np.ndarray]] = None,
        check_interval: int = 100,
        trust_threshold: float = 50.,
        rel_acor_threshold: float = 0.05,
        progress_desc: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run a round of sampling with at least `min_steps`, at most `max_steps` and
        only keep every `thin_by` step.

        Every `check_interval` iterations, convergence is checked based on
        three criteria:
        - did it run for at least `min_steps`?
        - has the acor time crossed the N / `trust_threshold`?
        - did the acor time change less than `rel_acor_threshold`?

        If a `progress_desc` is provided, progress will be displayed.

        Returns dictionary containing the autocorrelation times obtained during the
        run (along with the iteration number at which they were computed) and the
        numpy array with the coordinates of the last draw.
        """
        if max_steps < min_steps:
            warnings.warn(
                "Sampling param min_steps is larger than max_steps. Swapping."
            )
            tmp = max_steps
            max_steps = min_steps
            min_steps = tmp

        if initial_state is None:
            initial_state = np.random.uniform(
                low=0., high=1.,
                size=(self.nwalkers, self.ndim)
            )

        iterations = []
        acor_times = []
        accept_rates = []
        old_acor = np.inf
        is_converged = False

        samples_iterator = self.sample(
            initial_state=initial_state,
            iterations=max_steps,
            thin_by=thin_by,
            **kwargs
        )
        # wrap the iteration over samples in a rich tracker,
        # if verbosity is desired
        if progress_desc is not None:
            report_progress = CustomProgress(console=report)
            samples_iterator = report_progress.track(
                sequence=samples_iterator,
                total=max_steps,
                description=progress_desc,
            )
            report_progress.start()

        coords = None
        for sample in samples_iterator:
            # after `check_interval` number of samples...
            coords = sample.coords
            if self.iteration < min_steps or self.iteration % check_interval:
                continue

            # ...get the acceptance rate up to this point
            accept_rates.append(100. * np.mean(self.acceptance_fraction))

            # ...compute the autocorrelation time and store it in an array.
            new_acor = self.get_autocorr_time(tol=0)
            iterations.append(self.iteration)
            acor_times.append(np.mean(new_acor))

            logger.debug(
                f"Acceptance rate at {self.iteration}: {accept_rates[-1]:.2f}%"
            )
            logger.debug(
                f"Autocorrelation time at {self.iteration}: {acor_times[-1]:.2f}"
            )

            # check convergence
            is_converged = self.iteration >= min_steps
            is_converged &= np.all(new_acor * trust_threshold < self.iteration)
            rel_acor_diff = np.abs(old_acor - new_acor) / new_acor
            is_converged &= np.all(rel_acor_diff < rel_acor_threshold)

            # if it has converged, stop
            if is_converged:
                break

            old_acor = new_acor

        iterations.append(self.iteration)
        acor_times.append(np.mean(self.get_autocorr_time(tol=0)))
        accept_rates.append(100. * np.mean(self.acceptance_fraction))
        accept_rate_str = f"acceptance rate was {accept_rates[-1]:.2f}%"

        if progress_desc is not None:
            report_progress.stop()
            if is_converged:
                logging.info(f"{progress_desc} converged, {accept_rate_str}")
            else:
                logging.info(
                    f"{progress_desc} finished: Max. steps reached, {accept_rate_str}"
                )

        return {
            "iterations": iterations,
            "acor_times": acor_times,
            "accept_rates": accept_rates,
            "final_state": coords,
        }


def run_mcmc_with_burnin(
    nwalkers: int,
    ndim: int,
    log_prob_fn: Callable,
    nsteps: int,
    persistent_backend: emcee.backends.HDFBackend,
    sampling_kwargs: Optional[dict] = None,
    burnin: Optional[int] = None,
    keep_burnin: bool = False,
    thin_by: int = 1,
    npools: Optional[int] = None,
    verbose: bool = True,
    return_chain: bool = False,
    return_sampler: bool = False,
) -> Dict[str, Any]:
    """
    Draw samples from the `log_prob_fn` using the `ConvenienceSampler` (subclass of
    `emcee.EnsembleSampler`).

    This function first draws `burnin` samples (that are discarded if `keep_burnin`
    is set to `False`) and afterwards it performs a second sampling round, starting
    where the burnin left off, of length `nsteps` that will be stored in the
    `persistent_backend`.

    When `burnin` is not given, the burnin phase will still take place and it will
    sample until convergence using convergence criteria defined via `sampling_kwargs`,
    after which it will draw another `nsteps` samples.

    Only every `thin_by` step will be made persistent.

    If `verbose` is set to `True`, the progress will be displayed.

    Returns a dictionary with some information about the burnin phase.
    """
    if sampling_kwargs is None:
        sampling_kwargs = {}

    if npools is None:
        created_pool = mp.Pool(os.cpu_count())
    elif npools == 0:
        created_pool = DummyPool()
    elif 0 < npools:
        created_pool = mp.Pool(np.minimum(npools, os.cpu_count()))
    else:
        raise ValueError(
            "Number of pools must be integer larger or equal to 0 (or `None`)"
        )

    with created_pool as pool:
        # Burnin phase
        if keep_burnin:
            burnin_backend = persistent_backend
        else:
            burnin_backend = emcee.backends.Backend()

        burnin_sampler = ConvenienceSampler(
            nwalkers, ndim, log_prob_fn,
            pool=pool, backend=burnin_backend,
        )

        if burnin is None:
            burnin_result = burnin_sampler.run_sampling(
                progress_desc=f"Burn-in ({created_pool._processes} cores)" if verbose else None,
                **sampling_kwargs,
            )
        else:
            burnin_result = burnin_sampler.run_sampling(
                min_steps=burnin,
                max_steps=burnin,
                progress_desc=f"Burn-in ({created_pool._processes} cores)" if verbose else None,
            )

        sampler = ConvenienceSampler(
            nwalkers, ndim, log_prob_fn,
            pool=pool, backend=persistent_backend,
        )
        sampler.run_sampling(
            min_steps=nsteps,
            max_steps=nsteps,
            thin_by=thin_by,
            initial_state=burnin_result["final_state"],
            progress_desc=f"Sampling ({created_pool._processes} cores)" if verbose else None,
        )

        if return_sampler:
            return sampler
        if return_chain:
            return sampler.get_chain(flat=True)
        return burnin_result


MODEL = None
INV_TEMP = 1.

def log_prob_fn(theta: np.array) -> float:
    """log probability function using global variables because of pickling."""
    for t in theta:
        if t < 0 or 1 < t:
            return -10000
    llh = MODEL.likelihood(given_param_args=theta, log=True)
    if np.isinf(llh):
        return -np.inf, -np.inf
    return llh

# def log_prob_fn(theta: np.ndarray | list) -> float:
#     """log probability function using global variables because of pickling."""
#     for t in theta:
#         if t < 0 or 1 < t:
#             return -10000
#     llh = MODEL.likelihood(given_param_args=theta, log=True)
#     if np.isnan(llh):
#         llh = -10000
#     if np.isinf(llh):
#         llh = -10000
#     return llh


def sample_from_global_model_and_configs(
    log_prob_fn: callable,
    ndim: int,
    sampling_params: dict,
    hdf5_backend: Optional[emcee.backends.HDFBackend] = None,
    starting_point: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None,
    llh_args: Optional[List] = None,
    models: Optional[List] = None
):
    global MODELS
    if models is not None:
        MODELS = models

    nwalkers = sampling_params["walkers_per_dim"] *ndim
    thin_by = sampling_params.get("thin_by", 1)
    sampling_kwargs = {"initial_state": starting_point}

    print(log_prob_fn(np.random.random(ndim)))

    sampler = run_mcmc_with_burnin(
        nwalkers, ndim, log_prob_fn,
        nsteps=sampling_params["nsteps"],
        burnin=sampling_params["nburnin"],
        persistent_backend=hdf5_backend,
        sampling_kwargs=sampling_kwargs,
        keep_burnin=False, # To not use backend at all.??
        thin_by=thin_by,
        verbose=True,
        npools=None,
        return_chain=False,
        return_sampler=True
    )

    samples = sampler.get_chain(flat=True)
    log_probs = sampler.get_log_prob(flat=True)
    end_point = sampler.get_last_sample()[0]

    return samples, end_point, log_probs


def sample_from_model(model, params_sampling, sample_path: Path, sample_name: str, store_as_chain: bool = False):
    """ Initiate sampling from a given model (loaded with patients), given some parameters."""
    global MODEL
    MODEL = model
    ndim = len(model.get_params())
    nwalkers = ndim * params_sampling["walkers_per_dim"]
    thin_by = params_sampling["thin_by"]
    logger.info(
        f"Set up sampling for {type(MODEL)} model with {ndim} parameters and "
        f"{len(model.patient_data)} patients"
    )

    output = sample_path / sample_name
    if not output.suffix or output.suffix != ".hdf5":
        output = output.with_suffix(".hdf5")
    # make sure path to output file exists
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    # prepare backend
    hdf5_backend = emcee.backends.HDFBackend(output, name="mcmc")

    logger.info(f"Prepared sampling params & backend at {output}")

    sampling_chain = run_mcmc_with_burnin(
        nwalkers, ndim, log_prob_fn,
        nsteps=params_sampling["nsteps"],
        burnin=params_sampling["nburnin"],
        persistent_backend=hdf5_backend,
        sampling_kwargs=None,
        thin_by=thin_by,
        verbose=True,
        npools=None,
        return_chain=True
    )
    # Store the chain since some functions fetch the chain later.
    if store_as_chain:
        np.save(sample_path / sample_name, sampling_chain)

    return sampling_chain


def main(args: argparse.Namespace) -> None:
    """
    First, this program reads in the parameter YAML file and the CSV training data.
    After that, it parses the loaded configuration and creates an instance of the model
    that loads the training data.

    From there - depending on the user input - it either performs thermodynamic
    integration [^1] to compute the data's evidence given this particular model or it
    draws directly from the posterior of the parameters given the data. In the latter
    case it automatically samples until convergence. It uses the amazing
    [`emcee`](https://github.com/dfm/emcee) library for the MCMC process.

    The drawn samples are always stored in and HDF5 file and some progress about the
    sampling procedures is stored as well (acceptance rates of samples, estimates
    of the chains' autocorrelation times and the accuracy during the runs).

    Th help via `lyscripts sample --help` shows this output:

    ```
    USAGE: lyscripts sample [-h] [--params PARAMS]
                            [--modalities MODALITIES [MODALITIES ...]] [--plots PLOTS]
                            [--ti] [--pools POOLS] [--seed SEED]
                            input output

    Learn the spread probabilities of the HMM for lymphatic tumor progression using
    the preprocessed data as input and MCMC as sampling method.

    This is the central script performing for our project on modelling lymphatic
    spread in head & neck cancer. We use it for model comparison via the thermodynamic
    integration functionality and use the sampled parameter estimates for risk
    predictions. This risk estimate may in turn some day guide clinicians to make more
    objective decisions with respect to defining the *elective clinical target volume*
    (CTV-N) in radiotherapy.

    POSITIONAL ARGUMENTS:
      input                 Path to training data files
      output                Path to the HDF5 file to store the results in

    OPTIONAL ARGUMENTS:
      -h, --help            show this help message and exit
      --params PARAMS       Path to parameter file (default: ./params.yaml)
      --modalities MODALITIES [MODALITIES ...]
                            List of modalities for inference. Must be defined in
                            `params.yaml` (default: ['max_llh'])
      --plots PLOTS         Directory to store plot of acor times (default: ./plots)
      --ti                  Perform thermodynamic integration (default: False)
      --pools POOLS         Number of parallel worker pools (CPU cores) to use. If not
                            provided, it will use all cores. If set to zero,
                            multiprocessing will not be used. (default: None)
      --seed SEED           Seed value to reproduce the same sampling round. (default:
                            42)
    ```

    [^1]: https://doi.org/10.1007/s11571-021-09696-9
    """
    params = load_yaml_params(args.params, logger=logger)

    # Only read in two header rows when using the Unilateral model
    is_unilateral = params["model"]["class"] == "Unilateral"
    header = [0, 1] if is_unilateral else [0, 1, 2]
    inference_data = pd.read_csv(args.input, header=header)
    logger.info(f"Read in training data from {args.input}")

    global MODEL  # ugly, but necessary for pickling
    MODEL = model_from_config(
        graph_params=params["graph"],
        model_params=params["model"],
        modalities_params=get_modalities_subset(
            defined_modalities=params["modalities"],
            selection=args.modalities,
        ),
    )
    MODEL.patient_data = inference_data
    ndim = len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric
    nwalkers = ndim * params["sampling"]["walkers_per_dim"]
    thin_by = params["sampling"]["thin_by"]
    logger.info(
        f"Set up {type(MODEL)} model with {ndim} parameters and loaded "
        f"{len(inference_data)} patients"
    )

    # Not sure this is working. emcee sadly does not support numpy's new
    # random number generator yet.
    np.random.seed(args.seed)
    if args.ti:
        global INV_TEMP

        # make sure path to output file exists
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # set up sampling params
        temp_schedule = params["sampling"]["temp_schedule"]
        nsteps = params["sampling"]["nsteps"]
        burnin = params["sampling"]["burnin"]
        logger.info("Prepared thermodynamic integration.")

        # record some information about each burnin phase
        x_axis = temp_schedule.copy()
        plots = {
            "acor_times": [],
            "accept_rates": [],
        }
        for i,inv_temp in enumerate(temp_schedule):
            INV_TEMP = inv_temp
            logger.info(f"TI round {i+1}/{len(temp_schedule)} with β = {INV_TEMP}")

            # set up backend
            hdf5_backend = emcee.backends.HDFBackend(args.output, name=f"ti/{i+1:0>2d}")

            burnin_info = run_mcmc_with_burnin(
                nwalkers, ndim, log_prob_fn, nsteps,
                persistent_backend=hdf5_backend,
                burnin=burnin,
                keep_burnin=False,
                thin_by=thin_by,
                verbose=True,
                npools=args.pools,
            )
            plots["acor_times"].append(burnin_info["acor_times"][-1])
            plots["accept_rates"].append(burnin_info["accept_rates"][-1])

        # copy last sampling round over to a group in the HDF5 file called "mcmc"
        # because that is what other scripts expect to see, e.g. for plotting risks
        h5_file = h5py.File(args.output, "r+")
        h5_file.copy(f"ti/{len(temp_schedule):0>2d}", h5_file, name="mcmc")
        logger.info("Finished thermodynamic integration.")

    else:
        # make sure path to output file exists
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # prepare backend
        hdf5_backend = emcee.backends.HDFBackend(args.output, name="mcmc")

        logger.info(f"Prepared sampling params & backend at {args.output}")

        burnin_info = run_mcmc_with_burnin(
            nwalkers, ndim, log_prob_fn,
            nsteps=params["sampling"]["nsteps"],
            persistent_backend=hdf5_backend,
            sampling_kwargs=params["sampling"]["kwargs"],
            thin_by=thin_by,
            verbose=True,
            npools=args.pools,
        )
        x_axis = np.array(burnin_info["iterations"])
        plots = {
            "acor_times": burnin_info["acor_times"],
            "accept_rates": burnin_info["accept_rates"]
        }

    args.plots.mkdir(parents=True, exist_ok=True)

    for name, y_axis in plots.items():
        tmp_df = pd.DataFrame(
            np.array([x_axis, y_axis]).T,
            columns=["x", name],
        )
        tmp_df.to_csv(args.plots/(name + ".csv"), index=False)

    logger.info(f"Stored {len(plots)} plots about burnin phases at {args.plots}")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
