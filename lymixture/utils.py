"""
Module with utilities for the mixture model package.
"""
# pylint: disable=logging-fstring-interpolation

import itertools
import logging
import os

import emcee
import lymph
import numpy as np
import pandas as pd
import scipy as sp
from scipy.special import factorial
from lyscripts.sample import DummyPool, run_mcmc_with_burnin


logger = logging.getLogger(__name__)


def binom_pmf(k: np.ndarray, n: int, p: float):
    """Binomial PMF"""
    if p > 1.0 or p < 0.0:
        raise ValueError("Binomial prob must be btw. 0 and 1")
    q = 1.0 - p
    binom_coeff = factorial(n) / (factorial(k) * factorial(n - k))
    return binom_coeff * p**k * q ** (n - k)


def late_binomial(support: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Parametrized binomial distribution."""
    return binom_pmf(k=support, n=support[-1], p=p)


def create_models(
    num_models: int,
    graph_dict: dict[tuple[str, str], list[str]] | None = None,
    model_kwargs: dict | None = None,
    include_late: bool = True,
    ignore_t_stage: bool = False,
    first_binom_prob: float = 0.3,
    max_time: int = 10,
) -> list[lymph.models.Unilateral]:
    """Create ``num_models`` Unilateral models.

    They will all share the same ``graph_dict``, ``model_kwargs``, and distributions
    over diagnose times. The earliest T-category time distribution is parametrized by
    ``first_binom_prob`` and ``max_time``.
    """
    if graph_dict is None:
        graph_dict = {
            ("tumor", "primary"): ["I", "II", "III", "IV"],
            ("lnl", "I"): [],
            ("lnl", "II"): ["I", "III"],
            ("lnl", "III"): ["IV"],
            ("lnl", "IV"): [],
        }

    if model_kwargs is None:
        model_kwargs = {}

    diagnostic_spsn = {"max_llh": [1.0, 1.0],}
    time_steps = np.arange(max_time + 1)
    early_prior = sp.stats.binom.pmf(time_steps, max_time, first_binom_prob)

    models = []
    for _ in range(num_models):
        model = lymph.models.Unilateral.binary(graph_dict=graph_dict, **model_kwargs)
        model.modalities = diagnostic_spsn

        if ignore_t_stage:
            model.diag_time_dists["all"] = early_prior
        else:
            model.diag_time_dists["early"] = early_prior
            if include_late:
                model.diag_time_dists["late"] = late_binomial
        models.append(model)

    return models


def create_synth_data(
    params_0, params_1, n, ratio, graph_dict, t_stage_dist=None, header="Default"
):
    """Create synthetic dataset under the given model parameters."""
    gen_model = create_models(1, graph_dict=graph_dict)[0]
    gen_model.assign_params(*params_0)

    data_synth_s0_30 = gen_model.generate_dataset(
        int(n * ratio), t_stage_dist, column_index_levels=header
    )
    gen_model.assign_params(*params_1)
    data_synth_s1_30 = gen_model.generate_dataset(
        int(n * (1 - ratio)), t_stage_dist, column_index_levels=header
    )

    data_synth_s2 = pd.concat([data_synth_s0_30, data_synth_s1_30], ignore_index=True)
    return data_synth_s2


def convert_params(params, n_clusters, n_subsites):
    n_mixing = (n_clusters - 1) * (n_subsites)
    p_mixing = params[-n_mixing:]
    p_model = params[:-n_mixing]
    params_model = [
        [params[i + j] for j in range(n_clusters)]
        for i in range(0, len(p_model), n_clusters)
    ]

    params_mixing = [
        [p_mixing[i + j] for j in range(n_clusters - 1)]
        for i in range(0, len(p_mixing), n_clusters - 1)
    ]
    params_mixing = [[*mp, 1 - np.sum(mp)] for mp in params_mixing]
    return params_model, params_mixing


def emcee_sampling(llh_function, n_params, sample_name, llh_args=None):
    nwalkers, nstep, burnin = 20 * n_params, 1000, 1500
    thin_by = 1
    logger.info(f"Dimension: {n_params} with n walkers: {nwalkers}")
    output_name = sample_name

    created_pool = mp.Pool(os.cpu_count())
    with created_pool as pool:
        starting_points = np.random.uniform(size=(nwalkers, n_params))
        logger.info(
            f"Start Burning (steps = {burnin}) with {created_pool._processes} cores"
        )
        burnin_sampler = emcee.EnsembleSampler(
            nwalkers,
            n_params,
            llh_function,
            args=llh_args,
            pool=pool,
        )
        _ = burnin_sampler.run_mcmc(
            initial_state=starting_points, nsteps=burnin, progress=True
        )

        ar = np.mean(burnin_sampler.acceptance_fraction)
        logger.info(
            f"the HMM sampler for model 01 accepted {ar * 100 :.2f} % of samples."
        )
        last_sample = burnin_sampler.get_last_sample()[0]
        logger.info(f"The shape of the last sample is {last_sample.shape}")
        starting_points = np.random.uniform(size=(nwalkers, n_params))
        original_sampler_mp = emcee.EnsembleSampler(
            nwalkers,
            n_params,
            llh_function,
            args=llh_args,
            backend=None,
            pool=pool,
        )
        sampling_results = original_sampler_mp.run_mcmc(
            initial_state=last_sample, nsteps=nstep, progress=True, thin_by=thin_by
        )

        ar = np.mean(original_sampler_mp.acceptance_fraction)
        logger.info(f"the HMM sampler for model accepted {ar * 100 :.2f} % of samples.")
        samples = original_sampler_mp.get_chain(flat=True)
        np.save(f"./samples/" + output_name, samples)
        # plots["acor_times"].append(burnin_info["acor_times"][-1])
        # plots["accept_rates"].append(burnin_info["accept_rates"][-1])
    return samples


def emcee_sampling_ext(
    llh_function,
    n_params=None,
    sample_name=None,
    n_burnin=None,
    n_step=None,
    start_with=None,
    llh_args=None,
):
    nwalkers = 20 * n_params
    burnin = 1000 if n_burnin is None else n_burnin
    nstep = 1000 if n_step is None else n_step
    thin_by = 1
    logger.info(f"Dimension: {n_params} with n walkers: {nwalkers}")
    output_name = sample_name

    created_pool = DummyPool()
    with created_pool as pool:
        if start_with is None:
            starting_points = np.random.uniform(size=(nwalkers, n_params))
        else:
            if np.shape(start_with) != np.shape(
                np.random.uniform(size=(nwalkers, n_params))
            ):
                starting_points = np.tile(start_with, (nwalkers, 1))
            else:
                starting_points = start_with
        logger.info(
            f"Start Burning (steps = {burnin}) with {created_pool._processes} cores"
        )
        burnin_sampler = emcee.EnsembleSampler(
            nwalkers,
            n_params,
            llh_function,
            args=llh_args,
            pool=pool,
        )
        burnin_results = burnin_sampler.run_mcmc(
            initial_state=starting_points, nsteps=burnin, progress=True
        )

        ar = np.mean(burnin_sampler.acceptance_fraction)
        logger.info(
            f"the HMM sampler for model 01 accepted {ar * 100 :.2f} % of samples."
        )
        starting_points = burnin_sampler.get_last_sample()[0]
        # logger.info(f"The shape of the last sample is {starting_points.shape}")
        original_sampler_mp = emcee.EnsembleSampler(
            nwalkers,
            n_params,
            llh_function,
            args=llh_args,
            backend=None,
            pool=pool,
        )
        sampling_results = original_sampler_mp.run_mcmc(
            initial_state=starting_points,
            nsteps=nstep,
            progress=True,
            thin_by=thin_by,
        )

        ar = np.mean(original_sampler_mp.acceptance_fraction)
        logger.info(f"the HMM sampler for model accepted {ar * 100 :.2f} % of samples.")
        samples = original_sampler_mp.get_chain(flat=True)
        log_probs = original_sampler_mp.get_log_prob(flat=True)
        end_point = original_sampler_mp.get_last_sample()[0]
        if output_name is not None:
            np.save(f"./samples/" + output_name, samples)
        # plots["acor_times"].append(burnin_info["acor_times"][-1])
        # plots["accept_rates"].append(burnin_info["accept_rates"][-1])
    return samples, end_point, log_probs


def convert_lnl_to_filename(lnls):
    if not lnls:
        return "Empty_List"
    if len(lnls) == 1:
        return lnls[0]

    return f"{lnls[0]}_to_{lnls[-1]}"


def reverse_dict(original_dict: dict) -> dict:
    reverse_dict = {}
    for k, v in original_dict.items():
        if isinstance(v, list):
            for vs in v:
                reverse_dict[vs] = k
        else:
            reverse_dict[v] = k
    return reverse_dict


def create_states(lnls, total_lnls=True):
    """Create states (patterns) used for risk predictions.

    If total_lnls is set, then only total risk in lnls is considered.
    """
    if total_lnls:
        states_all = [
            {lnls[i]: True if i == j else None for i in range(len(lnls))}
            for j in range(len(lnls))
        ]
    else:
        states_all_raw = [
            list(combination)
            for combination in itertools.product([0, 1], repeat=len(lnls))
        ]
        states_all = [
            {lnls[-(i + 1)]: p[i] for i in range(len(lnls))} for p in states_all_raw
        ]

    return states_all


def sample_from_global_model_and_configs(
    log_prob_fn: callable,
    ndim: int,
    sampling_params: dict,
    backend: emcee.backends.Backend | None = None,
    starting_point: np.ndarray | None = None,
    models: list | None = None,
    verbose: bool = True,
):
    global MODELS
    if models is not None:
        MODELS = models

    if backend is None:
        backend = emcee.backends.Backend()

    nwalkers = sampling_params["walkers_per_dim"] * ndim
    thin_by = sampling_params.get("thin_by", 1)
    sampling_kwargs = {"initial_state": starting_point}

    _ = run_mcmc_with_burnin(
        nwalkers,
        ndim,
        log_prob_fn,
        nsteps=sampling_params["nsteps"],
        burnin=sampling_params["nburnin"],
        persistent_backend=backend,
        sampling_kwargs=sampling_kwargs,
        keep_burnin=False,  # To not use backend at all.??
        thin_by=thin_by,
        verbose=verbose,
        npools=0,
    )

    samples = backend.get_chain(flat=True)
    log_probs = backend.get_log_prob(flat=True)
    end_point = backend.get_last_sample()[0]

    return samples, end_point, log_probs


def get_param_labels(model):
    """Get parameter labels from a model."""
    return [
        t.replace("primary", "T").replace("_spread", "")
        for t in model.get_params(as_dict=True).keys()
    ]
