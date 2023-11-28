from math import factorial
import pandas as pd
import lymph
import numpy as np
from lyscripts.predict.prevalences import (
    compute_observed_prevalence,
    compute_predicted_prevalence,
)

# from lyscripts.helpers import add_tstage_marg
from pathlib import Path

import itertools, sys

import scipy as sp
import matplotlib.pyplot as plt


# Plotting styles
usz_blue = "#005ea8"
usz_blue_border = "#2387D5"
usz_green = "#00afa5"
usz_green_border = "#30DFD5"
usz_red = "#ae0060"
usz_red_border = "#FB4EAE"
usz_orange = "#f17900"
usz_orange_border = "#F8AB5C"
usz_gray = "#c5d5db"
usz_gray_border = "#DFDFDF"
# sn.set_theme()
usz_colors = [usz_blue, usz_green, usz_red, usz_orange, usz_gray]
edge_colors = [
    usz_blue_border,
    usz_green_border,
    usz_red_border,
    usz_orange_border,
    usz_gray_border,
]

PLOT_PATH = Path("./figures/")


def set_size(width="single", unit="cm", ratio="golden"):
    if width == "single":
        width = 10
    elif width == "full":
        width = 16
    else:
        try:
            width = width
        except:
            width = 10

    if unit == "cm":
        width = width / 2.54

    if ratio == "golden":
        ratio = 1.618
    else:
        ratio = ratio

    try:
        height = width / ratio
    except:
        height = width / 1.618

    return (width, height)


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
    n, graph=None, include_late=True, ignore_t_stage=False, n_mixture_components=1
) -> list[lymph.models.Unilateral] | lymph.models.Unilateral:
    if graph is None:
        graph = {
            ("tumor", "primary"): ["I", "II", "III", "IV"],
            ("lnl", "I"): [],
            ("lnl", "II"): ["I", "III"],
            ("lnl", "III"): ["IV"],
            ("lnl", "IV"): [],
        }

    diagnostic_spsn = {
        "max_llh": [1.0, 1.0],
    }

    max_t = 10
    first_binom_prob = 0.3

    models = []
    for i in range(n):
        model = lymph.models.Unilateral.binary(graph_dict=graph)
        model.modalities = diagnostic_spsn

        max_time = model.max_time
        time_steps = np.arange(max_time + 1)
        p = 0.3

        if ignore_t_stage:
            early_prior = sp.stats.binom.pmf(time_steps, max_time, p)
            model.diag_time_dists["all"] = early_prior
        else:
            early_prior = sp.stats.binom.pmf(time_steps, max_time, p)
            model.diag_time_dists["early"] = early_prior
            if include_late:
                model.diag_time_dists["late"] = late_binomial
        model.n_mixture_components = n_mixture_components
        models.append(model)
    if n > 1:
        return models
    else:
        return models[0]


def create_prev_vectors(
    data,
    lnls,
    t_stages=None,
    full_involvement=False,
    plot=False,
    title=None,
    save_figure=False,
    ax=None,
):
    states_all_raw = [
        list(combination) for combination in itertools.product([0, 1], repeat=len(lnls))
    ]
    states_all = [
        {lnls[-(i + 1)]: p[i] for i in range(len(lnls))} for p in states_all_raw
    ]
    if full_involvement:
        states_all = [
            {lnls[i]: True if i == j else None for i in range(len(lnls))}
            for j in range(len(lnls))
        ]
    if t_stages is None:
        t_stages = ["early", "late"]
    X_inv_list = []
    for state in states_all:
        # sys.stdout.write(f"\r State: {state}")
        inv, nd = 0, 0
        for t_stage in t_stages:
            observed_prevalence_result = compute_observed_prevalence(
                pattern={"ipsi": state},
                data=data,
                t_stage=t_stage,
                lnls=lnls,
            )
            inv += observed_prevalence_result[0]
            nd += observed_prevalence_result[1]
        X_inv_list.append((inv / nd) if nd != 0 else 0)

    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, figsize=set_size("full"))

        ax.barh(range(len(X_inv_list)), width=X_inv_list)
        maxx = max(X_inv_list)
        ax.set_title(f"{title} (n = {len(data)})")  # Adjust title properties
        if full_involvement:
            ax.set_yticks(range(len(X_inv_list)), lnls)
        else:
            ax.set_yticks(range(len(X_inv_list)), [str(s) for s in states_all_raw])
        ax.tick_params(axis="x", which="both")
        if save_figure:
            fig.savefig(PLOT_PATH / f"prev_vectors_{title}_{str(lnls)}.svg")
        if ax is None:
            plt.show()
    if len(states_all) > 4:
        print(f"Prev Vector: {X_inv_list}")
    else:
        prev_formatted = "".join(f"{l}: {p}" for l, p in zip(states_all, X_inv_list))
        print(f"Prev Vector: {prev_formatted}")
    return X_inv_list


def create_synth_data(
    params_0, params_1, n, ratio, graph, t_stage_dist=None, header="Default"
):
    gen_model = create_models(1, graph=graph)
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


import emcee
import multiprocess as mp
import os


def emcee_sampling(llh_function, n_params, sample_name, llh_args=None):
    nwalkers, nstep, burnin = 20 * n_params, 1000, 1500
    thin_by = 1
    print(f"Dimension: {n_params} with n walkers: {nwalkers}")
    output_name = sample_name

    if False:
        samples = np.load("samples/" + output_name + ".npy")
    else:
        created_pool = mp.Pool(os.cpu_count())
        with created_pool as pool:
            starting_points = np.random.uniform(size=(nwalkers, n_params))
            print(
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
            print(
                f"the HMM sampler for model 01 accepted {ar * 100 :.2f} % of samples."
            )
            last_sample = burnin_sampler.get_last_sample()[0]
            print(f"The shape of the last sample is {last_sample.shape}")
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
        print(f"the HMM sampler for model accepted {ar * 100 :.2f} % of samples.")
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
    skip_burnin=False,
    llh_args=None,
):
    nwalkers = 20 * n_params
    burnin = 1000 if n_burnin is None else n_burnin
    nstep = 1000 if n_step is None else n_step
    thin_by = 1
    print(f"Dimension: {n_params} with n walkers: {nwalkers}")
    output_name = sample_name

    if False:
        samples = np.load("samples/" + output_name + ".npy")
    else:
        created_pool = mp.Pool(os.cpu_count())
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
            print(
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
            print(
                f"the HMM sampler for model 01 accepted {ar * 100 :.2f} % of samples."
            )
            starting_points = burnin_sampler.get_last_sample()[0]
            # print(f"The shape of the last sample is {starting_points.shape}")
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
        print(f"the HMM sampler for model accepted {ar * 100 :.2f} % of samples.")
        samples = original_sampler_mp.get_chain(flat=True)
        log_probs = original_sampler_mp.get_log_prob(flat=True)
        end_point = original_sampler_mp.get_last_sample()[0]
        if output_name is not None:
            np.save(f"./samples/" + output_name, samples)
        # plots["acor_times"].append(burnin_info["acor_times"][-1])
        # plots["accept_rates"].append(burnin_info["accept_rates"][-1])
    return samples, end_point, log_probs
