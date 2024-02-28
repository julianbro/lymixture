import logging
from cycler import cycler
import pandas as pd
import lymph
import numpy as np
import emcee
import os
from lyscripts.predict.prevalences import (
    compute_observed_prevalence,
)
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.special import factorial
import itertools
import scipy as sp
import matplotlib.pyplot as plt
from lyscripts.sample import run_mcmc_with_burnin, DummyPool

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
usz_colors = [usz_blue, usz_green, usz_red, usz_orange, usz_gray]
edge_colors = [
    usz_blue_border,
    usz_green_border,
    usz_red_border,
    usz_orange_border,
    usz_gray_border,
]
COLORS = {
    "blue": "#005ea8",
    "orange": "#f17900",
    "green": "#00afa5",
    "red": "#ae0060",
    "gray": "#c5d5db",
}

PLOT_PATH = Path("./figures/")
logger = logging.getLogger(__name__)


# Define a function to set plot size based on desired width, unit, and ratio
def set_size(width="single", unit="cm", ratio="golden"):
    """
    Set the size of the plot for Matplotlib.

    Parameters:
    - width (int or str): The width of the plot.
    - unit (str): The unit of width ('cm', 'inch', etc.).
    - ratio (float or str): The aspect ratio of the plot.

    Returns:
    - tuple: A tuple (width, height) specifying the plot size.
    """
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
    """
    Creates n Unilateral models, all with the same graph, and same time distributions.
    """
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
        logger.info(f"Prev Vector: {X_inv_list}")
    else:
        prev_formatted = "".join(f"{l}: {p}" for l, p in zip(states_all, X_inv_list))
        logger.info(f"Prev Vector: {prev_formatted}")
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


def emcee_sampling(llh_function, n_params, sample_name, llh_args=None):
    nwalkers, nstep, burnin = 20 * n_params, 1000, 1500
    thin_by = 1
    logger.info(f"Dimension: {n_params} with n walkers: {nwalkers}")
    output_name = sample_name

    if False:
        samples = np.load("samples/" + output_name + ".npy")
    else:
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
            burnin_results = burnin_sampler.run_mcmc(
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
    skip_burnin=False,
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


def fade_to_white(initial_color, n):
    """
    Create an array of n colors, fading from the initial color to white.
    """
    # Convert the initial color to an RGB array
    initial_color_rgb = np.array(mcolors.to_rgb(initial_color))

    # Create an array of n steps fading to white (RGB for white is [1, 1, 1])
    fade_colors = [
        initial_color_rgb + (np.array([1, 1, 1]) - initial_color_rgb) * i / (n - 1)
        for i in range(n)
    ]

    return fade_colors


def convert_lnl_to_filename(lnls):
    if not lnls:
        return "Empty_List"
    if len(lnls) == 1:
        return lnls[0]

    return f"{lnls[0]}_to_{lnls[-1]}"


def plot_prevalences_icd(
    prev_icds, prev_loc, lnls_full, icds_codes, base_color, save_name
):
    x_labels = lnls_full
    fade_colors = fade_to_white(base_color, len(prev_icds) + 2)[:-2]

    w_data, sp = 0.85, 0.15
    overlap = 0.1
    w_icd = w_data / (2 * overlap + len(icds_codes) * (1 - 2 * overlap))
    if len(icds_codes) > 1:
        icd_spacing = (w_data - w_icd) / (len(icds_codes) - 1)
    else:
        icd_spacing = w_data - w_icd
    rel_pos_icds = [
        i * icd_spacing + (-w_data / 2 + w_icd / 2) for i in range(len(icds_codes))
    ]
    pos = [1 + 1 * i for i in range(len(x_labels))]
    # w_icd_fill = (w_data/len(icds_codes))
    # w_icd =w_icd_fill*(1-overlap)

    # rel_pos_icds = [w_icd_fill/2 + i*w_icd_fill - (w_data)/2 for i in range(len(oc_icds))]
    pos_icds = [[p + rel_pos_icds[i] for p in pos] for i in range(len(icds_codes))]
    # pos[1] += 0.075
    # pos[2] -= 0.075

    fig, ax = plt.subplots(1, 1, figsize=set_size(width="single"), tight_layout=True)
    bar_kwargs = {"width": w_icd, "align": "center"}
    plt.bar(
        pos,
        prev_loc,
        width=w_data,
        color=usz_red,
        hatch="////",
        label="mean",
        alpha=0.0,
    )

    for i, prev_icd in enumerate(prev_icds[::-1]):
        j = len(prev_icds) - i - 1
        plt.bar(
            pos_icds[j],
            height=prev_icd,
            color=fade_colors[j],
            label=icds_codes[j],
            edgecolor=base_color,
            alpha=0.92,
            **bar_kwargs,
        )

    ax.set_xticks(pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("LNLs")

    ax.set_yticks(np.arange(0.0, 1.1 * max(max(prev_icds)), 0.1))
    ax.set_yticklabels(
        [f"{100*tick:.0f}%" for tick in np.arange(0.0, 1.1 * max(max(prev_icds)), 0.1)]
    )
    ax.set_ylabel("ipsilateral involvement")
    ax.grid(axis="y")
    ax.legend()
    fig.savefig(save_name)
    # plt.show()


def plot_prevalences_categories(
    prev_loc, prev_mean, lnls_full, icds_codes, colors, edge_color, save_name
):
    x_labels = lnls_full
    # fade_colors = fade_to_white(base_color, len(prev_icds) + 2)[:-2]

    w_data, sp = 0.85, 0.15
    overlap = 0.15
    w_icd = w_data / (2 * overlap + len(icds_codes) * (1 - 2 * overlap))
    if len(icds_codes) > 1:
        icd_spacing = (w_data - w_icd) / (len(icds_codes) - 1)
    else:
        icd_spacing = w_data - w_icd
    rel_pos_icds = [
        i * icd_spacing + (-w_data / 2 + w_icd / 2) for i in range(len(icds_codes))
    ]
    pos = [1 + 1 * i for i in range(len(x_labels))]
    # w_icd_fill = (w_data/len(icds_codes))
    # w_icd =w_icd_fill*(1-overlap)

    # rel_pos_icds = [w_icd_fill/2 + i*w_icd_fill - (w_data)/2 for i in range(len(oc_icds))]
    pos_icds = [[p + rel_pos_icds[i] for p in pos] for i in range(len(icds_codes))]
    # pos[1] += 0.075
    # pos[2] -= 0.075

    fig, ax = plt.subplots(1, 1, figsize=set_size(width="full"), tight_layout=True)
    bar_kwargs = {"width": w_icd, "align": "center"}
    plt.bar(
        pos,
        prev_mean,
        width=w_data,
        color=usz_gray,
        hatch="////",
        label="mean",
        alpha=0.0,
    )
    for i, prev_icd in enumerate(prev_loc[::-1]):
        j = len(prev_loc) - i - 1
        plt.bar(
            pos_icds[j],
            height=prev_icd,
            color=colors[j],
            label=icds_codes[j],
            # edgecolor=edge_colors[j],
            alpha=0.95,
            **bar_kwargs,
        )

    ax.set_xticks(pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("LNLs")

    ticks_val = np.arange(0.0, 0.75, 0.1)
    # ticks_val = np.arange(0.0, 1.1 * max(max(prev_loc)), 0.1)
    ax.set_yticks(ticks_val)
    ax.set_yticklabels([f"{100*tick:.0f}%" for tick in ticks_val])
    ax.set_ylabel("ipsilateral involvement")
    ax.grid(axis="y")
    ax.legend()
    fig.savefig(save_name)
    # plt.show()


def reverse_dict(original_dict: dict) -> dict:
    reverse_dict = {}
    for k, v in original_dict.items():
        if isinstance(v, list):
            for vs in v:
                reverse_dict[vs] = k
        else:
            reverse_dict[v] = k
    return reverse_dict


# script_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append()


def load_data(datasets_names, logger=None):
    dataset = pd.DataFrame({})
    for ds in datasets_names:
        try:
            dataset_new = pd.read_csv(
                Path("data/datasets/enhanced/" + ds), header=[0, 1, 2]
            )
        except:
            dataset_new = pd.read_csv(
                Path("../data/datasets/enhanced/" + ds), header=[0, 1, 2]
            )
        dataset = pd.concat([dataset, dataset_new], ignore_index=True)
    if logger:
        logger.info(f"Succesfully loaded {len(dataset)} Patients")
    else:
        logger.info(f"Succesfully loaded {len(dataset)} Patients")
    return dataset


def enhance_data(dataset: pd.DataFrame, convert_t_stage: dict = None):
    """Enhance the loaded datasets. Essentially assigns t-stage and majorsubsite (icd codes) to the dataframe."""
    if convert_t_stage is None:
        convert_t_stage = {0: "early", 1: "early", 2: "early", 3: "late", 4: "late"}

    convert_t_stage_fun = lambda t: convert_t_stage[t]
    t_stages = list(dataset[("tumor", "1", "t_stage")].apply(convert_t_stage_fun))
    dataset[("info", "tumor", "t_stage")] = t_stages

    # Add the major subsites
    subsites_list = list(dataset["tumor"]["1"]["subsite"])
    major_subsites = [s[:3] for s in subsites_list]
    dataset["tumor", "1", "majorsubsites"] = major_subsites

    # # For reference (or initial clustering), cluster the subsites based on the location
    # location_to_cluster = {
    #     "oral cavity": 0,
    #     "oropharynx": 1,
    #     "hypopharynx": 2,
    #     "larynx": 3,
    # }
    # dataset["tumor", "1", "clustering"] = dataset["tumor"]["1"]["location"].apply(
    #     lambda x: location_to_cluster[x]
    # )
    return dataset


def create_states(lnls, total_lnls=True):
    """Creates states (patterns) used for risk predictions. If total_lnls is set, then only total risk in lnls is considered."""
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


def plot_histograms(data_in, states, single_line, loc, colors, bins=50, plot_path=None):
    for state in states:
        try:
            data = {k: v[state] for k, v in data_in.items()}
        except:
            state = str(state)
            data = {k: v[state] for k, v in data_in.items()}
        fig, axs = plt.subplots(1, figsize=set_size(width="full"), tight_layout=True)
        # fig.suptitle('S2')
        hist_cycl = cycler(histtype=["stepfilled", "step"]) * cycler(
            color=list(colors.values())
        )
        line_cycl = cycler(linestyle=["-", "--"]) * cycler(color=list(colors.values()))
        logger.info("Set up figure")

        values = []
        labels = []
        num_matches = []
        num_totals = []
        lines = []
        min_value = 1.0
        max_value = 0.0
        for label, d in data.items():
            values.append([ds * 100 for ds in d[0]])
            labels.append(label)
            num_matches.append(d[1][0])
            num_totals.append(d[1][1])
            lines.append(100.0 * num_matches[-1] / num_totals[-1])
            min_value = np.minimum(1, np.min(values))
            max_value = np.maximum(0, np.max(values))

        min_value = np.min(lines, where=~np.isnan(lines), initial=min_value)
        max_value = np.max(lines, where=~np.isnan(lines), initial=max_value)

        hist_kwargs = {
            "bins": np.linspace(min_value, max_value, bins),
            "density": True,
            "alpha": 0.6,
            "linewidth": 2.0,
        }

        x = np.linspace(min_value, max_value, 200)
        zipper = zip(values, labels, num_matches, num_totals, hist_cycl, line_cycl)
        # ax = axs[int(np.floor(i/2))][i%2]
        ax = axs
        for i, (vals, label, a, n, hstyle, lstyle) in enumerate(zipper):
            ax.hist(vals, label=label, **hist_kwargs, **hstyle)
            if not np.isnan(a) and not (i > 0 and single_line):
                post = sp.stats.beta.pdf(x / 100.0, a + 1, n - a + 1) / 100.0
                if single_line:
                    lstyle["color"] = usz_red
                ax.plot(x, post, label=f"{int(a)}/{int(n)}", **lstyle)
            ax.legend()
            ax.set_xlabel("probability [%]")
        ax.set_title(f"{state}", fontsize="small", fontweight="regular")
        logger.info(f"Plotted {len(values)} histograms")
        if plot_path is None:
            plt.savefig(PLOT_PATH / f"hist_{loc}_{state}.png")
        else:
            plt.savefig(plot_path / f"hist_{loc}_{state}.png")
        plt.show()


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

    nwalkers = sampling_params["walkers_per_dim"] *ndim
    thin_by = sampling_params.get("thin_by", 1)
    sampling_kwargs = {"initial_state": starting_point}

    _ = run_mcmc_with_burnin(
        nwalkers, ndim, log_prob_fn,
        nsteps=sampling_params["nsteps"],
        burnin=sampling_params["nburnin"],
        persistent_backend=backend,
        sampling_kwargs=sampling_kwargs,
        keep_burnin=False, # To not use backend at all.??
        thin_by=thin_by,
        verbose=verbose,
        npools=0,
    )

    samples = backend.get_chain(flat=True)
    log_probs = backend.get_log_prob(flat=True)
    end_point = backend.get_last_sample()[0]

    return samples, end_point, log_probs
