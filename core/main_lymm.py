# %%
# %load_ext autoreload
# %autoreload 2

import itertools
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import matplotlib.colors as mcolors

import corner
import lymph

import sys, os
from icd_definitions import desc_to_icd, icd_to_desc
from lyscripts.sample import sample_from_model
from em_sampling import em_sampler
from util_2 import (
    set_size,
    usz_red,
    usz_colors,
    create_models,
    create_prev_vectors,
)

# from model_functions import log_prob_fn, emcee_sampling_ext
import logging

logger = logging.getLogger(__name__)


# %%
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
    overlap = 0.2
    w_icd = w_data / (2 * overlap + len(icds_codes) * (1 - 2 * overlap))
    icd_spacing = (w_data - w_icd) / (len(icds_codes) - 1)
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

    for i, prev_icd in enumerate(prev_icds[::-1]):
        j = len(prev_icds) - i - 1
        print(icds_codes[j])
        plt.bar(
            pos_icds[j],
            height=prev_icd,
            color=fade_colors[j],
            label=icds_codes[j],
            **bar_kwargs,
        )
    plt.bar(pos, prev_loc, width=w_data, color=usz_red, hatch="////", alpha=0.0)
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
    plt.show()


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


def load_data(datasets_names):
    dataset = pd.DataFrame({})
    for ds in datasets_names:
        dataset_new = pd.read_csv(
            Path("../../data/datasets/enhanced/" + ds), header=[0, 1, 2]
        )
        dataset = pd.concat([dataset, dataset_new], ignore_index=True)
    print(f"Succesfully loaded {len(dataset)} Patients")
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

    # For reference (or initial clustering), cluster the subsites based on the location
    location_to_cluster = {
        "oral cavity": 0,
        "oropharynx": 1,
        "hypopharynx": 2,
        "larynx": 3,
    }
    dataset["tumor", "1", "clustering"] = dataset["tumor"]["1"]["location"].apply(
        lambda x: location_to_cluster[x]
    )
    return dataset


def log_prob_fn(theta: np.ndarray | list) -> float:
    global MODEL
    for t in theta:
        if t < 0 or 1 < t:
            return -10000
    llh = MODEL.likelihood(given_param_args=theta, log=True)
    if np.isnan(llh):
        llh = -10000
    if np.isinf(llh):
        llh = -10000
    return llh


# %%
if __name__ == "__main__":
    global N_SUBSITES, N_CLUSTERS, MODELS
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.debug("This is a debug message.")
    logger.info("This is a info message.")
    ############################
    # GLOBAL DEFINITIONS
    ############################
    PLOT_PATH = Path("./figures/expl_oc_hp")
    SAMPLE_PATH = Path("./samples/expl_oc_hp/")

    graph = {
        ("tumor", "primary"): ["I", "II", "III", "IV"],
        ("lnl", "I"): [],
        ("lnl", "II"): ["I", "III"],
        ("lnl", "III"): ["IV"],
        ("lnl", "IV"): [],
    }

    ############################
    # DATA HANDLING
    ############################

    ## Definitions
    ignore_t = True
    if ignore_t:
        convert_t_stage = {0: "all", 1: "all", 2: "all", 3: "all", 4: "all"}
    else:
        convert_t_stage = {0: "early", 1: "early", 2: "early", 3: "late", 4: "late"}
    t_stages = list(set(convert_t_stage.values()))
    datasets_names = [
        "2022_CLB_multisite_enhanced.csv",
        "2022_CLB_oropharynx_enhanced.csv",
        "2022-ISB-multisite_enhanced.csv",
    ]

    location_to_include = ["oral cavity", "oropharynx"]
    icds_to_neglect = ["C00", "C08"]

    dataset = load_data(datasets_names)
    dataset = enhance_data(dataset, convert_t_stage)

    # Check which icds we want to include in the mixture model. If a threshold is given, icd codes with patients less than the threshold will be excluded for training.
    # Check which subsites belong to OC and OP
    loc_to_mask = {
        loc: dataset["tumor"]["1"]["location"] == loc for loc in location_to_include
    }

    loc_to_icds_full = {
        loc: dataset[loc_to_mask[loc]]["tumor", "1", "majorsubsites"].unique()
        for loc in location_to_include
    }

    for k, v in loc_to_icds_full.items():
        print(f"Location {k}: {v}")
        print(f"{[icd_to_desc[icd] for icd in v]}")

    # Filtering out unwanted ICDs
    loc_to_icds_model = {
        k: [icd for icd in v if icd not in icds_to_neglect]
        for k, v in loc_to_icds_full.items()
    }

    icd_to_loc_model = reverse_dict(loc_to_icds_model)

    icd_to_masks = {}
    for icd in list(chain(*loc_to_icds_model.values())):
        icd_to_masks[icd] = dataset["tumor", "1", "majorsubsites"] == icd

    lnls_full = graph["tumor", "primary"]

    # %%
    ############################
    # Plots & Create Dataframes
    ############################
    PLOT_N_PATIENTS = False
    PLOT_OBSERVED_PREV = False
    # Color mapping for locations
    color_map = {loc: usz_colors[i] for i, loc in enumerate(location_to_include)}

    if PLOT_N_PATIENTS:
        # Counting the occurrence of each ICD
        icd_value_counts_sorted = dataset[
            np.logical_or.reduce(list(icd_to_masks.values()))
        ]["tumor", "1", "majorsubsites"].value_counts()
        # Assigning colors to each ICD based on its location
        colors = [
            color_map[icd_to_loc_model[icd]] for icd in icd_value_counts_sorted.keys()
        ]

        # labels = [icd_to_loc_model[icd] for icd in icd_value_counts_sorted.keys()]

        fig, ax = plt.subplots(1, figsize=set_size("full"))
        icd_value_counts_sorted.plot(kind="barh", ax=ax, color=colors)

        legend_elements = []
        for loc in location_to_include:
            legend_elements.append(
                mpatches.Patch(color=color_map[loc], label=loc.capitalize())
            )
        ax.set_xlabel("Number of Patients")
        ax.set_ylabel("ICD's")
        fig.legend(handles=[e for e in legend_elements])
        fig.savefig(PLOT_PATH / "n_patients_icd.png")
        plt.show()

    if PLOT_OBSERVED_PREV:
        # Plot every location in a single plot if one location has more than one subsite.
        if any(len(loc_to_icds_model[loc]) > 1 for loc in location_to_include):
            for loc in location_to_include:
                # Here we do not consider early and late
                prev_loc = create_prev_vectors(
                    dataset[loc_to_mask[loc]],
                    lnls_full,
                    t_stages=t_stages,
                    full_involvement=True,
                    plot=False,
                )
                prev_loc_icds = [
                    create_prev_vectors(
                        dataset[icd_to_masks[icd]],
                        lnls_full,
                        t_stages=t_stages,
                        full_involvement=True,
                    )
                    for icd in loc_to_icds_model[loc]
                ]

                plot_prevalences_icd(
                    prev_loc_icds,
                    prev_loc,
                    lnls_full,
                    loc_to_icds_model[loc],
                    color_map[loc],
                    save_name=PLOT_PATH
                    / f"prev_{loc}_{convert_lnl_to_filename(lnls_full)}.png",
                )

                prev_loc_icds.append(prev_loc)
                prev_loc_df = pd.DataFrame(
                    np.round(np.array(prev_loc_icds) * 100, 2),
                    index=[*loc_to_icds_model[loc], loc.capitalize()],
                    columns=[str(s) for s in lnls_full],
                ).T
                prev_loc_df.to_csv(
                    f"./prevalences/prev_{loc}_{convert_lnl_to_filename(lnls_full)}.csv"
                )

    # %%
    ############################
    # Independent Prediction of location
    ############################

    indep_nburnin = 250
    indep_nstep = 500
    params_sampling = {
        "walkers_per_dim": 20,
        "nsteps": 100,
        "thin_by": 1,
        "nburnin": 200,
    }

    indep_plot_corner = False

    # We need global variables, such that the likelihood functions work
    global MODEL

    models_loc = {}
    for loc in location_to_include:
        model = create_models(1, graph, ignore_t_stage=ignore_t)
        model.load_patient_data(
            dataset[loc_to_mask[loc]], mapping=lambda x: convert_t_stage[x], side="ipsi"
        )
        models_loc[loc] = model
        # n_params = len(model.get_params())
        sample_name = Path(
            f"samples_ind_{loc}_{convert_lnl_to_filename(lnls_full)}_{ignore_t}"
        )
        output_dir = SAMPLE_PATH / sample_name
        if output_dir.with_suffix(".npy").exists():
            sampling_results = np.load(output_dir.with_suffix(".npy"))
        else:
            sampling_results = sample_from_model(
                model,
                params_sampling,
                SAMPLE_PATH,
                sample_name=sample_name,
                store_as_chain=True,
            )

    # %%
    if indep_plot_corner:
        for loc in location_to_include:
            # The sample name has to be the same as defined in sampling.
            sample_name = (
                f"samples_ind_{loc}_{convert_lnl_to_filename(lnls_full)}_{ignore_t}"
            )
            sample_dir = SAMPLE_PATH / sample_name
            samples = np.load(sample_dir.with_suffix(".npy"))

            # Ugly, may change to more robust function (e.g. get_lables)
            label_ts = [
                t.replace("primary", "T").replace("_spread", "")
                for t in models_loc[loc].get_params(as_dict=True).keys()
            ]
            fig = corner.corner(
                samples,
                labels=label_ts,
                show_titles=True,
            )
            fig.suptitle(f"{loc}", fontsize=16)
            fig.tight_layout()
            fig.savefig(PLOT_PATH / f"corner_ind_{loc}.png")
            plt.show()

    # %%
    ############################
    # Run EM sampling for the icd codes
    ############################
    # if True:
    N_CLUSTERS = 2
    N_SUBSITES = len(icd_to_loc_model.keys())
    graph_debug = {
        ("tumor", "primary"): ["I", "II", "III"],
        ("lnl", "I"): [],
        ("lnl", "II"): ["I", "III"],
        ("lnl", "III"): [],
    }

    # For now, we create the models already here and pass it to the em_sampling.py file.
    # Later the em_sampling.py shoudl contain a function em_sampler which has the following structure:
    # Desired Structure:
    # final_weights, final_model_params, history? = em_sampler(data, icds, N_CLUSTERS, **em_params, **final_sampling_params)

    models_MM = create_models(
        N_SUBSITES,
        graph_debug,
        ignore_t_stage=ignore_t,
        n_mixture_components=N_CLUSTERS,
    )
    for i, (k, v) in enumerate(icd_to_masks.items()):
        models_MM[i].load_patient_data(dataset[v], mapping=lambda x: convert_t_stage[x])
        print(f"Loaded patients for ICD {k} (total {v.sum()} patients)")
    # %%
    # if True:
    #     %load_ext autoreload
    #     %autoreload 2
    #     import importlib
    #     import em_sampling
    #     importlib.reload(em_sampling)
    #     from em_sampling import em_sampler

    #     em_path = SAMPLE_PATH / Path("em_samples/script/")
    #     save_name = f"2samples_hc_op_test2"
    #     final_weights, final_model_params, history = em_sampler(
    #         models_MM, N_CLUSTERS, em_path, save_name
    #     )

    #     # %%
    #     from em_sampling import plot_history

    #     plot_history(history, icd_to_loc_model.keys(), models_MM, n_clusters=N_CLUSTERS)

    # %%
    # Do Final Sampling based on the icd codes.
    # if True:

    import importlib
    import em_sampling

    importlib.reload(em_sampling)
    from em_sampling import em_sampler

    from em_sampling import assign_mixing_parameters
    import mixture_model

    importlib.reload(mixture_model)
    from mixture_model import LymphMixtureModel

    LMM = LymphMixtureModel(
        models_MM,
        n_clusters=N_CLUSTERS,
        base_dir=SAMPLE_PATH.joinpath(Path("LMM_Test/")),
        name="LMM_Test",
    )

    em_config = {
        "max_steps": 10,
        "method": "Default",
        "sampling_params": {
            "params_for_expectation": {
                "walkers_per_dim": 20,
                "nsteps": 10,
                "nburnin": 5,
                "sampler": "SIMPLE",
            },
            "params_for_maximation": {"minimize_method": "SLSQP"},
        },
    }

    mcmc_config = {
        "sampler": "PRO",
        "sampling_params": {
            "walkers_per_dim": 20,
            "nsteps": 200,
            "nburnin": 300,
        },
    }
    LMM.cluster_assignments = [0.35, 0.12, 0.19, 0.4, 0.21, 0.89, 0.81, 0.67]
    # Run the EM algorithm and sample from the found cluster assignmnet
    LMM.fit(em_config=em_config, mcmc_config=mcmc_config, do_plot_history=True)
    # LMM.plot_cluster_parameters()

    LMM.plot_cluster_assignment_matrix(labels=list(icd_to_loc_model.keys()))

    LMM.model_labels = list(icd_to_loc_model.keys())

    # %%
    # if True:
    # Predictions!! under construction
    icd = "C02"
    # i know that c02 is at first position this is why i take the first model
    model_c02 = LMM.lymph_models[0]

    print(f"Prevalence for {icd}")

    lnls_debug = graph_debug["tumor", "primary"]

    # create the patterns dataframe for all LNL's
    states_all_raw = [
        list(combination)
        for combination in itertools.product([0, 1], repeat=len(lnls_debug))
    ]
    # states_all = [
    #     {lnls_debug[-(i + 1)]: p[i] for i in range(len(lnls_debug))}
    #     for p in states_all_raw
    # ]
    states_all = [
        {lnls_debug[i]: True if i == j else None for i in range(len(lnls_debug))}
        for j in range(len(lnls_debug))
    ]

    # a, b = LMM.predict_with_model(model_c02, states_all, lnls_debug, "test")
    # a, b = LMM.predict_with_model(model_c02, states_all, lnls_debug, "test")
    # LMM.load_data([dataset[m] for m in icd_to_masks.values()])
    # c = LMM.create_result_df(states_all, lnls_debug, save_name="test")

    # print(c)

    # %% Check if indepenent comparison is working
    for loc in location_to_include:
        logger.info(f"Result Dataframe for {loc}")
        # Create the model and load the independet samples
        model = create_models(1, graph_debug, ignore_t_stage=ignore_t)
        model.load_patient_data(
            dataset[loc_to_mask[loc]], mapping=lambda x: convert_t_stage[x], side="ipsi"
        )
        models_loc[loc] = model
        # n_params = len(model.get_params())
        sample_name = Path(
            f"samples_ind_{loc}_{convert_lnl_to_filename(lnls_full)}_{ignore_t}"
        )
        output_dir = SAMPLE_PATH / sample_name
        if output_dir.with_suffix(".npy").exists():
            sampling_results = np.load(output_dir.with_suffix(".npy"))

        c = LMM.create_result_df(
            states_all,
            lnls_debug,
            labels=loc_to_icds_model[loc],
            independent_model=model,
            independent_model_samples=sampling_results,
            save_name=f"test_{loc}",
        )
