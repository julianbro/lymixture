# %%
# %load_ext autoreload
# %autoreload 2

import itertools
from pathlib import Path
import matplotlib as mpl
import pandas as pd
import logging
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
    fade_to_white,
    convert_lnl_to_filename,
    plot_prevalences_icd,
    reverse_dict,
    load_data,
    enhance_data,
    create_states,
)

# Set styling params
mpl.rcParams.update(mpl.rcParamsDefault)
# plt.style.use('./styles/mplstyle_rl.txt')

# Get the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__))
# Calculate the parent directory's path
parent_directory = os.path.abspath(os.path.join(current_directory, ".."))
# Add the parent directory to the Python path
sys.path.append(parent_directory)

logger = logging.getLogger(__name__)


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
    """
    Simple script that creates the environment for the LymphMixtureModel class. The definitions for each step are done in the specific step.
    This includes:
        1) Data Loading and enhancing
        2) Optional Prevalence Plots
        3) Independent Sampling for the given locations
        4) Init and fit of the mixture components
        5) Predictions using the mixture model
    """
    ############################
    # GLOBAL DEFINITIONS &
    ############################

    global N_SUBSITES, N_CLUSTERS, MODELS

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    PLOT_PATH = Path("figures/expl_oc_hp/")
    SAMPLE_PATH = Path("samples/expl_oc_hp/")

    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.mkdir(parents=True, exist_ok=True)

    graph = {
        ("tumor", "primary"): ["I", "II", "III", "IV"],
        ("lnl", "I"): [],
        ("lnl", "II"): ["I", "III"],
        ("lnl", "III"): ["IV"],
        ("lnl", "IV"): [],
    }

    # Ignore the t-stages (TODO)
    ignore_t = True

    PLOT_N_PATIENTS = True
    PLOT_OBSERVED_PREV = True

    indep_plot_corner = False

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

    # Select which locations you want to analize
    location_to_include = ["oral cavity", "oropharynx"]
    # Icd codes in locations which you want to exlude
    icds_to_neglect = ["C00", "C08"]

    dataset = load_data(datasets_names)
    dataset = enhance_data(dataset, convert_t_stage)

    # Create a mask for the patients for each locations
    loc_to_mask = {
        loc: dataset["tumor"]["1"]["location"] == loc for loc in location_to_include
    }

    # Location to icd dict
    loc_to_icds_full = {
        loc: dataset[loc_to_mask[loc]]["tumor", "1", "majorsubsites"].unique()
        for loc in location_to_include
    }

    # Filtering out unwanted ICDs
    loc_to_icds_model = {
        k: [icd for icd in v if icd not in icds_to_neglect]
        for k, v in loc_to_icds_full.items()
    }

    for k, v in loc_to_icds_model.items():
        logger.info(f"In tumor location {k}: {v}")
        # print(f"{[icd_to_desc[icd] for icd in v]}")

    icd_to_loc_model = reverse_dict(loc_to_icds_model)

    # For each icd, create a mask for the corresponding patients
    icd_to_masks = {}
    for icd in list(chain(*loc_to_icds_model.values())):
        icd_to_masks[icd] = dataset["tumor", "1", "majorsubsites"] == icd

    lnls_full = graph["tumor", "primary"]

    # %%
    ############################
    # Plots & Create Dataframes
    ############################

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
        # plt.show()

        logger.info(f"Created n_patients_icd figure in {PLOT_PATH}")
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
                    PLOT_PATH / f"prev_{loc}_{convert_lnl_to_filename(lnls_full)}.csv"
                )
                logger.info(f"Created prevalence figures in {PLOT_PATH}")

    # %%
    ############################
    # Independent Prediction of each location 
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
            logger.warning(
                f"Found samples in {output_dir}. Skipping independent model sampling. "
            )
        else:
            logger.info(f"Start independent sampling for {loc}..")
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
            # plt.show()
            logger.info(f"Created corner plot for samples for {loc}")

    # %%
    ############################
    # Create the models and loads each model with patient data from a ICD code.
    ############################
    # if True:

    N_CLUSTERS = 2
    N_SUBSITES = len(icd_to_loc_model.keys())
    # For further analysis we use this simplified graph.
    graph_debug = {
        ("tumor", "primary"): ["I", "II", "III"],
        ("lnl", "I"): [],
        ("lnl", "II"): ["I", "III"],
        ("lnl", "III"): [],
    }

    models_MM = create_models(
        N_SUBSITES,
        graph_debug,
        ignore_t_stage=ignore_t,
        n_mixture_components=N_CLUSTERS,
    )
    for i, (k, v) in enumerate(icd_to_masks.items()):
        models_MM[i].load_patient_data(dataset[v], mapping=lambda x: convert_t_stage[x])
        # logger.info(f"Loaded patients for ICD {k} (total {v.sum()} patients)")
    # %%
    ############################
    # Using Mixture Model Class to train the mixture model
    ############################
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
        base_dir=Path("./"),
        name="LMM_Test",
        model_labels=list(icd_to_loc_model.keys()),
    )


    # The current config, is really only for debugging. 
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
            "convergence_ths": 0.015
        },
    }

    mcmc_config = {
        "sampler": "PRO",
        "sampling_params": {
            "walkers_per_dim": 20,
            "nsteps": 100,
            "nburnin": 300,
        },
    }

    # Enable EM Sampling by uncommenting this line.
    # LMM.cluster_assignments = [0.35, 0.12, 0.19, 0.4, 0.21, 0.89, 0.81, 0.67]
    
    # Run the EM algorithm and sample from the found cluster assignmnet
    LMM.fit(em_config=em_config, mcmc_config=mcmc_config, do_plot_history=True)

    LMM.plot_cluster_parameters()

    LMM.plot_cluster_assignment_matrix(labels=list(icd_to_loc_model.keys()))

    # %% Make Predictions with the model. !!Still under construction!!

    ############################
    # Using Mixture Model Class to make predictions for single ICD code
    ############################


    # if True:

    # The idea is to create a result dataframe with observed an predicted values for given 'patterns'
    icd = "C02"
    # i know that c02 is at first position this is why i take the first model
    model_c02 = LMM.lymph_models[0]
    logger.info(f"Single prevalence predictions for {icd}")
    lnls_debug = graph_debug["tumor", "primary"]

    # create the patterns dataframe for all LNL's, (total risk for the lnl's)
    states_all = create_states(lnls_debug)

    a, b = LMM.predict_with_model(model_c02, states_all, lnls_debug, "test")

    # %% Check if indepenent comparison is working

    ############################
    # Using Mixture Model Class to create a full obs/pred results dataframe, and compare it to the indepenently trained models for each location.
    ############################

    for loc in location_to_include:
        logger.info(f"Creating result data for {loc}")
        # Create the loc model and load the independet samples
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
        else:
            raise ValueError()

        c = LMM.create_result_df(
            states_all,
            lnls_debug,
            labels=loc_to_icds_model[loc],
            independent_model=model,
            independent_model_samples=sampling_results,
            save_name=f"test_{loc}",
        )


    #%%
    ############################
    # Predictions for new, unseen ICD code
    ############################
    # TODO