import random
from typing import Dict, Generator, List, Optional

import pandas as pd
import numpy as np
from lyscripts.predict.risks import predicted_risk
from lyscripts.predict.utils import complete_pattern
import lymph
from lyscripts.predict.prevalences import (
    create_patient_row,
    compute_observed_prevalence,
    generate_predicted_prevalences,
    # compute_predicted_prevalence_for_mixture,
)

# from lymixture.mixture_model import LymphMixtureModel

## Prevalences

def mm_generate_predicted_prevalences(
    cluster_assignments: np.ndarray,
    cluster_parameters: np.ndarray,
    pattern: Dict[str, Dict[str, bool]],
    model: lymph.models.Unilateral,
    t_stage: str = "early",
    modality_spsn: Optional[List[float]] = None,
    invert: bool = False,
    **_kwargs,
) -> Generator[float, None, None]:
    """Wrapper for the `lyscript predict` function."""

    # The number of clusters are extracted from the length of cluster assignments given:
    n_clusters = len(cluster_assignments)
    n_p = len(model.get_params())

    cluster_predictions = np.zeros(shape=(n_clusters))
    for k in range(n_clusters):
        k_cluster_parameters = cluster_parameters[:, k * n_p : (k + 1) * n_p]
        cluster_predictions[k] = next(
            generate_predicted_prevalences(
                pattern,
                model,
                k_cluster_parameters,
                t_stage=t_stage,
                modality_spsn=modality_spsn,
                invert=invert,
                **_kwargs,
            )
        )
    return cluster_assignments @ cluster_predictions


## Risks


def mm_predicted_risk(
    involvement: Dict[str, Dict[str, bool]],
    model: lymph.models.Unilateral,
    cluster_assignment: np.ndarray,
    cluster_parameters: np.ndarray,
    t_stage: str,
    midline_ext: bool = False,
    given_diagnosis: Optional[Dict[str, Dict[str, bool]]] = None,
    given_diagnosis_spsn: Optional[List[float]] = None,
    invert: bool = False,
    **_kwargs,
) -> Generator[float, None, None]:
    """Wrapper for the `lyscript predicted_risk` function."""

    if midline_ext is not False:
        raise NotImplementedError
    # The number of clusters are extracted from the length of cluster assignments given:
    n_clusters = len(cluster_assignment)
    n_p = len(model.get_params())

    cluster_risks = np.zeros(shape=(n_clusters))
    for k in range(n_clusters):
        k_cluster_parameters = cluster_parameters[:, k * n_p : (k + 1) * n_p]
        cluster_risks[k] = next(
            predicted_risk(
                involvement,
                model,
                k_cluster_parameters,
                t_stage,
                midline_ext,
                given_diagnosis,
                given_diagnosis_spsn,
                invert,
                **_kwargs,
            )
        )
    return cluster_assignment @ cluster_risks


def _create_obs_pred_df_single(
    samples, model, data, patterns, lnls, save_name=None, n_samples=100
):
    """
    Messy function which creates a dataframe comparing observed / predictions. Also
    returns object which can be used for histogram plots.
    """
    random_idx = random.sample(range(samples.shape[0]), n_samples)
    samples_to_use = [samples[i, :] for i in random_idx]
    obs_prev = {}
    pred_prev = {}
    pred_prev_std = {}
    pred_prev_list = {}
    df_dict = {}
    t_stages = list(model.diag_time_dists)
    for t_stage in t_stages:
        obs_prev_t = {}
        pred_prev_t = {}
        pred_prev_std_t = {}
        pred_prev_list_t = {}
        df_dict_t = {}

        for pattern in patterns:
            pname = pattern
            op = compute_observed_prevalence(
                pattern={"ipsi": pattern},
                data=data,
                t_stage=t_stage,
                lnls=lnls,
            )
            # logger.info(f"{loc}: Observed P({lnl}) = {round(op[0]/op[1], 2)} ({op})")
            pp_list = list(
                generate_predicted_prevalences(
                    pattern={"ipsi": pattern},
                    model=model,
                    samples=samples_to_use,
                    t_stage=t_stage,
                )
            )
            # predicted_prevalence overwrites the patient data, therefore we have to load it again.
            obs_prev_t[str(pname)] = op[0] / op[1] if op[1] != 0 else 0
            pred_prev_t[str(pname)] = np.mean(pp_list)
            pred_prev_std_t[str(pname)] = np.std(pp_list)
            pred_prev_list_t[str(pname)] = (pp_list, op)
            df_dict_t[str(pname)] = {
                "obs": op[0] / op[1] if op[1] != 0 else 0,
                "pred": np.mean(pp_list),
                "pred_tot": np.mean(pp_list) * op[1],
                "diff": (op[0] / op[1] if op[1] != 0 else 0) - np.mean(pp_list),
                "n/t": op,
                "n": op[0],
                "t": op[1],
            }

            # logger.info(f"{loc}: Predicted P({lnl}) = {round(pp_list.mean(), 2)}")
        obs_prev[t_stage] = obs_prev_t
        pred_prev[t_stage] = pred_prev_t
        pred_prev_std[t_stage] = pred_prev_std_t
        pred_prev_list[t_stage] = pred_prev_list_t
        df_dict[t_stage] = df_dict_t

    df_obs_pred = pd.DataFrame((df_dict), columns=[["oral cavity"]])
    df = pd.DataFrame.from_dict(
        {
            (stage, state): df_dict[stage][state]
            for stage in df_dict.keys()
            for state in df_dict[stage].keys()
        },
        orient="columns",
    )
    df = df.stack().unstack(level=0)
    if len(t_stages) == 2:
        df["tot", "obs"] = (df["early", "n"] + df["late", "n"]) / (
            df["early", "t"] + df["late", "t"]
        )
        df["tot", "pred"] = (
            df["early", "pred"] * df["early", "t"]
            + df["late", "pred"] * df["late", "t"]
        ) / (df["early", "t"] + df["late", "t"])
        # df['tot', "n/t"] = (df["early", "n"] + df["late", "n"], df["early", "t"]+df["late", "t"])
        pred_prev_list["all"] = {
            k: (
                random.sample(
                    pred_prev_list["early"][k][0] + pred_prev_list["late"][k][0],
                    len(pred_prev_list["early"][k][0]),
                ),
                (
                    pred_prev_list["early"][k][1][0] + pred_prev_list["late"][k][1][0],
                    pred_prev_list["early"][k][1][1] + pred_prev_list["late"][k][1][1],
                ),
            )
            for k in pred_prev_list["early"].keys()
        }
    if save_name is not None:
        df.to_csv(save_name)
    return df, [obs_prev, pred_prev, pred_prev_std, pred_prev_list]


# def generate_predicted_prevalences_for_mixture(
#     pattern: Dict[str, Dict[str, bool]],
#     model: lymph.models.Unilateral,
#     n_clusters: int,
#     cluster_assignment: np.ndarray,
#     samples: np.ndarray,
#     t_stage: str = "early",
#     modality_spsn: Optional[List[float]] = None,
#     invert: bool = False,
#     **_kwargs,
# ) -> Generator[float, None, None]:
#     """Compute the prevalence of a given `pattern` of lymphatic progression using a
#     `model` and trained `samples`.

#     Do this computation for the specified `t_stage` and whether or not the tumor has
#     a `midline_ext`. `modality_spsn` defines the values for specificity & sensitivity
#     of the diagnostic modality for which the prevalence is to be computed. Default is
#     a value of 1 for both.

#     Use `invert` to compute 1 - p.
#     """
#     lnls = list(model.graph.lnls.keys())
#     pattern = complete_pattern(pattern, lnls)

#     if modality_spsn is None:
#         modality_spsn = [1., 1.]

#     model.modalities = {"max_llh": modality_spsn}

#     if not isinstance(model, lymph.models.Unilateral):
#         raise NotImplementedError()

#     patient_row = create_patient_row(
#         pattern,
#         t_stage,
#         False,
#         make_unilateral=True,
#     )
#     if t_stage in ["early", "late"]:
#         mapping = None
#     else:
#         mapping = lambda x: "all"

#     # Create an instance of the mixture model
#     lmm = LymphMixtureModel(model, n_clusters=n_clusters, n_subpopulation=1)
#     # assign the cluster assignment to the model.
#     lmm.cluster_assignments = cluster_assignment[:-1]
#     # load the patient row data
#     lmm.load_data([patient_row], mapping=mapping)

#     # compute prevalence as likelihood of diagnose `prev`, which was defined above
#     for sample in samples:
#         prevalence = compute_predicted_prevalence_for_mixture(
#             loaded_mixture_model=lmm,
#             given_params=sample,
#         )
#         yield (1.0 - prevalence) if invert else prevalence


def create_obs_pred_df_single(
    samples_for_predictions: np.ndarray,
    model: lymph.models.Unilateral,
    n_clusters,
    cluster_assignment,
    data_input,
    patterns,
    lnls,
    save_name=None,
    n_predictions=None,
):
    """
    Uses lyscripts methods to create a DataFrame comparing observed and predicted prevalences.

    Args:
        samples: MCMC samples for the model.
        model: The lymph model object.
        data: Data used for predictions.
        patterns: Patterns to generate predictions for.
        lnls: Lymph node levels to consider.
        save_name: Path to save the DataFrame as CSV (optional).
        n_samples: Number of samples to use for predictions.

    Returns:
        A DataFrame with observed and predicted prevalences, and additional statistics.
    """
    # We need to create a copy since else it could lead to problems
    data = data_input.copy(deep=True)

    # Sample from the sample chain, if n_predictions is given
    samples = samples_for_predictions
    if n_predictions is not None:
        random_idx = random.sample(
            range(samples_for_predictions.shape[0]), n_predictions
        )
        samples = [samples_for_predictions[i, :] for i in random_idx]

    # Initialize dictionaries for storing results
    df_dict = {}

    for t_stage in model.diag_time_dists:
        df_dict_t = {}
        for pattern in patterns:
            op = compute_observed_prevalence(
                pattern={"ipsi": pattern}, data=data, t_stage=t_stage, lnls=lnls
            )
            pp_list = list(
                generate_predicted_prevalences_for_mixture(
                    pattern={"ipsi": pattern},
                    model=model,
                    n_clusters=n_clusters,
                    cluster_assignment=cluster_assignment,
                    samples=samples,
                    t_stage=t_stage,
                )
            )
            # After each prediciton, we need to set the patient data again (direct, since we do not have t_stage information here)
            model._patient_data = data_input

            observed = op[0] / op[1] if op[1] != 0 else 0
            predicted = np.mean(pp_list)

            df_dict_t[str(pattern)] = {
                "obs": observed,
                "pred": predicted,
                "pred_tot": predicted * op[1],
                "diff": observed - predicted,
                "n/t": op,
                "n": op[0],
                "t": op[1],
            }

        df_dict[t_stage] = df_dict_t

    # Create DataFrame from dictionary
    df = pd.DataFrame.from_dict(
        {
            (stage, state): df_dict[stage][state]
            for stage in df_dict
            for state in df_dict[stage]
        },
        orient="columns",
    )
    df = df.stack().unstack(level=0)

    # Additional calculations for stages if necessary
    if len(model.diag_time_dists) == 2:
        df["tot", "obs"] = (df["early", "n"] + df["late", "n"]) / (
            df["early", "t"] + df["late", "t"]
        )
        df["tot", "pred"] = (
            df["early", "pred"] * df["early", "t"]
            + df["late", "pred"] * df["late", "t"]
        ) / (df["early", "t"] + df["late", "t"])

    # Save DataFrame if a filename is provided
    if save_name:
        df.to_csv(save_name)
    return df, df_dict
