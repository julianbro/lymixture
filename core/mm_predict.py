import random
import lymph
from lyscripts.predict.prevalences import (
    compute_observed_prevalence,
    compute_predicted_prevalence,
    generate_predicted_prevalences,
)

import pandas as pd
import numpy as np


def _create_obs_pred_df_single(
    samples, model, data, patterns, lnls, save_name=None, n_samples=100
):
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
            # print(f"{loc}: Observed P({lnl}) = {round(op[0]/op[1], 2)} ({op})")
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

            # print(f"{loc}: Predicted P({lnl}) = {round(pp_list.mean(), 2)}")
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

    if save_name is not None:
        df.to_csv(save_name)
    return df, [obs_prev, pred_prev, pred_prev_std, pred_prev_list]


def create_obs_pred_df_single(
    samples,
    model: lymph.models.Unilateral,
    data_input,
    patterns,
    lnls,
    save_name=None,
    n_samples=100,
):
    """
    Creates a DataFrame comparing observed and predicted prevalences.

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
    random_idx = random.sample(range(samples.shape[0]), n_samples)
    samples_to_use = samples[random_idx, :]

    # We need to create a copy since else it could lead to problems
    data = data_input.copy(deep=True)

    # Initialize dictionaries for storing results
    df_dict = {}

    for t_stage in model.diag_time_dists:
        df_dict_t = {}
        for pattern in patterns:
            op = compute_observed_prevalence(
                pattern={"ipsi": pattern}, data=data, t_stage=t_stage, lnls=lnls
            )
            pp_list = list(
                generate_predicted_prevalences(
                    pattern={"ipsi": pattern},
                    model=model,
                    samples=samples_to_use,
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
    return df, None
