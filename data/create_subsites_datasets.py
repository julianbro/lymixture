import itertools
import os
from pathlib import Path
from typing import List
from scipy.special import factorial
from matplotlib import pyplot as plt
import yaml
import argparse
import sys

import numpy as np
import scipy as sp
import pandas as pd
from core.util_2 import set_size

import lymph
from lyscripts.predict.prevalences import compute_observed_prevalence


def read_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


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
):
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


def create_prev_vectors(data, lnls, plot=False, title=None, save_figure=False, ax=None):
    states_all_raw = [
        list(combination) for combination in itertools.product([0, 1], repeat=len(lnls))
    ]
    states_all = [
        {lnls[-(i + 1)]: p[i] for i in range(len(lnls))} for p in states_all_raw
    ]
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

        ax.set_yticks(range(len(X_inv_list)), [str(s) for s in states_all_raw])
        ax.tick_params(axis="x", which="both")
        if save_figure:
            fig.savefig(PLOT_PATH / f"prev_vectors_{title}_{str(lnls)}.svg")
        if ax is None:
            fig.show()
    if len(states_all) > 4:
        print(f"Prev Vector: {X_inv_list}")
    else:
        prev_formatted = "".join(f"{l}: {p}" for l, p in zip(states_all, X_inv_list))
        print(f"Prev Vector: {prev_formatted}")
    return X_inv_list


def generate_patients_df(
    params_0, params_1, n, ratio, graph, t_stage_dist=None, header="Default"
):
    "Creates a model and generates the synth dataset using the given parameters. If a ratio is params_1, then a mixed dataset is created."
    gen_model = create_models(1, graph=graph)
    gen_model.assign_params(*params_0)
    data_synth_0 = gen_model.generate_dataset(
        int(n * ratio), t_stage_dist, column_index_levels=header
    )
    if params_1 is not None:
        gen_model.assign_params(*params_1)
        data_synth_1 = gen_model.generate_dataset(
            int(n * (1 - ratio)), t_stage_dist, column_index_levels=header
        )

        data_synth_mixed = pd.concat([data_synth_0, data_synth_1], ignore_index=True)
        return data_synth_mixed
    return data_synth_0


import pandas as pd


def create_dataset(
    graph: dict,
    n: List[int],
    params: List[float],
    locations: List[str],
    t_dist: dict,
    name: str,
    save_dataset: bool = True,
    plot: bool = True,
):
    """Creates a synth dataset with the given parameters. Make sure to pass everything as a list"""
    lnls = graph.get(("tumor", "primary"), [])

    data_list = []
    for i, loc in enumerate(locations):
        loc_data = generate_patients_df(
            params[i], None, n[i], 1, graph, t_dist, header="nd"
        )
        loc_data[("tumor", "1", "location")] = [loc] * len(loc_data)
        loc_data[("tumor", "1", "subsite")] = [loc] * len(loc_data)
        data_list.append(loc_data)

        if plot:
            create_prev_vectors(loc_data, lnls=lnls, plot=True)

    data_stacked = pd.concat([d for d in data_list])

    file_dir = os.path.dirname(__file__)
    output_path = f"/datasets/enhanced/synth/{name}.csv"
    if save_dataset:
        data_stacked.to_csv(file_dir + output_path)
    return data_stacked


def main():
    """Create a synth data frame from config, or by settings"""
    parser = argparse.ArgumentParser(description="Run the script with YAML config")
    parser.add_argument(
        "-c", "--config", help="Path to configuration file", required=False
    )
    args = parser.parse_args()

    if args.config:
        config = read_yaml(args.config)
    else:
        # Define the parameters directly in the script if not using a YAML file
        config = {
            "name": "synth_s1_s2_n0",
            "n": [200, 200, 200],
            "params": [
                [0.3, 0.0, 0.3, 0.0, 0.0],
                [0.0, 0.6, 0.6, 0.0, 0.0],
                [0.01, 0.01, 0.01, 0.01, 0.01],
            ],
            "locations": ["S1", "S2", "S3"],
            "t_dist": {"early": 1, "late": 0},
            "graph_lnl_I_II": {
                ("tumor", "primary"): ["I", "II", "III"],
                ("lnl", "I"): [],
                ("lnl", "II"): ["I", "III"],
                ("lnl", "III"): [],
            },
        }

    create_dataset(**config)


if __name__ == "__main__":
    main()
