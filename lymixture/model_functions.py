from typing import Any, Generator

import lymph
import numpy as np
import pandas as pd


def gen_diagnose_matrices(
    datasets: list[pd.DataFrame],
    lymph_model: lymph.models.Unilateral,
    load_kwargs: dict[str, Any] | None = None,
) -> Generator[np.ndarray, None, None]:
    """Generate the diagnose matrices for the individual subgroups."""
    if load_kwargs is None:
        load_kwargs = {}

    for dataset in datasets:
        lymph_model.load_patient_data(dataset, **load_kwargs)
        for t_stage in lymph_model.t_stages:
            yield lymph_model.diagnose_matrices[t_stage]


def create_DS_list_from_models(models: list[lymph.models.Unilateral]):
    """
    Creates the diagnose matrices for each subsite S, given lymph models with loaded
    data. Returns a list of S arrays with shape (tau, 2^V, n_patients).
    """
    DS = []
    for m in models:
        DS.append(np.array([m.diagnose_matrices[t_stage] for t_stage in m.t_stages]))
    return DS


def compute_cluster_assignment_matrix(cluster_assignments, n_S, n_K):
    """
    returns the PI matrix with shape (S, K) containing the cluster assignment for each
    subsite S to cluster K. Return np.ndarray with shape (S, K)
    """
    PI = np.zeros(shape=(n_S, n_K))
    for s in range(n_S):
        ca = cluster_assignments[s * (n_K - 1) : (s + 1) * (n_K - 1)]
        PI[s] = [*ca, 1 - sum(ca)]
    return PI


def compute_cluster_state_probabilty_matrices(
    cluster_params,
    k_model,
    n_clusters,
):
    """
    Generates a np.array with shape (K, t , 2^V) containing all 2^V state probabilitiy
    matrices for each t_stage t and cluster K
    """
    n_V = len(k_model.state_list)
    n_K = n_clusters
    n_t = len(list(k_model.t_stages))
    n_p = len(k_model.get_params())
    mu_sum = np.zeros(shape=(n_K, n_t, n_V))
    for k in range(n_clusters):
        k_model.assign_params(*cluster_params[k * n_p : (k + 1) * n_p])
        evolved_model = k_model.comp_dist_evolution()
        for t, t_stage in enumerate(k_model.t_stages):
            mu_sum[k][t] = k_model.diag_time_dists[t_stage].distribution @ evolved_model
    return mu_sum


def compute_state_probability_matrices(
    cluster_assignment_matrix, cluster_state_probabilty_matrices
):
    """
    computes the :math:`\Kappa` matrices for each subsite S and t_stage tau. Generates
    an np.ndarray with shape (S, tau, 2^V)
    """
    return np.einsum(
        "sk,ktv->stv", cluster_assignment_matrix, cluster_state_probabilty_matrices
    )
