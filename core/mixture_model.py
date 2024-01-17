from functools import cached_property
import logging
import os
from pyexpat import model
import random
import numpy as np
import pandas as pd
import emcee
import lymph

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from corner import corner
from scipy import cluster

from core.costum_types import EMConfigType


from em_sampling import (
    ExpectationMaximization,
    History,
    assign_global_params,
    em_sampler,
    emcee_simple_sampler,
    exceed_param_bound,
    llh_theta_given_z,
    plot_history,
)
from lyscripts.sample import sample_from_global_model_and_configs
from lyscripts.plot.corner import get_param_labels
from lyscripts.plot.utils import save_figure
from model_functions import (
    compute_state_probability_matrices,
    compute_cluster_assignment_matrix,
    compute_cluster_state_probabilty_matrices,
    create_diagnose_matrices,
)
from mm_predict import (
    _create_obs_pred_df_single,
    create_obs_pred_df_single,
    mm_generate_predicted_prevalences,
    mm_predicted_risk,
)
from mm_plotting import plot_cluster_assignments, plot_cluster_parameters


# Define a global variable which can be used within this module.
global LMM_GLOBAL

# class CachedMethod:
#     def __init__(self, method):
#         self.method = method
#         self.cache = {}

#     def __get__(self, instance, owner):
#         def wrapper(*args, **kwargs):
#             # Create a key based on args and kwargs
#             key = (args, tuple(sorted(kwargs.items())))
#             if key not in self.cache:
#                 # Cache miss, call the method and cache the result
#                 self.cache[key] = self.method(instance, *args, **kwargs)
#             return self.cache[key]

#         return wrapper


def get_param_labels_temp(model):
    """
    Temporary method to get parameter labels from a model.

    Args:
        model: Lymph model instance.

    Returns:
        list: List of parameter labels.
    """
    return [
        t.replace("primary", "T").replace("_spread", "")
        for t in model.get_params(as_dict=True).keys()
    ]


def as_list(val):
    """Return val as list if it is not already a list"""
    if not isinstance(val, list):
        return [val]
    else:
        return val


def log_ll_cl_parameters(cluster_parameters):
    if exceed_param_bound(cluster_parameters):
        return -np.inf
    llh = LMM_GLOBAL.mm_hmm_likelihood(cluster_parameters=cluster_parameters)
    if np.isinf(llh):
        return -np.inf
    return llh


class LymphMixtureModel:
    """
    Wrapper model which handles the mixture components (clusters) in the lymph models.

    Args:
        lymph_models (list): List of lymph models.
        n_clusters (int): Number of clusters.
        base_dir (Path): Base directory for saving outputs.
        name (str, optional): Name for the model. Defaults to 'LMM'.
    """

    def __init__(
        self,
        lymph_model: lymph.models.Unilateral,
        n_clusters: int,
        n_subpopulation: int,
        base_dir: Optional[Path] = None,
        name: Optional[str] = None,
        **_kwargs,
    ):
        # Initialize logger
        self._setup_logger()

        # Initialize model configurations
        self.lymph_model = lymph_model
        if not isinstance(lymph_model, lymph.models.Unilateral):
            self.logger.error(
                "Mixture model is only implemented for Unilateral cluster model."
            )
            raise NotImplementedError
        self.n_clusters = n_clusters
        self.n_subpopulation = n_subpopulation

        # Compute number of parameters expected by the model
        self.compute_expected_n()

        # Set up directories
        self.name = name if name else "LMM"
        if base_dir is None:
            base_dir = Path(os.getcwd())
        self.base_dir = base_dir.joinpath(self.name)
        self._create_directories()

        self.logger.info(
            f"Create LymphMixtureModel of type {type(self.lymph_model)} with {self.n_clusters} clusters in {self.base_dir}"
        )

    def _create_directories(self):
        """
        Creates necessary directories for saving outputs.
        """
        self.samples_dir = self.base_dir.joinpath("samples/")
        self.figures_dir = self.base_dir.joinpath("figures/")
        self.predictions_dir = self.base_dir.joinpath("predictions/")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self):
        """
        Sets up the logger for the class.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def delete_cached_property(self, property: str):
        if property in self.__dict__:
            del self.__dict__[property]

    def compute_expected_n(self):
        """(Re-)Computes expected number of various parameters used by the model"""
        self.n_model_params = len(self.lymph_model.get_params())
        self.n_cluster_parameters = self.n_model_params * self.n_clusters
        self.n_cluster_assignments = self.n_subpopulation * (self.n_clusters - 1)
        self.n_states = len(self.lymph_model.state_list)
        self.n_tstages = len(list(self.lymph_model.diag_time_dists.keys()))

    @property
    def subpopulation_labels(self):
        """Get or set the labels for the subpopulations."""
        return self._subpopulation_labels

    @subpopulation_labels.setter
    def subpopulation_labels(self, value: List[str]):
        if len(value) != self.n_subpopulation:
            raise ValueError(
                "Number of labels do not match with number of subpopulations."
            )
        for i, v in enumerate(value):
            if not isinstance(v, str):
                raise ValueError("ICD codes should be passed as list of strings.")
            self.logger.info(
                f"Assigned {v} to supopulations with {len(self.subpopulation_data[i])} patients"
            )
        self._subpopulation_labels = value

    # @CachedMethod
    # def diagnose_matrices(self, **kwargs) -> List[List[np.ndarray]]:
    #     """Holds the diagnose matrices for each subpopulation."""
    #     if self.subpopulation_data is None:
    #         raise ValueError(
    #             "No subpopulation data is given. Please provide subpopulation data via the ``load_data`` method."
    #         )
    #     return create_diagnose_matrices(
    #         self.subpopulation_data, self.lymph_model, **kwargs
    #     )

    @cached_property
    def cluster_state_probabilty_matrices(self) -> np.ndarray:
        """Holds the cluster matrices (i.e the state probabilities of the cluster) for each t-stage and cluster."""
        if self.cluster_parameters is None:
            raise ValueError(
                "No cluster parameters are in the model. Please provide cluster parameters first."
            )
        return compute_cluster_state_probabilty_matrices(
            self.cluster_parameters, self.lymph_model, self.n_clusters
        )

    @cached_property
    def cluster_assignment_matrix(self) -> np.ndarray:
        """Holds the cluster assignment matrix."""
        if self.cluster_assignments is None:
            raise ValueError(
                "No cluster assignments are loaded in the model. Please provide cluster assignments first."
            )
        return compute_cluster_assignment_matrix(
            self.cluster_assignments, self.n_subpopulation, self.n_clusters
        )

    @cached_property
    def state_probability_matrices(self) -> np.ndarray:
        """Holds the state probability matrices for every subpopulation and every t_stage."""
        return compute_state_probability_matrices(
            self.cluster_assignment_matrix, self.cluster_state_probabilty_matrices
        )

    @property
    def cluster_assignments(self):
        """
        Property to get or set the cluster assignments.
        """
        return self._cluster_assignments

    @cluster_assignments.setter
    def cluster_assignments(self, value):
        """
        Setter for cluster assignments that deletes the current cluster matrices. Note that the model expects ``n_subsites`` less parameters for the cluster assignment due to the constraint that the cluster assignment of one subpopulation has to sum to 1.
        """
        if len(value) != self.n_cluster_assignments:
            raise ValueError(
                "Number of provided cluster assignments do not match the number of cluster assignments expected by the model."
            )
        self._cluster_assignments = value

        self.delete_cached_property("cluster_assignment_matrix")
        self.delete_cached_property("state_probability_matrices")

    @property
    def cluster_parameters(self):
        """Property to get or set the cluster parameters"""
        return self._cluster_parameters

    @cluster_parameters.setter
    def cluster_parameters(self, value):
        """Set the cluster parameters and delete the cluster state probability matrices and the state probability matrices."""
        if len(value) != self.n_cluster_parameters:
            raise ValueError(
                "Number of provided cluster_parameters do not match the number of cluster_parameters expected by the model."
            )
        self._cluster_parameters = value

        # Delete the cluster matrices and the state probability matrices
        self.delete_cached_property("cluster_state_probabilty_matrices")
        self.delete_cached_property("state_probability_matrices")

    # def _check_model_consistency(self):
    #     """
    #     Checks if all models have the same definitions. Raises an error if inconsistent.
    #     """
    #     param_counts = [len(lm.get_params()) for lm in self.lymph_models]

    #     if len(set(param_counts)) != 1:
    #         raise ValueError("All models must have the same number of parameters.")

    #     self.n_model_params = param_counts[0]
    #     # self.logger.info(
    #     #     f"Number of parameters per cluster is set to {self.n_model_params}"
    #     # )

    #     diag_time_dists = [list(lm.diag_time_dists) for lm in self.lymph_models]
    #     dtd_first = diag_time_dists[0]
    #     if any(dtd_i != dtd_first for dtd_i in diag_time_dists):
    #         raise ValueError(
    #             "All models must have the same diagnostic time distributions."
    #         )

    #     self.n_cluster_parameters = self.n_model_params * self.n_clusters

    def load_data(self, data: List[pd.DataFrame], labels=None, **kwargs):
        """
        Loads patient data into the model. Patient data is provided as a list of pd.DataFrames. Deletes the diagnose matrices.

        See Also:
            :py:class:`~lymph.models.Unilateral.load_patient_data`

        """
        if len(data) != self.n_subpopulation:
            raise ValueError(
                "Number of provided subpopulation data does not match with the expected number of subpopulations"
            )
        self.subpopulation_data = data

        # If given, add labels to the subpopulations
        if labels is not None:
            self.subpopulation_labels = labels

        # Delete the diagnose matrices
        self.delete_cached_property("diagnose_matrices")
        # And compute the diagnose matrices again.
        # This is done directly here, because only here we have information about how to load the data in the models
        self.diagnose_matrices = create_diagnose_matrices(
            self.subpopulation_data, self.lymph_model, **kwargs
        )

    def _mm_hmm_likelihood(self, log: bool = True) -> float:
        """
        Implements the likelihood function
        """
        llh = 0 if log else 1.0
        # Sum over all subsites..
        for diagnose_matrices_s, state_probability_matrices_s in zip(
            self.diagnose_matrices, self.state_probability_matrices
        ):
            # .. and sum over all t_stages within that subsite.
            for diagnose_matrix, state_probabilty in zip(
                diagnose_matrices_s, state_probability_matrices_s
            ):
                # sum the likelihoods of observing each patient diagnose given the state probability from the mixture.
                if log:
                    llh += np.sum(np.log(state_probabilty @ diagnose_matrix))
                else:
                    llh *= np.prod(state_probabilty @ diagnose_matrix)
        return llh

    def mm_hmm_likelihood(
        self,
        cluster_parameters=None,
        cluster_assignments=None,
        data: pd.DataFrame | None = None,
        load_data_kwargs: dict[str, Any] | None = None,
        log: bool = True,
    ):
        """Compute the likelihood of the ``data`` given the model, theta and cluster assignment.

        Parameters:
        - cluster_parameters: The cluster parameters for the mixture model. If not provided, the
          model uses the existing parameters set within the model.
        - cluster_assignments: The assignment of data points to specific clusters.
          If not provided, existing assignments within the model are used.
        - data: The data for which to compute the likelihood. If not provided, the
          model uses the data already loaded.
        - load_data_kwargs: Additional keyword arguments to pass to the
          :py:meth:`~load_patient_data` method when loading the data.

        Raises:
        - ValueError: If the necessary parameters or data are not provided and are
          also not available within the model.

        Returns:
        - The likelihood of the provided data given the model parameters and settings.
        """

        if cluster_parameters is not None:
            self.cluster_parameters = cluster_parameters

        if cluster_assignments is not None:
            self.cluster_assignments = cluster_assignments

        if data is not None:
            if load_data_kwargs is None:
                load_data_kwargs = {}
            self.load_data(data, **load_data_kwargs)

        return self._mm_hmm_likelihood(log=log)

    def estimate_cluster_assignments(self, em_config=None) -> (np.ndarray, History):
        """Estimates the cluster assignments using an EM algortihm"""
        self.em_algortihm = ExpectationMaximization(lmm=self, em_config=em_config)

        estimated_cluster_assignments, em_history = self.em_algortihm.run_em()

        return estimated_cluster_assignments, em_history

    def _mcmc_sampling(
        self,
        mcmc_config: dict,
        save_samples: bool = True,
        force_resampling: bool = False,
    ):
        """
        Performs MCMC sampling to determine final model parameters.
        """
        global LMM_GLOBAL

        LMM_GLOBAL = self
        sampler = mcmc_config.get("sampler", "SIMPLE")
        sampling_params = mcmc_config["sampling_params"]
        log_prob_fn = log_ll_cl_parameters

        sample_file_name = f"mcmc_sampling_chain_{sampling_params['nsteps']}_{sampling_params['nburnin']}"
        hdf5_backend = emcee.backends.HDFBackend(
            self.samples_dir / f"{sample_file_name}.hdf5", name="mcmc"
        )

        mcmc_chain_dir = self.samples_dir.joinpath(Path(sample_file_name)).with_suffix(
            ".npy"
        )
        if mcmc_chain_dir.exists() and not force_resampling:
            self.logger.info(
                f"MCMC sampling chain found in {mcmc_chain_dir}. Skipping Sampling."
            )
            sample_chain = np.load(mcmc_chain_dir)
            end_point, log_probs = None, None
        else:
            self.logger.info(
                f"Prepared sampling params & backend at {self.samples_dir}"
            )

            if sampler == "SIMPLE":
                self.logger.info("Using simple sampler for MCMC")
                sample_chain, end_point, log_probs = emcee_simple_sampler(
                    log_prob_fn,
                    ndim=self.n_cluster_parameters,
                    sampling_params=sampling_params,
                    starting_point=None,
                    save_dir=self.samples_dir,
                    models=self,
                )
            else:
                self.logger.info("Using lyscript sampler for MCMC")
                (
                    sample_chain,
                    end_point,
                    log_probs,
                ) = sample_from_global_model_and_configs(
                    log_prob_fn,
                    ndim=self.n_cluster_parameters,
                    sampling_params=sampling_params,
                    starting_point=None,
                    hdf5_backend=hdf5_backend,
                    save_dir=self.samples_dir,
                    models=self,
                    verbose=True,
                )

            if save_samples:
                np.save(
                    self.samples_dir / f"{sample_file_name}.npy",
                    sample_chain,
                )

        return sample_chain, end_point, log_probs

    def fit(
        self,
        em_config: Optional[dict] = None,
        mcmc_config: Optional[dict] = None,
        force_resampling: bool = False,
        do_plot_history: bool = False,
        cluster_assignments: np.ndarray = None,
    ):
        """
        Fits the mixture model, i.e. finds the optimal cluster assignments and the cluster parameters, using (1) the EM algorithm and (2) the MCMC sampling method.
        """
        global LMM_GLOBAL
        LMM_GLOBAL = self

        # Skip the EM-Algorithm if there is already a cluster asignments (Only for debug)
        if cluster_assignments is not None:
            # Only for debug
            self.logger.info(
                "Skipping EM Algortihm, since cluster assignment is already given"
            )
            try:
                history = np.load(self.base_dir / "EM/" / "history.npy")
            except:
                history = None
        else:
            cluster_assignments, history = self.estimate_cluster_assignments(em_config)
            np.save(
                self.samples_dir / "final_cluster_assignments.npy", cluster_assignments
            )
            # cluster_assignments = [0.89045091, 0.13857439, 0.89188093]

        # if do_plot_history:
        #     history.plot_history(
        #         self.subpopulation_labels,
        #         list(self.lymph_model.get_params(as_dict=True).keys()),
        #         self.n_clusters,
        #         None,
        #     )

        self.cluster_assignments = cluster_assignments
        # MCMC Sampling
        # Refine the cluster parameters using MCMC based on the assignments from EM
        # Store the final cluster assignments and parameters
        # TODO for now, if a sample exists already, skip sampling for perfomance
        sample_chain = []
        if mcmc_config is not None:
            sample_chain, _, _ = self._mcmc_sampling(
                mcmc_config, force_resampling=force_resampling
            )

        self.cluster_assignments = cluster_assignments
        self.cluster_parameters = sample_chain.mean(axis=0)
        self.cluster_parameters_chain = sample_chain

        return history

    def plot_cluster_parameters(self):
        """Corner Plot for the cluster parameters"""
        """Plots the corner plots for cluster parameters using an external function."""
        labels = get_param_labels_temp(self.lymph_model)
        plot_cluster_parameters(
            self.cluster_parameters_chain,
            self.n_clusters,
            labels,
            self.figures_dir,
            self.logger,
        )

    def plot_cluster_assignment_matrix(self, labels: Optional[List[str]] = None):
        """
        Plots the cluster assignmnet matrix
        """
        if labels is None:
            labels = self.subpopulation_labels
        plot_cluster_assignments(
            self.cluster_assignment_matrix,
            labels=labels,
            save_dir=self.figures_dir,
            logger=self.logger,
        )

    def get_cluster_assignment_for_label(self, label: str):
        """Get the cluster assignment for the given label."""
        try:
            index = self.subpopulation_labels.index(label)
            return self.cluster_assignment_matrix[index, :]

        except ValueError:
            self.logger.error(f"'{label}' is not in the supopulation labelk.")

    def predict_prevalence_for_cluster_assignment(
        self,
        cluster_assignment: np.ndarray,
        for_pattern: Dict[str, Dict[str, bool]],
        t_stage: str = "early",
        modality_spsn: Optional[List[float]] = None,
        invert: bool = False,
        cluster_parameters: Optional[np.ndarray] = None,
        n_samples_for_prediction: int = 200,
        **_kwargs,
    ):
        """Predict Prevalence for a given cluster assignment."""

        # When no cluster parameters are explicitly given, we take the one stored in the model
        if cluster_parameters is None:
            cluster_parameters = self.cluster_parameters_chain
        # Only use n samples to make the prediction
        random_idx = random.sample(
            range(cluster_parameters.shape[0]), n_samples_for_prediction
        )
        cluster_parameters_for_prediction = cluster_parameters[random_idx, :]

        return mm_generate_predicted_prevalences(
            cluster_assignment,
            cluster_parameters_for_prediction,
            for_pattern,
            self.lymph_model,
            t_stage,
            modality_spsn,
            invert,
            **_kwargs,
        )

    def predict_risk(
        self,
        cluster_assignment: np.ndarray,
        involvement: Dict[str, Dict[str, bool]],
        t_stage: str,
        midline_ext: bool = False,
        given_diagnosis: Optional[Dict[str, Dict[str, bool]]] = None,
        given_diagnosis_spsn: Optional[List[float]] = None,
        invert: bool = False,
        cluster_parameters: Optional[np.ndarray] = None,
        n_samples_for_prediction: int = 200,
        **_kwargs,
    ):
        """Predict Risk for a given cluster assignment."""

        # When no cluster parameters are explicitly given, we take the one stored in the model
        if cluster_parameters is None:
            cluster_parameters = self.cluster_parameters_chain
        # Only use n samples to make the prediction
        random_idx = random.sample(
            range(cluster_parameters.shape[0]), n_samples_for_prediction
        )
        cluster_parameters_for_pred = cluster_parameters[random_idx, :]

        return mm_predicted_risk(
            involvement=involvement,
            model=self.lymph_model,
            cluster_assignment=cluster_assignment,
            cluster_parameters=cluster_parameters_for_pred,
            t_stage=t_stage,
            midline_ext=midline_ext,
            given_diagnosis=given_diagnosis,
            given_diagnosis_spsn=given_diagnosis_spsn,
            invert=invert,
            **_kwargs,
        )

    def create_observed_predicted_df_for_cluster_assignment(
        self,
        cluster_assignment,
        for_states,
        data,
        save_name="model",
    ):
        """Create an observed / predicted results dataframe for the given `cluster_assignment` and given `data`, using the cluster parameters from the model."""

        oc_df, _ = create_obs_pred_df_single(
            samples=self.cluster_parameters_chain,
            model=self.lymph_model,
            n_clusters=self.n_clusters,
            cluster_assignment=cluster_assignment,
            data_input=data,
            patterns=for_states,
            lnls=list(self.lymph_model.graph.lnls.keys()),
            save_name=None,
            n_samples=50,
        )

        # print(oc_df)
        oc_df.to_csv(
            self.predictions_dir.joinpath(Path(f"predictions_{save_name}.csv"))
        )
        print(self.predictions_dir.joinpath(Path(f"predictions_{save_name}.csv")))
        return oc_df, _

    def create_result_df(
        self,
        for_states,
        lnls,
        save_name="result_df",
        independent_model=None,
        independent_model_samples=None,
        labels=None,
        for_t_stages=None,
        n_samples_for_prediction=50,
    ):
        """Creates a pd.Dataframe which holds the predictions and observations for the given labels.
        If no labels are given, then all loaded models are considered.
        If labels are given, only the matching models are considered.
        If independent_model is given, then the result df gets another column where the predictions are compared to the predictions of the independent model.
        """
        if labels is None:
            if self._subpopulation_labels is None:
                raise ValueError("Please provide model labels to create the dataframe")
            self.logger.info(
                "Labels not provided, proceed with generating the dataframe for all models"
            )
            labels = self._subpopulation_labels
        # Find the indices of matching models
        lm_idxs = [self.subpopulation_labels.index(l) for l in labels]

        # Only use n samples to make the prediction
        random_idx = random.sample(
            range(self.cluster_parameters_chain.shape[0]), n_samples_for_prediction
        )
        cluster_parameters_for_pred = self.cluster_parameters_chain[random_idx, :]

        if for_t_stages is None:
            for_t_stages = list(self.lymph_model.diag_time_dists)
        else:
            for_t_stages = as_list(for_t_stages)

        obs_pred_df_for_labels = []
        for i, l in zip(lm_idxs, labels):
            self.logger.info(f"Computing for {l}")
            obs_pred_df_for_labels.append(
                create_obs_pred_df_single(
                    cluster_parameters_for_pred,
                    self.lymph_model,
                    self.n_clusters,
                    self.get_cluster_assignment_for_label(l),
                    self.subpopulation_data[i],
                    for_states,
                    lnls,
                    None,
                )[0]
            )

        if independent_model is None:
            data_df = [
                item
                for d in obs_pred_df_for_labels
                for t_stage in for_t_stages
                for item in [
                    d[t_stage]["obs"],
                    d[t_stage]["pred"],
                    d[t_stage]["pred"] - d[t_stage]["obs"],
                ]
            ]

        else:
            # Very hacky, be cautions when using this.
            self.logger.info(f"Computing for independent model{l}")
            obs_pred_df_for_indp_model = _create_obs_pred_df_single(
                independent_model_samples,
                independent_model,
                independent_model.patient_data,
                for_states,
                lnls,
                None,
                n_samples=50,
            )[0]
            data_df = [
                item
                for t_stage in for_t_stages
                for d in [*obs_pred_df_for_labels, obs_pred_df_for_indp_model]
                for item in [
                    d[t_stage]["obs"],
                    d[t_stage]["pred"],
                    d[t_stage]["pred"] - d[t_stage]["obs"],
                    obs_pred_df_for_indp_model[t_stage]["pred"] - d[t_stage]["obs"],
                ]
            ]

        multiindex_lvl1 = for_t_stages
        multiindex_lvl2 = list(labels)
        multiindex_lvl3 = ["Observed", "Predicted", "Difference (MM)"]
        if independent_model is not None:
            multiindex_lvl3.append("Difference (Indp)")
            multiindex_lvl2.append("Ind")

        multiindex = pd.MultiIndex.from_product(
            [multiindex_lvl1, multiindex_lvl2, multiindex_lvl3],
            names=["T-Stage", "ICD", "Types"],
        )

        df = pd.DataFrame(data_df, index=multiindex)
        df = df.round(2)
        df.to_csv(self.predictions_dir.joinpath(Path(f"results_df_{save_name}.csv")))
        self.logger.info(
            f"Succesfully created results dataframe in {self.predictions_dir}"
        )
        return df

    def predict(self, cluster_assignmnet, for_states):
        """Implements the predict function for a new icd code (cluster assignmnet)"""
        # TODO todo
        self.logger.info("Predict method is not yet implemented.")
