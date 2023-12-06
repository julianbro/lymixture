import logging
from pathlib import Path
from typing import List, Optional, Union
import emcee
from core.costum_types import EMConfigType
import lymph
import numpy as np
import pandas as pd
from corner import corner
from em_sampling import (
    assign_global_params,
    em_sampler,
    emcee_simple_sampler,
    llh_theta_given_z,
    plot_history,
)
from lyscripts.sample import sample_from_global_model_and_configs
from lyscripts.plot.corner import get_param_labels
from lyscripts.plot.utils import save_figure
from mm_predict import create_obs_pred_df_single

from mm_plotting import plot_cluster_assignments, plot_cluster_parameters


# from shared import MODELS_SHARED


def group_mixing_params(z: np.ndarray, N: int, K: int):
    """
    Regroups the mixing parameters for the model structure.

    Args:
        z (np.ndarray): Array of mixing parameters.
        N (int): Number of models.
        K (int): Number of clusters.

    Returns:
        list: Grouped mixing parameters.
    """
    z_grouped = [z[i : i + (K - 1)] for i in range(0, N, K - 1)]
    return [[*zs, 1 - np.sum(zs)] for zs in z_grouped]


def assign_mixing_parameters(z: Union[np.ndarray, List[float]], models, n_clusters):
    """
    Assigns mixing parameters (latent variables) to models.

    Args:
        z (Union[np.ndarray, List[float]]): Mixing parameters.
        models (list): List of lymph models.
        n_clusters (int): Number of clusters.
    """
    z_grouped = group_mixing_params(z, len(models), n_clusters)
    for i, model in enumerate(models):
        model.mixture_components = z_grouped[i]
    return models


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
        lymph_models: List[lymph.models.Unilateral],
        n_clusters: int,
        base_dir: Path,
        name: Optional[str] = None,
        model_labels: Optional[List[str]] = None,
    ):
        # Initialize logger
        self._setup_logger()

        # Initialize model configurations
        self.lymph_models = lymph_models
        self.n_clusters = n_clusters
        self._cluster_assignments = None
        self.cluster_assignments_full_matrix = np.zeros(
            (len(self.lymph_models), n_clusters)
        )
        self.cluster_parameters = None
        self.n_model_params = None
        self.n_thetas = None
        self.n_cluster_assignments = len(self.lymph_models) * (n_clusters - 1)

        self._model_labels = None
        if model_labels is not None:
            self.model_labels = model_labels

        # Check model consistency
        self._check_model_consistency()

        # Set up directories
        self.name = name if name else "LMM"
        self.base_dir = base_dir.joinpath(self.name)
        self.samples_dir = self.base_dir.joinpath("samples/")
        self.figures_dir = self.base_dir.joinpath("figures/")
        self.predictions_dir = self.base_dir.joinpath("predictions/")
        self._create_directories()

        self.logger.info(
            f"Create LymphMixtureModel with {len(self.lymph_models)} models and {self.n_clusters} cluster components in {self.base_dir}"
        )

    def _create_directories(self):
        """
        Creates necessary directories for saving outputs.
        """
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

    @property
    def model_labels(self):
        return self._model_labels

    @model_labels.setter
    def model_labels(self, value: List[str]):
        if len(value) != len(self.lymph_models):
            raise ValueError(
                "Number of model labels do not match with number if lymph models."
            )

        for i, v in enumerate(value):
            if not isinstance(v, str):
                raise ValueError("ICD codes should be passed as list of strings.")
            self.logger.info(
                f"Assigned {v} to lymph model with {len(self.lymph_models[i].patient_data)} patients"
            )
        self._model_labels = value

    @property
    def cluster_assignments(self):
        """
        Property to get or set the cluster assignments.
        """
        return self._cluster_assignments

    @cluster_assignments.setter
    def cluster_assignments(self, value):
        """
        Setter for cluster assignments that triggers update logic.
        """
        self._cluster_assignments = value
        self._on_cluster_assignment_updated(value)

    def _on_cluster_assignment_updated(self, value):
        """
        Logic to execute when cluster assignments are updated.
        """
        self.cluster_assignments_full_matrix = np.array([value, [1 - v for v in value]])

    def _check_model_consistency(self):
        """
        Checks if all models have the same definitions. Raises an error if inconsistent.
        """
        param_counts = [len(lm.get_params()) for lm in self.lymph_models]

        if len(set(param_counts)) != 1:
            raise ValueError("All models must have the same number of parameters.")

        self.n_model_params = param_counts[0]
        # self.logger.info(
        #     f"Number of parameters per cluster is set to {self.n_model_params}"
        # )

        diag_time_dists = [list(lm.diag_time_dists) for lm in self.lymph_models]
        dtd_first = diag_time_dists[0]
        if any(dtd_i != dtd_first for dtd_i in diag_time_dists):
            raise ValueError(
                "All models must have the same diagnostic time distributions."
            )

        self.n_thetas = self.n_model_params * self.n_clusters

    def load_data(
        self, data: List[pd.DataFrame], t_stage_mapping: Optional[callable] = None
    ):
        """
        Loads patient data into the models. Patient data is provided as a list of pd.DataFrames
        """
        if len(data) != len(self.lymph_models):
            raise ValueError(
                "Length of provided data does not match length of loaded models."
            )

        for lm, d in zip(self.lymph_models, data):
            if t_stage_mapping:
                lm.load_patient_data(d, mapping=t_stage_mapping)
            else:
                lm.load_patient_data(d)

    def _em_sampling(
        self,
        em_config: EMConfigType,
        do_plot_history: bool = True,
        save_assignments: bool = True,
    ):
        """Handles the em sampling functionalities of the class."""

        file_name = f"cluster_assignments_final_{em_config['method']}_{str(em_config['convergence_ths']).replace('.', '_')}"

        cluster_assignments_dir = self.samples_dir.joinpath(
            Path(file_name)
        ).with_suffix(".npy")
        if cluster_assignments_dir.exists():
            self.logger.info(
                f"Cluster assignment found in {cluster_assignments_dir}. Skipping EM-Algortihm."
            )
            cluster_assignments = np.load(cluster_assignments_dir)
            end_point, log_probs = None, None

        else:
            # Create the path for the samples generated by the EM algorithm
            em_path = self.samples_dir.joinpath("EMSamples")
            em_path.mkdir(parents=True, exist_ok=True)

            # TODO: Make class structure for the EM sampler.
            cluster_assignments, final_model_params, history = em_sampler(
                self.lymph_models,
                self.n_clusters,
                em_path,
                self.name,
                em_params=em_config,
            )

            # Store the cluster assignmets
            np.save(
                self.samples_dir / f"{file_name}.npy",
                cluster_assignments,
            )

            if do_plot_history:
                plot_history(
                    history,
                    labels_w=self.model_labels,
                    models=self.lymph_models,
                    n_clusters=self.n_clusters,
                    save_dir=self.figures_dir,
                )

        self.lymph_models = assign_mixing_parameters(
            cluster_assignments, self.lymph_models, self.n_clusters
        )
        return cluster_assignments

    def _mcmc_sampling(self, mcmc_config: dict, save_samples: bool = True):
        """
        Performs MCMC sampling to determine final model parameters.
        """
        sampler = mcmc_config.get("sampler", "SIMPLE")
        sampling_params = mcmc_config["sampling_params"]
        log_prob_fn = llh_theta_given_z
        sample_file_name = f"mcmc_sampling_chain_{sampling_params['nsteps']}_{sampling_params['nburnin']}"
        hdf5_backend = emcee.backends.HDFBackend(
            self.samples_dir / f"{sample_file_name}.hdf5", name="mcmc"
        )

        mcmc_chain_dir = self.samples_dir.joinpath(Path(sample_file_name)).with_suffix(
            ".npy"
        )
        if mcmc_chain_dir.exists():
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
                    ndim=self.n_thetas,
                    sampling_params=sampling_params,
                    starting_point=None,
                    save_dir=self.samples_dir,
                    models=self.lymph_models,
                )
            else:
                self.logger.info("Using lyscript sampler for MCMC")
                assign_global_params(self.lymph_models, self.n_clusters)
                (
                    sample_chain,
                    end_point,
                    log_probs,
                ) = sample_from_global_model_and_configs(
                    log_prob_fn,
                    ndim=self.n_thetas,
                    sampling_params=sampling_params,
                    starting_point=None,
                    hdf5_backend=hdf5_backend,
                    save_dir=self.samples_dir,
                    models=self.lymph_models,
                )

            if save_samples:
                np.save(
                    self.samples_dir / f"{sample_file_name}.npy",
                    sample_chain,
                )

        return sample_chain, end_point, log_probs

    def fit(
        self,
        data: Optional[List[pd.DataFrame]] = None,
        do_plot_history: bool = False,
        em_config: Optional[dict] = None,
        mcmc_config: Optional[dict] = None,
    ):
        """
        Fits the mixture model using (1) the EM algorithm and (2) the MCMC sampling method.
        """
        # TODO clean up!!
        # Implementation of EM algorithm
        # - Expectation Step: Assign models to clusters or sample cluster parameters, depending on method
        # - Maximization Step: Estimate cluster parameters, or estimate cluster assignmnets

        # If no data is provided, the loaded models should already include data.
        if (
            not any(len(lm.patient_data) > 0 for lm in self.lymph_models)
            and data is None
        ):
            raise ValueError(
                f"There are models which do not contain any data. Please provide data for the lymph models."
            )
        if data is not None:
            self.load_data(data)

        # Skip the EM-Algorithm if there is already a cluster asignments (Only for debug)
        if self._cluster_assignments is not None:
            # Only for debug
            self.logger.warning(
                "Skipping EM Algortihm, since cluster assignment is already given"
            )
            cluster_assignments = self._cluster_assignments
        else:
            cluster_assignments = self._em_sampling(em_config)

        # MCMC Sampling
        # Refine the cluster parameters using MCMC based on the assignments from EM
        # Store the final cluster assignments and parameters
        # TODO for now, if a sample exists already, skip sampling for perfomance
        sample_chain = []
        if mcmc_config is not None:
            sample_chain, _, _ = self._mcmc_sampling(mcmc_config)

        self.cluster_assignments = cluster_assignments
        self.cluster_parameters = sample_chain

    def plot_cluster_parameters(self):
        """Corner Plot for the cluster parameters"""
        """Plots the corner plots for cluster parameters using an external function."""
        labels = get_param_labels_temp(self.lymph_models[0])
        plot_cluster_parameters(
            self.cluster_parameters,
            self.n_clusters,
            labels,
            self.figures_dir,
            self.logger,
        )

    def plot_cluster_assignment_matrix(self, labels: Optional[List[str]] = None):
        """
        Plots the cluster assignmnet matrix
        """

        plot_cluster_assignments(
            self.cluster_assignments_full_matrix,
            labels=labels,
            save_dir=self.figures_dir,
            logger=self.logger,
        )

    def predict_with_model(
        self,
        model: lymph.models.Unilateral,
        for_states,
        lnls,
        save_name="model",
        data=None,
    ):
        """Make predictions for a given model (cluster assignmnet), given a model containing data
        *** This method is still under construction ***
        """
        if data is None:
            if len(model.patient_data) > 0:
                data = model.patient_data.copy(deep=True)
            else:
                ValueError(
                    "No data is given and provided model does not contain any patient data."
                )
        oc_df, _ = create_obs_pred_df_single(
            self.cluster_parameters,
            model,
            data,
            for_states,
            lnls,
            save_name=None,
            n_samples=50,
        )

        # print(oc_df)
        oc_df.to_csv(
            self.predictions_dir.joinpath(Path(f"predictions_{save_name}.csv"))
        )
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
    ):
        """Creates a pd.Dataframe which holds the predictions and observations for the given labels.
        If no labels are given, then all loaded models are considered.
        If labels are given, only the matching models are considered.
        If independent_model is given, then the result df gets another column where the predictions are compared to the predictions of the independent model.
        """
        if labels is None:
            if self._model_labels is None:
                raise ValueError("Please provide model labels to create the dataframe")
            self.logger.info(
                "Labels not provided, proceed with generating the dataframe for all models"
            )
            labels = self._model_labels
        # Find the indices of matching models
        lm_idxs = [self.model_labels.index(l) for l in labels]

        if for_t_stages is None:
            for_t_stages = list(self.lymph_models[0].diag_time_dists)
        else:
            for_t_stages = as_list(for_t_stages)

        obs_pred_df_for_labels = []
        for i, l in zip(lm_idxs, labels):
            self.logger.info(f"Computing for {l}")
            obs_pred_df_for_labels.append(
                create_obs_pred_df_single(
                    self.cluster_parameters,
                    self.lymph_models[i],
                    self.lymph_models[i].patient_data,
                    for_states,
                    lnls,
                    None,
                    n_samples=50,
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
            obs_pred_df_for_indp_model = create_obs_pred_df_single(
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
                for d in [*obs_pred_df_for_labels, obs_pred_df_for_indp_model]
                for t_stage in for_t_stages
                for item in [
                    d[t_stage]["obs"],
                    d[t_stage]["pred"],
                    d[t_stage]["pred"] - d[t_stage]["obs"],
                    obs_pred_df_for_indp_model[t_stage]["pred"] - d[t_stage]["obs"],
                ]
            ]

        multiindex_lvl1 = for_t_stages
        multiindex_lvl2 = labels
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
