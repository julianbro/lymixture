"""
This module defines the class wrapping the base model and composing the mixture model
likelihood from the components and subgroups in the data.
"""
# pylint: disable=logging-fstring-interpolation

import logging
import random
from functools import cached_property
from typing import Any

import lymph
import numpy as np
import pandas as pd

from lymixture.em_sampling import (
    ExpectationMaximization,
    History,
    emcee_simple_sampler,
    exceed_param_bound,
)
from lymixture.mm_predict import (
    create_obs_pred_df_single,
    mm_generate_predicted_prevalences,
    mm_predicted_risk,
)
from lymixture.model_functions import (
    compute_cluster_assignment_matrix,
    compute_cluster_state_probabilty_matrices,
    compute_state_probability_matrices,
    gen_diagnose_matrices,
)
from lymixture.utils import sample_from_global_model_and_configs


logger = logging.getLogger(__name__)


# Define a global variable which can be used within this module.
global LMM_GLOBAL


def log_ll_cl_parameters(cluster_parameters):
    if exceed_param_bound(cluster_parameters):
        return -np.inf
    llh = LMM_GLOBAL.mm_hmm_likelihood(cluster_parameters=cluster_parameters)
    if np.isinf(llh):
        return -np.inf
    return llh


class LymphMixtureModel:
    """Class that handles the individual components of the mixture model."""

    def __init__(
        self,
        model_cls: lymph.models.Unilateral,
        model_kwargs: dict[str, Any] | None = None,
        num_components: int = 2,
    ):
        """Initialize the mixture model.

        The mixture will be based on the given `model_cls` (which is instantiated with
        the `model_kwargs`), and will have `num_components`.
        """
        if model_kwargs is None:
            model_kwargs = {}

        if not issubclass(model_cls, lymph.models.Unilateral):
            raise NotImplementedError(
                "Mixture model only implemented for `Unilateral` model."
            )

        self.lymph_model = model_cls(**model_kwargs)
        self.num_components = num_components

        self._compute_component_nums()

        logger.info(
            f"Created LymphMixtureModel based on {model_cls} model with "
            f"{self.num_components} components."
        )

    def _compute_component_nums(self):
        """(Re-)compute total number of model params and free assignment values."""
        num_model_params = len(self.lymph_model.get_params())
        self.num_component_params = num_model_params * self.num_components
        self.num_component_assignments = self.num_subgroups * (self.num_components - 1)

    @property
    def subgroup_labels(self):
        """Get or set the labels for the subgroups."""
        return self._subgroup_labels

    @subgroup_labels.setter
    def subgroup_labels(self, value: list[str]):
        if len(value) != self.num_subgroups:
            raise ValueError(
                "Number of labels do not match with number of subgroups."
            )
        for i, v in enumerate(value):
            if not isinstance(v, str):
                raise ValueError("ICD codes should be passed as list of strings.")
            logger.info(
                f"Assigned {v} to supopulations with {len(self.subgroup_datas[i])} "
                "patients"
            )
        self._subgroup_labels = value

    @cached_property
    def cluster_state_probabilty_matrices(self) -> np.ndarray:
        """
        Holds the cluster matrices (i.e the state probabilities of the cluster) for
        each t-stage and cluster.
        """
        if self.component_params is None:
            raise ValueError(
                "No cluster parameters are in the model. Please provide cluster parameters first."
            )
        return compute_cluster_state_probabilty_matrices(
            self.component_params, self.lymph_model, self.num_components
        )

    @cached_property
    def cluster_assignment_matrix(self) -> np.ndarray:
        """Holds the cluster assignment matrix."""
        if self.component_assignments is None:
            raise ValueError(
                "No cluster assignments are loaded in the model. Please provide "
                "cluster assignments first."
            )
        return compute_cluster_assignment_matrix(
            self.component_assignments, self.num_subgroups, self.num_components
        )

    @cached_property
    def state_probability_matrices(self) -> np.ndarray:
        """Holds the state probability matrices for every subgroup and every t_stage."""
        return compute_state_probability_matrices(
            self.cluster_assignment_matrix, self.cluster_state_probabilty_matrices
        )

    @property
    def num_subgroups(self):
        """The number of subgroups."""
        return len(self.subgroup_datas)

    @property
    def component_assignments(self):
        """The assignment of subgroups to mixture components."""
        return self._component_assignments

    @component_assignments.setter
    def component_assignments(self, value):
        """Set new component assignments and delete the current cluster matrices.

        Note that the model expects ``num_subgroups`` less parameters for the cluster
        assignment due to the constraint that the cluster assignment of one subgroup
        has to sum to 1.
        """
        if len(value) != self.num_component_assignments:
            raise ValueError(
                "Number of provided cluster assignments do not match the number of "
                "cluster assignments expected by the model."
            )
        self._component_assignments = value

        del self.cluster_assignment_matrix
        del self.state_probability_matrices

    @property
    def component_params(self):
        """The parameters of the individual mixture components."""
        return self._component_params

    @component_params.setter
    def component_params(self, value):
        """Set the component parameters.

        This deletes the cluster state probability matrices and the state probability
        matrices.
        """
        if len(value) != self.num_component_params:
            raise ValueError(
                "Number of provided cluster_parameters do not match the number of "
                "cluster_parameters expected by the model."
            )
        self._component_params = value

        del self.cluster_state_probabilty_matrices
        del self.state_probability_matrices

    def load_patient_data(
        self,
        patient_data: pd.DataFrame,
        split_by: tuple[str, str, str],
        **kwargs,
    ):
        """Split the ``patient_data`` into subgroups and load it into the model.

        This amounts to computing the diagnose matrices for the individual subgroups.
        The ``split_by`` tuple should contain the three-level header of the LyProX-style
        data. Any additional keyword arguments are passed to the
        :py:meth:`~lymph.models.Unilateral.load_patient_data` method.
        """
        grouped = patient_data.groupby(split_by)
        self.subgroup_datas = [df for _, df in grouped]
        self.subgroup_labels = [label for label, _ in grouped]

        self.diagnose_matrices = list(gen_diagnose_matrices(
            datasets=self.subgroup_datas,
            lymph_model=self.lymph_model,
            load_kwargs=kwargs,
        ))

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
                # sum the likelihoods of observing each patient diagnose given the
                # state probability from the mixture.
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
        """Compute the likelihood of the ``data`` given the model, theta and cluster
        assignment.

        Parameters:
        - cluster_parameters: The cluster parameters for the mixture model. If not
            provided, the model uses the existing parameters set within the model.
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
            self.component_params = cluster_parameters

        if cluster_assignments is not None:
            self.component_assignments = cluster_assignments

        if data is not None:
            if load_data_kwargs is None:
                load_data_kwargs = {}
            self.load_patient_data(data, **load_data_kwargs)

        return self._mm_hmm_likelihood(log=log)

    def estimate_cluster_assignments(self, em_config=None) -> tuple[np.ndarray, History]:
        """Estimates the cluster assignments using an EM algortihm"""
        self.em_algortihm = ExpectationMaximization(lmm=self, em_config=em_config)
        estimated_cluster_assignments, em_history = self.em_algortihm.run_em()
        return estimated_cluster_assignments, em_history

    def _mcmc_sampling(self, mcmc_config: dict):
        """
        Performs MCMC sampling to determine model parameters for the current cluster
        assignments.
        """
        global LMM_GLOBAL
        LMM_GLOBAL = self

        sampler = mcmc_config.get("sampler", "SIMPLE")
        sampling_params = mcmc_config["sampling_params"]
        log_prob_fn = log_ll_cl_parameters

        hdf5_backend = emcee.backends.HDFBackend(self.hdf5_output, name="mcmc")
        logger.debug(f"Prepared sampling backend at {self.hdf5_output}")

        if sampler == "SIMPLE":
            logger.info("Using simple sampler for MCMC")
            sample_chain, end_point, log_probs = emcee_simple_sampler(
                log_prob_fn,
                ndim=self.num_component_params,
                sampling_params=sampling_params,
                starting_point=None,
                hdf5_backend=hdf5_backend,
            )
        else:
            logger.info("Using lyscript sampler for MCMC")
            (
                sample_chain,
                end_point,
                log_probs,
            ) = sample_from_global_model_and_configs(
                log_prob_fn,
                ndim=self.num_component_params,
                sampling_params=sampling_params,
                starting_point=None,
                backend=hdf5_backend,
                models=self,
            )

        return sample_chain, end_point, log_probs

    def fit(
        self,
        em_config: dict | None = None,
        mcmc_config: dict | None = None,
    ):
        """
        Fits the mixture model, i.e. finds the optimal cluster assignments and the
        cluster parameters, using (1) the EM algorithm and (2) the MCMC sampling method.
        """
        # Ugly, but we need to do it for the mcmc sampling.
        global LMM_GLOBAL
        LMM_GLOBAL = self

        # Estimate the Cluster Assignments.
        self.component_assignments, history = self.estimate_cluster_assignments(em_config)

        # MCMC Sampling
        # Find the cluster parameters using MCMC based on the cluster assignments.
        # Store the final cluster assignments and parameters
        if mcmc_config is not None:
            sample_chain, _, _ = self._mcmc_sampling(mcmc_config)

        self.component_params = sample_chain.mean(axis=0)
        self.cluster_parameters_chain = sample_chain

        return sample_chain, self.component_assignments, history

    def get_cluster_assignment_for_label(self, label: str):
        """Get the cluster assignment for the given label."""
        try:
            index = self.subgroup_labels.index(label)
            return self.cluster_assignment_matrix[index, :]

        except ValueError:
            logger.error(f"'{label}' is not in the supopulation labelk.")

    def predict_prevalence_for_cluster_assignment(
        self,
        cluster_assignment: np.ndarray,
        for_pattern: dict[str, dict[str, bool]],
        t_stage: str = "early",
        modality_spsn: list[float] | None = None,
        invert: bool = False,
        cluster_parameters: np.ndarray | None = None,
        n_samples_for_prediction: int = 200,
        **_kwargs,
    ):
        """Predict Prevalence for a given cluster assignment."""
        # When no cluster parameters are given, we take the one stored in the model
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
        involvement: dict[str, dict[str, bool]],
        t_stage: str,
        midline_ext: bool = False,
        given_diagnosis: dict[str, dict[str, bool]] | None = None,
        given_diagnosis_spsn: list[float] | None = None,
        invert: bool = False,
        cluster_parameters: np.ndarray | None = None,
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
    ):
        """
        Create an observed / predicted results dataframe for the given
        `cluster_assignment` and given `data`, using the cluster parameters from the
        model.
        """
        oc_df, _ = create_obs_pred_df_single(
            samples_for_predictions=self.cluster_parameters_chain,
            model=self.lymph_model,
            n_clusters=self.num_components,
            cluster_assignment=cluster_assignment,
            data_input=data,
            patterns=for_states,
            lnls=list(self.lymph_model.graph.lnls.keys()),
            save_name=None,
        )

        return oc_df, _
