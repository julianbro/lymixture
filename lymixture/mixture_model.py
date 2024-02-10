"""
This module defines the class wrapping the base model and composing the mixture model
likelihood from the components and subgroups in the data.
"""
# pylint: disable=logging-fstring-interpolation

import logging
import random
from functools import cached_property
from typing import Any, Iterator

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
)
from lymixture.utils import split_params_over_components, sample_from_global_model_and_configs


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
    components: list[lymph.models.Unilateral]
    subgroups: dict[str, lymph.models.Unilateral]

    def __init__(
        self,
        model_cls: lymph.models.Unilateral,
        model_kwargs: dict[str, Any] | None = None,
        num_components: int = 2,
    ):
        """Initialize the mixture model.

        The mixture will be based on the given ``model_cls`` (which is instantiated with
        the ``model_kwargs``), and will have ``num_components``.
        """
        if model_kwargs is None:
            model_kwargs = {}

        if not issubclass(model_cls, lymph.models.Unilateral):
            raise NotImplementedError(
                "Mixture model only implemented for `Unilateral` model."
            )

        self._model_cls = model_cls
        self._model_kwargs = model_kwargs
        self._init_components(num_components)

        self._compute_component_nums()

        logger.info(
            f"Created LymphMixtureModel based on {model_cls} model with "
            f"{num_components} components."
        )


    def _init_components(self, num_components: int):
        """Initialize the component parameters and assignments."""
        self.components = []
        for _ in range(num_components):
            self.components.append(self._model_cls(**self._model_kwargs))


    @property
    def num_components(self) -> int:
        """The number of mixture components."""
        return len(self.components)


    def _compute_component_nums(self):
        """(Re-)compute total number of model params and free assignment values."""
        num_model_params = len(self.lymph_model.get_params())
        self.num_component_params = num_model_params * self.num_components
        self.num_component_assignments = self.num_subgroups * (self.num_components - 1)


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
        if self.mixture_coefs is None:
            raise ValueError(
                "No cluster assignments are loaded in the model. Please provide "
                "cluster assignments first."
            )
        return compute_cluster_assignment_matrix(
            self.mixture_coefs, self.num_subgroups, self.num_components
        )


    @cached_property
    def state_probability_matrices(self) -> np.ndarray:
        """Holds the state probability matrices for every subgroup and every t_stage."""
        return compute_state_probability_matrices(
            self.cluster_assignment_matrix, self.cluster_state_probabilty_matrices
        )


    @property
    def mixture_coefs(self):
        """The assignment of subgroups to mixture components."""
        return self._mixture_coefs


    @mixture_coefs.setter
    def mixture_coefs(self, value):
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
        self._mixture_coefs = value

        del self.cluster_assignment_matrix
        del self.state_probability_matrices


    def get_mixture_coefs(
        self,
        subgroup: str | None = None,
        component: int | None = None,
    ) -> pd.DataFrame:
        """Get mixture coefficients for the given subgroup and component.

        The mixture coefficients are sliced by the given ``subgroup`` and ``component``
        which means that if no subgroupd and/or component is given, multiple mixture
        coefficients are returned.
        """
        return self._mixture_coefs.loc[slice(component), slice(subgroup)]


    def assign_mixture_coefs(self,  new_mixture_coefs: np.ndarray) -> None:
        """Assign new mixture coefficients to the model."""
        self._mixture_coefs = pd.DataFrame(
            new_mixture_coefs,
            index=range(self.num_components),
            columns=self.subgroups.keys(),
        )


    def assign_component_params(
        self,
        *new_params_args,
        **new_params_kwargs,
    ) -> Iterator[float]:
        """Assign new spread params to the component models.

        Parameters can be set as positional arguments, in which case they are used up
        one by one for each component, or as keyword arguments.

        If provided as keyword arguments, the keys are the parameter names expected by
        the individual models, prefixed by the index of the component (e.g.
        ``0_param1``, ``1_param1``, etc.). When no index is found, the parameter is set
        for all components.
        """
        params_for_components, global_params = split_params_over_components(
            new_params_kwargs, num_components=self.num_components
        )
        for c, component in enumerate(self.components):
            component_params = {}
            component_params.update(global_params)
            component_params.update(params_for_components[c])
            new_params_args, _ = component.assign_params(
                *new_params_args, **component_params
            )

        # TODO: Think about whether these two attributes are necessary in this form
        del self.cluster_state_probabilty_matrices
        del self.state_probability_matrices
        return new_params_args


    def get_responsibilities(
        self,
        patient: int | None = None,
        subgroup: str | None = None,
        component: int | None = None,
    ) -> pd.DataFrame:
        """Get the repsonsibility of a ``patient`` for a ``component``.

        The ``patient`` index enumerates all patients in the mixture model unless
        ``subgroup`` is given, in which case the index runs over the patients in the
        given subgroup.
        """
        if subgroup is not None:
            resp = self.subgroups[subgroup].patient_data["_mixture", "responsibility"]
        else:
            resp = pd.concat([
                model.patient_data["_mixture", "responsibility"]
                for model in self.subgroups.values()
            ])
        return resp.loc[slice(patient), slice(component)]


    def assign_responsibilities(self, new_responsibilities: np.ndarray):
        """Assign responsibilities to the model.

        They should have the shape (num_patients, num_components) and summing them
        along the last axis should yield a vector of ones.

        Note that these responsibilities essentially become the latent variables
        of the model if they are "hard", i.e. if they are either 0 or 1 and thus
        represent a one-hot encoding of the component assignments.
        """
        if new_responsibilities.shape != (self.num_patients, self.num_components):
            raise ValueError(
                "Shape of responsibilities wrong. Expected "
                f"({self.num_patients}, {self.num_components})"
            )

        summed = np.sum(new_responsibilities, axis=-1)
        if not np.allclose(summed, np.ones(self.num_patients)):
            raise ValueError("Repsonsibilities for each patient should sum to one.")

        mixture_columns = pd.MultiIndex.from_tuples([
            ("_mixture", "responsibility", i) for i in range(self.num_components)
        ])
        for model in self.subgroups.values():
            subgroup_responsibilities = pd.DataFrame(
                new_responsibilities[:len(model.patient_data)],
                columns=mixture_columns,
            )
            new_responsibilities = new_responsibilities[len(model.patient_data):]
            model.patient_data.join(subgroup_responsibilities)


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

        for label, data in grouped:
            self.subgroups[label] = self._model_cls(**self._model_kwargs)
            self.subgroups[label].load_patient_data(data, **kwargs)

        self.diagnose_matrices = []
        for model in self.subgroups.values():
            for t_stage in model.t_stages:
                self.diagnose_matrices.append(model.diagnose_matrices[t_stage])


    @property
    def num_patients(self) -> int:
        """Number of patients in the mixture model. Sum of patients in all subgroups."""
        return sum(len(model.patient_data) for model in self.subgroups.values())


    @property
    def num_subgroups(self) -> int:
        """The number of subgroups."""
        return len(self.subgroups)


    def complete_data_likelihood(
        self,
        patient_data: pd.DataFrame | None = None,
        split_by: tuple[str, str, str] = ("tumor", "1", "subsite"),
        load_data_kwargs: dict[str, Any] | None = None,
        responsibilities: np.ndarray | None = None,
        mixture_coefs: np.ndarray | None = None,
        model_params: np.ndarray | None = None,
        log: bool = True,
    ) -> float:
        """Compute the complete data likelihood of the model."""
        if patient_data is not None:
            load_data_kwargs = {} if load_data_kwargs is None else load_data_kwargs
            self.load_patient_data(patient_data, split_by, **load_data_kwargs)

        if responsibilities is not None:
            self.assign_responsibilities(responsibilities)

        if mixture_coefs is not None:
            self.assign_mixture_coefs(mixture_coefs)

        if model_params is not None:
            self.assign_component_params(*model_params)

        llh = 0 if log else 1.0
        # TODO: Implement the complete data likelihood
        return llh


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
            self.mixture_coefs = cluster_assignments

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
        self.mixture_coefs, history = self.estimate_cluster_assignments(em_config)

        # MCMC Sampling
        # Find the cluster parameters using MCMC based on the cluster assignments.
        # Store the final cluster assignments and parameters
        if mcmc_config is not None:
            sample_chain, _, _ = self._mcmc_sampling(mcmc_config)

        self.component_params = sample_chain.mean(axis=0)
        self.cluster_parameters_chain = sample_chain

        return sample_chain, self.mixture_coefs, history


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
