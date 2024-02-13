"""
This module defines the class wrapping the base model and composing the mixture model
likelihood from the components and subgroups in the data.
"""
# pylint: disable=logging-fstring-interpolation

import logging
import random
from functools import cached_property
from typing import Any, Iterable, Iterator
import warnings

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
from lymixture.utils import (
    join_with_responsibilities,
    sample_from_global_model_and_configs,
    split_over_components,
    RESP_COL
)

pd.options.mode.copy_on_write = True
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
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


class LymphMixture:
    """Class that handles the individual components of the mixture model."""

    def __init__(
        self,
        model_cls: type = lymph.models.Unilateral,
        model_kwargs: dict[str, Any] | None = None,
        num_components: int = 2,
    ):
        """Initialize the mixture model.

        The mixture will be based on the given ``model_cls`` (which is instantiated with
        the ``model_kwargs``), and will have ``num_components``.
        """
        if model_kwargs is None:
            model_kwargs = {"graph_dict": {
                ("tumor", "T"): ["II", "III"],
                ("lnl", "II"): ["III"],
                ("lnl", "III"): [],
            }}

        if not issubclass(model_cls, lymph.models.Unilateral):
            raise NotImplementedError(
                "Mixture model only implemented for `Unilateral` model."
            )

        self._model_cls = model_cls
        self._model_kwargs = model_kwargs
        self._mixture_coefs = None

        self.subgroups: dict[str, self._model_cls] = {}
        self.components: list[self._model_cls] = self._create_components(num_components)

        logger.info(
            f"Created LymphMixtureModel based on {model_cls} model with "
            f"{num_components} components."
        )


    def _create_components(self, num_components: int) -> list[Any]:
        """Initialize the component parameters and assignments."""
        components = []
        for _ in range(num_components):
            components.append(self._model_cls(**self._model_kwargs))

        return components


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
            self.component_params, self.lymph_model, len(self.components)
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
            self.mixture_coefs, len(self.subgroups), len(self.components)
        )


    @cached_property
    def state_probability_matrices(self) -> np.ndarray:
        """Holds the state probability matrices for every subgroup and every t_stage."""
        return compute_state_probability_matrices(
            self.cluster_assignment_matrix, self.cluster_state_probabilty_matrices
        )


    def _create_empty_mixture_coefs(self) -> pd.DataFrame:
        nan_array = np.empty((len(self.components), len(self.subgroups)))
        nan_array[:] = np.nan
        return pd.DataFrame(
            nan_array,
            index=range(len(self.components)),
            columns=self.subgroups.keys(),
        )


    def get_mixture_coefs(
        self,
        component: int | None = None,
        subgroup: str | None = None,
    ) -> float | pd.Series | pd.DataFrame:
        """Get mixture coefficients for the given ``subgroup`` and ``component``.

        The mixture coefficients are sliced by the given ``subgroup`` and ``component``
        which means that if no subgroupd and/or component is given, multiple mixture
        coefficients are returned.
        """
        if getattr(self, "_mixture_coefs", None) is None:
            self._mixture_coefs = self._create_empty_mixture_coefs()

        component = slice(None) if component is None else component
        subgroup = slice(None) if subgroup is None else subgroup
        return self._mixture_coefs.loc[component, subgroup]


    def assign_mixture_coefs(
        self,
        new_mixture_coefs: float | np.ndarray,
        component: int | None = None,
        subgroup: str | None = None,
    ) -> None:
        """Assign new mixture coefficients to the model.

        As in :py:meth:`~get_mixture_coefs`, ``subgroup`` and ``component`` can be used
        to slice the mixture coefficients and therefore assign entirely new coefs to
        the entire model, to one subgroup, to one component, or to one component of one
        subgroup.
        """
        mixture_coefs = self.get_mixture_coefs()
        component = slice(None) if component is None else component
        subgroup = slice(None) if subgroup is None else subgroup
        mixture_coefs.loc[component, subgroup] = new_mixture_coefs


    def get_component_params(
        self,
        param: str | None = None,
        component: int | None = None,
        as_dict: bool = True,
        flatten: bool = False,
    ) -> float | Iterable[float] | dict[str, float]:
        """Get the spread parameters of the individual mixture component models.

        If ``component`` is the index of one of the mixture model's components, this
        method simply returns the call to :py:meth:`~lymph.models.Unilateral.get_params`
        for the given component.

        When no ``component`` is specified and ``flatten`` is set to ``False``, a list
        of calls to :py:meth:`~lymph.models.Unilateral.get_params` for all components is
        returned. This may then contain only floats (if ``param`` is given), iterables
        of parameters (when ``as_dict`` is set to ``False``), or dictionaries of the
        component's parameters.

        Lastly, when ``flatten`` is set to ``True``, a dictionary of the form
        ``<idx>_<param>: value`` is created, where ``<idx>`` is the index of the
        component and ``<param>`` is the name of the parameter. When ``param`` is not
        given and ``as_dict`` is ``True``, this dictionary is returned. When ``param``
        specifies a cluster parameter, the value of that parameter is returned. And
        when ``param`` is not given and ``as_dict`` is ``False``, the values of the
        dictionary are returned as an iterable.

        Examples:

        >>> graph_dict = {
        ...     ("tumor", "T"): ["II"],
        ...     ("lnl", "II"): [],
        ... }
        >>> mixture = LymphMixture(
        ...     model_kwargs={"graph_dict": graph_dict},
        ...     num_components=2,
        ... )
        >>> mixture.assign_component_params(0.1, 0.9)
        >>> mixture.get_component_params()              # doctest: +NORMALIZE_WHITESPACE
        [{'T_to_II_spread': 0.1},
         {'T_to_II_spread': 0.9}]
        >>> mixture.get_component_params(flatten=True)  # doctest: +NORMALIZE_WHITESPACE
        {'0_T_to_II_spread': 0.1,
         '1_T_to_II_spread': 0.9}
        >>> mixture.get_component_params(param="T_to_II_spread")
        [0.1, 0.9]
        >>> mixture.get_component_params(param="T_to_II_spread", component=0)
        0.1
        >>> mixture.get_component_params(as_dict=False)
        [dict_values([0.1]), dict_values([0.9])]
        >>> mixture.get_component_params(as_dict=False, flatten=True)
        dict_values([0.1, 0.9])
        >>> mixture.get_component_params(param="0_T_to_II_spread", flatten=True)
        0.1
        """
        if component is not None:
            return self.components[component].get_params(param=param, as_dict=as_dict)

        if not flatten:
            return [c.get_params(param=param, as_dict=as_dict) for c in self.components]

        flat_params = {
            f"{c}_{k}": v
            for c, component in enumerate(self.components)
            for k, v in component.get_params(as_dict=True).items()
        }

        if param is not None:
            return flat_params[param]

        return flat_params if as_dict else flat_params.values()


    def assign_component_params(
        self,
        *new_params_args,
        component: int | None = None,
        **new_params_kwargs,
    ) -> Iterator[float]:
        """Assign new spread params to the component models.

        If ``component`` is given, the arguments and keyword arguments are passed to the
        corresponding component model's :py:meth:`~lymph.models.Unilateral.assign_params`
        method.

        Parameters can be set as positional arguments, in which case they are used up
        one at a time by the individual component models. E.g., if each component has
        two parameters, the first two positional arguments are used for the first
        component, the next two for the second component, and so on.

        If provided as keyword arguments, the keys are the parameter names expected by
        the individual models, prefixed by the index of the component (e.g.
        ``0_param1``, ``1_param1``, etc.). When no index is found, the parameter is set
        for all components.
        """
        if component is not None:
            return self.components[component].assign_params(
                *new_params_args, **new_params_kwargs
            )

        params_for_components, global_params = split_over_components(
            new_params_kwargs, num_components=len(self.components)
        )
        for c, component in enumerate(self.components):
            component_params = {}
            component_params.update(global_params)
            component_params.update(params_for_components[c])
            new_params_args, _ = component.assign_params(
                *new_params_args, **component_params
            )


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

        Omitting ``component`` or ``patient`` (or both) will return corresponding slices
        of the responsibility table.
        """
        if subgroup is not None:
            resp_table = self.subgroups[subgroup].patient_data
        else:
            resp_table = self.patient_data

        pat_slice = slice(None) if patient is None else patient
        comp_slice = (*RESP_COL, slice(None) if component is None else component)
        res = resp_table.loc[pat_slice,comp_slice]
        try:
            return res[RESP_COL]
        except (KeyError, IndexError):
            return res


    def assign_responsibilities(
        self,
        new_responsibilities: float | np.ndarray,
        patient: int | None = None,
        subgroup: str | None = None,
        component: int | None = None,
    ):
        """Assign responsibilities to the model.

        They should have the shape ``(num_patients, num_components)`` and summing them
        along the last axis should yield a vector of ones.

        Note that these responsibilities essentially become the latent variables
        of the model if they are "hard", i.e. if they are either 0 or 1 and thus
        represent a one-hot encoding of the component assignments.
        """
        pat_slice = slice(None) if patient is None else patient
        comp_slice = (*RESP_COL, slice(None) if component is None else component)

        if subgroup is not None:
            sub_data = self.subgroups[subgroup].patient_data
            sub_data.loc[pat_slice,comp_slice] = new_responsibilities
            return

        patient_idx = 0
        for subgroup in self.subgroups.values():
            sub_data = subgroup.patient_data
            patient_idx += len(sub_data)

            if patient is not None:
                if patient_idx > patient:
                    sub_data.loc[pat_slice,comp_slice] = new_responsibilities
                    return

            else:
                sub_resp = new_responsibilities[:len(sub_data)]
                sub_data.loc[pat_slice,comp_slice] = sub_resp
                new_responsibilities = new_responsibilities[len(sub_data):]


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
        self._mixture_coefs = None
        grouped = patient_data.groupby(split_by)

        for label, data in grouped:
            self.subgroups[label] = self._model_cls(**self._model_kwargs)
            data = join_with_responsibilities(
                data, num_components=len(self.components)
            )
            self.subgroups[label].load_patient_data(data, **kwargs)


    @property
    def patient_data(self) -> pd.DataFrame:
        """Return all patients stored in the individual subgroups."""
        return pd.concat([
            subgroup.patient_data for subgroup in self.subgroups.values()
        ], ignore_index=True)


    def complete_data_likelihood(
        self,
        responsibilities: np.ndarray | None = None,
        mixture_coefs: np.ndarray | None = None,
        model_params: np.ndarray | None = None,
        log: bool = True,
    ) -> float:
        """Compute the complete data likelihood of the model."""
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
            n_clusters=len(self.components),
            cluster_assignment=cluster_assignment,
            data_input=data,
            patterns=for_states,
            lnls=list(self.lymph_model.graph.lnls.keys()),
            save_name=None,
        )

        return oc_df, _
