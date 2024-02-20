"""
This module defines the class wrapping the base model and composing the mixture model
likelihood from the components and subgroups in the data.
"""
# pylint: disable=logging-fstring-interpolation

import logging
import warnings
from typing import Any, Iterable, Iterator, Literal

import lymph
import numpy as np
import pandas as pd
from lymph.diagnose_times import DistributionsUserDict

from lymixture.utils import (
    RESP_COL,
    T_STAGE_COL,
    join_with_responsibilities,
    split_over_components,
)

pd.options.mode.copy_on_write = True
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)


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
        normalize: bool = True,
    ) -> float | pd.Series | pd.DataFrame:
        """Get mixture coefficients for the given ``subgroup`` and ``component``.

        The mixture coefficients are sliced by the given ``subgroup`` and ``component``
        which means that if no subgroupd and/or component is given, multiple mixture
        coefficients are returned.

        If ``normalize`` is set to ``True``, the mixture coefficients are normalized
        along the component axis before being returned.
        """
        if getattr(self, "_mixture_coefs", None) is None:
            self._mixture_coefs = self._create_empty_mixture_coefs()

        if normalize:
            self.normalize_mixture_coefs()

        component = slice(None) if component is None else component
        subgroup = slice(None) if subgroup is None else subgroup
        return self._mixture_coefs.loc[component, subgroup]


    def set_mixture_coefs(
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
        if getattr(self, "_mixture_coefs", None) is None:
            self._mixture_coefs = self._create_empty_mixture_coefs()

        component = slice(None) if component is None else component
        subgroup = slice(None) if subgroup is None else subgroup
        self._mixture_coefs.loc[component, subgroup] = new_mixture_coefs


    def normalize_mixture_coefs(self) -> None:
        """Normalize the mixture coefficients to sum to one."""
        self._mixture_coefs = self._mixture_coefs / self._mixture_coefs.sum(axis=0)


    def repeat_mixture_coefs(self, t_stage: str, log: bool = True) -> np.ndarray:
        """Stretch the mixture coefficients to match the number of patients."""
        res = np.empty(shape=(0, len(self.components)))
        for label, subgroup in self.subgroups.items():
            num_patients = subgroup.diagnose_matrices[t_stage].shape[1]
            res = np.vstack([
                res,
                np.tile(self.get_mixture_coefs(subgroup=label), (num_patients, 1))
            ])

        return np.log(res) if log else res


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
        filter_by: tuple[str, str, str] | None = None,
        filter_value: Any | None = None,
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

        if filter_by is not None and filter_value is not None:
            filter_idx = resp_table[filter_by] == filter_value
            resp_table = resp_table.loc[filter_idx]

        pat_slice = slice(None) if patient is None else patient
        comp_slice = (*RESP_COL, slice(None) if component is None else component)
        res = resp_table.loc[pat_slice,comp_slice]
        try:
            return res[RESP_COL]
        except (KeyError, IndexError):
            return res


    def set_responsibilities(
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


    def normalize_responsibilities(self) -> None:
        """Normalize the responsibilities to sum to one."""
        for label in self.subgroups:
            sub_resps = self.get_responsibilities(subgroup=label)
            self.set_responsibilities(sub_resps / sub_resps.sum(axis=1), subgroup=label)


    def harden_responsibilities(self) -> None:
        """Make the responsibilities hard, i.e. convert them to one-hot encodings."""
        resps = self.get_responsibilities().to_numpy()
        max_resps = np.max(resps, axis=1)
        hard_resps = np.where(resps == max_resps[:,None], 1, 0)
        self.set_responsibilities(hard_resps)


    @property
    def diag_time_dists(self) -> DistributionsUserDict:
        """Distributions over diagnose times of the mixture, delegated from components."""
        return self.components[0].diag_time_dists


    def update_modalities(
        self,
        new_modalities,
        subgroup: str | None = None,
        clear: bool = False,
    ):
        """Update the modalities of the mixture's subgroup models.

        The subgroups' modalities are updated with the given ``new_modalities``. If
        ``subgroup`` is given, only the modalities of that subgroup are updated. When
        ``clear`` is set to ``True``, the existing modalities are cleared before
        updating.
        """
        subgroup_keys = [subgroup] if subgroup is not None else self.subgroups.keys()

        for key in subgroup_keys:
            if clear:
                self.subgroups[key].modalities.clear()
            self.subgroups[key].modalities.update(new_modalities)


    def update_diag_time_dists(
        self,
        new_diag_time_dists: DistributionsUserDict,
        component: int | None = None,
        clear: bool = False,
    ):
        """Update the diagnose time distributions of the mixture's components.

        The diagnose time distributions of the components are updated with the given
        ``new_diag_time_dists``. If ``component`` is given, only the diagnose time
        distributions of that component are updated. When ``clear`` is set to ``True``,
        the existing diagnose time distributions are cleared before updating.
        """
        comp_slice = slice(None) if component is None else component
        components = self.components[comp_slice]

        for component in components:
            if clear:
                component.diag_time_dists.clear()
            component.diag_time_dists.update(new_diag_time_dists)


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
            if label not in self.subgroups:
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


    def get_t_stage_intersection(
        self, get_for: Literal["subgroups", "components"],
    ) -> set[str]:
        """Get the intersection of T-stages defined in subgroups or components.

        This method returns those T-stages that are defined in all subgroups or all
        components (depending on the value of ``get_for``).

        In case of the subgroups, the T-stages are taken from the patient data. For the
        components, the T-stages are taken from the diagnose time distributions.
        """
        if get_for == "subgroups":
            generator = (
                set(sub.patient_data[T_STAGE_COL].unique())
                for sub in self.subgroups.values()
            )
        elif get_for == "components":
            generator = (comp.diag_time_dists.keys() for comp in self.components)
        else:
            raise ValueError(
                f"Unknown value for 'get_for': {get_for}. Must be 'subgroups' or "
                "'components'."
            )

        t_stages = None
        for item in generator:
            if t_stages is None:
                t_stages = item
            else:
                t_stages &= item
        return t_stages


    @property
    def t_stages(self) -> set[str]:
        """Compute the intersection of T-stages defined in subgroups and components."""
        return (
            self.get_t_stage_intersection("components")
            & self.get_t_stage_intersection("subgroups")
        )


    def comp_component_patient_likelihood(
        self,
        t_stage: str,
        log: bool = True,
    ) -> np.ndarray:
        """Compute the (log-)likelihood of all patients, given the components.

        The returned array has shape ``(num_components, num_patients)`` and contains
        the likelihood of each patient under each component. If ``log`` is set to
        ``True``, the likelihoods are returned in log-space.
        """
        stacked_diag_matrices = np.hstack([
            subgroup.diagnose_matrices[t_stage] for subgroup in self.subgroups.values()
        ])
        llhs = np.empty(shape=(stacked_diag_matrices.shape[1], len(self.components)))
        for i, component in enumerate(self.components):
            llhs[:,i] = component.comp_state_dist(t_stage=t_stage) @ stacked_diag_matrices

        return np.log(llhs) if log else llhs


    def comp_patient_mixture_likelihood(
        self,
        t_stage: str,
        log: bool = True,
        marginalize_components: bool = False,
    ) -> np.ndarray:
        """Compute the (log-)likelihood of all patients under the mixture model.

        This is essentially the (log-)likelihood of all patients given the individual
        components, but weighted by the mixture coefficients.

        If ``marginalize_components`` is set to ``True``, the likelihoods are summed
        over the components, effectively marginalizing the components out of the
        likelihoods.
        """
        component_patient_likelihood = self.comp_component_patient_likelihood(t_stage, log)
        full_mixture_coefs = self.repeat_mixture_coefs(t_stage, log)

        if log:
            llh = full_mixture_coefs + component_patient_likelihood
        else:
            llh = full_mixture_coefs * component_patient_likelihood

        if marginalize_components:
            return np.logaddexp.reduce(llh, axis=0) if log else np.sum(llh, axis=0)

        return llh


    def complete_data_likelihood(
        self,
        t_stage: str | None = None,
        log: bool = True,
    ) -> float:
        """Compute the complete data likelihood of the model."""
        if t_stage is None:
            t_stages = self.t_stages
        else:
            t_stages = [t_stage]

        llh = 0 if log else 1.0
        for t in t_stages:
            llhs = self.comp_patient_mixture_likelihood(t, log)
            resps = self.get_responsibilities(
                filter_by=T_STAGE_COL, filter_value=t
            ).to_numpy()

            if log:
                llh += np.sum(resps * llhs)
            else:
                llh *= np.prod(llhs ** resps)

        return llh
