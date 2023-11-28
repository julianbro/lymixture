"""
A `streamlit` app for computing, displaying and reproducing prevalence estimates.

The primary goal with this little GUI is that one can quickly draft some data &
prediction comparisons visually and then copy & paste the configuration in YAML format
that is necessary to reproduce this via the `lyscripts.predict.prevalences` script.
"""
import argparse
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Union

import lymph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from lyscripts.plot.utils import COLOR_CYCLE, Histogram, Posterior, draw
from lyscripts.predict.prevalences import (
    compute_observed_prevalence,
    generate_predicted_prevalences,
)
from lyscripts.predict.utils import complete_pattern, reduce_pattern
from lyscripts.utils import (
    LymphModel,
    create_model_from_config,
    get_lnls,
    load_data_for_model,
    load_hdf5_samples,
    load_yaml_params,
)


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """
    Add an `ArgumentParser` to the subparsers action.
    """
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """
    Add arguments needed to run this `streamlit` app.
    """
    parser.add_argument(
        "--message", type=str,
        help="Print our this little message."
    )

    parser.set_defaults(run_main=launch_streamlit)


def launch_streamlit(*_args, discard_args_idx: int = 3, **_kwargs):
    """
    Regardless of the entry point into this script, this function will start
    `streamlit` and pass on the provided command line arguments.

    It will discard all entries in the `sys.argv` that come before the
    `discard_args_idx`, because this also usually contains e.g. the name of the current
    file that might not be relevant to the streamlit app.
    """
    try:
        from streamlit.web.cli import main as st_main
    except ImportError as mnf_err:
        raise ImportError(
            "Install lyscripts with the `apps` option to install the necessary "
            "requirements for running the streamlit apps."
        ) from mnf_err

    sys.argv = ["streamlit", "run", __file__, "--", *sys.argv[discard_args_idx:]]
    st_main()


def _get_lnl_pattern_label(selected: Optional[bool] = None) -> str:
    """Return labels for the involvement options of an LNL."""
    if selected is None:
        return "Unknown"
    elif selected:
        return "Involved"
    elif not selected:
        return "Healthy"
    else:
        raise ValueError("Selected option can only be `True`, `False` or `None`.")


def _get_midline_ext_label(selected: Optional[bool] = None) -> str:
    """Return labels for the options of the midline extension."""
    if selected is None:
        return "Unknown"
    elif selected:
        return "Extension"
    elif not selected:
        return "Lateralized"
    else:
        raise ValueError("Selected option can only be `True`, `False` or `None`.")


def interactive_load(streamlit):
    """
    Load the YAML file defining the parameters, the CSV file with the patient data
    and the HDF5 file with the drawn model samples interactively.
    """
    params_file = streamlit.file_uploader(
        label="YAML params file",
        type=["yaml", "yml"],
        help="Parameter YAML file containing configurations w.r.t. the model etc.",
    )
    params = load_yaml_params(params_file)
    model = create_model_from_config(params)
    is_unilateral = isinstance(model, lymph.models.Unilateral)

    streamlit.write("---")

    data_file = streamlit.file_uploader(
        label="CSV file of patient data",
        type=["csv"],
        help="CSV spreadsheet containing lymphatic patterns of progression",
    )
    header_rows = [0,1] if is_unilateral else [0,1,2]
    patient_data = load_data_for_model(data_file, header_rows=header_rows)

    streamlit.write("---")

    samples_file = streamlit.file_uploader(
        label="HDF5 sample file",
        type=["hdf5", "hdf", "h5"],
        help="HDF5 file containing the samples."
    )
    samples = load_hdf5_samples(samples_file)

    return model, patient_data, samples


def interactive_pattern(
    streamlit,
    is_unilateral: bool,
    lnls: List[str],
    side: str
) -> Dict[str, bool]:
    """
    Create a `streamlit` panel for all specified `lnls` in one `side` of a patient's
    neck to specify the lymphatic pattern of interest, which is then returned.
    """
    streamlit.subheader(f"{side}lateral")
    side_pattern = {}

    if side == "contra" and is_unilateral:
        return side_pattern

    for lnl in lnls:
        side_pattern[lnl] = streamlit.radio(
            label=f"LNL {lnl}",
            options=[False, None, True],
            index=1,
            key=f"{side}_{lnl}",
            format_func=_get_lnl_pattern_label,
            horizontal=True,
        )

    return side_pattern


def interactive_additional_params(
    streamlit: ModuleType,
    model: LymphModel,
    data: pd.DataFrame,
    samples: np.ndarray,
) -> Dict[str, Any]:
    """
    Allow the user to select T-category, midline extension and whether to invert the
    computed prevalence (meaning computing $1 - p$, when $p$ is the prevalence).

    The respective controls are presented next to each other in three dedicated columns.
    """
    control_cols = streamlit.columns([1,2,1,1,1])
    t_stage = control_cols[0].selectbox(
        label="T-category",
        options=model.diag_time_dists.keys(),
    )
    modalities_in_data = data.columns.get_level_values(level=0).difference(
        ["patient", "tumor", "positive_dissected", "total_dissected", "info"]
    )
    selected_modality = control_cols[1].selectbox(
        label="Modality",
        options=modalities_in_data,
        index=5,
    )
    midline_ext = control_cols[2].radio(
        label="Midline Extension",
        options=[False, None, True],
        index=0,
        format_func=_get_midline_ext_label,
    )

    invert = control_cols[3].radio(
        label="Invert?",
        options=[False, True],
        index=0,
        format_func=lambda x: "Yes" if x else "No",
    )

    thin = control_cols[4].slider(
        label="Sample thinning",
        min_value=1,
        max_value=len(samples) // 100,
        value=100,
    )

    return {
        "t_stage": t_stage,
        "modality": selected_modality,
        "midline_ext": midline_ext,
        "invert": invert,
        "thin": thin,
    }


def reset(session_state: Dict[str, Any]):
    """Reset `streamlit` session state."""
    for key in session_state.keys():
        del session_state[key]


def add_current_scenario(
    session_state: Dict[str, Any],
    pattern: Dict[str, Dict[str, bool]],
    model: LymphModel,
    samples: np.ndarray,
    data: pd.DataFrame,
    prevs_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Union[Histogram, Posterior]]:
    """
    Compute the prevalence of a `pattern` as observed in the `data` and as predicted
    by the `model` (using a set of `samples`). The results are then stored in the
    `contents` list ready to be plotted. The `prevs_kwargs` are directly passed on to
    the functions `lyscripts.predict.prevalences.compute_observed_prevalence`
    and `lyscripts.predict.prevalences.generate_predicted_prevalences`.
    """
    num_success, num_total = compute_observed_prevalence(
        pattern=pattern,
        data=data,
        lnls=get_lnls(model),
        **prevs_kwargs,
    )

    prevs_gen = generate_predicted_prevalences(
        pattern=pattern,
        model=model,
        samples=samples,
        **prevs_kwargs,
    )
    computed_prevs = np.zeros(shape=len(samples))
    for i, prevalence in enumerate(prevs_gen):
        computed_prevs[i] = prevalence

    next_color = next(COLOR_CYCLE)
    beta_posterior = Posterior(num_success, num_total, kwargs={"color": next_color})
    histogram = Histogram(computed_prevs, kwargs={"color": next_color})

    session_state["contents"].append(beta_posterior)
    session_state["contents"].append(histogram)

    session_state["scenarios"].append({
        "pattern": reduce_pattern(pattern), **prevs_kwargs
    })


def main(args: argparse.Namespace):
    """
    The main function that contains the `streamlit` code and main functionality.
    """
    import streamlit as st

    st.title("Prevalence")

    with st.sidebar:
        model, patient_data, samples = interactive_load(st)

    st.write("---")

    contra_col, ipsi_col = st.columns(2)
    container = {"ipsi": ipsi_col, "contra": contra_col}

    lnls = get_lnls(model)
    is_unilateral = isinstance(model, lymph.models.Unilateral)

    pattern = {}
    for side in ["ipsi", "contra"]:
        with container[side]:
            pattern[side] = interactive_pattern(st, is_unilateral, lnls, side)

    pattern = complete_pattern(pattern, lnls)
    st.write("---")

    prevs_kwargs = interactive_additional_params(st, model, patient_data, samples)
    thin = prevs_kwargs.pop("thin")

    st.write("---")

    if "contents" not in st.session_state:
        st.session_state["contents"] = []

    if "scenarios" not in st.session_state:
        st.session_state["scenarios"] = []

    button_cols = st.columns(6)
    button_cols[0].button(
        label="Reset plot",
        on_click=reset,
        args=(st.session_state,),
        type="secondary",
    )
    button_cols[1].button(
        label="Add figures",
        on_click=add_current_scenario,
        kwargs={
            "session_state": st.session_state,
            "pattern": pattern,
            "model": model,
            "samples": samples[::thin],
            "data": patient_data,
            "prevs_kwargs": prevs_kwargs,
        },
        type="primary",
    )

    fig, ax = plt.subplots()
    draw(axes=ax, contents=st.session_state.get("contents", []), xlims=(0., 100.))
    ax.legend()
    st.pyplot(fig)

    st.write("---")

    for scenario in st.session_state["scenarios"]:
        st.code(yaml.dump(scenario))


if __name__ == "__main__":
    if "__streamlit__" in locals():
        parser = argparse.ArgumentParser(description=__doc__)
        _add_arguments(parser)

        args = parser.parse_args()
        main(args)

    else:
        launch_streamlit(discard_args_idx=1)
