"""
Generate a corner plot of the drawn samples.

A corner plot is a combination of 1D and 2D marginals of probability distributions.
The library I use for this is built on `matplotlib` and is called
[`corner`](https://github.com/dfm/corner.py).
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import logging
from pathlib import Path
from typing import List, Union

import corner
import emcee
import lymph

from lyscripts.plot.utils import save_figure
from lyscripts.utils import load_yaml_params, model_from_config

logger = logging.getLogger(__name__)


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
    Add arguments needed to run this script to a `subparsers` instance
    and run the respective main function when chosen.
    """
    parser.add_argument(
        "model", type=Path,
        help="Path to model output files (HDF5)."
    )
    parser.add_argument(
        "output", type=Path,
        help="Path to output corner plot (SVG)."
    )
    parser.add_argument(
        "-p", "--params", default="./params.yaml", type=Path,
        help="Path to parameter file"
    )

    parser.set_defaults(run_main=main)


def get_param_labels(
    model: Union[lymph.models.Unilateral, lymph.models.Bilateral],
) -> List[str]:
    """Create labels from a `model`.

    An example:
    >>> graph = {
    ...     ("tumor", "primary"): ["II", "III"],
    ...     ("lnl", "II"): ["III"],
    ...     ("lnl", "III"): [],
    ... }
    >>> model = lymph.models.Unilateral(graph)
    >>> from lyscripts.utils import add_tstage_marg
    >>> add_tstage_marg(model, ["early", "late"], 0.3, 10)
    >>> get_param_labels(model)
    ['primary->II', 'primary->III', 'II->III', 'late']
    """
    binom_labels = []
    for t_stage,dist in model.diag_time_dists.items():
        if dist.is_updateable:
            binom_labels.append(t_stage)

    if isinstance(model, lymph.models.Unilateral):
        base_labels = [f"{e.start}->{e.end}" for e in model.base_edges]
        trans_labels = [f"{e.start}->{e.end}" for e in model.trans_edges]
        return [*base_labels, *trans_labels, *binom_labels]

    if isinstance(model, lymph.models.Bilateral):
        base_ipsi_labels = [f"i {e.start}->{e.end}" for e in model.ipsi.base_edges]
        base_contra_labels = [f"c {e.start}->{e.end}" for e in model.contra.base_edges]
        trans_labels = [f"{e.start}->{e.end}" for e in model.ipsi.trans_edges]
        return [*base_ipsi_labels, *base_contra_labels, *trans_labels, *binom_labels]

    # if isinstance(model, lymph.MidlineBilateral):
    #     base_ipsi_labels = [f"i {e.start}->{e.end}" for e in model.ext.ipsi.base_edges]
    #     base_contra_ext_labels = [
    #         f"ce {e.start}->{e.end}" for e in model.ext.contra.base_edges
    #     ]
    #     base_contra_noext_labels = [
    #         f"cn {e.start}->{e.end}" for e in model.noext.contra.base_edges
    #     ]
    #     trans_labels = [f"{e.start}->{e.end}" for e in model.ext.ipsi.trans_edges]
    #     return [
    #         *base_ipsi_labels,
    #         *base_contra_noext_labels,
    #         "mixing $\\alpha$",
    #         *trans_labels,
    #         *binom_labels,
    #     ] if model.use_mixing else [
    #         *base_ipsi_labels,
    #         *base_contra_ext_labels,
    #         *base_contra_noext_labels,
    #         *trans_labels,
    #         *binom_labels,
    #     ]


def main(args: argparse.Namespace):
    """
    This (sub)subrogram shows the following help message when asking for it
    via `lyscripts plot corner --help`

    ```
    USAGE: lyscripts plot corner [-h] [-p PARAMS] model output

    Generate a corner plot of the drawn samples.

    A corner plot is a combination of 1D and 2D marginals of probability
    distributions. The library I use for this is built on `matplotlib` and is called
    [`corner`](https://github.com/dfm/corner.py).

    POSITIONAL ARGUMENTS:
        model                 Path to model output files (HDF5).
        output                Path to output corner plot (SVG).

    OPTIONAL ARGUMENTS:
        -h, --help            show this help message and exit
        -p, --params PARAMS   Path to parameter file (default: ./params.yaml)
    ```
    """
    params = load_yaml_params(args.params, logger=logger)

    backend = emcee.backends.HDFBackend(args.model, read_only=True)
    logger.info(f"Opened model as emcee backend from {args.model}")

    model = model_from_config(
        graph_params=params["graph"],
        model_params=params["model"],
    )
    labels = get_param_labels(model)
    labels = [label.replace("->", "➜") for label in labels]

    chain = backend.get_chain(flat=True)
    if len(labels) != chain.shape[1]:
        raise RuntimeError(f"length labels: {len(labels)}, shape chain: {chain.shape}")
    fig = corner.corner(
        chain,
        labels=labels,
        show_titles=True,
    )

    save_figure(args.output, fig, formats=["png", "svg"], logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
