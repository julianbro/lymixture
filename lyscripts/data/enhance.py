"""
Enhance a LyProX-style CSV dataset in two ways:

1. Add consensus diagnoses based on all available modalities using on of two methods:
`max_llh` infers the most likely true state of involvement given only the available
diagnoses. `rank` uses the available diagnositc modalities and ranks them based on
their respective sensitivity and specificity.

2. Complete sub- & super-level fields. This means that if a dataset reports LNLs IIa
and IIb separately, this script will add the column for LNL II and fill it with the
correct values. Conversely, if e.g. LNL II is reported to be healthy, we can assume
the sublevels IIa and IIb would have been reported as healthy, too.
"""
# pylint: disable=singleton-comparison,logging-fstring-interpolation
import argparse
import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from lyscripts.data.utils import load_csv_table, save_table_to_csv
from lyscripts.decorators import log_state
from lyscripts.utils import (
    CustomProgress,
    get_modalities_subset,
    load_yaml_params,
    report,
)

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
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
        "input", type=Path,
        help="Path to a LyProX-style CSV file"
    )
    parser.add_argument(
        "output", type=Path,
        help="Destination for LyProX-style output file including the consensus"
    )
    parser.add_argument(
        "-c", "--consensus", nargs="+", default=["max_llh"],
        choices=CONSENSUS_FUNCS.keys(),
        help="Choose consensus method(s)"
    )
    parser.add_argument(
        "-p", "--params", default="params.yaml", type=Path,
        help="Path to parameter file"
    )
    parser.add_argument(
        "--modalities", nargs="+",
        default=["CT", "MRI", "PET", "FNA", "diagnostic_consensus", "pathology", "pCT"],
        help="List of modalities for enhancement. Must be defined in `params.yaml`"
    )
    parser.add_argument(
        "--sublvls", nargs="+", default=["a", "b"],
        help="Indicate what kinds of sublevels exist"
    )
    parser.add_argument(
        "--lnls-with-sub", nargs="+", default=["I", "II", "V"],
        help="List of LNLs where sublevel reporting has been performed or is common"
    )

    parser.set_defaults(run_main=main)


def get_sublvl_values(
    data_frame: pd.DataFrame,
    lnl: str,
    sub_ids: List[str],
):
    """
    Get the values of sublevels (e.g. 'IIa' and 'IIb') for a given LNL and a
    dataframe.
    """
    has_sublvls = all(lnl+sub in data_frame for sub in sub_ids)
    if not has_sublvls:
        return None
    return data_frame[[lnl+sub for sub in sub_ids]].values


@log_state(
    success_msg="Inferred super-level's involvement from sub-levels",
    logger=logger,
)
def infer_superlvl_from_sublvls(
    table: pd.DataFrame,
    modalities: List[str],
    lnls_with_sub: List[str],
    sublvls: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Infer the involvement state of all `lnls_with_sub`, i.e. LNLs where sub-levels were
    reported, for each patient in the `table`. Do this for all defined `modalities` and
    take into account all specified `sublvls`.

    This means that if e.g. sub-LNL IIa reports involvement and sub-LNL IIb shows no
    sign of metastasis, this method will infer that the superlevel II must be involved
    as well.
    """
    if sublvls is None:
        sublvls = ["a", "b"]

    fixed_table = table.copy()

    for mod in modalities:
        for side in ["ipsi", "contra"]:
            for lnl in lnls_with_sub:
                sublvl_values = get_sublvl_values(
                        table[mod,side], lnl, sublvls
                    )
                if sublvl_values is None:
                    continue

                # sometimes, the sublevels both report `False` (healthy) but the
                # superlevel is involved. In this case, we want to keep the superlevel
                # as involved.
                if lnl in table[mod,side]:
                    is_superlvl_involved = table[mod,side,lnl] == True
                else:
                    is_superlvl_involved = False

                has_sublvl_involved = np.any(sublvl_values==True , axis=1)
                all_sublvls_healthy = np.all(sublvl_values==False, axis=1)

                fixed_table.loc[has_sublvl_involved, (mod,side,lnl)] = True
                fixed_table.loc[all_sublvls_healthy & ~is_superlvl_involved, (mod,side,lnl)] = False

    return fixed_table


def get_lnl_observations(
    patient: pd.Series,
    side: str,
    lnl: str,
    modalities: List[str],
) -> Tuple[bool]:
    """
    Collect the observations for an `lnl` from every one of the available `modalities`
    in a tuple. Do this for one `side` of the neck of a particular `patient`.
    """
    observations = ()

    for mod in modalities.keys():
        try:
            add_obs = patient[mod,side,lnl]
            add_obs = None if pd.isna(add_obs) else add_obs
        except KeyError:
            add_obs = None
        observations = (*observations, add_obs)

    return observations


@lru_cache
def has_all_none(obs_tuple: Tuple[np.ndarray]):
    """
    Check if all entries in the observation tuple are ``None``.
    """
    return all(obs is None for obs in obs_tuple)

@lru_cache
def or_consensus(obs_tuple: Tuple[np.ndarray]):
    """
    Compute the consensus of different diagnostic modalities by computing the
    logical OR.
    """
    if has_all_none(obs_tuple):
        return None

    return any(obs_tuple)

@lru_cache
def and_consensus(obs_tuple: Tuple[np.ndarray]):
    """
    Compute the consensus of different diagnostic modalities by computing the
    logical AND.
    """
    if has_all_none(obs_tuple):
        return None

    return not(
        any(not(obs) if obs is not None else None for obs in obs_tuple)
    )

@lru_cache
def maxllh_consensus(
    obs_tuple: Tuple[np.ndarray],
    modalities_spsn: Tuple[List[float]]
):
    """
    Compute the consensus of different diagnostic modalities using their
    respective specificity & sensitivity.

    Args:
        obs_tuple: Tuple with the involvement (``True``, ``False`` or
            ``None``).
        modalities_spsn: Tuple with 2-element lists of the specificity &
            sensitivity of the modalities corresponding to the diagnoses in the
            parameter ``obs_tuple``.

    Returns:
        The most likely true state according to the consensus from the
        diagnoses provided.
    """
    if has_all_none(obs_tuple):
        return None

    healthy_llh = 1.
    involved_llh = 1.
    for obs, spsn in zip(obs_tuple, modalities_spsn):
        if obs is None:
            continue
        spsn = np.array(spsn)
        obs = int(obs)
        spsn2x2 = np.diag(spsn) + np.diag(1. - spsn)[::-1]
        healthy_llh *= spsn2x2[obs,0]
        involved_llh *= spsn2x2[obs,1]

    healthy_vs_involved = np.array([healthy_llh, involved_llh])
    return bool(np.argmax(healthy_vs_involved))

@lru_cache
def rank_consensus(
    obs_tuple: Tuple[np.ndarray],
    modalities_spsn: Tuple[List[float]]
):
    """
    Compute the consensus of different diagnostic modalities using a ranking
    based on sensitivity & specificity.

    Args:
        obs_tuple: Tuple with the involvement (``True``, ``False`` or
            ``None``).
        modalities_spsn: Tuple with 2-element lists of the specificity &
            sensitivity of the modalities corresponding to the diagnoses in the
            parameter ``obs_tuple``.

    Returns:
        The most likely true state based on the ranking.
    """
    if has_all_none(obs_tuple):
        return None

    modalities_spsn = list(modalities_spsn)

    healthy_sens = [
        modalities_spsn[i][1] for i,obs in enumerate(obs_tuple) if obs == False
    ]
    involved_spec = [
        modalities_spsn[i][0] for i,obs in enumerate(obs_tuple) if obs == True
    ]
    if np.max([*healthy_sens, 0.]) > np.max([*involved_spec, 0.]):
        return False

    return True


CONSENSUS_FUNCS = {
    "max_llh": maxllh_consensus,
    "rank": rank_consensus,
    "logic_or": lambda obs, *_args, **_kwargs: or_consensus(obs),
    "logic_and": lambda obs, *_args, **_kwargs: and_consensus(obs),
}


def main(args: argparse.Namespace):
    """
    Below is the help output (call with `lyscripts enhance --help`)

    ```
    USAGE: lyscripts data enhance [-h]
                                  [-c {max_llh,rank,logic_or,logic_and} [{max_llh,rank,logic_or,logic_and} ...]]
                                  [-p PARAMS]
                                  [--modalities MODALITIES [MODALITIES ...]]
                                  [--sublvls SUBLVLS [SUBLVLS ...]]
                                  [--lnls-with-sub LNLS_WITH_SUB [LNLS_WITH_SUB ...]]
                                  input output

    Enhance a LyProX-style CSV dataset in two ways:

    1. Add consensus diagnoses based on all available modalities using on of two
    methods: `max_llh` infers the most likely true state of involvement given only the
    available diagnoses. `rank` uses the available diagnositc modalities and ranks
    them based on their respective sensitivity and specificity.

    2. Complete sub- & super-level fields. This means that if a dataset reports LNLs
    IIa and IIb separately, this script will add the column for LNL II and fill it
    with the correct values. Conversely, if e.g. LNL II is reported to be healthy, we
    can assume the sublevels IIa and IIb would have been reported as healthy, too.

    POSITIONAL ARGUMENTS:
      input                 Path to a LyProX-style CSV file
      output                Destination for LyProX-style output file including the
                            consensus

    OPTIONAL ARGUMENTS:
      -h, --help            show this help message and exit
      -c, --consensus {max_llh,rank,logic_or,logic_and} [{max_llh,rank,logic_or,logic_and} ...]
                            Choose consensus method(s) (default: ['max_llh'])
      -p, --params PARAMS   Path to parameter file (default: params.yaml)
      --modalities MODALITIES [MODALITIES ...]
                            List of modalities for enhancement. Must be defined in
                            `params.yaml` (default: ['CT', 'MRI', 'PET', 'FNA',
                            'diagnostic_consensus', 'pathology', 'pCT'])
      --sublvls SUBLVLS [SUBLVLS ...]
                            Indicate what kinds of sublevels exist (default: ['a',
                            'b'])
      --lnls-with-sub LNLS_WITH_SUB [LNLS_WITH_SUB ...]
                            List of LNLs where sublevel reporting has been performed
                            or is common (default: ['I', 'II', 'V'])
    ```
    """
    input_table = load_csv_table(args.input, header_row=[0,1,2], logger=logger)
    params = load_yaml_params(args.params, logger=logger)

    modalities = get_modalities_subset(
        defined_modalities=params["modalities"],
        selection=args.modalities,
    )

    available_mod_keys = sorted(set(
        input_table.columns.get_level_values(0)
    ).intersection(
        modalities.keys()
    ))
    available_mods = {key: modalities[key] for key in available_mod_keys}
    lnl_union = sorted(set().union(
        *[input_table[mod,"ipsi"].columns for mod in available_mod_keys]
    ))
    consensus = pd.DataFrame(
        index=input_table.index,
        columns=pd.MultiIndex.from_product(
            [args.consensus, ["ipsi", "contra"], lnl_union]
        )
    )

    with CustomProgress(console=report) as report_progress:
        enhance_task = report_progress.add_task(
            description=f"Compute {args.consensus} consensus of modalities...",
            total=2 * len(input_table),
        )
        for side in ["ipsi", "contra"]:
            # go through patients and LNLs and compute consensus for each
            for p,patient in input_table.iterrows():
                for lnl in lnl_union:
                    observations = get_lnl_observations(
                        patient, side, lnl, available_mods
                    )
                    for cons in args.consensus:
                        consensus[cons, side, lnl].iloc[p] = CONSENSUS_FUNCS[cons](
                            observations, available_mods.values()
                        )
                report_progress.update(enhance_task, advance=1)
        table_with_consensus = input_table.join(consensus)


    data_modalities = sorted(set(
        table_with_consensus.columns.get_level_values(0)
    ).intersection(
        [*modalities.keys(), *args.consensus]
    ))
    consensus_and_fixed_sublvlvs = infer_superlvl_from_sublvls(
        table_with_consensus,
        data_modalities,
        lnls_with_sub=args.lnls_with_sub,
        sublvls=args.sublvls,
    )
    logger.info("Fixed sub- & super level fields.")

    save_table_to_csv(args.output, consensus_and_fixed_sublvlvs, logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
