"""
Functions and methods that are shared by the prediction submodules in
`lyscripts.predict`.
"""
from typing import Dict, List, Optional


def complete_pattern(
    pattern: Optional[Dict[str, Dict[str, bool]]],
    lnls: List[str],
) -> Dict[str, Dict[str, bool]]:
    """
    Make sure the provided involvement `pattern` is correct. For each side of the neck,
    and for each of the `lnls` this should in the end contain `True`, `False` or `None`.

    Example:
    >>> pattern = {"ipsi": {"II": True}}
    >>> lnls = ["II", "III"]
    >>> complete_pattern(pattern, lnls)
    {'ipsi': {'II': True, 'III': None}, 'contra': {'II': None, 'III': None}}
    """
    if pattern is None:
        pattern = {}

    for side in ["ipsi", "contra"]:
        if side not in pattern:
            pattern[side] = {}

        for lnl in lnls:
            if lnl not in pattern[side]:
                pattern[side][lnl] = None
            elif pattern[side][lnl] is None:
                continue
            else:
                pattern[side][lnl] = bool(pattern[side][lnl])

    return pattern


def reduce_pattern(pattern: Dict[str, Dict[str, bool]]) -> Dict[str, Dict[str, bool]]:
    """
    Reduce a `pattern` by removing all entries that are `None`. This way, it should
    be completely recoverable by the `complete_pattern` function but be shorter to
    store.

    Example:
    >>> full = {
    ...     "ipsi": {"I": None, "II": True, "III": None},
    ...     "contra": {"I": None, "II": None, "III": None},
    ... }
    >>> reduce_pattern(full)
    {'ipsi': {'II': True}}
    """
    tmp_pattern = pattern.copy()
    reduced_pattern = {}
    for side in ["ipsi", "contra"]:
        if not all(v is None for v in tmp_pattern[side].values()):
            reduced_pattern[side] = {}
            for lnl, val in tmp_pattern[side].items():
                if val is not None:
                    reduced_pattern[side][lnl] = val

    return reduced_pattern
