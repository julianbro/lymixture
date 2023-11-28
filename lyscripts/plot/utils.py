"""
Utility functions for the plotting commands.
"""
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.axes._axes import Axes as MPLAxes
from matplotlib.figure import Figure

from lyscripts.decorators import (
    check_input_file_exists,
    check_output_dir_exists,
    log_state,
)

# define USZ colors
COLORS = {
    "blue": '#005ea8',
    "orange": '#f17900',
    "green": '#00afa5',
    "red": '#ae0060',
    "gray": '#c5d5db',
}
COLOR_CYCLE = cycle(COLORS.values())
CM_PER_INCH = 2.54


def _floor_at_decimal(value: float, decimal: int) -> float:
    """
    Compute the floor of `value` for the specified `decimal`, which is the distance
    to the right of the decimal point. May be negative.
    """
    power = 10**decimal
    return np.floor(power * value) / power

def _ceil_at_decimal(value: float, decimal: int) -> float:
    """
    Compute the ceiling of `value` for the specified `decimal`, which is the distance
    to the right of the decimal point. May be negative.
    """
    return - _floor_at_decimal(-value, decimal)

def _floor_to_step(value: float, step: float) -> float:
    """
    Compute the next closest value on a ladder of stepsize `step` that is below `value`.
    """
    return (value // step) * step

def _ceil_to_step(value: float, step: float) -> float:
    """
    Compute the next closest value on a ladder of stepsize `step` that is above `value`.
    """
    return _floor_to_step(value, step) + step


def _clean_and_check(filename: Union[str, Path]) -> Path:
    """
    Check if file with `filename` exists. If not, raise error, otherwise return
    cleaned `PosixPath`.
    """
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(
            f"File with the name {filename} does not exist at {filepath.resolve()}"
        )
    return filepath


@dataclass
class Histogram:
    """Class containing data for plotting a histogram."""
    values: np.ndarray
    scale: float = field(default=100.)
    kwargs: Dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        self.values = self.scale * self.values

    @classmethod
    def from_hdf5(cls, filename, dataname, scale: float = 100., **kwargs):
        """Create a histogram from an HDF5 file."""
        filename = _clean_and_check(filename)
        with h5py.File(filename, mode="r") as h5file:
            dataset = h5file[dataname]
            if "label" not in kwargs:
                kwargs["label"] = get_label(dataset.attrs)
            return cls(values=dataset[:], scale=scale, kwargs=kwargs)

    def left_percentile(self, percent: float) -> float:
        """Compute the point where `percent` of the values are to the left."""
        return np.percentile(self.values, percent)

    def right_percentile(self, percent: float) -> float:
        """Compute the point where `percent` of the values are to the right."""
        return np.percentile(self.values, 100. - percent)

@dataclass
class Posterior:
    """Class for storing plot configs for a Beta posterior."""
    num_success: int
    num_total: int
    scale: float = field(default=100.)
    kwargs: Dict[str, Any] = field(default_factory=lambda: {})

    @classmethod
    def from_hdf5(cls, filename, dataname, scale: float = 100., **kwargs) -> None:
        """Initialize data container for Beta posteriors from HDF5 file."""
        filename = _clean_and_check(filename)
        with h5py.File(filename, mode="r") as h5file:
            dataset = h5file[dataname]
            try:
                num_success = int(dataset.attrs["num_match"])
                num_total = int(dataset.attrs["num_total"])
            except KeyError as key_err:
                raise KeyError(
                    "Dataset does not contain observed prevalence data"
                ) from key_err

        return cls(num_success, num_total, scale=scale, kwargs=kwargs)

    @property
    def num_fail(self):
        return self.num_total - self.num_success

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute the probability density function."""
        return sp.stats.beta.pdf(
            x,
            a=self.num_success+1,
            b=self.num_fail+1,
            scale=self.scale
        )

    def left_percentile(self, percent: float) -> float:
        """Return the point where the CDF reaches `percent`."""
        return sp.stats.beta.ppf(
            percent / 100.,
            a=self.num_success+1,
            b=self.num_fail+1,
            scale=self.scale,
        )

    def right_percentile(self, percent: float) -> float:
        """Return the point where 100% minus the CDF equals `percent`."""
        return sp.stats.beta.ppf(
            1. - (percent / 100.),
            a=self.num_success+1,
            b=self.num_fail+1,
            scale=self.scale,
        )


def get_size(width="single", unit="cm", ratio="golden"):
    """
    Return a tuple of figure sizes in inches, as the `matplotlib` keyword argument
    `figsize` expects it. This figure size is computed from a `width`, in the `unit` of
    centimeters by default, and a `ratio` which is set to the golden ratio by default.

    Examples:
    >>> get_size(width="single", ratio="golden")
    (3.937007874015748, 2.4332557935820445)
    >>> get_size(width="full", ratio=2.)
    (6.299212598425196, 3.149606299212598)
    >>> get_size(width=10., ratio=1.)
    (3.937007874015748, 3.937007874015748)
    >>> get_size(width=5, unit="inches", ratio=2./3.)
    (5, 7.5)
    """
    if width == "single":
        width = 10
    elif width == "full":
        width = 16

    ratio = 1.618 if ratio == "golden" else ratio
    width = width / CM_PER_INCH if unit == "cm" else width
    height = width / ratio
    return (width, height)


def get_label(attrs) -> str:
    """Extract label of a histogram from the HDF5 `attrs` object of the dataset."""
    label = []
    transforms = {
        "label": str,
        "modality": str,
        "t_stage": str,
        "midline_ext": lambda x: "ext" if x else "noext"
    }
    for key,func in transforms.items():
        if key in attrs and attrs[key] is not None:
            label.append(func(attrs[key]))
    return " | ".join(label)


def get_xlims(
    contents: List[Union[Histogram, Posterior]],
    percent_lims: Tuple[float] = (10., 10.),
) -> Tuple[float]:
    """
    Compute the `xlims` of a plot containing histograms and probability density
    functions by considering their smallest and largest percentiles.
    """
    left_percentiles = np.array(
        [c.left_percentile(percent_lims[0]) for c in contents]
    )
    left_lim = np.min(left_percentiles)
    right_percentiles = np.array(
        [c.right_percentile(percent_lims[0]) for c in contents]
    )
    right_lim = np.max(right_percentiles)
    return left_lim, right_lim


def draw(
    axes: MPLAxes,
    contents: List[Union[Histogram, Posterior]],
    percent_lims: Tuple[float] = (10., 10.),
    xlims: Optional[Tuple[float]] = None,
    hist_kwargs: Optional[Dict[str, Any]] = None,
    plot_kwargs: Optional[Dict[str, Any]] = None,
) -> MPLAxes:
    """
    Draw histograms and Beta posterior from `contents` into `axes`.

    The limits of the x-axis is computed to be the smallest and largest left and right
    percentile of all provided `contents` respectively via the `percent_lims` tuple.

    The `hist_kwargs` define general settings that will be applied to all histograms.
    One additional key `'nbins'` may be used to adjust only the numbers, not the spacing
    of the histogram bins.
    Similarly, `plot_kwargs` adjusts the default settings for the Beta posteriors.

    Both these keyword arguments can be overwritten by what the individual `contents`
    have defined.
    """
    if not all(isinstance(c, (Histogram, Posterior)) for c in contents):
        raise TypeError("Contents must be `Histogram` or `Posterior` instances")

    if xlims is None:
        xlims = get_xlims(contents, percent_lims)
    elif len(xlims) != 2 or xlims[0] > xlims[-1]:
        raise ValueError("`xlims` must be tuple of two increasing values")

    x = np.linspace(*xlims, 300)

    hist_kwargs = {} if hist_kwargs is None else hist_kwargs
    nbins = hist_kwargs.pop("nbins", 60)
    default_hist_kwargs = {
        "density": True,
        "bins": np.linspace(*xlims, nbins),
        "histtype": "stepfilled",
        "alpha": 0.7,
    }
    default_hist_kwargs.update(hist_kwargs)

    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    default_plot_kwargs = {}
    default_plot_kwargs.update(plot_kwargs)

    for content in contents:
        if isinstance(content, Histogram):
            tmp_hist_kwargs = default_hist_kwargs.copy()
            tmp_hist_kwargs.update(content.kwargs)
            axes.hist(content.values, **tmp_hist_kwargs)
        elif isinstance(content, Posterior):
            tmp_plot_kwargs = default_plot_kwargs.copy()
            tmp_plot_kwargs["label"] = f"{content.num_success} / {content.num_total}"
            tmp_plot_kwargs.update(content.kwargs)
            axes.plot(x, content.pdf(x), **tmp_plot_kwargs)

    axes.set_xlim(*xlims)
    return axes


@log_state(success_msg="Loaded MPL stylesheet")
@check_input_file_exists
def use_mpl_stylesheet(file_path: Union[str, Path]):
    """Load a `.mplstyle` stylesheet from `file_path`."""
    plt.style.use(file_path)


@log_state(success_msg="Saved matplotlib figure")
@check_output_dir_exists
def save_figure(
    output_path: Union[str, Path],
    figure: Figure,
    formats: Optional[List[str]],
):
    """Save a `figure` to `output_path` in every one of the provided `formats`."""
    for frmt in formats:
        figure.savefig(output_path.with_suffix(f".{frmt}"))
