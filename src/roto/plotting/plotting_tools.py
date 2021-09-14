from typing import Optional

import numpy as np
from matplotlib.figure import Axes, Figure
from matplotlib.ticker import Formatter, ScalarFormatter
from scipy.stats import binned_statistic


def split_phase(phase, data, timeseries=None, buff=0.9, nperiods=1):
    """
    returns list of lists of data & phases for complete periods
    :param phase: (requires sorted phased timeseries)
    :param data:
    :param timeseries:
    :param nperiods:
    :param buff: require this much phase coverage in first and last segments
    """
    phases = []
    datas = []
    timeseriess = [] if timeseries is not None else None

    idx_changes = np.where(np.diff(phase) < 0)[0][::nperiods]
    if len(idx_changes) > 0:
        use_first = True if (phase[0] < 1.0 - buff) else False
        use_last = True if (phase[-1] > buff) else False

        if use_first:
            phases.append(phase[: idx_changes[0]])
            datas.append(data[: idx_changes[0]])
            if timeseriess is not None:
                timeseriess.append(timeseries[: idx_changes[0]])

        for i, idx in enumerate(idx_changes[:-1]):
            phases.append(phase[idx + 1 : idx_changes[i + 1]])
            datas.append(data[idx + 1 : idx_changes[i + 1]])
            if timeseriess is not None:
                timeseriess.append(timeseries[idx + 1 : idx_changes[i + 1]])

        if use_last or np.any(np.diff(phase[idx_changes[-1] + 1 :]) < 0):
            phases.append(phase[idx_changes[-1] :])
            datas.append(data[idx_changes[-1] :])
            if timeseriess is not None:
                timeseriess.append(timeseries[idx_changes[-1] :])
        if timeseriess is not None:
            return phases, datas, timeseriess
        else:
            return phases, datas
    return [phase], [data]


def calculate_phase(timeseries, period, epoch):
    """
    Phase fold time series on period and epoch
    """
    return np.mod(np.array(timeseries) - epoch, period) / period


def rescale_phase(phase, max_phase=0.2):
    """
    Shift phase points if greater than max_phase to negative
    """
    return [p - 1 if p > 1 - max_phase else p for p in phase]


def append_to_phase(phase, data, amt=0.05):
    """
    Add additional data outside of phase 0-1.
    """
    indexes_before = [i for i, p in enumerate(phase) if p > 1 - amt]
    indexes_after = [i for i, p in enumerate(phase) if p < amt]

    phase_before = [phase[i] - 1 for i in indexes_before]
    data_before = [data[i] for i in indexes_before]

    phase_after = [phase[i] + 1 for i in indexes_after]
    data_after = [data[i] for i in indexes_after]

    return (
        np.concatenate((phase_before, phase, phase_after)),
        np.concatenate((data_before, data, data_after)),
    )


def bin_phase_curve(phase, data, statistic="median", bins=20):
    """
    Bin phase curve.
    """
    bin_medians, bin_edges, _ = binned_statistic(
        phase, data, statistic=statistic, bins=bins
    )
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[1:] - bin_width / 2

    return bin_centers, bin_medians


def create_axis_with_formatter(
    fig: Figure, gridspec_position, formatter: Optional[Formatter] = None, **kwargs
) -> Axes:
    """Create subplot figure and apply formatter to x/y axis.

    Args:
        fig (Figure): Matplotlib Figure
        gridspec_position (gridspec slice): gridspec slice / position.
        formatter (Optional[Formatter], optional): Matplotlib Ticker Formatter.

    Returns:
        Axes: [description]
    """
    if not formatter:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)

    ax = fig.add_subplot(gridspec_position, **kwargs)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    return ax


def rel_flux_to_ppt(
    flux_arr: np.ndarray,
    normalise: bool = False,
    normalisation_value: float = 1.0,
    center_around: float = 0.0,
) -> np.ndarray:
    """Convert an array in relative flux into ppt

    Args:
        flux_arr (np.ndarray): [description]
        normalise (bool, optional): [description]. Defaults to False.
        normalisation_value (float, optional): [description]. Defaults to 1.0.
        center_around (float, optional): [description]. Defaults to 0.0.

    Returns:
        np.ndarray: [description]
    """
    if not normalise:
        return flux_arr * 1.0e3
    else:
        return (flux_arr / normalisation_value - center_around) * 1.0e3


def ppt_to_rel_flux(
    flux_arr: np.ndarray,
    normalise: bool = False,
    normalisation_value: float = 1.0,
    center_around: float = 0.0,
) -> np.ndarray:
    """Convert an array in ppt into relative flux

    Args:
        flux_arr (np.ndarray): [description]
        normalise (bool, optional): [description]. Defaults to False.
        normalisation_value (float, optional): [description]. Defaults to 1.0.
        center_around (float, optional): [description]. Defaults to 0.0.

    Returns:
        np.ndarray: [description]
    """
    if not normalise:
        return flux_arr / 1.0e3
    else:
        return (flux_arr / 1.0e3 + center_around) * normalisation_value
