"""Plotting and visualization utilities."""

from __future__ import annotations

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import stats

from featherpy.weighting import ONE_STD, sigmoid


def plot_feather(
    low_res_vis: NDArray[np.complex128],
    high_res_vis: NDArray[np.complex128],
    uv_distance_2d: u.Quantity,
    high_res_weighted: NDArray[np.float64],
    low_res_weighted: NDArray[np.float64],
    feathered_vis: NDArray[np.complex128],
    feather_centre: u.Quantity,
    feather_sigma: u.Quantity,
    n_uv_bins: int = 10,
) -> Figure:
    """Plot the feathering

    Args:
        low_res_vis (NDArray[np.complex128]): Low resolution visibilities
        high_res_vis (NDArray[np.complex128]): High resolution visibilities
        uv_distance_2d (u.Quantity): 2D array of uv distances
        high_res_weighted (NDArray[np.float64]): Weighted high resolution visibilities
        low_res_weighted (NDArray[np.float64]): Weighted low resolution visibilities
        feathered_vis (NDArray[np.complex128]): Feathered visibilities
        feather_centre (u.Quantity): Feather centre (in meters)
        feather_sigma (u.Quantity): Feather sigma (in meters)
        n_uv_bins (int, optional): Number of bins for smoothing. Defaults to 10.

    Returns:
        plt.Figure: Feather plot
    """
    plot_bound = (feather_centre + 25 * feather_sigma).to(u.m).value
    uv_bins = np.linspace(0, plot_bound, n_uv_bins)
    uv_bin_centers = (uv_bins[:-1] + uv_bins[1:]) / 2
    high_res_binned = stats.binned_statistic(
        x=uv_distance_2d.flatten(),
        values=np.abs(high_res_vis.flatten()),
        statistic="median",
        bins=uv_bins,
    )
    low_res_binned = stats.binned_statistic(
        x=uv_distance_2d.flatten(),
        values=np.abs(low_res_vis.flatten()),
        statistic="median",
        bins=uv_bins,
    )

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 18))
    sc_alpha = 0.1
    _ = ax1.plot(
        uv_distance_2d.ravel(),
        np.abs(high_res_vis).ravel(),
        ".",
        color="tab:blue",
        alpha=sc_alpha,
    )
    _ = ax1.plot(
        uv_bin_centers,
        high_res_binned.statistic,
        label="high resolution",
        color="tab:blue",
    )

    _ = ax1.plot(
        uv_distance_2d.ravel(),
        np.abs(low_res_vis).ravel(),
        ".",
        color="tab:orange",
        alpha=sc_alpha,
    )
    _ = ax1.plot(
        uv_bin_centers,
        low_res_binned.statistic,
        label="low resolution",
        color="tab:orange",
    )

    uvdists = np.linspace(0, plot_bound, 1000)

    sigmoid_slope = ONE_STD / feather_sigma.to(u.m).value

    ax2.plot(
        uvdists,
        sigmoid(uvdists, feather_centre.to(u.m).value, sigmoid_slope),
        label="high resolution",
    )
    ax2.plot(
        uvdists,
        sigmoid(-uvdists, -feather_centre.to(u.m).value, sigmoid_slope),
        label="low resolution",
    )

    ax3.plot(
        uv_distance_2d.ravel(),
        np.abs(high_res_weighted).ravel(),
        ".",
        label="high resolution (weighted)",
    )
    ax3.plot(
        uv_distance_2d.ravel(),
        np.abs(low_res_weighted).ravel(),
        ".",
        label="low resolution (weighted)",
    )
    ax3.plot(
        uv_distance_2d.ravel(),
        np.abs(feathered_vis).ravel(),
        ".",
        label="feathered",
    )
    ax3.set(
        xlim=(0, plot_bound),
        yscale="log",
        ylabel="Visibility Amplitude",
        xlabel="$uv$-distance / m",
    )
    ax2.set(ylabel="Visibility weights")
    ax1.set(xlim=(0, plot_bound), yscale="log", ylabel="Visibility Amplitude")
    for ax in (ax1, ax2, ax3):
        ax.axvline(
            (feather_centre + -5 * feather_sigma).to(u.m).value,
            color="black",
            linestyle="--",
        )
        ax.axvline(
            (feather_centre + 5 * feather_sigma).to(u.m).value,
            color="black",
            linestyle="--",
            label=r"Feather $\pm5\sigma$",
        )
        ax.axvline(
            feather_centre.to(u.m).value,
            color="black",
            linestyle=":",
            label="Feather centre",
        )
        ax.legend()
    return fig
