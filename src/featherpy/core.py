from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import NamedTuple

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import reproject as rp
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from radio_beam import Beam
from scipy import fft, stats

from featherpy.exceptions import ShapeError, UnitError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)
logging.captureWarnings(True)


class Visibilities(NamedTuple):
    """Data in the Fourier domain."""

    low_res_data_fft_corr: ArrayLike
    """ The low resolution data, deconvolved with the low resolution beam and reconvolved with the high resolution beam. """
    high_res_data_fft: ArrayLike
    """ The high resolution data. """
    uv_distance_2d: u.Quantity
    """ The 2D array of uv distances. """


class FitsData(NamedTuple):
    """Data from a FITS file."""

    data: u.Quantity
    """ The data. """
    beam: Beam
    """ The beam. """
    wcs: WCS
    """ The WCS. """


class FeatheredData(NamedTuple):
    """Feathered data."""

    feathered_image: u.Quantity
    """ The feathered image. """
    feathered_fft: ArrayLike
    """ The feathered data in the Fourier domain. """
    low_res_fft_weighted: ArrayLike
    """ The low resolution data in the Fourier domain, weighted for feathering. """
    high_res_fft_weighted: ArrayLike
    """ The high resolution data in the Fourier domain, weighted for feathering. """
    low_res_weights: ArrayLike
    """ The weights for the low resolution data. """
    high_res_weights: ArrayLike
    """ The weights for the high resolution data. """


def sigmoid(x: ArrayLike, x0: float = 0, k: float = 1) -> ArrayLike:
    """Sigmoid function

    Args:
        x (ArrayLike): x values
        x0 (float, optional): x offset. Defaults to 0.
        k (float, optional): growth rate. Defaults to 1.

    Returns:
        ArrayLike: sigmoid values
    """
    return 1 / (1 + np.exp(-k * (x - x0)))


def make_beam_fft(
    beam: Beam,
    data_shape: tuple[int, int],
    pix_scale: u.Quantity,
) -> ArrayLike:
    """Make the FFT of a beam in the same shape as the data

    Args:
        beam (Beam): The beam
        data_shape (tuple[int, int]): Shape of the data
        pix_scale (u.Quantity): Pixel scale (assumed to be the same in x and y)

    Returns:
        ArrayLike: Beam FFT array
    """
    beam_image = beam.as_kernel(
        pix_scale, x_size=data_shape[1], y_size=data_shape[0]
    ).array
    beam_image /= beam_image.sum()
    return fft.fftshift(fft.fft2(beam_image))


def fft_data(
    low_res_data: u.Quantity,
    high_res_data: u.Quantity,
    low_res_beam: Beam,
    high_res_beam: Beam,
    wcs: WCS,
    wavelength: u.Quantity,
    outer_uv_cut: u.Quantity | None = None,
) -> Visibilities:
    """Apply FFTs and deconvolution to the data

    Args:
        low_res_data (u.Quantity): Low resolution data (must be in Jy/sr)
        high_res_data (u.Quantity): High resolution data (must be in Jy/sr)
        low_res_beam (Beam): Low resolution beam
        high_res_beam (Beam): High resolution beam
        wcs (WCS): WCS of the data
        wavelength (u.Quantity): Wavelength of the data
        outer_uv_cut (u.Quantity | None, optional): UV cut to apply to low res data. Defaults to None.

    Raises:
        UnitError: If low_res_data is not in Jy/sr
        UnitError: If high_res_data is not in Jy/sr
        ShapeError: If the data shapes do not match

    Returns:
        Visibilities: low_res_data_fft_corr, high_res_data_fft, uv_distance_2d
    """

    # Sanity checks
    if low_res_data.unit != u.Jy / u.sr:
        msg = "low_res_data  must be in Jy/sr"
        raise UnitError(msg)
    if high_res_data.unit != u.Jy / u.sr:
        msg = "high_res_data must be in Jy/sr"
        raise UnitError(msg)

    if low_res_data.shape != high_res_data.shape:
        msg = (
            f"Data shapes do not match ({low_res_data.shape=}, {high_res_data.shape=})"
        )
        raise ShapeError(msg)

    # Convert the data to Jy/sr
    low_res_data = low_res_data.to(
        u.Jy / u.sr, equivalencies=u.beam_angular_area(low_res_beam.sr)
    )
    high_res_data = high_res_data.to(
        u.Jy / u.sr, equivalencies=u.beam_angular_area(high_res_beam.sr)
    )

    pix_scales = proj_plane_pixel_scales(wcs.celestial)
    assert pix_scales[0] == pix_scales[1]
    pix_scale = pix_scales[0] * u.deg

    # Make the beam FFTs
    low_res_beam_fft = make_beam_fft(low_res_beam, low_res_data.shape, pix_scale)
    high_res_beam_fft = make_beam_fft(high_res_beam, high_res_data.shape, pix_scale)

    # FFT the data
    low_res_data_fft = fft.fftshift(fft.fftn(low_res_data))
    high_res_data_fft = fft.fftshift(fft.fftn(high_res_data))
    v_size, u_size = low_res_data_fft.shape
    u_array = fft.fftshift(fft.fftfreq(u_size, d=pix_scale.to(u.radian).value))
    v_array = fft.fftshift(fft.fftfreq(v_size, d=pix_scale.to(u.radian).value))
    u_2d_array, v_2d_array = np.meshgrid(u_array, v_array)
    uv_distance_2d = np.hypot(u_2d_array, v_2d_array) * wavelength.to(u.m)

    # Deconvolve the low resolution beam
    low_res_data_fft_corr = low_res_data_fft / np.abs(low_res_beam_fft)
    # Reconvolve with the high resolution beam
    low_res_data_fft_corr *= np.abs(high_res_beam_fft)
    low_res_data_fft_corr[~np.isfinite(low_res_data_fft_corr)] = 0

    if outer_uv_cut is not None:
        low_res_data_fft_corr[
            uv_distance_2d.to(u.m).value > outer_uv_cut.to(u.m).value
        ] = 0

    return Visibilities(low_res_data_fft_corr, high_res_data_fft, uv_distance_2d)


def feather(
    low_res_data_fft_corr: ArrayLike,
    high_res_data_fft: ArrayLike,
    high_res_beam: Beam,
    uv_distance_2d: u.Quantity,
    feather_centre: u.Quantity,
    feather_sigma: u.Quantity,
    low_res_scale_factor: float | None = None,
) -> FeatheredData:
    """Feather the data

    Args:
        low_res_data_fft_corr (ArrayLike): A 2D array of the low resolution data in the Fourier domain
        high_res_data_fft (ArrayLike): A 2D array of the high resolution data in the Fourier domain
        high_res_beam (Beam): The high resolution beam
        uv_distance_2d (u.Quantity): A 2D array of the uv distances
        feather_centre (u.Quantity): The centre of the feathering function (in meters)
        feather_sigma (u.Quantity): The width of the feathering function (in meters)
        low_res_scale_factor (float | None, optional): Scaling factor for the low res data. Defaults to None.

    Raises:
        ShapeError: If the data shapes do not match
        UnitError: If feather_centre is not in meters
        UnitError: If feather_sigma is not in meters
        UnitError: If uv_distance_2d is not in meters

    Returns:
        FeatheredData: feathered_image, feathered_fft, low_res_fft_weighted, high_res_fft_weighted, low_res_weights, high_res_weights
    """
    if low_res_data_fft_corr.shape != high_res_data_fft.shape:
        msg = f"Data shapes do not match ({low_res_data_fft_corr.shape=}, {high_res_data_fft.shape=})"
        raise ShapeError(msg)
    try:
        _ = feather_centre.to(u.m)
    except u.UnitConversionError as e:
        msg = "feather_centre must be in meters (or convertible to meters)"
        raise UnitError(msg) from e
    try:
        _ = feather_sigma.to(u.m)
    except u.UnitConversionError as e:
        msg = "feather_sigma must be in meters (or convertible to meters)"
        raise UnitError(msg) from e
    try:
        _ = uv_distance_2d.to(u.m)
    except u.UnitConversionError as e:
        msg = "uv_distance_2d must be in meters (or convertible to meters)"
        raise UnitError(msg) from e

    if low_res_scale_factor is None:
        logger.info("Calculating low resolution scale factor")
        uv_start = (feather_centre + -5 * feather_sigma).to(u.m)
        uv_end = (feather_centre + 5 * feather_sigma).to(u.m)
        overlap_index = (uv_distance_2d > uv_start) & (uv_distance_2d < uv_end)
        auto_low_res_scale_factor = np.nanmedian(
            np.abs(high_res_data_fft[overlap_index])
            / np.abs(low_res_data_fft_corr[overlap_index])
        )
        low_res_scale_factor = np.round(auto_low_res_scale_factor, 3)

    msg = f"Feathering with low resolution scale factor: {low_res_scale_factor}"
    logger.info(msg)
    low_res_data_fft_corr *= low_res_scale_factor

    # Approximately convert 1 sigma to the slope of the sigmoid
    one_std = -1 * np.log((1 - stats.norm.cdf(1)) * 2)
    sigmoid_slope = one_std / feather_sigma.to(u.m).value

    high_res_weights = sigmoid(
        x=uv_distance_2d.to(u.m).value, x0=feather_centre.to(u.m).value, k=sigmoid_slope
    )
    low_res_weights = sigmoid(
        x=-uv_distance_2d.to(u.m).value,
        x0=-feather_centre.to(u.m).value,
        k=sigmoid_slope,
    )

    high_res_fft_weighted = high_res_data_fft * high_res_weights
    low_res_fft_weighted = low_res_data_fft_corr * low_res_weights

    feathered_fft = high_res_fft_weighted + low_res_fft_weighted
    feathered_fft /= high_res_weights + low_res_weights

    feathered_image = np.real(fft.ifftn(fft.ifftshift(feathered_fft))) * u.Jy / u.sr
    feathered_image.to(
        u.Jy / u.beam, equivalencies=u.beam_angular_area(high_res_beam.sr)
    )

    return FeatheredData(
        feathered_image=feathered_image,
        feathered_fft=feathered_fft,
        low_res_fft_weighted=low_res_fft_weighted,
        high_res_fft_weighted=high_res_fft_weighted,
        low_res_weights=low_res_weights,
        high_res_weights=high_res_weights,
    )


def reproject_low_res(
    low_res_data: u.Quantity,
    low_res_wcs: WCS,
    high_res_wcs: WCS,
) -> u.Quantity:
    """Reproject the low resolution data to the high resolution WCS

    Args:
        low_res_data (u.Quantity): Low resolution data
        low_res_wcs (WCS): Low resolution WCS
        high_res_wcs (WCS): High resolution WCS

    Returns:
        u.Quantity: Reprojected low resolution data
    """

    low_res_data_rp, _ = rp.reproject_adaptive(
        (low_res_data, low_res_wcs), high_res_wcs
    )

    return low_res_data_rp * low_res_data.unit


def get_data_from_fits(
    file_path: Path,
    unit: u.Unit | None = None,
    ext: int = 0,
) -> FitsData:
    """Get data from a FITS file

    Args:
        file_path (Path): Path to the FITS file
        unit (u.Unit | None, optional): Units of the data. Defaults to None.
        ext (int, optional): FITS extension to use. Defaults to 0.

    Raises:
        ShapeError: If the data is not 2D
        UnitError: If no unit is provided and no BUNIT keyword is found in the header

    Returns:
        FitsData: data, beam, wcs
    """
    with fits.open(file_path) as hdul:
        hdu = hdul[ext]
        data = hdu.data.squeeze()
        header = hdu.header

    if data.ndim != 2:
        msg = f"Data must be 2D (got {data.ndim}D)"
        raise ShapeError(msg)

    wcs = WCS(header).celestial
    beam = Beam.from_fits_header(header)

    if unit is None:
        try:
            bunit = header["BUNIT"]
        except KeyError as e:
            msg = "No unit provided and no BUNIT keyword found in header"
            raise UnitError(msg) from e
        unit = u.Unit(bunit)

    data = data * unit
    return FitsData(data=data, beam=beam, wcs=wcs)


def kelvin_to_jansky_per_beam(
    data_kelvin: u.Quantity, beam: Beam, frequency: u.Quantity
) -> u.Quantity:
    """Convert data from Kelvin to Jansky per beam

    Args:
        data_kelvin (u.Quantity): Data in Kelvin
        beam (Beam): Beam
        frequency (u.Quantity): Frequency of the data

    Returns:
        u.Quantity: Data in Jansky per beam
    """

    return data_kelvin.to(u.Jy / u.beam, equivalencies=beam.jtok_equiv(frequency))


def jansky_per_beam_to_jansky_per_sr(data_jy: u.Quantity, beam: Beam) -> u.Quantity:
    """Convert data from Jansky per beam to Jansky per steradian

    Args:
        data_jy (u.Quantity): Data in Jansky per beam
        beam (Beam): Beam

    Returns:
        u.Quantity: Data in Jansky per steradian
    """
    return data_jy.to(u.Jy / u.sr, equivalencies=u.beam_angular_area(beam.sr))


def jansky_per_sr_to_jansky_per_beam(data_jy_sr: u.Quantity, beam: Beam) -> u.Quantity:
    """Convert data from Jansky per steradian to Jansky per beam

    Args:
        data_jy_sr (u.Quantity): Data in Jansky per steradian
        beam (Beam): Beam

    Returns:
        u.Quantity: Data in Jansky per beam
    """
    return data_jy_sr.to(u.Jy / u.beam, equivalencies=u.beam_angular_area(beam.sr))


def plot_feather(
    low_res_vis: ArrayLike,
    high_res_vis: ArrayLike,
    uv_distance_2d: u.Quantity,
    high_res_weighted: ArrayLike,
    low_res_weighted: ArrayLike,
    feathered_vis: ArrayLike,
    feather_centre: u.Quantity,
    feather_sigma: u.Quantity,
    n_uv_bins: int = 10,
) -> Figure:
    """Plot the feathering

    Args:
        low_res_vis (ArrayLike): Low resolution visibilities
        high_res_vis (ArrayLike): High resolution visibilities
        uv_distance_2d (u.Quantity): 2D array of uv distances
        high_res_weighted (ArrayLike): Weighted high resolution visibilities
        low_res_weighted (ArrayLike): Weighted low resolution visibilities
        feathered_vis (ArrayLike): Feathered visibilities
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

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 18))
    ax1, ax2, ax3 = axs
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

    one_std = -1 * np.log((1 - stats.norm.cdf(1)) * 2)
    sigmoid_slope = one_std / feather_sigma.to(u.m).value

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


def write_feathered_fits(
    output_file: Path,
    feathered_data: u.Quantity,
    wcs: WCS,
    beam: Beam,
    overwrite: bool = False,
) -> None:
    """Write feathered data to a FITS file

    Args:
        output_file (Path): Path to the output FITS file
        feathered_data (u.Quantity): Feathered data (must be in Jy/sr)
        wcs (WCS): WCS of the data
        beam (Beam): Beam of the data
        overwrite (bool, optional): Whether to overwrite. Defaults to False.
    """
    header = wcs.to_header()
    header = beam.attach_to_header(header)
    data_jy = jansky_per_sr_to_jansky_per_beam(data_jy_sr=feathered_data, beam=beam)
    header["BUNIT"] = data_jy.unit.to_string(format="fits")
    fits.writeto(output_file, data_jy.value, header, overwrite=overwrite)


def feather_from_fits(
    low_res_file: Path,
    high_res_file: Path,
    output_file: Path,
    feather_centre: u.Quantity,
    feather_sigma: u.Quantity,
    frequency: u.Quantity,
    outer_uv_cut: u.Quantity | None = None,
    low_res_unit: u.Unit | None = None,
    high_res_unit: u.Unit | None = None,
    do_feather_plot: bool = False,
    overwrite: bool = False,
) -> None:
    """Feather two FITS files

    Args:
        low_res_file (Path): Path to the low resolution FITS file
        high_res_file (Path): Path to the high resolution FITS file
        output_file (Path): Path to the output feathered FITS file
        feather_centre (u.Quantity): Overall UV centre of the feathering function
        feather_sigma (u.Quantity): Width of the feathering function
        frequency (u.Quantity): Frequency of the data
        outer_uv_cut (u.Quantity | None, optional): UV cut to apply to low res data. Defaults to None.
        low_res_unit (u.Unit | None, optional): Units of low resolution data. Defaults to None.
        high_res_unit (u.Unit | None, optional): Units of high resolution data. Defaults to None.
        do_feather_plot (bool, optional): Make feather plots. Defaults to False.
        overwrite (bool, optional): Overwrite output data. Defaults to False.

    Raises:
        FileExistsError: If output file exists and overwrite is False
        ValueError: If outer_uv_cut is negative
        ValueError: If outer_uv_cut is less than feather_centre
        UnitError: If low_res_data is not in (or convertible to) Jy/beam
        UnitError: If high_res_data is not in (or convertible to) Jy/beam
    """

    if output_file.exists() and not overwrite:
        msg = f"Output file {output_file} already exists and overwrite is False"
        raise FileExistsError(msg)

    if outer_uv_cut is not None:
        if outer_uv_cut < 0:
            msg = f"outer_uv_cut must be positive (got {outer_uv_cut})"
            raise ValueError(msg)
        if outer_uv_cut <= feather_centre:
            msg = f"outer_uv_cut ({outer_uv_cut}) must be greater than feather_centre ({feather_centre})"
            raise ValueError(msg)

    low_res_data, low_res_beam, low_res_wcs = get_data_from_fits(
        file_path=low_res_file,
        unit=low_res_unit,
    )

    if low_res_data.unit == u.K:
        low_res_data = kelvin_to_jansky_per_beam(
            data_kelvin=low_res_data, beam=low_res_beam, frequency=frequency
        )

    if low_res_data.unit != u.Jy / u.beam:
        msg = f"low_res_data must be in Jy/beam (got {low_res_data.unit})"
        raise UnitError(msg)

    low_res_data = jansky_per_beam_to_jansky_per_sr(
        data_jy=low_res_data, beam=low_res_beam
    )

    high_res_data, high_res_beam, high_res_wcs = get_data_from_fits(
        file_path=high_res_file,
        unit=high_res_unit,
    )

    if high_res_unit != u.Jy / u.beam:
        msg = f"high_res_data must be in Jy/beam (got {high_res_data.unit})"
        raise UnitError(msg)

    high_res_data = jansky_per_beam_to_jansky_per_sr(
        data_jy=high_res_data, beam=high_res_beam
    )

    low_res_data_rp = reproject_low_res(
        low_res_data=low_res_data,
        low_res_wcs=low_res_wcs,
        high_res_wcs=high_res_wcs,
    )

    visibilities = fft_data(
        low_res_data=low_res_data_rp,
        high_res_data=high_res_data,
        low_res_beam=low_res_beam,
        high_res_beam=high_res_beam,
        wcs=high_res_wcs,
        wavelength=frequency.to(u.m, equivalencies=u.spectral()),
        outer_uv_cut=outer_uv_cut,
    )

    feathered_data = feather(
        low_res_data_fft_corr=visibilities.low_res_data_fft_corr,
        high_res_data_fft=visibilities.high_res_data_fft,
        high_res_beam=high_res_beam,
        uv_distance_2d=visibilities.uv_distance_2d,
        feather_centre=feather_centre,
        feather_sigma=feather_sigma,
    )

    if do_feather_plot:
        fig = plot_feather(
            low_res_vis=visibilities.low_res_data_fft_corr,
            high_res_vis=visibilities.high_res_data_fft,
            uv_distance_2d=visibilities.uv_distance_2d,
            high_res_weighted=feathered_data.high_res_fft_weighted,
            low_res_weighted=feathered_data.low_res_fft_weighted,
            feathered_vis=feathered_data.feathered_fft,
            feather_centre=feather_centre,
            feather_sigma=feather_sigma,
        )
        output_figure = output_file.with_suffix(".png")
        fig.savefig(output_figure, bbox_inches="tight", dpi=150)

    write_feathered_fits(
        output_file=output_file,
        feathered_data=feathered_data.feathered_image,
        wcs=high_res_wcs,
        beam=high_res_beam,
        overwrite=overwrite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feather two FITS files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("low_res_file", type=Path, help="Low resolution FITS file")
    parser.add_argument("high_res_file", type=Path, help="High resolution FITS file")
    parser.add_argument("output_file", type=Path, help="Output feathered FITS file")
    parser.add_argument("frequency", type=float, help="Frequency of the data in Hz")
    parser.add_argument(
        "--feather-centre",
        type=float,
        default=0,
        help="UV centre of the feathering function in meters",
    )
    parser.add_argument(
        "--feather-sigma",
        type=float,
        default=1,
        help="UV width of the feathering function in meters",
    )
    parser.add_argument(
        "--outer-uv-cut",
        type=float,
        help="Outer UV cut in meters",
        default=None,
    )
    parser.add_argument(
        "--low-res-unit",
        type=str,
        help="Unit of the low resolution data. Will try to read from BUNIT if not provided",
        default=None,
    )
    parser.add_argument(
        "--high-res-unit",
        type=str,
        help="Unit of the high resolution data. Will try to read from BUNIT if not provided",
        default=None,
    )
    parser.add_argument(
        "--do-feather-plot",
        action="store_true",
        help="Make a plot of the feathering",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists",
    )

    args = parser.parse_args()

    feather_from_fits(
        low_res_file=args.low_res_file,
        high_res_file=args.high_res_file,
        output_file=args.output_file,
        feather_centre=args.feather_centre * u.m,
        feather_sigma=args.feather_sigma * u.m,
        frequency=args.frequency * u.Hz,
        outer_uv_cut=args.outer_uv_cut * u.m if args.outer_uv_cut is not None else None,
        low_res_unit=u.Unit(args.low_res_unit)
        if args.low_res_unit is not None
        else None,
        high_res_unit=u.Unit(args.high_res_unit)
        if args.high_res_unit is not None
        else None,
        do_feather_plot=args.do_feather_plot,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
