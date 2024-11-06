from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import astropy.units as u
import numpy as np
import reproject as rp
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from radio_beam import Beam
from scipy import fft, stats
import matplotlib.pyplot as plt

from pyfeather.exceptions import ShapeError, UnitError


class Visibilities(NamedTuple):
    """ Data in the Fourier domain. """
    low_res_data_fft_corr: np.ndarray
    """ The low resolution data, deconvolved with the low resolution beam and reconvolved with the high resolution beam. """
    high_res_data_fft: np.ndarray
    """ The high resolution data. """
    uv_distance_2d: u.Quantity
    """ The 2D array of uv distances. """

class FitsData(NamedTuple):
    """ Data from a FITS file. """
    data: u.Quantity
    """ The data. """
    beam: Beam
    """ The beam. """
    wcs: WCS
    """ The WCS. """

class FeatheredData(NamedTuple):
    """ Feathered data. """
    feathered_image: u.Quantity
    """ The feathered image. """
    feathered_fft: np.ndarray
    """ The feathered data in the Fourier domain. """
    low_res_fft_weighted: np.ndarray
    """ The low resolution data in the Fourier domain, weighted for feathering. """
    high_res_fft_weighted: np.ndarray
    """ The high resolution data in the Fourier domain, weighted for feathering. """
    low_res_weights: np.ndarray
    """ The weights for the low resolution data. """
    high_res_weights: np.ndarray
    """ The weights for the high resolution data. """
    

def sigmoid(x: np.ndarray, x0: float = 0, k: float = 1) -> np.ndarray:
    """Sigmoid function

    Args:
        x (np.ndarray): x values
        x0 (float, optional): x offset. Defaults to 0.
        k (float, optional): growth rate. Defaults to 1.

    Returns:
        np.ndarray: sigmoid values
    """
    return 1 / (1 + np.exp(-k * (x - x0)))

def make_beam_fft(
        beam: Beam,
        data_shape: tuple[int, int],
        pix_scale: u.Quantity,
) -> np.ndarray:
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
    
    # Sanity checks
    if low_res_data.unit != u.Jy / u.sr:
        msg = "low_res_data  must be in Jy/sr"
        raise UnitError(msg)
    if high_res_data.unit != u.Jy / u.sr:
        msg = "high_res_data must be in Jy/sr"
        raise UnitError(msg)

    if low_res_data.shape != high_res_data.shape:
        msg = f"Data shapes do not match ({low_res_data.shape=}, {high_res_data.shape=})"
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
        low_res_data_fft_corr[uv_distance_2d > outer_uv_cut.to(u.m)] = 0

    return Visibilities(low_res_data_fft_corr, high_res_data_fft, uv_distance_2d)


def feather(
    low_res_data_fft_corr: np.ndarray,
    high_res_data_fft: np.ndarray,
    high_res_beam: Beam,
    uv_distance_2d: u.Quantity,
    feather_centre: u.Quantity,
    feather_sigma: u.Quantity,
    low_res_scale_factor: float | None = None,
) -> FeatheredData:
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
        uv_start = (feather_centre + -5 * feather_sigma).to(u.m)
        uv_end = (feather_centre + 5 * feather_sigma).to(u.m)
        overlap_index = (uv_distance_2d > uv_start) & (uv_distance_2d < uv_end)
        low_res_scale_factor = np.median(
            np.abs(high_res_data_fft[overlap_index])
            / np.abs(low_res_data_fft_corr[overlap_index])
        )
        low_res_scale_factor = np.round(low_res_scale_factor, 3)

    low_res_data_fft_corr *= low_res_scale_factor

    # Approximately convert 1 sigma to the slope of the sigmoid
    one_std = -1 * np.log((1 - stats.norm.cdf(1)) * 2)
    sigmoid_slope = one_std / feather_sigma.to(u.m).value

    high_res_weights = sigmoid(
        x=uv_distance_2d.to(u.m).value, x0=feather_centre.to(u.m).value, k=sigmoid_slope
    )
    low_res_weights = sigmoid(
        x=-uv_distance_2d.to(u.m).value, x0=-feather_centre.to(u.m).value, k=sigmoid_slope
    )

    high_res_fft_weighted = high_res_data_fft * high_res_weights
    low_res_fft_weighted = low_res_data_fft_corr * low_res_weights

    feathered_fft = high_res_fft_weighted + low_res_fft_weighted
    feathered_fft /= high_res_weights + low_res_weights

    feathered_image = fft.ifftn(fft.ifftshift(feathered_fft)).real * u.Jy / u.sr
    feathered_image.to(u.Jy / u.beam, equivalencies=u.beam_angular_area(high_res_beam.sr))

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

    low_res_data_rp, _ = rp.reproject_adaptive((low_res_data, low_res_wcs), high_res_wcs)

    return low_res_data_rp * low_res_data.unit

def get_data_from_fits(
    file_path: Path,
    unit: u.Unit | None = None,
    ext: int = 0,
) -> FitsData:

    with fits.open(file_path) as hdul:
        hdu = hdul[ext]
        data = hdu.data.squeeze()
        header = hdu.header
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
    return FitsData(
        data=data, 
        beam=beam, 
        wcs=wcs
    )

def kelvin_to_jansky_per_beam(
        data_kelvin: u.Quantity,
        beam: Beam, 
        frequency: u.Quantity
) -> u.Quantity:
    
    return data_kelvin.to(u.Jy / u.beam, equivalencies=beam.jtok_equiv(frequency))

def jansky_per_beam_to_jansky_per_sr(
        data_jy: u.Quantity,
        beam: Beam
) -> u.Quantity:
    
    return data_jy.to(u.Jy / u.sr, equivalencies=u.beam_angular_area(beam.sr))

def plot_feather(
        low_res_vis: np.ndarray,
        high_res_vis: np.ndarray,
        uv_distance_2d: u.Quantity,
        high_res_weighted: np.ndarray,
        low_res_weighted: np.ndarray,
        feathered_vis: np.ndarray,
        feather_centre: u.Quantity,
        feather_sigma: u.Quantity,
        n_uv_bins: int = 10,
) -> plt.Figure:
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
        alpha=sc_alpha
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
        alpha=sc_alpha
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
        label="high resolution"
    )
    ax2.plot(
        uvdists, 
        sigmoid(-uvdists, -feather_centre.to(u.m).value, sigmoid_slope),
        label="low resolution"
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
    ax3.set(xlim=(0,plot_bound), yscale="log", ylabel="Visibility Amplitude", xlabel="$uv$-distance / m")
    ax2.set(ylabel="Visibility weights")
    ax1.set(xlim=(0,plot_bound), yscale="log", ylabel="Visibility Amplitude")
    for ax in (ax1, ax2, ax3):
        ax.axvline((feather_centre + -5 * feather_sigma).to(u.m).value, color="black", linestyle="--")
        ax.axvline(
            (feather_centre + 5 * feather_sigma).to(u.m).value, 
            color="black", 
            linestyle="--",
            label="Feather $\pm5\sigma$"
        )
        ax.axvline(
            feather_centre.to(u.m).value, 
            color="black", 
            linestyle=":",
            label="Feather centre"
        )
        ax.legend()
    return fig

def write_feathered_fits(
    output_file: Path,
    feathered_data: u.Quantity,
    wcs: WCS,
    beam: Beam,
    overwrite: bool = False,
):
    header = wcs.to_header()
    header = beam.attach_to_header(header)
    fits.writeto(output_file, feathered_data.value, header, overwrite=overwrite)

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

    if output_file.exists() and not overwrite:
        msg = f"Output file {output_file} already exists and overwrite is False"
        raise FileExistsError(msg)
    
    low_res_data, low_res_beam, low_res_wcs = get_data_from_fits(
        file_path=low_res_file,
        unit=low_res_unit,
    )

    if low_res_data.unit == u.K:
        low_res_data = kelvin_to_jansky_per_beam(
            data_kelvin=low_res_data,
            beam=low_res_beam,
            frequency=frequency
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

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Feather two FITS files", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
        low_res_unit=u.Unit(args.low_res_unit) if args.low_res_unit is not None else None,
        high_res_unit=u.Unit(args.high_res_unit) if args.high_res_unit is not None else None,
        do_feather_plot=args.do_feather_plot,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()