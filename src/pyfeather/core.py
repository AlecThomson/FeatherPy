from __future__ import annotations

from typing import NamedTuple

import astropy.units as u
import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from radio_beam import Beam
from scipy import fft

from pyfeather.exceptions import ShapeError, UnitError


class Visibilities(NamedTuple):
    """ Data in the Fourier domain. """
    low_res_data_fft_corr: np.ndarray
    """ The low resolution data, deconvolved with the low resolution beam and reconvolved with the high resolution beam. """
    high_res_data_fft: np.ndarray
    """ The high resolution data. """
    uv_distance_2d: u.Quantity
    """ The 2D array of uv distances. """

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

def fft_data(
    low_res_data: u.Quantity,
    high_res_data: u.Quantity,
    low_res_beam: Beam,
    high_res_beam: Beam,
    wcs: WCS,
    outer_uv_cut: u.Quantity | None = None,
) -> Visibilities:
    
    # Sanity checks
    if low_res_data.unit != u.Jy / u.beam:
        msg = "low_res_data  must be in Jy/beam"
        raise UnitError(msg)
    if high_res_data.unit != u.Jy / u.beam:
        msg = "high_res_data must be in Jy/beam"
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

    # Construct beam FFTs
    pix_scales = proj_plane_pixel_scales(wcs.celestial)
    assert pix_scales[0] == pix_scales[1]
    pix_scale = pix_scales * u.deg
    # Construct the beams
    low_res_beam_image = low_res_beam.as_kernel(
        pix_scale, x_size=low_res_data.shape[1], y_size=low_res_data.shape[0]
    ).array
    high_res_beam_image = high_res_beam.as_kernel(
        pix_scale, x_size=low_res_data.shape[1], y_size=low_res_data.shape[0]
    ).array
    low_res_beam_image /= low_res_beam_image.sum()
    high_res_beam_image /= high_res_beam_image.sum()
    low_res_beam_fft = fft.fftshift(fft.fft2(low_res_beam_image))
    high_res_beam_fft = fft.fftshift(fft.fft2(high_res_beam_image))

    # FFT the data
    low_res_data_fft = fft.fftshift(fft.fftn(low_res_data))
    high_res_data_fft = fft.fftshift(fft.fftn(high_res_data))
    v_size, u_size = low_res_data_fft.shape
    u_array = fft.fftshift(fft.fftfreq(u_size, d=pix_scale.to(u.radian).value))
    v_array = fft.fftshift(fft.fftfreq(v_size, d=pix_scale.to(u.radian).value))
    u_2d_array, v_2d_array = np.meshgrid(u_array, v_array)
    uv_distance_2d = np.hypot(u_2d_array, v_2d_array) * u.m


    # Deconvolve the low resolution beam
    low_res_data_fft_corr = low_res_data_fft / np.abs(low_res_beam_fft)
    # Reconvolve with the high resolution beam
    low_res_data_fft_corr *= np.abs(high_res_beam_fft)
    low_res_data_fft_corr[~np.isfinite(low_res_data_fft_corr)] = 0

    if outer_uv_cut is not None:
        low_res_data_fft_corr[uv_distance_2d > outer_uv_cut.to(u.m).value] = 0

    return Visibilities(low_res_data_fft_corr, high_res_data_fft, uv_distance_2d)


def feather(
    low_res_data_fft_corr: np.ndarray,
    high_res_data_fft: np.ndarray,
    high_res_beam: Beam,
    uv_distance_2d: u.Quantity,
    feather_centre: u.Quantity,
    feather_sigma: u.Quantity,
) -> u.Quantity:
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
                        
    high_res_weights = sigmoid(
        uv_distance_2d, 
        feather_centre.to(u.m), 
        feather_sigma.to(u.m)
    )
    low_res_weights = sigmoid(
        -uv_distance_2d, feather_centre.to(u.m), feather_sigma.to(u.m)
    )

    high_res_fft_weighted = high_res_data_fft * high_res_weights
    low_res_fft_weighted = low_res_data_fft_corr * low_res_weights

    feathered_fft = high_res_fft_weighted + low_res_fft_weighted
    feathered_fft /= (high_res_fft_weighted + low_res_fft_weighted)

    feathered_data = fft.ifftn(fft.ifftshift(feathered_fft)).real * u.Jy / u.sr
    feathered_data.to(u.Jy / u.beam, equivalencies=u.beam_angular_area(high_res_beam.sr))

    return feathered_data


def main():
    ...

if __name__ == "__main__":
    main()