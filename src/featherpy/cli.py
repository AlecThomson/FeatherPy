#!/usr/bin/env python3
"""Command line interface for featherpy."""

from __future__ import annotations

import argparse
from pathlib import Path

import astropy.units as u

from featherpy.core import feather_from_fits


def main() -> None:
    """Command line interface for featherpy."""
    parser = argparse.ArgumentParser(
        description="Feather two FITS files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("low_res_file", type=Path, help="Low resolution FITS file")
    parser.add_argument("high_res_file", type=Path, help="High resolution FITS file")
    parser.add_argument("output_file", type=Path, help="Output feathered FITS file")
    parser.add_argument("frequency", type=float, help="Frequency of the data in Hz")
    parser.add_argument(
        "-c",
        "--feather-centre",
        type=float,
        default=0,
        help="UV centre of the feathering function in meters",
    )
    parser.add_argument(
        "-s",
        "--feather-sigma",
        type=float,
        default=1,
        help="UV width of the feathering function in meters",
    )
    parser.add_argument(
        "-u",
        "--outer-uv-cut",
        type=float,
        help="Outer UV cut in meters",
        default=None,
    )
    parser.add_argument(
        "-lu",
        "--low-res-unit",
        type=str,
        help="Unit of the low resolution data. Will try to read from BUNIT if not provided",
        default=None,
    )
    parser.add_argument(
        "-hu",
        "--high-res-unit",
        type=str,
        help="Unit of the high resolution data. Will try to read from BUNIT if not provided",
        default=None,
    )
    parser.add_argument(
        "-p",
        "--do-feather-plot",
        action="store_true",
        help="Make a plot of the feathering",
    )
    parser.add_argument(
        "-o",
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
