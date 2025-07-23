import argparse
import sys
from pathlib import Path

from open_dive.viz import plot_nifti


def main():
    # Create args
    parser = argparse.ArgumentParser(
        description="OpenDIVE (Open Diffusion Imaging Visualization for Everyone) is a command line interface tool for generating accessible, interpretable visualizations from diffusion MRI.",
        epilog="If you use this tool in research, please cite the following: "
        "Saunders, A.M., McMaster, E.M., Rorden, C., Jis, J.K., Sun, M., Sadriddinov, A., VanTilburg, L., Kim, M.E., Landman, B.A., and Schilling, K. "
        "OpenDIVE: Streamlining Tractography Visualization. Medical Imaging with Deep Learning - Short Papers, 2025. (https://openreview.net/pdf?id=EjqP4vHnHL)",
    )

    scalar_group = parser.add_argument_group("Scalar options")
    tractography_group = parser.add_argument_group("Tractography options")
    glyph_group = parser.add_argument_group(
        "Diffusion glyph options (tensors and ODFs)"
    )
    window_group = parser.add_argument_group("Window options")

    scalar_group.add_argument(
        "-n",
        "--nifti_path",
        type=Path,
        help="Path to scalar-valued NIFTI to plot. Can be 3D or 4D. If 4D, --volume_idx must be provided.",
    )
    scalar_group.add_argument(
        "-s",
        "--slice",
        default="m",
        help='Slice index (integer) or "m" for middle slice. Default is "m".',
    )
    scalar_group.add_argument(
        "-o",
        "--orientation",
        default="axial",
        help='Orientation to plot for --nifti_path. Can be "axial", "sagittal" or "coronal". Default is "axial".',
    )
    scalar_group.add_argument(
        "--value_range",
        type=int,
        nargs=2,
        help="Value range to pass to slicer. Default is (min, max) of --nifti_path.",
    )
    scalar_group.add_argument(
        "--volume_idx",
        type=int,
        help="4D index of --nifti_path to display. Must be provided if the image is 4D.",
    )
    scalar_group.add_argument(
        "--interpolation",
        default="nearest",
        help='Interpolation method to use. Can be "nearest" or "linear". Default is "nearest".',
    )
    scalar_group.add_argument(
        "--scalar_colorbar",
        action="store_true",
        help="Whether to show a colorbar. Default is False",
    )
    scalar_group.add_argument(
        "--glass_brain",
        type=Path,
        help="Path to binary mask to generate a glass brain.",
    )

    tractography_group.add_argument(
        ## plot tractogram with slices
        "--tractography_path",
        type=Path,
        nargs="+",  # Accept one or more arguments
        help="List of tractogram(s) to plot, in .trk or .tck format.",
    )
    tractography_group.add_argument(
        "--tractography_values",
        type=float,
        nargs="+",
        help="Values to use for coloring each tractogram (must match number of tractography files)",
    )
    tractography_group.add_argument(
        "--tractography_cmap",
        help='Matplotlib or cmcrameri colormap to use for tractography. Default is "plasma" if --tractography_values is provided, otherwise "Set1".',
    )
    tractography_group.add_argument(
        "--tractography_cmap_range",
        type=float,
        nargs=2,
        help="Range to use for the colormap. Default is (0, 1).",
    )
    tractography_group.add_argument(
        "--tractography_opacity",
        type=float,
        nargs="+",
        default=0.6,
        help="Value to use for the tractogram opacity in range (0, 1].If a list, each value corresponds to a tractogram in --tractography_path.  Default is 0.6.",
    )
    tractography_group.add_argument(
        "--use_orientation_rgb",
        action = "store_true",
        help="Use RGB coloring based on local streamline orientation.",
    )
    tractography_group.add_argument(
        "--tractography_colorbar",
        action="store_true",
        help="Whether to show a tractography values colorbar. Default is False.",
    )

    glyph_group.add_argument(
        "--tensor_path",
        type=Path,
        help="Path to tensor image, format is Dxx, Dxy, Dyy, Dxz, Dyz, Dzz.",
    )
    glyph_group.add_argument(
        "--odf_path",
        type=Path,
        help="Path to orientation distribution function image represented as spherical harmonics.",
    )
    glyph_group.add_argument(
        "--sh_basis",
        default="descoteaux07",
        help='Spherical harmonic basis for --odf_path, either "descoteaux07" or "tournier07". Default is "descoteaux07".',
    )
    glyph_group.add_argument(
        "--scale",
        type=float,
        default=1,
        help="Scale of the tensor glyphs or ODF glyphs. Determines offset from --nifti_path. Default is 1.",
    )

    window_group.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=(600, 400),
        help="Size of window. Default is (600, 400).",
    )
    window_group.add_argument("--save_path", help="Optional path for saving the image.")
    window_group.add_argument(
        "--headless",
        action="store_true",
        help="If provided, do not show the scene in an interactive window.",
    )
    window_group.add_argument(
        "--azimuth",
        "--az",
        type=float,
        default=None,
        help="Azimuthal angle of the view.",
    )
    window_group.add_argument(
        "--elevation",
        "--el",
        type=float,
        default=None,
        help="Elevation angle of the view.",
    )
    window_group.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Zoom factor; Values >1 zoom in, <1 zoom out. Default zoom value is 1.0. Must be positive."
    )

    args = parser.parse_args()

    # If negative or 0 zoom provided, give feedback
    if args.zoom<=0:
        parser.error("--zoom must be a positive number. Values > 1 zoom in, values < 1 zoom out.")

    # If provided nothing, give help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Plot the NIFTI
    plot_nifti(
        nifti_path=args.nifti_path,
        data_slice=args.slice,
        orientation=args.orientation,
        size=args.size,
        volume_idx=args.volume_idx,
        save_path=args.save_path,
        headless=args.headless,
        value_range=args.value_range,
        interpolation=args.interpolation,
        scalar_colorbar=args.scalar_colorbar,
        tractography_path=args.tractography_path,
        tractography_opacity=args.tractography_opacity,
        use_orientation_rgb=args.use_orientation_rgb,
        tractography_values=args.tractography_values,
        tractography_cmap=args.tractography_cmap,
        tractography_cmap_range=args.tractography_cmap_range,
        tractography_colorbar=args.tractography_colorbar,
        tensor_path=args.tensor_path,
        odf_path=args.odf_path,
        sh_basis=args.sh_basis,
        scale=args.scale,
        azimuth=args.azimuth,
        elevation=args.elevation,
        zoom=args.zoom,
        glass_brain_path=args.glass_brain,
    )
