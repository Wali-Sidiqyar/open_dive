import os

import cmcrameri
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import vtk
from dipy.align.reslice import reslice
from dipy.core.geometry import cart2sphere, sphere2cart
from dipy.data import get_sphere
from dipy.io.image import load_nifti
from dipy.reconst.dti import decompose_tensor, from_lower_triangular
from dipy.reconst.shm import calculate_max_order, sh_to_sf_matrix
from dipy.viz import actor, window
from fury.actor import (
    _color_fa,
    _fa,
    contour_from_roi,
    ellipsoid,
    odf_slicer,
    slicer,
)
from fury.lib import Actor
from fury.utils import apply_affine
from matplotlib.colors import Colormap
from scipy.ndimage import binary_dilation, gaussian_filter


def plot_nifti(
    nifti_path: os.PathLike | None = None,
    data_slice: str | int = "m",
    orientation: str = "axial",
    size: tuple[int, int] = (600, 400),
    azimuth: float | None = None,
    elevation: float | None = None,
    value_range: tuple[int, int] | None = None,
    save_path: os.PathLike | None = None,
    headless: bool = False,
    scalar_colorbar: bool = True,
    tractography_path: list[os.PathLike] | None = None,
    tractography_opacity: list[float] | None = [0.6],
    use_orientation_rgb: bool = False,
    tractography_values: list[float] | None = None,
    tractography_cmap: str | None = None,
    tractography_cmap_range: tuple[int, int] | None = None,
    tractography_colorbar: bool = False,
    volume_idx: int | None = None,
    tensor_path: os.PathLike | None = None,
    odf_path: os.PathLike | None = None,
    sh_basis: str = "descoteaux07",
    scale: int = 1,
    zoom: float = 1.0,
    glass_brain_path: os.PathLike | None = None,
    **kwargs,
) -> None:
    """Create a 2D rendering of a NIFTI slice.

    Parameters
    ----------
    nifti_path : os.PathLike, optional
        Path to NIFTI to plot
    data_slice : str or int, default "m"
        Slice to plot or "m" for middle slice
    orientation : str, default "axial"
        Can be "axial", "sagittal" or "coronal"
    size : tuple, default (600, 400)
        Size of window
    azimuth : float, optional
        Azimuth angle in degrees
    elevation : float, optional
        Elevation angle in degrees
    value_range: tuple of float, optional
        Range for color mapping of image, by default automatically selected
    save_path : os.PathLike, optional
        Optional path to save to
    headless : bool, default False
        If True, do not show an interactive window
    scalar_colorbar : bool, default True
        Whether to show a scalar colorbar (for FA, T1, etc.)
    tractography_path : list of os.Pathlike, optional
        Optional tractogram(s) to plot with slices. Can provide multiple files
    tractography_opacity : list of float, default 0.6
        Optional opacity value for tractograms between (0, 1]
    use_orientation_rgb: boolean, default false
        Use RGB colors based on local fiber orientation for tractography
    tractography_values : list of float, optional
        Optional values to color the tractography with
    tractography_cmap : str, optional
        Colormap for tractography coloring; ignored if orientation RGB is used.
    tractography_cmap_range : tuple of float, default (0, 1) if tractography_values is not None
        Optional range to use for the colormap
    tractography_colorbar : bool, default False
        Whether to show a colorbar for the tractography
    volume_idx : int, optional
        Index of the volume to display if the image is 4D
    tensor_path : os.PathLike, optional
        Path to a tensor image to visualize (format is Dxx, Dxy, Dyy, Dxz, Dyz, Dzz)
    odf_path : os.PathLike, optional
        Path to an ODF image to visualize
    sh_basis : str, default "descoteaux07"
        Basis type for the spherical harmonics, either "descoteaux07" or "tournier07"
    scale : float, default 1
        Scale of the tensor glyphs or ODF glyphs
    zoom : float, default 1
        Zoom factor for the camera (1.0 = default; <1.0 zooms in; >1.0 zooms out).
    glass_brain_path : os.PathLike, optional
        Optional glass brain mask to overlay

    **kwargs
        Additional keyword arguments to pass to fury.actor.slicer
    """
    # Set defaults
    if orientation == "axial":
        azimuth = 0 if azimuth is None else azimuth
        elevation = 0 if elevation is None else elevation
    elif orientation == "coronal":
        azimuth = 180 if azimuth is None else azimuth
        elevation = 90 if elevation is None else elevation
    elif orientation == "sagittal":
        azimuth = 90 if azimuth is None else azimuth
        elevation = 90 if elevation is None else elevation

    # Set slice to int if not "m"
    if data_slice != "m":
        data_slice = int(data_slice)

    # Color by local orientation RGB if neither tractography values nor colormap are provided
    use_orientation_rgb = tractography_values is None and tractography_cmap is None
    if not use_orientation_rgb:
        if tractography_cmap is None:
            tractography_cmap = "plasma"
    if tractography_cmap_range is None:
        tractography_cmap_range = (
            (0, 1) if tractography_values is None else (min(tractography_values), max(tractography_values))
        )
    tractography_cbar_labels = tractography_values is not None

    # Set up scene and bounds
    scene = window.Scene()

    # Variable to store size of things we are rendering
    scene_bound_data = None
    scene_bound_affine = None

    # If we have a NIFTI file, use it to get the bounds of the scene
    scene_bound_nifti_path = nifti_path or tensor_path or odf_path or glass_brain_path
    if scene_bound_nifti_path is not None:
        scene_bound_nifti = nib.load(scene_bound_nifti_path)
        data = scene_bound_nifti.get_fdata()
        scene_bound_data = np.ones_like(data)
        scene_bound_affine = scene_bound_nifti.affine

        # Get slice if not defined
        if orientation == "axial":
            data_slice = data.shape[2] // 2 if data_slice == "m" else data_slice
            extent = (0, data.shape[0], 0, data.shape[1], data_slice, data_slice)
            offset = np.array([0, 0, scale])

        elif orientation == "coronal":
            data_slice = data.shape[1] // 2 if data_slice == "m" else data_slice
            extent = (0, data.shape[0], data_slice, data_slice, 0, data.shape[2])
            offset = np.array([0, scale, 0])

        elif orientation == "sagittal":
            data_slice = data.shape[0] // 2 if data_slice == "m" else data_slice
            extent = (data_slice, data_slice, 0, data.shape[1], 0, data.shape[2])
            offset = np.array([scale, 0, 0])

    if nifti_path is not None:
        slice_actor = _create_nifti_actor(
            nifti_path,
            volume_idx=volume_idx,
            value_range=value_range,
            **kwargs,
        )
        scene.add(slice_actor)
        slice_actor.display_extent(*extent)

    if scalar_colorbar:
        scalar_bar = _create_colorbar_actor(
            value_range=value_range,
            colorbar_position=(0.8, 0.1),
            colorbar_height=0.5,
            colorbar_width=0.1,
        )
        scene.add(scalar_bar)

    # Add tractography
    if tractography_path is not None:

        # Decide coloring logic: either orientation RGB or colormap-based
        if not use_orientation_rgb:
            cmap = plt.get_cmap(tractography_cmap)

            # Set to range
            if tractography_values is not None:
                vmin, vmax = tractography_cmap_range if tractography_cmap_range else (min(tractography_values), max(tractography_values))
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                colors = [cmap(norm(val)) for val in tractography_values]

            else:
                colors = [cmap(i) for i in range(len(tractography_path))]


            # Apply colorbar
            if tractography_colorbar:
                tract_bar = _create_colorbar_actor(
                    value_range=tractography_cmap_range,
                    colorbar_position=(0.1, 0.1),
                    colorbar_height=0.5,
                    colorbar_width=0.1,
                    cmap=cmap,
                    labels=tractography_cbar_labels,
                )
                scene.add(tract_bar)
        else:
            # Orientation RGB uses no colormap or colorbar
            colors = None
        # Add each tractography with its corresponding color and opacity
        stream_actors= []
        if tractography_opacity is not None:
            # If a single float is provided, expand it to match number of tracts
            if isinstance(tractography_opacity, float):
                tractography_opacities = [tractography_opacity] * len(tractography_path)
            else:
                tractography_opacities = tractography_opacity

            if len(tractography_opacities) != len(tractography_path):
                raise ValueError("Length of tractography_opacity list must match number of tractography files.")
            for i, (path, opacity) in enumerate(zip(tractography_path, tractography_opacities)):
                    color = None if use_orientation_rgb else colors[i]
                    actor_list = _create_tractography_actor(
                        [path],
                        colors = [color],
                        tractography_opacity = opacity,
                        use_orientation_rgb=use_orientation_rgb,
                    )
                    stream_actors.extend(actor_list)
        else:
            stream_actors = _create_tractography_actor(
                tractography_path,
                colors = colors,
                tractography_opacity = 0.6,
                use_orientation_rgb=use_orientation_rgb,
            )
        for stream_actor in stream_actors:
            scene.add(stream_actor)

    # Add diffusion tensor visualization
    if tensor_path is not None:
        tensor_actor = _create_tensor_actor(
            tensor_path,
            extent=extent,
            offset=offset,
            scale=scale,
        )
        scene.add(tensor_actor)

    # Add orientation distribution function visualization
    if odf_path:
        odf_actor = _create_odf_actor(
            odf_path,
            offset=offset,
            scale=scale,
            sh_basis=sh_basis,
        )
        scene.add(odf_actor)
        odf_actor.display_extent(*extent)

    if glass_brain_path:
        glass_brain_actor = _create_glass_brain_actor(glass_brain_path)
        scene.add(glass_brain_actor)

        if scene_bound_data is None:
            glass_brain = nib.load(glass_brain_path)
            scene_bound_data = np.ones_like(glass_brain.get_fdata())
            scene_bound_affine = glass_brain.affine

    _set_camera(
        scene=scene,
        azimuth=azimuth,
        elevation=elevation,
        zoom=zoom,
        scene_bound_data=scene_bound_data,
        scene_bound_affine=scene_bound_affine,
    )

    # Show the scene
    if save_path:
        window.record(scene=scene, out_path=save_path, size=size, reset_camera=False)

    if not headless:
        window.show(scene, size=size, reset_camera=False)


def _create_glass_brain_actor(
    mask_nifti: os.PathLike,
    resample_factor: int = 2,
    smooth_sigma: float = 2,
    dilation_iters: int = 2,
    opacity: float = 0.33,
) -> Actor:
    """Create a "glass brain" visualization from a binary mask.

    Parameters
    ----------
    mask_nifti : os.PathLike
        Path to binary mask NIFTI image
    resample_factor : int, default 3
        Factor to upsample the mask by
    smooth_sigma : float, default 2
        Standard deviation for Gaussian smoothing
    dilation_iters : int, default 2
        Number of iterations for binary dilation
    opacity : float, default 0.25
        Opacity of the glass brain

    Returns
    -------
    glass_brain : fury.actor.surface
        Glass brain surface actor
    """
    # Load the mask
    mask_nifti = nib.load(mask_nifti)
    mask = mask_nifti.get_fdata()
    affine = mask_nifti.affine
    zooms = mask_nifti.header.get_zooms()[:3]

    # Step 1: Upsample (regrid) the mask by a factor of 5
    new_zooms = tuple(z / resample_factor for z in zooms)
    mask_up, new_affine = reslice(mask, affine, zooms, new_zooms)

    # Step 2: Apply Gaussian smoothing with standard deviation 2
    mask_smooth = gaussian_filter(mask_up, sigma=smooth_sigma)

    # Step 3: Threshold the smoothed mask at 0.5
    mask_thres = (mask_smooth > 0.5).astype(np.uint8)

    # Step 4: Dilate the thresholded mask with 2 passes
    mask_dilated = binary_dilation(mask_thres, iterations=dilation_iters).astype(np.uint8)

    # Create a surface actor
    glass_brain_actor = contour_from_roi(mask_dilated, affine=new_affine, opacity=opacity, color=(0.5, 0.5, 0.5))
    return glass_brain_actor


def _create_nifti_actor(
    nifti_path: os.PathLike,
    volume_idx: int | None = None,
    value_range: tuple[int, int] | None = None,
    **kwargs,
) -> Actor:
    # Load the data and convert to RAS
    nifti = nib.load(nifti_path)
    nifti = nib.as_closest_canonical(nifti)
    data = nifti.get_fdata()

    if len(data.shape) == 4:
        if volume_idx is None:
            raise ValueError(
                "Input is a 4D image but no volume index specified. "
                "Please provide a volume_idx parameter to select which 3D volume to display."
            )
        if not 0 <= volume_idx < data.shape[3]:
            raise ValueError(f"volume_idx {volume_idx} is out of bounds for image with {data.shape[3]} volumes")
        data = data[..., volume_idx]
    elif len(data.shape) != 3:
        raise ValueError(f"Expected 3D or 4D image, but got image with {len(data.shape)} dimensions")

    # Get the data and affine
    affine = nifti.affine

    # value range
    if value_range is None:
        value_range = [np.min(data), np.max(data)]

    # Set up slicer and window
    slice_actor = slicer(data, value_range=value_range, affine=affine, **kwargs)
    return slice_actor


def _create_colorbar_actor(
    value_range: tuple[int, int] | None = None,
    colorbar_position: tuple[float, float] = (0.8, 0.1),
    colorbar_height: float = 0.5,
    colorbar_width: float = 0.1,
    cmap: Colormap | None = None,
    labels: bool = True,
) -> vtk.vtkScalarBarActor:
    """Create a colorbar actor for the scene."""

    # Create a grayscale colormap (from black to white)
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)  # Full grayscale (256 levels)
    lut.Build()  # Initialize the LUT

    if cmap is None:
        grayscale_cmap = plt.get_cmap("gray")
        for i in range(256):
            # Grayscale colors
            color = grayscale_cmap(i / 255.0)
            lut.SetTableValue(i, color[0], color[1], color[2], 1)
    else:
        for i in range(256):
            # Sample the colormap
            color = cmap(i / 255.0)
            lut.SetTableValue(i, color[0], color[1], color[2], 1)

    # Set the full grayscale range (e.g., 0 to 255 for typical image data)
    if value_range is None:
        lut.SetRange(0, 1)
    else:
        lut.SetRange(value_range[0], value_range[1])

    # Create the scalar bar (colorbar)
    colorbar = vtk.vtkScalarBarActor()
    colorbar.SetLookupTable(lut)  # Attach the grayscale LUT
    colorbar.SetPosition(*colorbar_position)  # Position of the colorbar
    colorbar.SetHeight(colorbar_height)  # Adjust height (increase size)
    colorbar.SetWidth(colorbar_width)  # Adjust width (increase size)

    if not labels:
        colorbar.SetLabelFormat("")

    return colorbar


def _create_tractography_actor(
    tractography_path: list[os.PathLike],
    colors: list[tuple[float, float, float]],
    tractography_opacity:list[float] = [0.6],
    use_orientation_rgb: bool = False,
) -> list[Actor]:
    """
    Create tractography actors from a list of NIFTI files.
    If `use_orientation_rgb` is True, colors are computed locally per point based on fiber orientation.
    Otherwise, use provided colors list for whole tracts.
    """

    # Loop over each tractography file and create an actor unless orientation RGB
    stream_actors = []
    if isinstance(tractography_opacity, float):
        tractography_opacity = [tractography_opacity]
    if len(tractography_opacity) == 1:
        tractography_opacity = tractography_opacity * len(tractography_path)
    for i, tract_file in enumerate(tractography_path):
        streamlines_nifti = nib.streamlines.load(tract_file)
        streamlines = streamlines_nifti.streamlines
        if use_orientation_rgb:
            local_colors = []
            for streamline in streamlines:
                directions = np.diff(streamline, axis=0)
                norms = np.linalg.norm(directions, axis=1, keepdims=True)
                norms[norms == 0] =1
                unit_directions = directions / norms
                rgb_colors = (unit_directions + 1) / 2
                last_color = rgb_colors[-1]
                rgb_colors = np.vstack([rgb_colors, last_color])
                local_colors.extend(rgb_colors)
            stream_actor = actor.line(streamlines, colors=local_colors, linewidth=0.2, opacity=tractography_opacity[i])
        else:
            color = colors[i]
            stream_actor = actor.line(streamlines, colors=color, linewidth=0.2, opacity=tractography_opacity[i])

        stream_actors.append(stream_actor)
    return stream_actors


def _create_tensor_actor(
    tensor_path: os.PathLike,
    extent: tuple[int, int, int, int, int, int],
    offset: np.ndarray,
    scale: float = 1,
) -> Actor:
    """Create a tensor actor from a NIFTI file."""

    # Load the tensor data and affine
    tensor_data, tensor_affine = load_nifti(tensor_path)
    tensor_matrix = from_lower_triangular(tensor_data)
    eigvals, eigvecs = decompose_tensor(tensor_matrix)
    mask = np.ones(tensor_data.shape[:3])
    centers = np.ascontiguousarray(np.array(np.nonzero(mask)).T)

    # Reshape to N x 3
    centers = centers.reshape(-1, 3)
    eigvals = eigvals.reshape(-1, 3)
    eigvecs = eigvecs.reshape(-1, 3, 3)

    # Only keep centers in extent
    indices = (
        (centers[:, 0] >= extent[0])
        & (centers[:, 0] <= extent[1])
        & (centers[:, 1] >= extent[2])
        & (centers[:, 1] <= extent[3])
        & (centers[:, 2] >= extent[4])
        & (centers[:, 2] <= extent[5])
    )

    centers = centers[indices, :]
    eigvals = eigvals[indices, :]
    eigvecs = eigvecs[indices, :, :]

    # Apply affine to centers
    centers = apply_affine(tensor_affine, centers)

    # Get colors
    colors = _color_fa(_fa(eigvals), eigvecs)

    # Due to a bug in fury's tensor actor, we need to draw the ellipsoids ourselves.
    tensor_actor = ellipsoid(
        centers=centers,
        axes=eigvecs,
        lengths=eigvals,
        scales=scale * 2,
        colors=colors,
    )

    # Set offset so it doesn't clash with slice
    position = tensor_actor.GetPosition()
    position += offset
    tensor_actor.SetPosition(position)

    return tensor_actor


def _create_odf_actor(
    odf_path: os.PathLike,
    offset: np.ndarray,
    scale: float = 1,
    sh_basis: str = "descoteaux07",
) -> Actor:
    """Create an ODF actor from a NIFTI file."""

    # Load the ODF data and render it
    odf_data, odf_affine = load_nifti(odf_path)
    sphere = get_sphere(name="repulsion724")  # Use a precomputed sphere
    sh_order_max = calculate_max_order(odf_data.shape[-1])
    B, _ = sh_to_sf_matrix(sphere=sphere, sh_order_max=sh_order_max, basis_type=sh_basis)
    odf_actor = odf_slicer(
        odf_data,
        sphere=sphere,
        B_matrix=B,
        scale=scale,
        norm=False,
        affine=odf_affine,
    )

    # Set offset so it doesn't clash with slice
    position = odf_actor.GetPosition()
    position += offset
    odf_actor.SetPosition(position)

    return odf_actor


def _set_camera(
    scene: window.Scene,
    azimuth: float,
    elevation: float,
    zoom: float = 1.0,
    scene_bound_data: np.ndarray | None = None,
    scene_bound_affine: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set the camera position, orientation, and zoom."""
    if scene_bound_data is not None:
        camera_pos = np.array([0, 0, 1])
        camera_focal = np.array([0, 0, 0])
        camera_up = np.array([0, 1, 0])

        camera_pos_r, camera_pos_theta, camera_pos_phi = cart2sphere(*camera_pos)
        camera_up_r, camera_up_theta, camera_up_phi = cart2sphere(*camera_up)

        # Rotate by azimuth
        camera_pos_phi += np.deg2rad(azimuth)
        camera_up_phi += np.deg2rad(azimuth)

        # Rotate by elevation
        camera_pos_theta -= np.deg2rad(elevation)
        camera_pos_phi += np.deg2rad(90)
        camera_up_theta -= np.deg2rad(elevation)

        # Convert back to cartesian
        camera_pos = sphere2cart(camera_pos_r, camera_pos_theta, camera_pos_phi)
        camera_up = sphere2cart(camera_up_r, camera_up_theta, camera_up_phi)

        # Default distance is 1.5*max dimenson. Divide by zoom to simulate zooming in or out. Shift to middle of array
        camera_pos = np.array(camera_pos) *( 1.5 * max(scene_bound_data.shape)/zoom) + np.array(
            [
                scene_bound_data.shape[0] // 2,
                scene_bound_data.shape[1] // 2,
                scene_bound_data.shape[2] // 2,
            ]
        )
        camera_focal = np.array(camera_focal) + np.array(
            [
                scene_bound_data.shape[0] // 2,
                scene_bound_data.shape[1] // 2,
                scene_bound_data.shape[2] // 2,
            ]
        )

        # Apply affine to translate into world coordinates
        camera_pos = apply_affine(scene_bound_affine, camera_pos)
        camera_focal = apply_affine(scene_bound_affine, camera_focal)

        # Set camera
        scene.set_camera(position=camera_pos, focal_point=camera_focal, view_up=camera_up)
    else:
        scene.reset_camera()
        camera_pos, camera_focal, _ = scene.get_camera()
        view_up = (0, 1, 0)

        # Subtract focal point to get camera position
        camera_pos = np.array(camera_pos)
        camera_pos_r, camera_pos_theta, camera_pos_phi = cart2sphere(*camera_pos)
        view_up_r, view_up_theta, view_up_phi = cart2sphere(*view_up)

        # Rotate by azimuth
        camera_pos_phi += np.deg2rad(azimuth)
        view_up_phi += np.deg2rad(azimuth)

        # Rotate by elevation (theta)
        camera_pos_theta -= np.deg2rad(elevation)
        view_up_theta -= np.deg2rad(elevation)

        # Convert back to cartesian
        camera_pos = sphere2cart(camera_pos_r, camera_pos_theta, camera_pos_phi)
        camera_pos = np.array(camera_pos)
        view_up = sphere2cart(view_up_r, view_up_theta, view_up_phi)
        view_up = np.array(view_up)

        scene.set_camera(position=camera_pos, focal_point=camera_focal, view_up=view_up)
