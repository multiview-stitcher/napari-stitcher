"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
import xarray as xr

from aicsimageio import AICSImage
import tifffile

import spatial_image as si
# import multiscale_spatial_image as msi

from napari_stitcher import _spatial_image_utils, _viewer_utils, _utils, _msi_utils


READER_METADATA_TRANSFORM_KEY = 'affine_metadata'


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    # otherwise we return the *function* that can read ``path``.
    if path.endswith(".czi"):
        return read_mosaic_czi
    else:
        return None
    

def read_mosaic_image_into_list_of_spatial_xarrays(path, scene_index=None):
    """
    Read CZI mosaic dataset into xarray containing all information needed for stitching.
    Could eventually be based on https://github.com/spatial-image/spatial-image.
    Use list instead of dict to make sure xarray metadata (coordinates + perhaps attrs)
    are self explanatory for downstream code (and don't depend e.g. on view/tile numbering).
    """

    aicsim = AICSImage(path, reconstruct_mosaic=False)

    if len(aicsim.scenes) > 1 and scene_index is None:
        from magicgui.widgets import request_values
        scene_index = request_values(
            scene_index=dict(annotation=int,
                        label="Which scene should be loaded?",
                        options={'min': 0, 'max': len(aicsim.scenes) - 1}),
            )['scene_index']
        aicsim.set_scene(scene_index)
    else:
        scene_index = 0

    xim =  aicsim.get_xarray_dask_stack()

    xim = xim.sel(I=scene_index)

    # xim coords to lower case
    xim = xim.rename({dim: dim.lower() for dim in xim.dims})

    # remove singleton Z
    # for axis in ['z', 't']:
    for axis in ['z']:
        if axis in xim.dims and len(xim.coords[axis]) < 2:
            xim = xim.sel({axis: 0}, drop=True)
    
    # ensure time dimension is present
    xim = _spatial_image_utils.ensure_time_dim(xim)

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xim)
    # ndim = _spatial_image_utils.get_ndim_from_xim(xim)
    # spacing = _spatial_image_utils.get_spacing_from_xim(xim)
    # origin = _spatial_image_utils.get_origin_from_xim(xim)

    views = range(len(xim.coords['m']))
    
    # pixel_sizes = aicsim.physical_pixel_sizes._asdict()
    pixel_sizes = dict()
    pixel_sizes['x'] = aicsim.physical_pixel_sizes.X
    pixel_sizes['y'] = aicsim.physical_pixel_sizes.Y
    if 'z' in spatial_dims:
        pixel_sizes['z'] = aicsim.physical_pixel_sizes.Z
        
    view_sims = []
    for iview, view in enumerate(views):

        view_xim = xim.sel(m=view)

        view_sim = _spatial_image_utils.get_sim_from_xim(view_xim)

        tile_mosaic_position = aicsim.get_mosaic_tile_position(view)
        origin_values = {mosaic_axis: tile_mosaic_position[ima] * pixel_sizes[mosaic_axis]
                  for ima, mosaic_axis in enumerate(['y', 'x'])}
        
        if 'z' in spatial_dims:
            origin_values['z'] = 0

        affine = _utils.shift_to_matrix(
            np.array([origin_values[dim] for dim in spatial_dims]))
        
        affine_xr = xr.DataArray(np.stack([affine] * len(view_sim.coords['t'])),
                                 dims=['t', 'x_in', 'x_out'])
        

        _spatial_image_utils.set_xim_affine(
            view_sim,
            affine_xr,
            READER_METADATA_TRANSFORM_KEY
            )

        view_sim.name = str(iview)

        view_sims.append(view_sim)

    return view_sims


def read_tiff_into_spatial_xarray(
        filename,
        scale=None, translation=None,
        affine_transform=None,
        **kwargs):

    if scale is None:
        scale = {ax: 1 for ax in ['z', 'y', 'x']}

    if translation is None:
        translation = {ax: 0 for ax in ['z', 'y', 'x']}

    aicsimage = AICSImage(filename)
    xim = aicsimage.get_xarray_dask_stack().squeeze(drop=True)
    xim = xim.rename({dim: dim.lower() for dim in xim.dims})
    xim = _spatial_image_utils.ensure_time_dim(xim)

    if 'c' not in xim.dims:
        xim = xim.expand_dims(['c'])

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xim)
    xim = xim.transpose(*(('t', 'c') + tuple(spatial_dims)))

    sim = si.to_spatial_image(
        xim.data,
        dims=xim.dims,
        scale=scale,
        translation=translation,
        )

    ndim = _spatial_image_utils.get_ndim_from_xim(xim)

    if affine_transform is None:
        affine_transform = np.eye(ndim + 1)

    affine_xr = xr.DataArray(
        np.stack([affine_transform] * len(xim.coords['t'])),
        dims=['t', 'x_in', 'x_out'],
        coords={'t': xim.coords['t']})

    _spatial_image_utils.set_xim_affine(
        sim,
        affine_xr,
        READER_METADATA_TRANSFORM_KEY,
    )

    return sim


# acisimageio can have problems, namely shape of xim is different from shape of computed xim.data
# therefore first load xim, then get xim.data, then create spatial image from xim.data and back on disk
# for multiscale always require zarr format


def read_mosaic_czi(path, scene_index=None):
    """
    
    Read in tiles as layers.
    
    Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path

    xims = read_mosaic_image_into_list_of_spatial_xarrays(paths[0], scene_index=scene_index)
    msims = [_msi_utils.get_msim_from_xim(xim) for xim in xims]

    out_layers = _viewer_utils.create_image_layer_tuples_from_msims(
        msims,
        transform_key=READER_METADATA_TRANSFORM_KEY)

    return out_layers


if __name__ == "__main__":

    import napari
    # from napari_stitcher import StitcherQWidget

    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/04_stretch-01_AcquisitionBlock2_pt2.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/yu_220829_WT_quail_st6_x10_zoom0.7_1x3_488ZO1-568Sox2-647Tbra.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/MAX_LSM900.czi"
    filename = "/Users/malbert/software/napari-stitcher/image-datasets/mosaic_test.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20210216_highres_TR2.czi" # somehow doesn't work here. the reader gives xarrays that have a shape different from their computed shape
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20230223_02_before_ablation-02_20X_max.czi"

    # xims = read_mosaic_image_into_list_of_spatial_xarrays(filename, scene_index=0)
    # msims =[_msi_utils.get_msim_from_xim(xim) for xim in xims]

    viewer = napari.Viewer()
    viewer.open(filename, scene_index=0)
