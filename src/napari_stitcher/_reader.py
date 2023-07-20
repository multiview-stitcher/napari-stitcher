"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
import xarray as xr

import xml

import dask.array as da
from dask import delayed

# from aicspylibczi import CziFile
from aicsimageio import AICSImage

from napari_stitcher import _spatial_image_utils, _viewer_utils, _utils


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

    views = range(len(xim.coords['m']))
    
    # pixel_sizes = aicsim.physical_pixel_sizes._asdict()
    pixel_sizes = dict()
    pixel_sizes['x'] = aicsim.physical_pixel_sizes.X
    pixel_sizes['y'] = aicsim.physical_pixel_sizes.Y
    if 'z' in spatial_dims:
        pixel_sizes['z'] = aicsim.physical_pixel_sizes.Z

    # pixel_sizes = {k: v for k, v in pixel_sizes.items() if v is not None}
    

    # # For some Zeiss files, the tiles contain duplicate image data at the borders.
    # # This trims the data at the borders.

    # m_string = xml.etree.ElementTree.tostring(aicsim.metadata)
    # if (
    #     b'Zeiss' in m_string and
    #    (b'Detector: Airyscan' in m_string) and
    #    (b'TileRegionCoveringMode' in m_string) and # not sure about this one
    #    (b'AlignedToLocalTileRegion' in m_string) # not sure about this one
    # ):
    #     tile_poss = np.array(aicsim.get_mosaic_tile_positions())
    #     tile_diffs = np.diff(tile_poss, axis=0)
    #     mosaic_dim = len(tile_poss[0])
    #     spatial_tile_slices = []
    #     for itile in range(len(tile_poss)):
    #         spatial_tile_slice = [slice(None) for _ in range(len(spatial_dims) - mosaic_dim)]
    #         for dim in range(mosaic_dim):
    #             if tile_poss[itile][dim] == np.max(tile_poss[:,dim]):
    #                 dim_slice = slice(0, xim.shape[-mosaic_dim+dim])
    #             else:
    #                 dim_slice = slice(0, tile_diffs[itile][dim])
    #             spatial_tile_slice.append(dim_slice)
    #         spatial_tile_slices.append(spatial_tile_slice)
    #     print('LSLSLSL', spatial_tile_slices)
    # else:
    #     spatial_tile_slices = [slice(None) for dim in spatial_dims]
        
    view_xims = []
    for iview, view in enumerate(views):

        view_xim = xim.sel(m=view)
        # import pdb; pdb.set_trace()

        # # preprocess tiles in case of specific mosaic format
        # # such as in the case of airyscan
        # view_xim = view_xim.isel(
        #     {dim: spatial_tile_slices[iview][idim]
        #      for idim, dim in enumerate(spatial_dims)})

        tile_mosaic_position = aicsim.get_mosaic_tile_position(view)
        origin_values = {mosaic_axis: tile_mosaic_position[ima] * pixel_sizes[mosaic_axis]
                  for ima, mosaic_axis in enumerate(['y', 'x'])}
        
        if 'z' in spatial_dims:
            origin_values['z'] = 0

        origin = xr.DataArray([origin_values[dim] for dim in spatial_dims],
                              dims=['dim'],
                              coords={'dim': spatial_dims})
        
        for dim in spatial_dims:
            view_xim = view_xim.assign_coords({dim: view_xim.coords[dim] + origin.loc[dim]})

        # affine = _utils.shift_to_matrix(
        #     np.array([origin_values[dim] for dim in spatial_dims]))

        view_xim.attrs.update(dict(
            # affine_metadata=affine,
            scene_index=scene_index,
            source=path,
        ))

        view_xim.name = str(view)

        view_xims.append(view_xim)

    return view_xims


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

    out_layers = _viewer_utils.create_image_layer_tuples_from_xims(xims)

    return out_layers


if __name__ == "__main__":

    import napari
    # from napari_stitcher import StitcherQWidget

    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/04_stretch-01_AcquisitionBlock2_pt2.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/yu_220829_WT_quail_st6_x10_zoom0.7_1x3_488ZO1-568Sox2-647Tbra.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/MAX_LSM900.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/mosaic_test.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20210216_highres_TR2.czi"
    filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20230223_02_before_ablation-02_20X_max.czi"

    viewer = napari.Viewer()
    
    # viewer.open("/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220609_WT_emb2_5X_part1_max.czi")

    # wdg = StitcherQWidget(viewer)
    # viewer.window.add_dock_widget(wdg)

    viewer.open(filename)

    # napari.run()