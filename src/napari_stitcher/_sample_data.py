"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import dask.array as da
from scipy import ndimage

from pathlib import Path

from napari_stitcher._reader import read_mosaic_czi, READER_METADATA_TRANSFORM_KEY
from napari_stitcher._viewer_utils import create_image_layer_tuples_from_msims
from napari_stitcher._utils import shift_to_matrix
from napari_stitcher import _spatial_image_utils, _msi_utils


def get_sample_data_path():

    sample_path = Path(__file__).parent.parent.parent /\
                             "image-datasets" / "mosaic_test.czi"
    
    return sample_path


def make_sample_data():
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image

    sample_path = get_sample_data_path()

    return read_mosaic_czi([sample_path])


def generate_tiled_dataset(ndim=2, N_c=2, N_t=20,
                           tile_size=30, tiles_x=2, tiles_y=2, tiles_z=1,
                           overlap=5, zoom=6, dtype=np.uint16,
                           spacing_x=0.5, spacing_y=0.5, spacing_z=2.,
                           shift_scale=2., drift_scale=2.,
                           transform_key=READER_METADATA_TRANSFORM_KEY):

    def transform_input(x, shifts, drifts, im_gt, overlap=0, zoom=10., block_info=None):

        x = x.squeeze()
        
        output_shape = np.array(x.shape)

        shift = shifts[block_info[0]['chunk-location']]
        drift = drifts[block_info[0]['chunk-location']]
        
        eff_shape = output_shape - overlap
        offset = np.array(block_info[None]['chunk-location'][1:]) * eff_shape

        offset = offset + drift + shift

        offset = offset / zoom
                
        x = ndimage.affine_transform(im_gt,
                                    matrix=np.eye(x.ndim) / zoom,
                                    offset=offset,
                                    output_shape=output_shape,
                                    mode='reflect',
                                    )[None]
            
        return x

    # build the array
    tiles = da.empty((N_t,) + tuple([tile_size * f
                                     for f in [tiles_z, tiles_y, tiles_x][-ndim:]]),
                    chunks=(1,) + (tile_size, ) * ndim, dtype=dtype)

    # simulate shifts and drifts
    shifts = (np.random.random(tiles.numblocks + (ndim, )) - 0.5) * shift_scale
    drifts = np.cumsum(np.ones(tiles.numblocks + (ndim, )) * drift_scale, axis=0)

    np.random.seed(0)
    tls = []
    for ch in range(N_c):
        # the channel ground truth
        im_gt = da.random.randint(
            0, 100, [2 * f * tile_size // zoom
                for f in [tiles_z, tiles_y, tiles_x][-ndim:]], dtype=np.uint16)
        tl = tiles.map_blocks(transform_input, shifts, drifts, im_gt, zoom=zoom,
                              overlap=overlap, dtype=tiles.dtype)
        tls.append(tl[None])
        
    tls = da.concatenate(tls, axis=0)

    # generate xims
    xims = []
    spatial_dims = ['z', 'y', 'x'][-ndim:]
    spacing = [spacing_z, spacing_y, spacing_x][-ndim:]
    for tile_index in np.ndindex(tls.numblocks[2:]):
        tile_index = np.array(tile_index)
        tile = tls.blocks[tuple([slice(0, N_c), slice(0, N_t)] + \
                                [slice(ti, ti+1) for ti in tile_index])]
        origin = tile_index * tile_size * spacing - overlap * (tile_index) * spacing
        xim = xr.DataArray(
            tile,
            dims=['c','t'] + spatial_dims,
            coords={spatial_dims[dim]:
                    # origin[dim] +\
                    np.arange(tile.shape[2+dim]) * spacing[dim] for dim in range(ndim)} |
            {'c': ['channel ' + str(c) for c in range(N_c)]},
        )

        affine = shift_to_matrix(origin)
        
        affine_xr = xr.DataArray(np.stack([affine] * len(xim.coords['t'])),
                                 dims=['t', 'x_in', 'x_out'])
        
        xim.attrs['transforms'] = xr.Dataset(
            {transform_key: affine_xr}
        )

        xims.append(xim)
    
    return xims


def drifting_timelapse_with_stage_shifts_no_overlap_2d():

    xims = generate_tiled_dataset(
        ndim=2, N_t=20, N_c=1,
        tile_size=30, tiles_x=3, tiles_y=3, tiles_z=1,
        drift_scale=2., shift_scale=2.,
        overlap=0, zoom=8, dtype=np.uint8)
    
    msims = [_msi_utils.get_msim_from_xim(xim) for xim in xims]

    layer_tuples = create_image_layer_tuples_from_msims(
        msims, transform_key=READER_METADATA_TRANSFORM_KEY)

    return layer_tuples


def timelapse_with_stage_shifts_with_overlap_3d():

    xims = generate_tiled_dataset(
        ndim=3, N_t=20, N_c=1,
        tile_size=30, tiles_x=3, tiles_y=3, tiles_z=1,
        drift_scale=0., shift_scale=2.,
        overlap=3, zoom=8, dtype=np.uint8)
    
    msims = [_msi_utils.get_msim_from_xim(xim) for xim in xims]

    layer_tuples = create_image_layer_tuples_from_msims(
        msims, transform_key=READER_METADATA_TRANSFORM_KEY)

    return layer_tuples


# if __name__ == "__main__":

#     import napari

#     viewer = napari.Viewer()

#     xims = generate_tiled_dataset(
#         ndim=2, N_t=20, N_c=1,
#         tile_size=1000, tiles_x=1, tiles_y=1, tiles_z=1,
#         overlap=15, zoom=5, shift_scale=5, drift_scale=2, dtype=np.uint8)
#     layer_tuples = create_image_layer_tuples_from_xims(xims, n_colors=2)

#     # layer_tuples = make_sample_data()

#     for lt in layer_tuples:
#         lt[1]['contrast_limits'] = [0, 100]
#         viewer.add_image(lt[0], **lt[1])

#     from napari_stitcher import StitcherQWidget

#     wdg = StitcherQWidget(viewer)
#     viewer.window.add_dock_widget(wdg)
