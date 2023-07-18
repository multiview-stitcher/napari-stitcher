import numpy as np
import dask.array as da
import xarray as xr

from napari_stitcher import _fusion, _registration, _sample_data, _spatial_image_utils, _reader, _mv_graph


def test_fuse_field():

    xims = _reader.read_mosaic_image_into_list_of_spatial_xarrays(
    _sample_data.get_sample_data_path())

    for ixim, xim in enumerate(xims):
        xims[ixim] = xim.sel(c=xim.coords['c'][0],
                             t=xim.coords['t'][0])
        # xims[ixim].name = ixim
        # xim.name = ixim

    params = [_registration.identity_transform(_spatial_image_utils.get_ndim_from_xim(xim))
                         for xim in xims]

    xfused = _fusion.fuse_field(
        xims,
        # [xim.sel(t=xim.coords['t'][0]) for xim in xims],
        params,
        output_origin=np.min([_spatial_image_utils.get_origin_from_xim(xim, asarray=True) for xim in xims], 0),
        output_spacing=_spatial_image_utils.get_spacing_from_xim(xims[0], asarray=True),
        output_shape=_spatial_image_utils.get_shape_from_xim(xr.merge(xims), asarray=True),
        )
    
    # check output is dask array and hasn't been converted into numpy array
    assert(type(xfused.data) == da.core.Array)
    
    xfused = xfused.compute()

    # compare to xarray merge
    xims_merge = xr.merge(xims)
    xims_merge_min = np.nanmin([xims_merge.data_vars[ixim]
                                   for ixim in xims_merge], 0)
    xims_merge_max = np.nanmax([xims_merge.data_vars[ixim]
                                   for ixim in xims_merge], 0)
    xims_merge_fused = np.nanmean([xims_merge.data_vars[ixim]
                                   for ixim in xims_merge], 0)
    
    # check basic properties
    assert(np.allclose(xims_merge_fused.shape, xfused.data.shape))
    assert(xfused.dtype == xims[0].dtype)

    # import tifffile
    # tifffile.imsave('test1.tif', xfused.data)
    # tifffile.imsave('test3.tif', (xims_merge_fused.astype(float) - xfused.data))

    diff_im = np.abs(xims_merge_fused.astype(float)-xfused.data)

    # check that most pixels are the same
    assert(np.percentile(diff_im, 90) < 1)

    # check that fused intensities lie between min and max input
    assert(np.all((xims_merge_min <= xfused.data)))
    assert(np.all((xims_merge_max >= xfused.data)))


def test_fuse_xims():

    xims = _reader.read_mosaic_image_into_list_of_spatial_xarrays(
    _sample_data.get_sample_data_path())

    # test with two channels
    for ixim, xim in enumerate(xims):
        xims[ixim] = xr.concat([xim] * 2, dim='c')\
        .assign_coords(c=[
            xim.coords['c'].data[0],
            xim.coords['c'].data[0] + '_2']
        )

    params = [_registration.identity_transform(_spatial_image_utils.get_ndim_from_xim(xim))
                        for xim in xims]

    xfused = _fusion.fuse_xims(
        xims,
        params,
        output_origin=[0,0],
        output_shape=[10,11],
        output_spacing=[1,1.],
        )

    # check output is dask array and hasn't been converted into numpy array
    assert(type(xfused.data) == da.core.Array)    
    assert(xfused.dtype == xims[0].dtype)

    # xfused.compute()
    xfused = xfused.compute(scheduled='threads')

    assert(xfused.dtype == xims[0].dtype)


# def test_calc_stack_properties_from_xims_and_params()
