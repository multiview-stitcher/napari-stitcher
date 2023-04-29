import numpy as np
import dask.array as da
import xarray as xr

from napari_stitcher import _fusion, _registration, _sample_data, _spatial_image_utils, _reader, _mv_graph

def test_fuse_fields():

    xims = _reader.read_mosaic_czi_into_list_of_spatial_xarrays(
    _sample_data.get_sample_data_path())

    for ixim, xim in enumerate(xims):
        xims[ixim] = xim.sel(C=xim.coords['C'][0])
        # xims[ixim].name = ixim
        # xim.name = ixim

    # params = xr.Dataset({xim.name: _registration.identity_transform(_spatial_image_utils.get_ndim_from_xim(xim))
    #                      for xim in xims})
    params = [_registration.identity_transform(_spatial_image_utils.get_ndim_from_xim(xim))
                         for xim in xims]

    # _fusion.calc_stack_properties_from_views_and_params

    # spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xims[0])

    xfused = _fusion.fuse_field(
        xims,
        params,
        output_origin=np.min([_spatial_image_utils.get_origin_from_xim(xim, asarray=True) for xim in xims], 0),
        output_spacing=_spatial_image_utils.get_spacing_from_xim(xims[0], asarray=True),
        output_shape=_spatial_image_utils.get_shape_from_xim(xr.merge(xims), asarray=True),
        )
    
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

