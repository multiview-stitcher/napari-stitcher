import numpy as np
import xarray as xr
import dask.array as da
from dask import delayed

from scipy import ndimage
from dask_image import ndinterp

from napari_stitcher import _spatial_image_utils


def combine_stack_props(stack_props_list):

    combined_stack_props = {}
    combined_stack_props['origin'] = np.min([sp['origin'] for sp in stack_props_list], axis=0)
    combined_stack_props['spacing'] = np.min([sp['spacing'] for sp in stack_props_list], axis=0)
    combined_stack_props['size'] = np.max([np.ceil((sp['origin'] + sp['size'] * sp['spacing']\
                                    - combined_stack_props['origin']) / combined_stack_props['spacing'])
                                    for sp in stack_props_list], axis=0).astype(np.uint16)
    return combined_stack_props


# def fuse_tiles(viewims: dict,
#                params: dict,
#                view_dict: dict,
#             #    blending_widths: list,
# ):

#     input_channels = sorted(viewims.keys())
#     input_times = sorted(viewims[input_channels[0]].keys())
#     views = sorted(viewims[input_channels[0]][input_times[0]].keys())

#     for view in views:
#         view_dict[view]['size'] = view_dict[view]['shape']

#     field_stack_props = [calc_stack_properties_from_views_and_params(
#                 [view_dict[view] for view in views],
#                 params[t],
#                 view_dict[views[0]]['spacing'],
#                 mode='union',
#             )
#         for t in input_times]

#     fusion_stack_props = combine_stack_props(field_stack_props)

#     fused_da = \
#         da.stack([
#             da.stack([
#                 # da.from_delayed(delayed(fuse_field)(
#                     fuse_field(
#                         viewims[ch][t],
#                         params[t],
#                         view_dict,
#                         fusion_stack_props,
#                         )
#                 # shape=tuple(fusion_stack_props['size']),
#                 # dtype=np.uint16,
#                 # dtype=viewims[ch][t][views[0]].dtype,
                
#             for t in input_times])
#         for ch in input_channels])

#     return fused_da, fusion_stack_props, field_stack_props


# def fuse_tiles(xims: list,
#                params: list,
#             #    view_dict: dict,
#             #    blending_widths: list,
# ):
#     """
#     - create xr dataset containing all info
#     - apply fuse_field to each tp independently
#     """
    
#     xds = xr.Dataset(
#         {xim.name: xim for xim in xims} |\
#         {(xim.name, 'transform'): params.data_vars[xim.name] for xim in xims}
#         )
    

    
#     xds.groupby('T')
    
#     xfused = xr.apply_ufunc(
#         fuse_field,
#         xds.groupby('T')) if 'T' in xds else xds,
        
    

#     xdss = []
#     for xim in xims:
        
#     xr.Dataset({xims.name: xim}.update({'params': }))
    
#     return


def fuse_tiles(xims: list,
               params: list,
            #    view_dict: dict,
            #    blending_widths: list,
):

    # input_channels = sorted(viewims.keys())
    # input_times = sorted(viewims[input_channels[0]].keys())
    # views = sorted(viewims[input_channels[0]][input_times[0]].keys())

    for view in views:
        view_dict[view]['size'] = view_dict[view]['shape']

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xims[0])

    target_spacing = _spatial_image_utils.get_spacing_from_xim(xims[0])

    # size, spacing, origin
    field_stack_props = [calc_stack_properties_from_views_and_params(
                [{'size': np.array([len(xim.coords[sd] for sd in spatial_dims)]),
                  'origin': np.array([_spatial_image_utils.get_origin_from_xim(xim, asarray=True)[sd]
                                      for sd in spatial_dims]),
                  'spacing': np.array([_spatial_image_utils.get_spacing_from_xim(xim, asarray=True)[sd]
                                       for sd in spatial_dims]),
                  } for xim in xims],
                params.isel(T=t),
                np.array([target_spacing[sd] for sd in spatial_dims]),
                mode='union')
        for t in input_times]

    fusion_stack_props = combine_stack_props(field_stack_props)

    fused_da = \
        da.stack([
            da.stack([
                # da.from_delayed(delayed(fuse_field)(
                    fuse_field(
                        viewims[ch][t],
                        params[t],
                        view_dict,
                        fusion_stack_props,
                        )
                # shape=tuple(fusion_stack_props['size']),
                # dtype=np.uint16,
                # dtype=viewims[ch][t][views[0]].dtype,
                
            for t in input_times])
        for ch in input_channels])

    return fused_da, fusion_stack_props, field_stack_props


def fuse_field(xims,
               params,
               output_origin=None,
               output_spacing=None,
               output_shape=None,
               do_interpolate_missing_pixels=True,
               ):
    """
    fuse tiles from single timepoint and channel
    """

    # views = sorted(field_ims.keys())
    input_dtype = xims[0].dtype

    ndim = _spatial_image_utils.get_ndim_from_xim(xims[0])

    field_ims_t = []
    field_ws_t = []
    # for view in views:
    for ixim, xim in enumerate(xims):

        p = np.array(params[ixim])
        matrix = p[:ndim, :ndim]
        offset = p[:ndim, ndim]

        # spacing matrices
        Sx = np.diag(output_spacing)
        Sy = np.diag(_spatial_image_utils.get_spacing_from_xim(xim, asarray=True))

        matrix_prime = np.dot(np.linalg.inv(Sy), np.dot(matrix, Sx))
        offset_prime = np.dot(np.linalg.inv(Sy),
            offset - _spatial_image_utils.get_origin_from_xim(xim, asarray=True) +
            np.dot(matrix, output_origin))
        
        field_ims_t.append(ndinterp.affine_transform(
                                            xim.data,
                                            matrix=matrix_prime,
                                            offset=offset_prime,
                                            order=1,
                                            output_shape=tuple(output_shape),
                                            # output_chunks=tuple([600 for _ in output_shape]),
                                            output_chunks=tuple(output_shape),
                                            mode='constant',
                                            cval=0.,
                                            )
        )

        if ndim == 2:
            blending_widths = [10] * 2
        else:
            blending_widths = [3] + [10] * 2

        field_ws = get_smooth_border_weight_from_shape(xim.shape[-ndim:], widths=blending_widths)

        field_ws_t.append(ndinterp.affine_transform(
                                            field_ws,
                                            matrix=matrix_prime,
                                            offset=offset_prime,
                                            order=1,
                                            output_shape=tuple(output_shape),
                                            output_chunks=tuple(output_shape),
                                            mode='constant',
                                            cval=np.nan, # nan indicates no data
                                            )
        )

    field_ims_t = da.stack(field_ims_t)
    field_ws_t = da.stack(field_ws_t)

    wsum = da.nansum(field_ws_t, axis=0)
    wsum[wsum==0] = 1

    field_ws_t = field_ws_t / wsum

    fused_field = da.nansum(field_ims_t * field_ws_t, axis=0)

    if do_interpolate_missing_pixels:

        # find empty spaces
        empty_mask = da.min(da.isnan(field_ws_t), 0)

        # convert to input dtype
        fused_field = fused_field.astype(input_dtype)

        fused_field = da.from_delayed(delayed(interpolate_missing_pixels)(
                            fused_field, empty_mask),
                            shape=fused_field.shape,
                            dtype=fused_field.dtype)

    fused_field = xr.DataArray(fused_field,
                             dims=xims[0].dims
    )

    fused_field = _spatial_image_utils.assign_si_coords_from_params(
        fused_field,
        _spatial_image_utils.compose_params(output_origin, output_spacing)
        )

    return fused_field


def calc_stack_properties_from_views_and_params(views_props, params, spacing=None, mode='sample'):

    spacing = np.array(spacing).astype(np.float64)

    if mode == 'sample':
        volume = get_sample_volume(views_props, params)
    elif mode == 'union':
        volume = get_union_volume(views_props, params)
    elif mode == 'intersection':
        volume = get_intersection_volume(views_props, params)

    stack_properties = calc_stack_properties_from_volume(volume, spacing)

    return stack_properties


def get_sample_volume(stack_properties_list, params):
    """
    back project first planes in every view to get maximum volume
    """

    ndim = len(stack_properties_list[0]['spacing'])
    generic_vertices = np.array([i for i in np.ndindex(tuple([2]*ndim))])
    vertices = np.zeros((len(stack_properties_list)*len(generic_vertices),ndim))
    for iim, sp in enumerate(stack_properties_list):
        tmp_vertices = generic_vertices * np.array(sp['size']) * np.array(sp['spacing']) + np.array(sp['origin'])
        inv_params = np.linalg.inv(((params[iim])))
        tmp_vertices_transformed = np.dot(inv_params[:ndim,:ndim], tmp_vertices.T).T + inv_params[:ndim,ndim]
        vertices[iim*len(generic_vertices):(iim+1)*len(generic_vertices)] = tmp_vertices_transformed

    lower = np.min(vertices,0)
    upper = np.max(vertices,0)

    return lower,upper


def get_union_volume(stack_properties_list, params):
    """
    back project first planes in every view to get maximum volume
    """
    ndim = len(stack_properties_list[0]['spacing'])
    generic_vertices = np.array([i for i in np.ndindex(tuple([2]*ndim))])
    vertices = np.zeros((len(stack_properties_list)*len(generic_vertices),ndim))
    for iim, sp in enumerate(stack_properties_list):
        tmp_vertices = generic_vertices * np.array(sp['size']) * np.array(sp['spacing']) + np.array(sp['origin'])
        inv_params = np.linalg.inv(((params[iim])))
        tmp_vertices_transformed = np.dot(inv_params[:ndim,:ndim], tmp_vertices.T).T + inv_params[:ndim,ndim]
        vertices[iim*len(generic_vertices):(iim+1)*len(generic_vertices)] = tmp_vertices_transformed

    lower = np.min(vertices,0)
    upper = np.max(vertices,0)

    return lower,upper


def get_intersection_volume(stack_properties_list, params):
    """
    back project first planes in every view to get maximum volume
    """

    ndim = len(stack_properties_list[0]['spacing'])
    # generic_vertices = np.array([[i,j,k] for i in [0,1] for j in [0,1] for k in [0,1]])
    generic_vertices = np.array([i for i in np.ndindex(tuple([2]*ndim))])
    vertices = np.zeros((len(stack_properties_list)*len(generic_vertices),ndim))
    for iim, sp in enumerate(stack_properties_list):
        tmp_vertices = generic_vertices * np.array(sp['size']) * np.array(sp['spacing']) + np.array(sp['origin'])
        inv_params = np.linalg.inv(((params[iim])))
        tmp_vertices_transformed = np.dot(inv_params[:ndim,:ndim], tmp_vertices.T).T + inv_params[:ndim,ndim]
        vertices[iim,:] = tmp_vertices_transformed

    lower = np.max(np.min(vertices,1),0)
    upper = np.min(np.max(vertices,1),0)

    return lower,upper


def calc_stack_properties_from_volume(volume, spacing):

    """
    :param volume: lower and upper edge of final volume (e.g. [edgeLow,edgeHigh] as calculated by calc_final_stack_cube)
    :param spacing: final spacing
    :return: dictionary containing size, origin and spacing of final stack
    """

    origin                      = volume[0]
    size                        = np.ceil((volume[1]-volume[0]) / spacing).astype(np.uint16)

    properties_dict = dict()
    properties_dict['size']     = size
    properties_dict['spacing']  = spacing
    properties_dict['origin']   = origin

    return properties_dict


def smooth_transition(x, x_offset=0.5, x_stretch=None, k=3):
    """
    Transform the distance from the border to a weight for blending.
    """
    # https://math.stackexchange.com/questions/1832177/sigmoid-function-with-fixed-bounds-and-variable-steepness-partially-solved
    
    if x_stretch is None:
        x_stretch = x_offset

    w = np.zeros(x.shape).astype(np.float32)

    xp = x.astype(np.float32)
    xp = (xp - x_offset) # w is 0 at offset
    xp = xp / x_stretch / 2. # w is +/-0.5 at offset +/- x_stretch

    mask = (xp > -.5) * (xp < .5)
    w[mask] = 1 - 1 / (1 + (1 / (xp[mask] + 0.5) - 1) ** (-k))

    w[xp <= -.5] = 0.
    w[xp >= .5] = 1.

    return w


def get_smooth_border_weight_from_shape(shape, widths=None):
    """
    Get a weight image for blending that is 0 at the border and 1 at the center.
    Transition widths can be specified for each dimension.
    """

    ndim = len(shape)

    # get distance to border for each dim

    # zero at the border
    # dim_dists = [ndimage.distance_transform_edt(
    #                 ndimage.binary_erosion(
    #                     np.ones(shape[dim]).astype(bool)))
    #                         for dim in range(ndim)]

    # nonzero at the border
    dim_dists = [ndimage.distance_transform_edt(
                    np.ones(shape[dim]).astype(bool))
                            for dim in range(ndim)]

    dim_ws = [smooth_transition(dim_dists[dim],
                                x_offset=widths[dim],
                                x_stretch=widths[dim]) for dim in range(ndim)]

    # get product of weights for each dim
    w = np.ones(shape).astype(np.float32)
    for dim in range(len(shape)):
        tmp_dim_w = dim_ws[dim]
        for _ in range(ndim - dim - 1):
            tmp_dim_w = tmp_dim_w[:, None]
        w *= tmp_dim_w

    return w


from scipy import interpolate
def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    # np.meshgrid(*tuple([np.arange(s) for s in [2,3,4]]))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


if __name__ == "__main__":

    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/MAX_LSM900.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/yu_220829_WT_quail_st6_x10_zoom0.7_1x3_488ZO1-568Sox2-647Tbra.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/04_stretch-01_AcquisitionBlock2_pt2.czi"
    filename = "/Users/malbert/software/napari-stitcher/image-datasets/mosaic_test.czi"

    from napari_stitcher import _utils, _reader
    # from mvregfus import io_utils, mv_utils

    # xims = 

    # view_dict = io_utils.build_view_dict_from_multitile_czi(filename, max_project=False)
    # views = np.array([view for view in sorted(view_dict.keys())])
    # pairs = mv_utils.get_registration_pairs_from_view_dict(view_dict)

    # viewims = _utils.load_tiles(view_dict, [0],
    #                 [0], max_project=False)
    
    # params = {0: {view: mv_utils.matrix_to_params(np.eye(len(view_dict[0]['origin'])+1)) for view in views}}

    # fused_da, fusion_stack_props_d, field_stack_props_d = \
    #     fuse_tiles(viewims, params, view_dict)

    # fused = fused_da.compute()

    # import tifffile
    # tifffile.imwrite('delme1.tif', fused.astype(np.float32))

    # for view in views:
    #     tifffile.imwrite('delme_%s.tif' % view, viewims[0][0][view].compute().astype(np.float32))
