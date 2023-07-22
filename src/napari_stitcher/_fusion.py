import itertools
import os
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
    combined_stack_props['shape'] = np.max([np.ceil((sp['origin'] + sp['shape'] * sp['spacing']\
                                    - combined_stack_props['origin']) / combined_stack_props['spacing'])
                                    for sp in stack_props_list], axis=0).astype(np.uint16)
    
    return combined_stack_props


def fuse_xims(xims: list,
               params: list,
               output_origin=None,
               output_spacing=None,
               output_shape=None,
               output_chunksize=512,
               interpolate_missing_pixels=True,
):
    
    """
    Fuse all fields from CT(Z)YX views
    """

    sdims = _spatial_image_utils.get_spatial_dims_from_xim(xims[0])
    nsdims = [dim for dim in xims[0].dims
              if dim not in sdims]

    xds = xr.Dataset(
        {(view, 'xim'): xims[view] for view in range(len(xims))} |
        {(view, 'param'): params[view] for view in range(len(xims))},
    )
    
    merges = []
    for ns_coords in itertools.product(*tuple([xds.coords[nsdim] for nsdim in nsdims])):
        
        xim_coord_dict = {ndsim: ns_coords[i] for i, ndsim in enumerate(nsdims)}
        params_coord_dict = {ndsim: ns_coords[i]
                                for i, ndsim in enumerate(nsdims) if ndsim in params[0].dims}
        
        sxims = [xim.sel(xim_coord_dict) for xim in xims]
        sparams = [param.sel(params_coord_dict) for param in params]

        merge = fuse_field(
            sxims,
            sparams,
            output_origin=output_origin,
            output_shape=output_shape,
            output_spacing=output_spacing,
            output_chunksize=output_chunksize,
            interpolate_missing_pixels=interpolate_missing_pixels, 
        )
        
        merge = merge.expand_dims(nsdims)
        merge = merge.assign_coords({ns_coord.name: [ns_coord.values] for ns_coord in ns_coords})

        merges.append(merge)
    
    if len(merges) > 1:
        # if xims are named, combine_by_coord returns a dataset
        res = xr.combine_by_coords([m.rename(None) for m in merges])
    else:
        res = merge

    return res


def fuse_field(xims,
               params,
               output_origin=None,
               output_spacing=None,
               output_shape=None,
               output_chunksize=512,
               interpolate_missing_pixels=True,
               ):
    """
    fuse tiles from single timepoint and channel
    """

    # views = sorted(field_ims.keys())
    input_dtype = xims[0].dtype
    ndim = _spatial_image_utils.get_ndim_from_xim(xims[0])
    # spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xims[0])

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
            output_chunks=tuple([output_chunksize for _ in output_shape]),
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
            output_chunks=tuple([output_chunksize for _ in output_shape]),
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

    if interpolate_missing_pixels:

        # find empty spaces
        empty_mask = da.min(da.isnan(field_ws_t), 0)

        # convert to input dtype
        fused_field = fused_field.astype(input_dtype)

        # # currently not chunking when interpolating
        # fused_field = da.from_delayed(delayed(get_interpolated_image)(
        #                     fused_field, empty_mask),
        #                     shape=fused_field.shape,
        #                     dtype=fused_field.dtype,
        #                     )
        
        fused_field = da.map_overlap(
            get_interpolated_image,
            fused_field,
            empty_mask,
            depth=tuple([0] + \
                [0 if not idim else np.min([s, output_chunksize]) // 4
                 for idim, s in enumerate(fused_field.shape)]),
            dtype=input_dtype,
        )

    if fused_field.dtype != input_dtype:
        fused_field = fused_field.astype(input_dtype)

    fused_field = xr.DataArray(fused_field, dims=xims[0].dims)

    fused_field = _spatial_image_utils.assign_si_coords_from_params(
        fused_field,
        _spatial_image_utils.compose_params(output_origin, output_spacing)
        )

    return fused_field


def calc_stack_properties_from_xims_and_params(
        xims,
        params,
        spacing,
        mode='union',
        ):
    """
    considers time
    """

    views_props = []
    for _, xim in enumerate(xims):
        views_props.append({
            'shape': _spatial_image_utils.get_shape_from_xim(xim, asarray=True),
            'spacing': _spatial_image_utils.get_spacing_from_xim(xim, asarray=True),
            'origin': _spatial_image_utils.get_origin_from_xim(xim, asarray=True),
        })

    params_ds = xr.Dataset({i: p for i, p in enumerate(params)})

    if 't' in params_ds.dims:
        stack_properties = combine_stack_props(
            [calc_stack_properties_from_view_properties_and_params(
                views_props,
                [params_ds.sel(t=t).data_vars[ip].data for ip in range(len(params))],
                spacing=spacing,
                mode=mode,
            ) for t in params_ds.coords['t']]
        )

    else:
        stack_properties = calc_stack_properties_from_view_properties_and_params(
            views_props,
            [p.data for p in params],
            spacing=spacing,
            mode=mode,
            )

    return stack_properties

def calc_stack_properties_from_view_properties_and_params(
        views_props,
        params,
        spacing,
        mode='union',
        ):
    """
    view props contains
    - shape
    - spacing
    - origin
    """

    spacing = np.array(spacing).astype(float)
    ndim = len(spacing)

    # import pdb; pdb.set_trace()

    stack_vertices = np.array([i for i in np.ndindex(tuple([2] * ndim))]).astype(float)

    if mode == 'sample':
        zero_z_face_vertices = stack_vertices[np.where(stack_vertices[:, 0]==0)]
        zero_z_face_vertices[:, 1] = np.mean(zero_z_face_vertices[:, 1]) # take mean in x
        transformed_vertices = get_transformed_stack_vertices(zero_z_face_vertices, views_props, params)
        volume = (np.min(transformed_vertices, 0),
                  np.max(transformed_vertices, 0)) # lower, upper
        
    elif mode == 'union':
        transformed_vertices = get_transformed_stack_vertices(stack_vertices, views_props, params)
        volume = np.min(transformed_vertices, 0), np.max(transformed_vertices, 0)

    elif mode == 'intersection':
        transformed_vertices = get_transformed_stack_vertices(stack_vertices, views_props, params)
        volume = np.max(np.min(transformed_vertices, 1), 0), np.min(np.max(transformed_vertices, 1), 0)

    # if mode == 'sample':
    #     modified_vertices = vertices[np.where(generic_vertices[:, 0]==0)] # front face of each view
    #     # modified_vertices = np.mean(modified_vertices, -1) # take mean in x
    #     volume = (np.min(modified_vertices,0),
    #               np.max(modified_vertices,0)) # lower, upper
    # if mode == 'union':
    #     volume = np.min(vertices,0), np.max(vertices,0)
    # elif mode == 'intersection':
    #     volume = np.max(np.min(vertices,1),0), np.min(np.max(vertices,1),0)

    stack_properties = calc_stack_properties_from_volume(volume, spacing)

    return stack_properties


def get_transformed_stack_vertices(stack_keypoints, stack_properties_list, params):

    ndim = len(stack_properties_list[0]['spacing'])
    # generic_vertices = np.array([i for i in np.ndindex(tuple([2]*ndim))])
    vertices = np.zeros((len(stack_properties_list) * len(stack_keypoints), ndim))
    for iim, sp in enumerate(stack_properties_list):
        tmp_vertices = stack_keypoints * np.array(sp['shape']) * np.array(sp['spacing']) + np.array(sp['origin'])
        inv_params = np.linalg.inv(((params[iim])))
        tmp_vertices_transformed = np.dot(inv_params[:ndim,:ndim], tmp_vertices.T).T + inv_params[:ndim,ndim]        
        vertices[iim*len(stack_keypoints):(iim+1)*len(stack_keypoints)] = tmp_vertices_transformed
    
    return vertices


def calc_stack_properties_from_volume(volume, spacing):

    """
    :param volume: lower and upper edge of final volume (e.g. [edgeLow,edgeHigh] as calculated by calc_final_stack_cube)
    :param spacing: final spacing
    :return: dictionary containing size, origin and spacing of final stack
    """

    origin                      = volume[0]
    shape                       = np.ceil((volume[1]-volume[0]) / spacing).astype(np.uint16)

    properties_dict = dict()
    properties_dict['shape']    = shape
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


# from scipy import interpolate
def get_interpolated_image(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """

    Currently only 2d!

    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """

    # in case of no known pixels
    if mask.min():
        return image

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


def fuse(
    xims,
    params,
    output_spacing, # dict
    tmpdir=None,
    interpolate_missing_pixels=None,
    output_chunksize=256,
    ):

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xims[0])
    ndim = len(spatial_dims)

    if interpolate_missing_pixels is None:
        interpolate_missing_pixels = True if ndim == 2 else False

    output_stack_properties = calc_stack_properties_from_xims_and_params(
        xims,
        params,
        spacing=np.array([output_spacing[dim] for dim in spatial_dims]),
        mode='union',
        )
    
    xfused = fuse_xims(
        xims,
        params,
        output_origin=output_stack_properties['origin'],
        output_spacing=output_stack_properties['spacing'],
        output_shape=output_stack_properties['shape'],
        output_chunksize=output_chunksize,
        interpolate_missing_pixels=interpolate_missing_pixels,
    )

    if not tmpdir is None:
        xfused.data = da.to_zarr(
            xfused.data,
            os.path.join(tmpdir.name, xfused.data.name+'.zarr'),
            return_stored=True,
            overwrite=True,
            compute=True,
            )
    
    if 'C' in xims[0].coords:
        xfused = xfused.assign_coords(C=xims[0].coords['C'])
    
    return xfused


if __name__ == "__main__":

    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/MAX_LSM900.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/yu_220829_WT_quail_st6_x10_zoom0.7_1x3_488ZO1-568Sox2-647Tbra.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/04_stretch-01_AcquisitionBlock2_pt2.czi"
    filename = "/Users/malbert/software/napari-stitcher/image-datasets/mosaic_test.czi"

    from napari_stitcher import _utils, _reader
