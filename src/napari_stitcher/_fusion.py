import numpy as np

import dask.array as da
from dask import delayed

from mvregfus.multiview import calc_stack_properties_from_views_and_params

from scipy import ndimage
from dask_image import ndinterp


def combine_stack_props(stack_props_list):

    combined_stack_props = {}
    combined_stack_props['origin'] = np.min([sp['origin'] for sp in stack_props_list], axis=0)
    combined_stack_props['spacing'] = np.min([sp['spacing'] for sp in stack_props_list], axis=0)
    combined_stack_props['size'] = np.max([np.ceil((sp['origin'] + sp['size'] * sp['spacing']\
                                    - combined_stack_props['origin']) / combined_stack_props['spacing'])
                                    for sp in stack_props_list], axis=0).astype(np.uint16)
    return combined_stack_props


def fuse_tiles(viewims: dict,
               params: dict,
               view_dict: dict,
            #    blending_widths: list,
):

    input_channels = sorted(viewims.keys())
    input_times = sorted(viewims[input_channels[0]].keys())
    views = sorted(viewims[input_channels[0]][input_times[0]].keys())

    for view in views:
        view_dict[view]['size'] = view_dict[view]['shape']

    field_stack_props = [calc_stack_properties_from_views_and_params(
                [view_dict[view] for view in views],
                params[t],
                view_dict[views[0]]['spacing'],
                mode='union',
            )
        for t in input_times]

    fusion_stack_props = combine_stack_props(field_stack_props)

    fused_da = \
        da.stack([
            da.stack([
                da.from_delayed(delayed(fuse_field)(
                        viewims[ch][t],
                        params[t],
                        view_dict,
                        fusion_stack_props,
                        ),
                shape=tuple(fusion_stack_props['size']),
                # dtype=np.uint16,
                dtype=viewims[ch][t][views[0]].dtype,
                )
            for t in input_times])
        for ch in input_channels])

    return fused_da, fusion_stack_props, field_stack_props


def fuse_field(field_ims, params, view_dict, out_stack_props):
    """
    fuse tiles from single timepoint and channel
    """

    views = sorted(field_ims.keys())
    input_dtype = field_ims[views[0]].dtype

    ndim = field_ims[0].ndim
    field_ims_t = []
    field_ws_t = []
    for view in views:

        p = np.array(params[view])
        matrix = p[:ndim*ndim].reshape(ndim,ndim)
        offset = p[ndim*ndim:]

        # spacing matrices
        Sx = np.diag(out_stack_props['spacing'])
        Sy = np.diag(view_dict[view]['spacing'])

        matrix_prime = np.dot(np.linalg.inv(Sy), np.dot(matrix, Sx))
        offset_prime = np.dot(np.linalg.inv(Sy),
            offset - view_dict[view]['origin'] + np.dot(matrix, out_stack_props['origin']))

        field_ims_t.append(ndinterp.affine_transform(
                                            field_ims[view].astype(np.uint16) + 1, # add 1 so that 0 indicates no data
                                            matrix=matrix_prime,
                                            offset=offset_prime,
                                            order=3,
                                            output_shape=tuple(out_stack_props['size']),
                                            output_chunks=tuple([600 for _ in out_stack_props['size']]),
                                            # output_chunks=tuple(out_stack_props['size']),
                                            mode='constant',
                                            cval=0.,
                                            )
        )

        if ndim == 2:
            blending_widths = [10] * 2
        else:
            blending_widths = [3] + [10] * 2

        field_ws = get_smooth_border_weight_from_shape(field_ims[view].shape[-ndim:], widths=blending_widths)

        field_ws_t.append(ndinterp.affine_transform(
                                            field_ws, # add 1 so that 0 indicates no data
                                            matrix=matrix_prime,
                                            offset=offset_prime,
                                            order=1,
                                            output_shape=tuple(out_stack_props['size']),
                                            output_chunks=tuple([600 for _ in out_stack_props['size']]),
                                            # output_chunks=tuple(out_stack_props['size']),
                                            mode='constant',
                                            cval=0.,
                                            )
        )

    field_ims_t = da.stack(field_ims_t)
    field_ws_t = da.stack(field_ws_t)

    # field_ims_t = field_ims_t.rechunk((len(views), ) + field_ims_t[0].shape)

    # border_width_px = 30

    # field_weights = da.map_blocks(
    #     lambda x, *args, **kwargs: get_smooth_border_weight_im_from_mask(x[0], *args, **kwargs)[None],
    #     field_ims_t > 0,
    #     dtype=np.float16,
    #     **{'width': border_width_px},
    #     )

    # overlap_per_dim = {dim + 1: np.min([field_ims_t.shape[dim] // 2, (border_width_px + 1)])
    #     for dim in range(ndim)}
    # overlap_per_dim[0] = 0

    # # use overlap to ensure proper border weights
    # field_weights = da.map_overlap(
    #     lambda x, *args, **kwargs: get_smooth_border_weight_im_from_mask(x[0], *args, **kwargs)[None],
    #     field_ims_t,
    #     dtype=np.float16,
    #     depth=overlap_per_dim,
    #     trim=True,
    #     boundary='none',
    #     **{'width': border_width_px},
    #     )

    wsum = da.sum(field_ws_t, axis=0)
    wsum[wsum==0] = 1

    field_ws_t = field_ws_t / wsum

    fused_field = da.sum(field_ims_t * field_ws_t, axis=0)

    fused_field = fused_field - 1  # subtract 1 because of earlier addition

    fill_empty_spaces_from_neighbouring_pixels = True
    if fill_empty_spaces_from_neighbouring_pixels:

        # find empty spaces
        empty_mask = fused_field < 0
        # empty_coords = delayed(lambda x: np.array(np.where(x)))(empty_mask)
        # empty_coords = da.where(empty_mask)

        # convert to smaller dtype
        fused_field = fused_field.astype(input_dtype)

        # empty_intensities = da.from_delayed(delayed(ndimage.map_coordinates)(fused_field, empty_coords, order=0),
        #                                     shape=(np.nan, ),
        #                                     dtype=input_dtype)

        # fused_field[(empty_mask,)] = empty_intensities
        # fused_field[empty_mask] = empty_intensities

        # def fill(field, coords, intensities):
        #     field = np.copy(field)
        #     # field[(coords,)] = intensities
        #     field[tuple(coords)] = intensities
        #     return field
        
        # print('OKOKOKOK', fused_field.shape)
        
        # fused_field = da.from_delayed(delayed(fill)(fused_field, empty_coords, empty_intensities),
        #                               shape=fused_field.shape,
        #                               dtype=fused_field.dtype)

        fused_field = da.from_delayed(delayed(interpolate_missing_pixels)(
                                fused_field, empty_mask),
                            shape=fused_field.shape,
                            dtype=fused_field.dtype)

    else:
        fused_field = fused_field * (fused_field >= 0)
        fused_field = fused_field.astype(input_dtype)


    # fused_field = fused_field.astype(input_dtype)

    # fill empty space with nearest neighbor
    # fused_field
    # fused_field = ndinterp.map_coordinates(fused_field, np.indices(fused_field.shape), order=0)

    return fused_field


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
    dim_dists = [ndimage.distance_transform_edt(
                    ndimage.binary_erosion(
                        np.ones(shape[dim]).astype(bool)))
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

    # # get min of weights for each dim
    # ws = []
    # for dim in range(len(shape)):
    #     tmp_dim_w = dim_ws[dim]
    #     for _ in range(ndim - dim - 1):
    #         tmp_dim_w = tmp_dim_w[:, None]
    #     w *= tmp_dim_w
    #     ws.append(w)
    # w = np.min(ws, axis=0)

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

    filename = "/Users/malbert/software/napari-stitcher/image-datasets/MAX_LSM900.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/yu_220829_WT_quail_st6_x10_zoom0.7_1x3_488ZO1-568Sox2-647Tbra.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/04_stretch-01_AcquisitionBlock2_pt2.czi"

    from napari_stitcher import _utils
    from mvregfus import io_utils, mv_utils

    view_dict = io_utils.build_view_dict_from_multitile_czi(filename, max_project=False)
    views = np.array([view for view in sorted(view_dict.keys())])
    pairs = mv_utils.get_registration_pairs_from_view_dict(view_dict)

    viewims = _utils.load_tiles(view_dict, [0],
                    [0], max_project=False)
    
    params = {0: {view: mv_utils.matrix_to_params(np.eye(len(view_dict[0]['origin'])+1)) for view in views}}

    fused_da, fusion_stack_props_d, field_stack_props_d = \
        fuse_tiles(viewims, params, view_dict)

    fused = fused_da.compute()

    # import tifffile
    # tifffile.imwrite('delme1.tif', fused.astype(np.float32))

    # for view in views:
    #     tifffile.imwrite('delme_%s.tif' % view, viewims[0][0][view].compute().astype(np.float32))
