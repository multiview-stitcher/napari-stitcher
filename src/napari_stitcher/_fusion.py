import numpy as np

import dask.array as da
from dask import delayed

from mvregfus.mv_utils import get_sigmoidal_border_weights_ndim_only_one
from mvregfus.multiview import calc_stack_properties_from_views_and_params
from mvregfus.mv_utils import params_to_matrix

from scipy import ndimage
from dask_image import ndinterp


def combine_stack_props(stack_props_list):

    combined_stack_props = {}
    combined_stack_props['origin'] = np.min([sp['origin'] for sp in stack_props_list], axis=0)
    combined_stack_props['spacing'] = np.min([sp['spacing'] for sp in stack_props_list], axis=0)
    combined_stack_props['size'] = np.max([np.ceil((sp['origin'] + sp['size'] * sp['spacing']\
                                    - combined_stack_props['origin']) / combined_stack_props['spacing'])
                                    for sp in stack_props_list], axis=0).astype(np.uint16)
    # import pdb; pdb.set_trace()
    return combined_stack_props


def fuse_tiles(viewims: dict,
               params: dict,
               view_dict: dict,
):

    input_channels = sorted(viewims.keys())
    input_times = sorted(viewims[input_channels[0]].keys())
    views = sorted(viewims[input_channels[0]][input_times[0]].keys())

    for view in views:
        view_dict[view]['size'] = view_dict[view]['shape']

    # field_stack_props = [delayed(calc_stack_properties_from_views_and_params)(
    #             [view_dict[view] for view in views],
    #             params[t],
    #             view_dict[views[0]]['spacing'],
    #             mode='union',
    #         )
    #     for t in input_times]

    # fusion_stack_props = delayed(combine_stack_props)(field_stack_props)

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
                # shape=tuple(viewims[ch][t][views[0]].shape),
                shape=tuple(fusion_stack_props['size']),
                dtype=np.uint16,
                )
            for t in input_times])
        for ch in input_channels])

    return fused_da, fusion_stack_props, field_stack_props


def fuse_field(field_ims, params, view_dict, out_stack_props):
    """
    fuse tiles from single timepoint and channel
    """

    # return field_ims[0]

    views = sorted(field_ims.keys())

    ndim = field_ims[0].ndim
    field_ims_t = []
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
                                            field_ims[view] + 1, # add 1 so that 0 indicates no data
                                            matrix=matrix_prime,
                                            offset=offset_prime,
                                            order=1,
                                            output_shape=tuple(out_stack_props['size']),
                                            output_chunks=tuple([600 for _ in out_stack_props['size']]),
                                            # output_chunks=tuple(out_stack_props['size']),
                                            # output_chunks=tuple(out_stack_props['size']),
                                            mode='constant',
                                            cval=0.,
                                            )
        )

    field_ims_t = da.stack(field_ims_t)

    # field_ims_t = field_ims_t.rechunk((len(views), ) + field_ims_t[0].shape)

    border_width_px = 30

    # field_weights = da.map_blocks(
    #     lambda x, *args, **kwargs: get_smooth_border_weight_im_from_mask(x[0], *args, **kwargs)[None],
    #     field_ims_t > 0,
    #     dtype=np.float16,
    #     **{'width': border_width_px},
    #     )

    overlap_per_dim = {dim + 1: np.min([field_ims_t.shape[dim] // 2, (border_width_px + 1)])
        for dim in range(ndim)}
    overlap_per_dim[0] = 0

    # use overlap to ensure proper border weights
    field_weights = da.map_overlap(
        lambda x, *args, **kwargs: get_smooth_border_weight_im_from_mask(x[0], *args, **kwargs)[None],
        field_ims_t,
        dtype=np.float16,
        depth=overlap_per_dim,
        trim=True,
        boundary='none',
        **{'width': border_width_px},
        )

    wsum = da.sum(field_weights, axis=0)
    wsum[wsum==0] = 1

    field_weights = field_weights / wsum

    import tifffile
    tifffile.imwrite('delme.tif', field_weights.compute().astype(np.float32))

    import pdb; pdb.set_trace()

    # fused_field = da.sum(field_ims_t * field_weights, axis=0)
    fused_field = da.sum(field_ims_t * field_weights, axis=0)

    fused_field = fused_field - 1  # subtract 1 because of earlier addition

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


def get_smooth_border_weight_im_from_mask(mask, width=10):
    """
    Get a weight image for blending from a mask image.
    """

    # width is the total width of the transition
    # weights are at 0.5 at width/2 from the border

    if np.all(mask):
        return np.ones(mask.shape, dtype=np.float16)
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.float16)

    b = mask>0
    # distance transform is zero at the border, so we need to dilate to distinguish the border from the interior
    # b = ndimage.binary_erosion(b)
    b = ndimage.binary_dilation(b)
    a = 1*width
    b2 = ndimage.binary_erosion(b, iterations=a)
    b3 = ndimage.binary_erosion(b2, iterations=a)
    b = b ^ b3
    dist = ndimage.distance_transform_edt(b)
    # dist = dist + 1 * b
    dist[b2] = a

    # import tifffile
    # tifffile.imwrite('dist.tif', dist.astype(np.float32))

    # import tifffile
    # tifffile.imwrite('b.tif', b.astype(np.float32))
    # tifffile.imwrite('b2.tif', b.astype(np.float32))
    # tifffile.imwrite('b3.tif', b.astype(np.float32))

    # sigmoid
    # w = 1 / (1 + np.exp(-(dist-width)/(width/7)))

    w = smooth_transition(dist, x_offset=width / 2.)

    return w.astype(np.float16)


if __name__ == "__main__":

    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi"
    filename = "/Users/malbert/software/napari-stitcher/image-datasets/yu_220829_WT_quail_st6_x10_zoom0.7_1x3_488ZO1-568Sox2-647Tbra.czi"

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

    import tifffile
    tifffile.imwrite('delme1.tif', fused.astype(np.float32))

    # for view in views:
    #     tifffile.imwrite('delme_%s.tif' % view, viewims[0][0][view].compute().astype(np.float32))