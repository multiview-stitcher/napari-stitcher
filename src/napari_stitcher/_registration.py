import numpy as np
import xarray as xr
from scipy import ndimage
from tqdm import tqdm
import networkx as nx

from dask import delayed, compute
import dask.array as da

import skimage.registration
import skimage.exposure

from napari_stitcher import _spatial_image_utils, _mv_graph, _utils, _transformation


def apply_recursive_dict(func, d):
    res = {}
    if isinstance(d, dict):
        for k, v in d.items():
            res[k] = apply_recursive_dict(func, v)
    else:
        return func(d)
    return res


def get_optimal_registration_binning(
        xim1, xim2,
        max_total_pixels_per_stack=(400)**3,
        use_only_overlap_region=False,
        ):
    """
    Heuristic to find good registration binning.

    - assume inputs contain only spatial dimensions.
    - assume inputs have same spacing
    - assume x and y have same spacing
    - so far, assume orthogonal axes

    Ideas:
      - sample physical space homogeneously
      - don't occupy too much memory

    """

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xim1)
    ndim = len(spatial_dims)
    spacings = [_spatial_image_utils.get_spacing_from_xim(xim, asarray=False)
                for xim in [xim1, xim2]]

    registration_binning = {dim: 1 for dim in spatial_dims}

    if use_only_overlap_region:
        raise(NotImplementedError("use_only_overlap_region"))

        # _, overlap_structure = _mv_graph.get_overlap_between_pair_of_xims(xim1, xim2)
        # # overlap = overlap_coords[1] - overlap_coords[0]
        # overlap = {dim: overlap_coords[1][dim] - overlap_coords[0][dim] for dim in spatial_dims}

    overlap = {dim: max(xim1.shape[idim], xim2.shape[idim])
               for idim, dim in enumerate(spatial_dims)}
        
    while max([np.product([overlap[dim] / spacings[ixim][dim] / registration_binning[dim]
                           for dim in spatial_dims])
               for ixim in range(2)]) >= max_total_pixels_per_stack:

        dim_to_bin = np.argmin([min([spacings[ixim][dim] for ixim in range(2)]) for dim in spatial_dims])

        if ndim == 3 and dim_to_bin == 0:
            registration_binning['z'] = registration_binning['z'] * 2
        else:
            for dim in ['x', 'y']:
                registration_binning[dim] = registration_binning[dim] * 2

        spacings = [{dim: spacings[ixim][dim] * registration_binning[dim]
                    for dim in spatial_dims} for ixim in range(2)]

    return registration_binning


def register_pair_of_spatial_images(
                   xim1, xim2,
                   transform_key=None,
                   registration_binning=None,
                   use_only_overlap_region=False,
                   ) -> dict:
    """
    Register input spatial images. Assume there's no C and T.

    Return: Transform in homogeneous coordinates
    """

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xim1) 
    spacing = _spatial_image_utils.get_spacing_from_xim(xim1, asarray=True)
    ndim = len(spatial_dims)

    if use_only_overlap_region:
        overlap_area, coords = _mv_graph.get_overlap_between_pair_of_xims(
            xim1, xim2, transform_key=transform_key)

        overlap_xims = [xim.sel({
            dim: slice(coords[0][dim], coords[1][dim])
                for dim in spatial_dims})
                    for xim in [xim1, xim2]]
        reg_xims = overlap_xims
    else:
        reg_xims = [xim1, xim2]

    if registration_binning is None:
        registration_binning = get_optimal_registration_binning(reg_xims[0], reg_xims[1])

    if registration_binning is not None and min(registration_binning.values()) > 1:
        reg_xims_b = [xim.coarsen(registration_binning,
                                boundary="trim").mean().astype(xim.dtype) for xim in reg_xims]
    else:
        reg_xims_b = reg_xims

    # # CLAHE
    # for i in range(2):
    #     # reg_xims_b[i].data = da.from_delayed(
    #     #     delayed(skimage.exposure.equalize_adapthist)(
    #     #         reg_xims_b[i].data, kernel_size=10, clip_limit=0.02, nbins=2 ** 13),
    #     #     shape=reg_xims_b[i].shape, dtype=float)

    #     reg_xims_b[i].data = da.map_overlap(
    #         skimage.exposure.equalize_adapthist,
    #         reg_xims_b[i].data,
    #         kernel_size=10,
    #         clip_limit=0.02,
    #         nbins=2 ** 13,
    #         depth={idim: 10 for idim, k in enumerate(spatial_dims)},
    #         dtype=float
    #         )

    # get images into the same physical space (that of xim1)

    ndim = _spatial_image_utils.get_ndim_from_xim(reg_xims_b[0])
    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(reg_xims_b[0])
    affines = [xim.attrs['transforms'][transform_key].squeeze().data for xim in reg_xims_b]
    transf_affine = np.matmul(np.linalg.inv(affines[1]), affines[0])

    corners = np.concatenate([_mv_graph.get_faces_from_xim(xim, transform_key=transform_key).reshape(-1, ndim)
                            for xim in reg_xims_b], axis=0)

    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=corners.dtype)))
    corners_xim1_phys = np.matmul(np.linalg.inv(affines[0]), corners.T).T[:,:ndim]

    lower, upper = np.min(corners_xim1_phys, axis=0), np.max(corners_xim1_phys, axis=0)

    spacing = _spatial_image_utils.get_spacing_from_xim(reg_xims_b[0], asarray=True)

    reg_xims_b_t = [_transformation.transform_xim(
        xim,
        [None, transf_affine][ixim],
        output_origin=lower,
        output_spacing=spacing,
        output_shape=np.ceil(np.array(upper-lower) / spacing).astype(np.uint16),
    ) for ixim, xim in enumerate(reg_xims_b)]

    # # trim to strictly the same shape
    # # (seems there can be a 1 pixel difference)
    # reg_shape = np.min([[xim.shape[idim] for idim in range(ndim)]
    #                     for xim in overlap_xims_b], 0)

    # overlap_xims_b = [xim[tuple([slice(reg_shape[idim]) for idim in range(ndim)])]
    #                     for xim in overlap_xims_b]

    shift_pc = da.from_delayed(
        delayed(np.array)(
        delayed(skimage.registration.phase_cross_correlation)(
            delayed(lambda x: np.array(x))(reg_xims_b_t[0].data),
            delayed(lambda x: np.array(x))(reg_xims_b_t[1].data),
            upsample_factor=(10 if ndim==2 else 2),
            disambiguate=True,
            normalization='phase',
            )[0]), shape=(ndim, ), dtype=float)

    shift = - shift_pc * spacing

    displ_endpts = [np.zeros(ndim), shift]
    pt_world = [np.dot(affines[0], np.concatenate([pt, np.ones(1)]))[:ndim] for pt in displ_endpts]
    displ_world = pt_world[1] - pt_world[0]

    param = _utils.shift_to_matrix(-displ_world)

    xparam = get_xparam_from_param(param)

    return xparam


def register_pair_of_xims_over_time(xim1, xim2,
                          registration_binning=None,
                          transform_key=None):
    """
    Register over time.
    """

    xim1 = _spatial_image_utils.ensure_time_dim(xim1)
    xim2 = _spatial_image_utils.ensure_time_dim(xim2)
    
    xp = xr.concat([register_pair_of_spatial_images(
            _spatial_image_utils.xim_sel_coords(xim1, {'t': t}),
            _spatial_image_utils.xim_sel_coords(xim2, {'t': t}),
            transform_key=transform_key,
            registration_binning=registration_binning)
                for t in xim1.coords['t'].values], dim='t')
    
    xp = xp.assign_coords({'t': xim1.coords['t'].values})
    
    return xp


def get_registration_graph_from_overlap_graph(
        g,
        registration_binning = None,
        transform_key=None,
        ):

    g_reg = g.to_directed()

    ref_node = _mv_graph.get_node_with_maximal_overlap_from_graph(g)

    # invert overlap to use as weight in shortest path
    for e in g_reg.edges:
        g_reg.edges[e]['overlap_inv'] = 1 / (g_reg.edges[e]['overlap'] + 1) # overlap can be zero

    # get shortest paths to ref_node
    paths = nx.shortest_path(g_reg, source=ref_node, weight='overlap_inv')

    # get all pairs of views that are connected by a shortest path
    for n, sp in paths.items():
        g_reg.nodes[n]['reg_path'] = sp

        if len(sp) < 2: continue

        # add registration edges
        for i in range(len(sp) - 1):
            pair = (sp[i], sp[i + 1])

            g_reg.edges[(pair[0], pair[1])]['transform'] = \
                (register_pair_of_xims_over_time)(
                    g.nodes[pair[0]]['xim'],
                    g.nodes[pair[1]]['xim'],
                    registration_binning=registration_binning,
                    transform_key=transform_key,
                    )
        
    g_reg.graph['pair_finding_method'] = 'shortest_paths_considering_overlap'

    return g_reg


def get_node_params_from_reg_graph(g_reg):
    """
    Get final transform parameters by concatenating transforms
    along paths of pairwise affine transformations.
    """

    ndim = len(_spatial_image_utils.get_spatial_dims_from_xim(
        g_reg.nodes[list(g_reg.nodes)[0]]['xim']))

    for n in g_reg.nodes:

        reg_path = g_reg.nodes[n]['reg_path']

        path_pairs = [[reg_path[i], reg_path[i+1]]
                      for i in range(len(reg_path) - 1)]
        
        if 't' in g_reg.nodes[n]['xim'].dims:
            path_params = xr.DataArray([np.eye(ndim + 1) for t in g_reg.nodes[n]['xim'].coords['t']],
                                        dims=['t', 'x_in', 'x_out'],
                                        coords={'t': g_reg.nodes[n]['xim'].coords['t']})
        else:
            path_params = _spatial_image_utils.identity_transform(ndim)
        
        for pair in path_pairs:
            path_params = xr.apply_ufunc(np.matmul,
                                         g_reg.edges[(pair[0], pair[1])]['transform'],
                                         path_params,
                                         input_core_dims=[['x_in', 'x_out']]*2,
                                         output_core_dims=[['x_in', 'x_out']],
                                         vectorize=True)
        
        g_reg.nodes[n]['transforms'] = path_params

    return g_reg


def register(xims, reg_channel_index=None, transform_key=None, registration_binning=None):

    if reg_channel_index is None:
        if xims[0].coords['c'].ndim > 0:
            raise(Exception('Please choose a registration channel.'))
    else:
        xims = [xim.sel(c=xim.coords['c'][reg_channel_index]) for xim in xims]

    g = _mv_graph.build_view_adjacency_graph_from_xims(xims, transform_key=transform_key)

    g_reg = get_registration_graph_from_overlap_graph(g, registration_binning=registration_binning, transform_key=transform_key)

    if not len(g_reg.edges):
        raise(Exception('No overlap between views for stitching. Consider stabilizing the tiles instead.'))

    g_reg_computed = _mv_graph.compute_graph_edges(g_reg)#, scheduler='threads')

    g_reg_nodes = get_node_params_from_reg_graph(g_reg_computed)

    node_transforms = _mv_graph.get_nodes_dataset_from_graph(g_reg_nodes, node_attribute='transforms')
    params = [node_transforms[dv] for dv in node_transforms.data_vars]

    return params


def stabilize(xims, reg_channel_index=0, sigma=2):

    xims = [xim.sel(c=xim.coords['c'][reg_channel_index]) for xim in xims]

    if len(xims[0].coords['t']) < 8:
        raise(Exception('Need at least 8 time points to perform stabilization.'))
    
    params = [get_stabilization_parameters_from_xim(xim, sigma=sigma) for xim in xims]

    params = compute(params)[0]

    return params


def correct_random_drift(ims, reg_ch=0, zoom_factor=10, particle_reinstantiation_stepsize=30, sigma_t=3):
    """
    ## Stage shift correction (currently 2d, but easily extendable)

    Goal: Correct for random stage shifts in timelapse movies in the absence of reference points that are fixed relative to the stage.

    Method: Assume that in the absence of random shifts particle trajectories are smooth in time. Find raw (uncorrected) trajectories, smooth them over time and determine the random stage shifts as the mean deviation of the actual trajectories from the smoothed ones.

    Steps:
    1) Calculate PIV fields (or optical flow) for the timelapse movie
    2) Track the initial coordinates as virtual particles throughout the timelapse
    3) Smooth trajectories and determine the mean deviations (N_t, d_x, d_y) from the actual trajectories
    4) Use the obtained deviations to correct the timelapse and verify result quality visually

    Comments:
    - assumes input data to be in the format (t, c, y, x)
    """

    import dask.delayed as d
    import dask
    from dask.diagnostics import ProgressBar

    # for some reason float images give a different output
    ims = ims.astype(np.uint16)

    of_channel = reg_ch
    regims = np.array([ndimage.zoom(ims[it, of_channel], 1./zoom_factor, order=1) for it in range(len(ims[:]))])

    fs = [d(skimage.registration.optical_flow_tvl1)(regims[t],
                                                    regims[t+1],
                                                attachment=10000,
                                                )
                for t in tqdm(range(len(ims) - 1)[:])]

    print('Computing optical flow...')
    with ProgressBar():
        fs=np.array(dask.compute(fs)[0])#*zoom_factor

    print('Tracking virtual particles...')

    # coordinates to be tracked
    x,y = np.mgrid[0:regims[0].shape[0], 0:regims[0].shape[1]]

    # define starting point(s) for coordinates to be tracked
    # for long movies, it's good to have several starting tps
    # as the coordinates can move out of the FOV
    
    starting_tps = range(0, len(ims), particle_reinstantiation_stepsize)

    posss = []
    for starting_tp in starting_tps:
        poss = [np.array([x, y])]
        for t in tqdm(range(starting_tp, len(ims) - 1)):
            displacement = np.array([ndimage.map_coordinates(
                                fs[t][dim],
                                np.array([poss[-1][0].flatten(), poss[-1][1].flatten()]),
                                order=1, mode='constant', cval=np.nan)
                            .reshape(x.shape) for dim in range(2)])
            poss.append(displacement + poss[-1])
        posss.append(np.array(poss))    


    print('Smoothing trajectories...')
    posss_smooth = [ndimage.gaussian_filter(posss[istp], [sigma_t, 0, 0, 0], mode='nearest')
                        for istp, starting_tp in enumerate(starting_tps)]

    devs = np.array([np.nanmean([posss_smooth[istp][t-starting_tp] - posss[istp][t-starting_tp]
                        for istp, starting_tp in enumerate(starting_tps) if starting_tp <= t], axis=(0, -1, -2))
                    for t in range(len(ims))])

    devs = devs * zoom_factor

    print('Correct drifts...')
    imst = np.array([[ndimage.affine_transform(ims[t,ch],
                                            matrix=[[1,0],[0,1]],
                                            offset=-devs[t], order=1)
                    for ch in range(ims.shape[1])]
                    for t in tqdm(range(len(ims)))])

    return imst, devs


import dask_image
def get_stabilization_parameters(tl, sigma=2):
    """

    Correct for random stage shifts in timelapse movies in the absence of reference points that are fixed relative to the stage.
    - obtain shifts between consecutive frames
    - get cumulative sum of shifts
    - smooth
    - consider difference between smoothed and unsmoothed shifts as the random stage shifts

    Assume first dimension is time.

    tl: dask array of shape (N_T, ...)
    """

    ndim = tl[0].ndim

    ps = da.stack([da.from_delayed(delayed(skimage.registration.phase_cross_correlation)(
            tl[t-1],
            tl[t],
            upsample_factor=10,
            normalization=None)[0], shape=(ndim, ), dtype=float)
            for t in range(1, tl.shape[0])])

    ps = da.concatenate([da.zeros((1, ndim)), ps], axis=0)

    ps_cum = da.cumsum(ps, axis=0)
    ps_cum_filtered = dask_image.ndfilters.gaussian_filter(ps_cum, [sigma, 0], mode='nearest')
    # deltas = ps_cum_filtered - ps_cum
    deltas = ps_cum - ps_cum_filtered
    deltas = -deltas

    # tl_t = da.stack([da.from_delayed(delayed(ndimage.affine_transform)(tl[t],
    #                                         matrix=np.eye(2),
    #                                         offset=-params[t], order=1), shape=tl[0].shape, dtype=tl[0].dtype)
    #             for t in range(N_t)]).compute()
    
    return deltas


def get_stabilization_parameters_from_xim(xim, sigma=2):

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xim)

    params = get_stabilization_parameters(xim.transpose(*tuple(['t'] + spatial_dims)), sigma=sigma)

    params = [_utils.shift_to_matrix(
        params[it] * _spatial_image_utils.get_spacing_from_xim(xim, asarray=True))
                    for it, t in enumerate(xim.coords['t'])]

    # params = xr.DataArray(params,
    #         dims=['t', 'x_in', 'x_out'],
    #         coords={
    #                 # 'c': xim1.coords['c'],
    #                 't': xim.coords['t'],
    #                 'x_in': np.arange(ndim+1),
    #                 'x_out': np.arange(ndim+1)})
    
    params = xr.concat([get_xparam_from_param(p) for p in params], dim='t')
    params = params.assign_coords({'t': xim.coords['t']})
    
    return params


def get_xparam_from_param(params):
    """
    Homogeneous matrix to xparams
    """

    ndim = params.shape[-1] - 1
    xparam = xr.DataArray(params,
            dims=['x_in', 'x_out'])
    
    return xparam


import dask_image
def get_drift_correction_parameters(tl, sigma=2):
    """
    Assume first dimension is time

    tl: dask array of shape (N_T, ...)
    """

    ndim = tl[0].ndim

    ps = da.stack([da.from_delayed(delayed(skimage.registration.phase_cross_correlation)(
            tl[t-1],
            tl[t],
            upsample_factor=2,
            normalization=None)[0], shape=(ndim, ), dtype=float)
            for t in range(1, tl.shape[0])])

    ps = da.concatenate([da.zeros((1, ndim)), ps], axis=0)

    ps_cum = -da.cumsum(ps, axis=0)

    # ps_cum_filtered = dask_image.ndfilters.gaussian_filter(ps_cum, [sigma, 0], mode='nearest')
    # deltas = ps_cum_filtered - ps_cum

    # tl_t = da.stack([da.from_delayed(delayed(ndimage.affine_transform)(tl[t],
    #                                         matrix=np.eye(2),
    #                                         offset=-params[t], order=1), shape=tl[0].shape, dtype=tl[0].dtype)
    #             for t in range(N_t)]).compute()

    return ps_cum


def crop_xim_to_references(
    xim_input_to_crop,
    reference_xims,
    transform_key_input,
    transform_keys_reference,
    input_time_index=0,
):
    """
    Crop input image to the minimal region fully covering the reference xim(s).
    """

    ref_corners_world = []
    for irefxim, reference_xim in enumerate(reference_xims):
        ref_corners_world += list(
            np.unique(_mv_graph.get_faces_from_xim(
                reference_xim,
                transform_key=transform_keys_reference[irefxim]).reshape((-1, 2)), axis=0))
        
    input_affine = _spatial_image_utils.get_affine_from_xim(xim_input_to_crop, transform_key=transform_key_input)

    if 't' in input_affine.dims:
        input_affine = input_affine.sel({'t': input_affine.coords['t'][input_time_index]})

    input_affine_inv = np.linalg.inv(np.array(input_affine))
    ref_corners_input_dataspace = _transformation.transform_pts(ref_corners_world, np.array(input_affine_inv))
    ref_corners_input_dataspace

    lower, upper = [func(ref_corners_input_dataspace, axis=0) for func in [np.min, np.max]]

    sdims = _spatial_image_utils.get_spatial_dims_from_xim(xim_input_to_crop)

    xim_cropped = _spatial_image_utils.xim_sel_coords(
        xim_input_to_crop,
        {dim: (xim_input_to_crop.coords[dim] > lower[idim]) * (xim_input_to_crop.coords[dim] < upper[idim])
         for idim, dim in enumerate(sdims)
        })
    
    return xim_cropped


if __name__ == "__main__":

    from napari_stitcher import _reader, _spatial_image_utils
    import xarray as xr
    import dask.array as da
    filename = "/Users/malbert/software/napari-stitcher/image-datasets/mosaic_test.czi"

    xims = _reader.read_mosaic_image_into_list_of_spatial_xarrays(filename, scene_index=0)
