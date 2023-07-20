import numpy as np
import xarray as xr
from scipy import ndimage
from tqdm import tqdm
import networkx as nx

from dask import delayed, compute
import dask.array as da

import skimage.registration
import skimage.exposure

from napari_stitcher import _spatial_image_utils, _mv_graph, _utils


def apply_recursive_dict(func, d):
    res = {}
    if isinstance(d, dict):
        for k, v in d.items():
            res[k] = apply_recursive_dict(func, v)
    else:
        return func(d)
    return res


def get_optimal_registration_binning(xim1, xim2, max_total_pixels_per_stack=(400)**3):
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

    _, overlap_coords = _mv_graph.get_overlap_between_pair_of_xims(xim1, xim2)
    # overlap = overlap_coords[1] - overlap_coords[0]
    overlap = {dim: overlap_coords[1][dim] - overlap_coords[0][dim] for dim in spatial_dims}

    # bin coordinate with largest spacing until we're below the threshold
    # while max([np.product([o / s / b for o, s, b in zip(overlap, spacings[ixim], registration_binning)])
    #            for ixim in range(2)]) >= max_total_pixels_per_stack:
        
    while max([np.product([overlap[dim] / spacings[ixim][dim] / registration_binning[dim]
                           for dim in spatial_dims])
               for ixim in range(2)]) >= max_total_pixels_per_stack:

        # dim_to_bin = np.argmin(np.min(np.array(spacings), axis=0))
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
                   use_only_overlap_region=True,
                   ) -> dict:
    """
    Register input spatial images. Assume there's no C and T.

    Return: Transform in homogeneous coordinates
    """

    # xim1, xim2 = [xim.sel(t=xim.coords['t'][0], C=xim.coords['c'][0]) for xim in [xim1, xim2]]

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xim1) 
    spacing = _spatial_image_utils.get_spacing_from_xim(xim1, asarray=True)
    ndim = len(spatial_dims)

    overlap_area, coords = _mv_graph.get_overlap_between_pair_of_xims(
        xim1, xim2, transform_key=transform_key)

    overlap_xims = [xim.sel({
        dim: slice(coords[0][dim], coords[1][dim])
        # dim: (xim.coords[dim] > coords[ixim][idim]) *  (xim.coords[dim] > coords[ixim][idim])
            for dim in spatial_dims})
                for xim in [xim1, xim2]]

    if registration_binning is None:
        registration_binning = get_optimal_registration_binning(overlap_xims[0], overlap_xims[1])

    if registration_binning is not None or min(registration_binning) > 1:
        overlap_xims_b = [xim.coarsen(registration_binning,
                                boundary="trim").mean().astype(xim.dtype) for xim in overlap_xims]
    else:
        overlap_xims_b = overlap_xims

    for i in range(2):
        overlap_xims_b[i].data = da.from_delayed(
            delayed(skimage.exposure.equalize_adapthist)(
                overlap_xims_b[i].data, kernel_size=10, clip_limit=0.02, nbins=2 ** 13),
            shape=overlap_xims_b[i].shape, dtype=float)

    # trim to strictly the same shape
    # (seems there can be a 1 pixel difference)
    reg_shape = np.min([[xim.shape[idim] for idim in range(ndim)]
                         for xim in overlap_xims_b], 0)

    overlap_xims_b = [xim[tuple([slice(reg_shape[idim]) for idim in range(ndim)])]
                         for xim in overlap_xims_b]

    param = da.from_delayed(delayed(skimage.registration.phase_cross_correlation)(
            overlap_xims_b[0].data,
            overlap_xims_b[1].data,
            upsample_factor=10,
            normalization=None)[0], shape=(ndim, ), dtype=float)

    param = - param * spacing * np.array([registration_binning[dim] for dim in spatial_dims])

    param = _utils.shift_to_matrix(param)

    param = get_xparam_from_param(param)

    return param


def register_pair_of_xims(xim1, xim2,
                          registration_binning=None,
                          transform_key=None):
    """
    Register over time.
    """

    xim1 = _spatial_image_utils.ensure_time_dim(xim1)
    xim2 = _spatial_image_utils.ensure_time_dim(xim2)
    
    xp = xr.concat([register_pair_of_spatial_images(
            xim1.sel(t=t),
            xim2.sel(t=t),
            transform_key=transform_key,
            registration_binning=registration_binning)
                for t in xim1.coords['t'].values], dim='t')
    
    xp = xp.assign_coords({'t': xim1.coords['t'].values})
    
    return xp


def get_registration_graph_from_overlap_graph(
        g, registration_binning = None,
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
    # reg_pairs = []

    for n, sp in paths.items():
        g_reg.nodes[n]['reg_path'] = sp

        if len(sp) < 2: continue

        # add registration edges
        for i in range(len(sp) - 1):
            pair = (sp[i], sp[i + 1])

            g_reg.edges[(pair[0], pair[1])]['transform'] = \
                (register_pair_of_xims)(
                    g.nodes[pair[0]]['xim'],
                    g.nodes[pair[1]]['xim'],
                    registration_binning=registration_binning,
                    transform_key=transform_key,
                    )
        
    # g_reg.graph['ref_node'] = ref_node
    g_reg.graph['pair_finding_method'] = 'shortest_paths_considering_overlap'

    return g_reg


def identity_transform(ndim):

    params = xr.DataArray(
        np.eye(ndim + 1),
        dims=['x_in', 'x_out'])
    
    return params


def get_node_params_from_reg_graph(g_reg):
    """
    Get final transform parameters by concatenating transforms
    along paths of pairwise affine transformations.
    """

    ndim = len(_spatial_image_utils.get_spatial_dims_from_xim(
        g_reg.nodes[list(g_reg.nodes)[0]]['xim']))

    # final_params = []
    for n in g_reg.nodes:

        reg_path = g_reg.nodes[n]['reg_path']

        path_pairs = [[reg_path[i], reg_path[i+1]]
                      for i in range(len(reg_path) - 1)]
        
        # path_params = identity_transform(ndim)

        if 't' in g_reg.nodes[n]['xim'].dims:
            path_params = xr.DataArray([np.eye(ndim + 1) for t in g_reg.nodes[n]['xim'].coords['t']],
                                        dims=['t', 'x_in', 'x_out'],
                                        coords={'t': g_reg.nodes[n]['xim'].coords['t']})
        else:
            path_params = identity_transform(ndim)
        
        for pair in path_pairs:
            path_params = xr.apply_ufunc(np.matmul,
                                         g_reg.edges[(pair[0], pair[1])]['transform'],
                                         path_params,
                                         input_core_dims=[['x_in', 'x_out']]*2,
                                         output_core_dims=[['x_in', 'x_out']],
                                         vectorize=True)
            # path_params = da.from_delayed(
            #     delayed(xr.apply_ufunc)(
            #         np.matmul,
            #         g_reg.edges[(pair[0], pair[1])]['transform'],
            #         path_params,
            #         input_core_dims=[['x_in', 'x_out']]*2,
            #         output_core_dims=[['x_in', 'x_out']],
            #         vectorize=True)
                
            # )
        
        g_reg.nodes[n]['transforms'] = path_params

    return g_reg


def register(xims, reg_channel_index=0, transform_key=None):

    xims = [xim.sel(c=xim.coords['c'][reg_channel_index]) for xim in xims]

    g = _mv_graph.build_view_adjacency_graph_from_xims(xims, transform_key=transform_key)

    g_reg = get_registration_graph_from_overlap_graph(g)

    if not len(g_reg.edges):
        raise(Exception('No overlap between views for stitching. Consider stabilizing the tiles instead.'))

    g_reg_computed = _mv_graph.compute_graph_edges(g_reg, scheduler='threads')

    g_reg_nodes = get_node_params_from_reg_graph(g_reg_computed)

    node_transforms = _mv_graph.get_nodes_dataset_from_graph(g_reg_nodes, node_attribute='transforms')
    
    return [node_transforms[dv] for dv in node_transforms.data_vars]


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

    # ndim = len(spatial_dims)

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
            upsample_factor=10,
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


if __name__ == "__main__":

    from napari_stitcher import _reader, _spatial_image_utils
    import xarray as xr
    import dask.array as da
    filename = "/Users/malbert/software/napari-stitcher/image-datasets/mosaic_test.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20210216_highres_TR2.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20230223_02_before_ablation-02_20X_max.czi"

    xims = _reader.read_mosaic_image_into_list_of_spatial_xarrays(filename, scene_index=0)
