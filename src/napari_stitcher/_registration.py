import numpy as np
import xarray as xr
from scipy import ndimage
import skimage
from tqdm import tqdm
import networkx as nx

from dask import delayed
import dask.array as da

from mvregfus import mv_utils, multiview
from mvregfus.image_array import ImageArray

from napari_stitcher import _spatial_image_utils, _mv_graph


def apply_recursive_dict(func, d):
    res = {}
    if isinstance(d, dict):
        for k, v in d.items():
            res[k] = apply_recursive_dict(func, v)
    else:
        return func(d)
    return res


def bin_views(view_ims, registration_binning):
        return apply_recursive_dict(
            lambda x: delayed(mv_utils.bin_stack)(x, registration_binning),
            view_ims)


def bin_xim(xim, registration_binning):
    return xr.apply_ufunc(mv_utils.bin_stack, xim, registration_binning,
                   input_core_dims=_spatial_image_utils.get_spatial_dims_from_xim(xim),
                   output_core_dims=_spatial_image_utils.get_spatial_dims_from_xim(xim),
                   )


def register_pair_of_spatial_images(
                   xims: list,
                   registration_binning = None,
                   ) -> dict:
    """
    Register input spatial images. Assume there's no C and T.

    Return: Transform in homogeneous coordinates
    """

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xims[0])

    if registration_binning is not None:
        rxims = [xim.coarsen({dim: registration_binning[dim] for dim in spatial_dims},
                             boundary="trim").mean().astype(xim.dtype) for xim in xims]
    else:
        rxims = xims
        
    ims = []
    for rxim in rxims:
        origin = _spatial_image_utils.get_origin_from_xim(rxim)
        spacing = _spatial_image_utils.get_spacing_from_xim(rxim)
        ims.append(
            # delayed(ImageArray)(rxim.data,
            (ImageArray)(rxim.squeeze().data,

                       origin=[origin[dim] for dim in spatial_dims],
                       spacing=[spacing[dim] for dim in spatial_dims])
        )

    p = (multiview.register_linear_elastix)(
            ims[0], ims[1],
            None,
            '',
            f'',
            None,
            )
    
    p = (mv_utils.params_to_matrix)(p)

    return p


def register_pair_of_xims(xim1,
                            xim2,
                            registration_binning=None):
    """
    Register over time.
    """

    xim1 = _spatial_image_utils.ensure_time_dim(xim1)
    xim2 = _spatial_image_utils.ensure_time_dim(xim2)

    # spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xim1)
    ndim = len(_spatial_image_utils.get_spatial_dims_from_xim(xim1))
    
    xp = xr.DataArray(da.stack([
        da.from_delayed(delayed(register_pair_of_spatial_images)
            (xims=[xim1.sel(T=t), xim2.sel(T=t)], registration_binning=registration_binning),
                shape=(ndim+1, ndim+1), dtype=float)
                for t in xim1.coords['T']]),
            # dims=['C', 'T', 'x_in', 'x_out'],
            dims=['T', 'x_in', 'x_out'],
            coords={
                    # 'C': xim1.coords['C'],
                    'T': xim1.coords['T'],
                    'x_in': np.arange(ndim+1),
                    'x_out': np.arange(ndim+1)})
    
    return xp


# def get_registration_pair_graph(
#         g,
#         # method=,
#         registration_binning = None,
# ):
#     """
#         - determine a reference_view and 
#         - get shortest paths between all views and the reference view (deambiguate with largest overlap)
#         - return directed graph of xims and registration edges with (delayed) transforms
#     """
#     g_reg = g.to_directed() # returns a deep copy


def get_registration_graph_from_overlap_graph(g,
                                            registration_binning = None,
                                              ):

    g_reg = g.to_directed()

    ref_node = _mv_graph.get_node_with_maximal_overlap_from_graph(g)

    # invert overlap to use as weight in shortest path
    for e in g_reg.edges:
        g_reg.edges[e]['overlap_inv'] = 1 / g_reg.edges[e]['overlap']

    # get shortest paths to ref_node
    paths = nx.shortest_path(g_reg, source=ref_node, weight='overlap_inv')

    # get all pairs of views that are connected by a shortest path
    # reg_pairs = []
    # import pdb; pdb.set_trace()

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

    # g = networkx.DiGraph()
    # for ipair,pair in enumerate(pairs):
    #     # g.add_edge(pair[0],pair[1],{'p': params[ipair]})

    #     # import pdb; pdb.set_trace()
    #     if consider_reg_quality and views is not None:
    #         from scipy import stats
    #         imf = views[view_indices[pair[0]]] + 1
    #         imm = views[view_indices[pair[1]]] + 1
    #         immt = transform_stack_sitk(imm, params[ipair],
    #                                     out_origin=imf.origin,
    #                                     out_spacing=imf.spacing,
    #                                     out_shape=imf.shape,
    #                                     )
    #         mask = (imf > 0) * (immt > 0)
    #         weight = 5 - stats.spearmanr(imf[mask], immt[mask]).correlation
    #         print('weights: ', pair, weight, params[ipair])
    #         # if pair[0] == 0 and pair[1] == 7: import pdb; pdb.set_trace()
    #     else:
    #         weight = 1

        # g.add_edge(pair[0],pair[1], p = params[ipair], weight=weight) # after update 201809 networkx seems to have changed
        # g.add_edge(pair[1], pair[0], p = invert_params(params[ipair]), weight=weight) # after update 201809 networkx seems to have changed

    # all_views = np.unique(np.array(pairs).flatten())
    # views_to_transform = np.sort(np.array(list(set(all_views).difference(set([ref_view])))))

    # ref_view = g_pairs.graph['ref_node']

    ndim = len(_spatial_image_utils.get_spatial_dims_from_xim(
        g_reg.nodes[list(g_reg.nodes)[0]]['xim']))

    # final_params = []
    for n in g_reg.nodes:

        reg_path = g_reg.nodes[n]['reg_path']

        path_pairs = [[reg_path[i], reg_path[i+1]]
                      for i in range(len(reg_path) - 1)]
        
        path_params = identity_transform(ndim)

        # path_params = xr.DataArray([np.eye(ndim + 1) for t in g_reg.nodes[n]['xim'].coords['T']],
        #                             dims=['T', 'x_in', 'x_out'],
        #                             coords={'T': g_reg.nodes[n]['xim'].coords['T']})
        
        for pair in path_pairs:
            path_params = xr.apply_ufunc(np.matmul,
                                         g_reg.edges[(pair[0], pair[1])]['transform'],
                                         path_params,
                                         input_core_dims=[['x_in', 'x_out']]*2,
                                         output_core_dims=[['x_in', 'x_out']],
                                         vectorize=True)
        
        g_reg.nodes[n]['transforms'] = path_params

        # else:
        #     #import pdb; pdb.set_trace()
        #     paths = nx.all_shortest_paths(g,ref_view, view, weight='weight')
        #     # print('PATHs for view %s: ' %view, [p for p in paths])
        #     paths_params = []
        #     for ipath,path in enumerate(paths):
        #         print('processing PATH for view %s: ' %view, path)
        #         # if ipath > 0: break # is it ok to take mean affine params?
        #         path_pairs = [[path[i],path[i+1]] for i in range(len(path)-1)]
        #         # print(path_pairs)
        #         path_params = np.eye(ndim+1)
        #         for edge in path_pairs:
        #             tmp_params = params_to_matrix(g.get_edge_data(edge[0], edge[1])['p'])
        #             path_params = np.dot(tmp_params,path_params)
        #             # print(path_params)
        #         paths_params.append(matrix_to_params(path_params))

        #     final_view_params = np.mean(paths_params,0)

        # # concatenate with time alignment if given
        # if time_alignment_params is not None:
        #     final_view_params = concatenate_view_and_time_params(time_alignment_params,final_view_params)

        # final_params.append(final_view_params)

    # print(final_params)

    return g_reg


# def register_graph(
#         g,
#         registration_binning = None,
#         ) -> dict:
    
#     """
#     Attach (delayed) transforms to all edges with 'to_register: True'
#     attribute in the directed input graph
#     """

#     # gd = g.to_directed() # create deep copy
    
#     # register all pairs of views along the shortest paths
#     for e in g.edges:
#         if not 'marked_for_registration' in g.edges[e].keys()\
#             or not g.edges[e]['marked_for_registration']: continue
        
#         g.edges[e]['transform'] = \
#             delayed(register_pair_of_spatial_images)(
#                 [g.nodes[e[0]]['xim'], g.nodes[e[1]]['xim']],
#                 registration_binning=registration_binning,
#                     )
    
#     return g
#     # return _mv_graph.compute_graph(gd)

# def register_

#     ds_xims = xr.merge([xim.rename({dim: "%s_%s" %(dim, ixim)
#                                     for dim in spatial_dims}).to_dataset(name='im%s' %ixim)
#                         for ixim, xim in enumerate(xims[:2])])
    
    # xr.apply_ufunc(lambda x: x,
    #                ds_xims.sel(C=ds_xims.coords['C'][0]),
    #                dask='allowed',
    #             #    input_core_dims=[['%s_%s' %(dim, iim)  for iim in range(2) for dim in spatial_dims]],
    #                input_core_dims=[['T']],
    #                output_core_dims=('M1', 'M2'),
    #                )

    # delayed(multiview.register_linear_elastix)(
    #     ImageArray()
    # )

    # # perform pairwise registrations
    # pair_ps = {t: {(view1, view2):
    #                     delayed(multiview.register_linear_elastix)
    #                              (view_reg_ims[reg_channel][t][view1],
    #                               view_reg_ims[reg_channel][t][view2],
    #                             #  (
    #                             # delayed(ImageArray)(view_reg_ims[reg_channel][t][view1],
    #                             #         spacing=view_dict[view1]['spacing'],
    #                             #         origin=view_dict[view1]['origin']),
    #                             #   delayed(ImageArray)(view_reg_ims[reg_channel][t][view2],
    #                             #              spacing=view_dict[view2]['spacing'],
    #                             #              origin=view_dict[view2]['origin']),
    #                               -1, #degree
    #                               None,
    #                               '',
    #                               f'{view1}_{view2}',
    #                               None
    #                              )
    #             for view1, view2 in pairs}
    #         for t in times}

    # # get final transform parameters
    # ps = {t: delayed(multiview.get_params_from_pairs)(
    #                             views[ref_view_index],
    #                             pairs,
    #                             [pair_ps[t][(v1,v2)] for v1, v2 in pairs],
    #                             None, # time_alignment_params
    #                             True, # consider_reg_quality
    #                             [view_reg_ims[reg_channel][t][view] for view in views],
    #                             {view: iview for iview, view in enumerate(views)}
    #                             )
    #         for t in times}

    # ps = {t: delayed(lambda x: {view: x[iview] for iview, view in enumerate(views)})
    #             (ps[t])
    #         for t in times}

    # return ps


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
    Assume first dimension is time
    """

    ndim = tl[0].ndim

    ps = da.stack([da.from_delayed(delayed(skimage.registration.phase_cross_correlation)(
            tl[t-1],
            tl[t],
            upsample_factor=3,
            normalization=None)[0], shape=(ndim, ), dtype=float)
            for t in range(1, tl.shape[0])])
    
    ps = da.concatenate([da.zeros((1, ndim)), ps], axis=0)

    ps_cum = da.cumsum(ps, axis=0)
    ps_cum_filtered = dask_image.ndfilters.gaussian_filter(ps_cum, [sigma, 0], mode='nearest')

    # deltas = ps_cum - ps_cum_filtered
    deltas = ps_cum_filtered - ps_cum

    # imst = da.stack([da.from_delayed(delayed(ndimage.affine_transform)(tl[t],
    #                                         matrix=np.eye(2),
    #                                         offset=-ps[t-1] if t else np.zeros(2), order=1), shape=tl[0].shape, dtype=tl[0].dtype)

    return deltas


if __name__ == "__main__":

    from napari_stitcher import _reader, _spatial_image_utils
    import xarray as xr
    import dask.array as da
    filename = "/Users/malbert/software/napari-stitcher/image-datasets/mosaic_test.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20210216_highres_TR2.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20230223_02_before_ablation-02_20X_max.czi"

    xims = _reader.read_mosaic_czi_into_list_of_spatial_xarrays(filename, scene_index=0)

    # def func(x):
    #     print(x.im1)
    #     return x
    
    # p = xr.apply_ufunc(
    #     # register_pair_of_spatial_images,
    #     func,
    #     xr.Dataset({'im1': xims[0],
    #                 'im2': xims[1]}),
    #     # input_core_dims=[_spatial_image_utils.get_spatial_dims_from_xim(xim) for xim in xims],
    #     input_core_dims=[_spatial_image_utils.get_spatial_dims_from_xim(xims[0])],

    #     output_core_dims=[['Y', 'X']],
    #     dask='allowed',
    #     vectorize=False,
    #     output_dtypes=float,
    #     # output_sizes={'Y': ndim + 1, 'x_out': ndim + 1},
    #     join='left',
    # )

    # def func(x):
    #     print(x.im1, x.im2)
    #     res = xr.DataArray(register_pair_of_spatial_images([x.im1, x.im2]), dims=['x_in', 'x_out'])
    #     return res

    # p = xr.apply_ufunc(
    #     # register_pair_of_spatial_images,
    #     func,
    #     xr.Dataset({'im1': xims[0],
    #                 'im2': xims[1]}),
    #     # input_core_dims=[_spatial_image_utils.get_spatial_dims_from_xim(xim) for xim in xims],
    #     input_core_dims=[_spatial_image_utils.get_spatial_dims_from_xim(xims[0])],

    #     output_core_dims=[['x_in', 'x_out']],
    #     dask='parallelized',
    #     vectorize=False,
    #     output_dtypes=float,
    #     output_sizes={'x_in': ndim + 1, 'x_out': ndim + 1},
    #     join='left',
    # )

    # def func(x):
    #     print(x.im1.shape, x.im2.shape)
    #     res = xr.DataArray(register_pair_of_spatial_images([x.im1, x.im2]), dims=['x_in', 'x_out'])
    #     return res

    # p = xr.Dataset({'im1': xims[0], 'im2': xims[1]}).groupby('C')
    # # t = p.apply(lambda x: xr.DataArray(da.ones((3,3)), dims=['x_in', 'x_out']))
    # t = p.apply(func)

    # from flox import xarray as fxarray

    # fxarray.xarray_reduce(
    #     xr.Dataset({'im1': xims[0], 'im2': xims[1]}),
    #     'C',
    #     func=lambda x: xr.Dataset({'M': register_pair_of_spatial_images([x.im1, x.im2])}),
    #     method='blockwise',
    #     )

    # for c in xims[0].coords['C']:
    #     print(c)
    #     print(register_pair_of_spatial_images([xims[0].sel(C=c), xims[1].sel(C=c)]))


    
    # xp = register_pair_of_xims(xims[0], xims[1])
