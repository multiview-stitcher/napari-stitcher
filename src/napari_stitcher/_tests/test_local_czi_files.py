import os
from napari_stitcher import _mv_graph, _sample_data, _reader, _spatial_image_utils


filenames = [
    '20230412_arthur_03_EE_sub.czi',
    'arthur_20230223_02_before_ablation-02_20X_max.czi',
    '20230412_arthur_03_e2_part1-10X_max.czi',
    'arthur_20230613_e1_t0.czi'
]

dir = '/Users/malbert/software/napari-stitcher/image-datasets/'

if __name__ == "__main__":

    import napari

    viewer = napari.Viewer()

    from napari_stitcher import StitcherQWidget

    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    viewer.open(os.path.join(dir, filenames[3]))

    # layer_tuples = make_sample_data()

    # for lt in layer_tuples:
    #     lt[1]['contrast_limits'] = [0, 100]
    #     viewer.add_image(lt[0], **lt[1])

    # from napari_stitcher import StitcherQWidget

    # wdg = StitcherQWidget(viewer)
    # viewer.window.add_dock_widget(wdg)

