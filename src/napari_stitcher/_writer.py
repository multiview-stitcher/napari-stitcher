"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/guides.html?#writers

Replace code below according to your needs.
"""
from __future__ import annotations

from tifffile import imwrite
import numpy as np
import dask.array as da

from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = Tuple[DataType, dict, str]


def write_single_image(path: str, data: Any, meta: dict) -> List[str]:
    """Writes a single image layer"""

    # implement your writer logic here ...

    # return path to any file(s) that were successfully written
    return [path]


def write_multiple(path: str, data: List[FullLayerData]) -> List[str]:
    """Writes multiple layers of different types.
    3-tuple with (data, meta, layer_type)
    """

    # implement your writer logic here ...

    if not np.all([d[1]['metadata']['view'] == -1 for d in data]):
        raise ValueError('Only saving of fused images is supported.')
        return

    spacing = data[0][1]['metadata']['stack_props']['spacing']
    times = data[0][1]['metadata']['times']
    channels = range(len(data))

    image_data = da.stack([d[0].squeeze() for d in data])

    axes = 'YX'
    if len(channels) > 1:
        axes = 'C' + axes
    if len(times) > 1:
        image_data = da.swapaxes(image_data, 0, 1)
        axes = 'T' + axes

    imwrite(
        path,
        image_data,
        imagej=True,
        resolution=tuple([1. / s for s in spacing]),
        metadata={
            # 'spacing': spacing[0],
            'unit': 'um',
            'finterval': 1/10,
            'axes': axes,
        }
    )

    # return path to any file(s) that were successfully written

    return [path]


if __name__ == "__main__":

    import napari
    viewer = napari.Viewer()

    viewer.add_image(np.random.random((10, 10, 10)).astype(np.uint8), metadata={'metadata':{'view':-1,
                                                               'stack_props': {'spacing': [0.1, 0.1]},
                                                               'times': range(10)}}, name='image1_ch000')
    viewer.add_image(np.random.random((10, 10, 10)).astype(np.uint8), metadata={'metadata':{'view':-1,
                                                               'stack_props': {'spacing': [0.1, 0.1]},
                                                               'times': range(10)}}, name='image1_ch001')

    # viewer.add_image(np.random.random((10, 10)).astype(np.uint8), metadata={'view':-1,
    #                                                            'stack_props': {'spacing': [0.1, 0.1]},
    #                                                            'times': [0]}, name='image1_ch000')
    # viewer.add_image(np.random.random((10, 10)).astype(np.uint8), metadata={'view':-1,
    #                                                            'stack_props': {'spacing': [0.1, 0.1]},
    #                                                            'times': [0]}, name='image1_ch001')

    write_multiple('delme.tif', [(l.data, l.metadata, '') for l in viewer.layers])
    # viewer.layers.save('delme.tif', plugin='napari_stitcher') # doesn't work
