__version__ = "0.0.1"

from ._reader import napari_get_reader
from ._sample_data import make_sample_data
from ._widget import ExampleQWidget, example_magic_widget
from ._writer import write_multiple, write_single_image

__all__ = (
    "napari_get_reader",
    "write_single_image",
    "write_multiple",
    "make_sample_data",
    "ExampleQWidget",
    "example_magic_widget",
)
