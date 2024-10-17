# Layer naming convention

!!! note "Napari and multi-channel images"
    In napari, different channels are represented as different layers. However `napari-stitcher` needs to know which layers correspond to the same tile. To do this, the layers need to be named in a specific way.

For single channel datasets, layer names can be arbitrary. For multi-channel datasets, the layers need to be named in the following way for best compatibility with napari-stitcher:

`{tile} :: {channel}`.

Example:

![](images/naming_convention_example.png){: style="width:40%"}

This is the convention followed by [`bioio`](https://github.com/bioio-devs/bioio) when loading images into napari.

!!! note "How to edit layer names?"
    To rename a layer, double-click on the layer name in the layer list and type the new name. To programmatically rename layers, you can use e.g. something like the following code in the napari console:

    ```python
    for il, l in enumerate(viewer.layers):
        l.name = f'Tile {il // 2} :: {[GFP, RFP][il % 2]}'
    ```