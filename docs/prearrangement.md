# Prepositioning tiles

!!! note "Prepositioning tiles"
    As a starting point for registration, `napari-stitcher` uses the positions of the layers as they are shown in napari.

There are two ways to preposition the layers in napari:

### 1. Manual positioning

You can manually position the tiles by dragging and dropping them in the viewer. This is useful when you have a small number of tiles and you want to freely position them.

!!! note "Visualization tip"
    To visualize tiles on top of each other in napari, you can set the "blending" property (in the layer controls) of all layers to "additive". Additionally, you can choose a different color for each layer to help you distinguish them.

#### 2D case

![](images/preposition.gif)

#### 3D case

<!--
for il, l in enumerate(viewer.layers):
    if l.data.shape[0] == 3:
        l.data = l.data[1]
    l.scale = [2, 0.3, 0.3][-l.data.ndim:]
    l.blending = 'additive'
    l.name = f'Layer {il} :: GFP'
viewer.layers[0].colormap = 'red'
viewer.layers[1].colormap = 'cyan'
if len(viewer.layers) > 2:
    viewer.layers[2].colormap = 'cyan'
    viewer.layers[3].colormap = 'red'
    viewer.layers[4].colormap = 'cyan'
    viewer.layers[5].colormap = 'red'
-->

!!! note "Pixel spacing"
    Make sure to set the pixel spacing of the layers in Z, Y, X correctly. You can do this by opening the console and running something like the following code:

    ```python
    for l in viewer.layers:
        l.scale = [2, 0.3, 0.3]
    ```

Install the awesome [`napari-threedee`](https://github.com/napari-threedee/napari-threedee) plugin to enable 3D positioning.

![](images/preposition3d.gif)


### 2. Aligning on a grid

The widget under Plugins->napari-stitcher->Mosaic arrangement allows you to align the layers on a grid. This is useful when you have a large number of tiles which should be placed on a regular grid. This works for 2D and 3D data.

![](images/preposition_mosaic.gif)