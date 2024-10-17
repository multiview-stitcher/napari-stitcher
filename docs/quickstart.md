
# Quickstart

![](images/napari-stitcher-loaded-mosaic-annotated.png)

## Opening napari

- Activate your conda environment by running `conda activate napari-stitcher` in the terminal (MacOS) or Miniconda Prompt (Windows)
- Start napari by running `napari`
- Start the plugin in the menu plugins > napari-stitcher > Stitcher

## Load images into napari

`napari-stitcher` can stitch any image layers that are loaded into napari. You can load images into napari by dragging and dropping files into the viewer, or by using the `File > Open Files` menu.

Additionally, `napari-stitcher` supports reading mosaic files which contain multiple pre-positioned tiles. If more than one scene is present, you'll be prompted to select a scene to load. The following file formats are supported:
- `.czi`
- `.lif`

Image files / napari layers to stitch can be 2D/3D(+time).

## Some napari basics

- The display settings (contrast, etc.) can be set using the panel at the top left (1)
- Tiles and channels can be displayed or hidden by clicking on the eye icon on the bottom left panel (2). Tiles are displayed with a color code to help the overlap inspection: an alternation of red and green is used so the overlapping structure appears yellow. 
- The plugin panel is at the top right (3).


## Using the napari-stitcher plugin

- The timepoints used to run the stitching can be adjusted with the “Timepoints” range slider. This is useful to quickly test the stitching in case of a large time series.
- Choose the channel used for calculating the registration with the “Reg channel” dropdown menu.
- Once the stitching or stabilization is performed, you can toggle between the “Original” image and the “Registered” image. Inspect the result by zooming on the overlap region.
- If you are satisfied with the result, click on “Fuse” to fuse all the tiles together in a single image. 
- Export the output by slecting the output layer (4), and clicking on “File > Save selected layers” (you can also use “Cmd+S” or “Ctrl+S”). Use .tif format for the output. 
