
# Saving fused images

Select the fused image layer(s) and press Ctrl/Command + S or use the `File > Save Selected Layers`. When using the `napari-stitcher` writer plugin, you can save all channel layers as a single tif file.

!!! note "Selecting a writer plugin"
    In the save dialog, you can select the writer plugin to use.

## Pre-computed fusion output

With **Pre-compute fusion** enabled, **Fusion output folder** selects a parent
directory, not the output store itself. Existing parent directories are allowed;
missing ones are created. Each run creates a unique `fusion_...` subfolder,
containing `channel_0.zarr`, `channel_1.zarr`, etc. Multiscale results use
the `.ome.zarr` extension. Channel indices follow the loaded channel order.
The run folder location is shown in a notification.

You can repeat fusion with the same parent folder: each run gets a separate
folder and earlier results and unrelated files are not deleted or overwritten.
Files and symbolic links cannot be selected as parent folders. All channel
stores are reserved before fusion starts; an interrupted run may leave partial
results in its own folder, without removing existing data.

Leave the parent folder empty to use temporary storage with the same per-run
layout. Temporary results remain available while the widget is alive; save
important results to a persistent location.
