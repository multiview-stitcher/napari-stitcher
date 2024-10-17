# Installation

You can install `napari-stitcher` via pip:

```bash
pip install napari-stitcher
```

To install latest development version:
```bash
pip install git+https://github.com/multiview-stitcher/napari-stitcher.git
```

!!! note "Don't have conda installed?"
    We recommend using python and `napari-stitcher` within a conda environment. Follow the instructions below to install `conda` on your computer.

## Installing Python

### Overview
This how-to teaches how to install Python on your computer using `conda`, which is both a

- environment manager and a
- package manager.


### Instructions

Select the installation instructions for your operating system and processor from the tabs below.

=== "Linux"

    1. In your web browser, navigate to this [download page](https://github.com/conda-forge/miniforge#miniforge).
    2. Scroll down to the "Miniforge" header of the "Downloads" section. Click the link to download link for `Miniforge-Linux-x86_64`.
    3. Open your terminal application
    4. Navigate to the folder you downloaded the installer to (usually this is in your Downloads folder). If the file was downloaded to your Downloads folder, you would enter:

    ```bash
    cd ~/Downloads
    ```   

    5. Execute the installer with the command below. You can use your arrow keys to scroll up and down to read the license agreement and enter "yes" if you agree.

    ```bash
    bash Miniforge-Linux-x86_64.sh
    ```

    6. After installation, you will be asked if you would like to initialize your terminal with "conda init". Enter "yes" to initalize your terminal.
    7. To verify your installation worked, close your Terminal window and open a new one. You should see `(base)` to the left of your prompt.

    !!! note "Don't see (base)?"
        You can manually initialize conda by entering the command below and re-opening your terminal application.


        ```bash
        conda init
        ```

=== "Mac OS (Intel)"

    1. In your web browser, navigate to this [download page](https://github.com/conda-forge/miniforge#miniforge).
    2. Scroll down to the "Miniforge" header of the "Downloads" section. Click the link to download link for `Miniforge-MacOSX-x86_64`.
    3. Open your Terminal (you can search for it in spotlight - `cmd` + `space`)
    4. Navigate to the folder you downloaded the installer to (usually this is in your Downloads folder). If the file was downloaded to your Downloads folder, you would enter:

    ```bash
    cd ~/Downloads
    ```
    
    5. Execute the installer with the command below.  You can use your arrow keys to scroll up and down to read the license agreement and enter "yes" if you agree.

    ```bash
    bash Miniforge-MacOSX-x86_64.sh
    ```

    6. After installation, you will be asked if you would like to initialize your terminal with "conda init". Enter "yes" to initalize your terminal.   
    7. To verify your installation worked, close your Terminal window and open a new one. You should see `(base)` to the left of your prompt.
    
    !!! note "Don't see (base)?"
        You can manually initialize conda by entering the command below and re-opening your terminal application.
        ```bash
        conda init
        ```


=== "Mac OS (M1/M2)"
    1. In your web browser, navigate to this [download page](https://github.com/conda-forge/miniforge#miniforge).
    2. Scroll down to the "Miniforge" header of the "Downloads" section. Click the link to download link for `Miniforge-MacOSX-arm64`.
    3. Open your Terminal (you can search for it in spotlight - `cmd` + `space`)
    4. Navigate to the folder you downloaded the installer to (usually this is in your Downloads folder). If the file was downloaded to your Downloads folder, you would enter:

    ```bash
    cd ~/Downloads
    ```
    
    5. Execute the installer with the command below. You can use your arrow keys to scroll up and down to read it/agree to it.

    ```bash
    bash Miniforge-MacOSX-arm64.sh
    ```
    
    6. After installation, you will be asked if you would like to initialize your terminal with "conda init". Enter "yes" to initalize your terminal. 
    7. To verify your installation worked, close your Terminal window and open a new one. You should see `(base)` to the left of your prompt.

    !!! note "Don't see (base)?"
        You can manually initialize conda by entering the command below and re-opening your terminal application.
        ```bash
        conda init
        ```

=== "Windows"
    1. In your web browser, navigate to this [download page](https://github.com/conda-forge/miniforge#miniforge).
    2. Scroll down to the "Miniforge" header of the "Downloads" section. Click the link to download link for `Miniforge-Windows-x86_64`.
    3. Find the file you downloaded (Miniforge-Windows-x86_64.exe) and double click to execute it. Follow the instructions to complete the installation.
    4. Once the installation has completed, you can verify it was correctly installed by searching for the "miniforge prompt" in your Start menu.


## Creating a conda environment

- Create a new conda environment with the necessary packages:

    ```bash
    conda create -n napari-stitcher python=3.10 napari pyqt=5
    ```

## Opening napari

=== "MacOS/Linux"
    1. Open a terminal
    - Activate the right conda environment: `conda activate napari-stitcher` in the terminal (MacOS) or Miniconda Prompt (Windows)
    - Start napari by running `napari`
    - Start the plugin in the menu plugins > napari-stitcher > Stitcher

=== "Windows"
    1. Open the Miniconda Prompt (Windows)
    - Activate the right conda environment: `conda activate napari-stitcher`
    - Start napari by running `napari`
    - Start the plugin in the menu plugins > napari-stitcher > Stitcher