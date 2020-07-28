# NEP Fitting

Nested-loop Ensemble PSF Fitting (NEP Fitting) is a method of simultaneously measuring image feature size and image resolution. The method is described in [doi.org/10.1016/j.bpj.2018.07.028](https://doi.org/10.1016/j.bpj.2018.07.028).


## Installation

NEP-fitting is implemented as a plugin for the Python Microscopy Environment (PYME). PYME installation isntructions can be found at
[python-microscopy.org](https://python-microscopy.org/).
Once installed, nep-fitting is accessible in the PYME image viewer (dh5view) by clicking 'sted_psf_fitting' in the modules drop-down menu.

The plugin itself can be installed via conda (general use) or from source (use and software development).


### Note - 2020/07

We are currently having some packaging issues with PYME and nep-fitting. For the time being, users wishing for the simplest installation should install nep-fitting and PYME into a new environment:

0.  Install python 3 [miniconda](https://docs.conda.io/en/latest/miniconda.html).

1. Open a terminal/command prompt and run

    `conda create -n nepfitting -c barentine nep-fitting=1.09`

2. Activate the nepfitting conda environment

    `conda activate nepfitting`

3. Test the installation or use the plugin by running

    `STEDFitter`


### Install using conda:

0. You may want to make sure conda is up to date, e.g. `conda update conda` or `conda update -n <your environment name> conda`
1. Run 
    
    `conda install -c barentine nep-fitting`
2. Test installation by running 

    `STEDFitter` 

3. Alternatively, run
    
    `dh5view`
    
    open an image, and select 'sted_psf_fitting' from the 'modules' drop-down menu.

### Install from source:

0. Clone this repository

1. `python nep-fitting/setup.py install`

2. `python nep-fitting/install_plugin.py`

3. Test installation by running `dh5view`, loading an image, and selecting 'sted_psf_fitting' from the 'modules'
drop-down menu.

## Contact
If something isn't working for you, you'd like to contribute, or have any questions about nep-fitting, please get in touch with us on the image.sc forum using the `pyme` tag.