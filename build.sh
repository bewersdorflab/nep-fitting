#!/bin/bash

$PYTHON setup.py install
#$PYTHON pymecompress/setup.py install

# Add more build steps here, if they are necessary.

echo "Trying to install as PYME plugin"

$PYTHON install_plugin.py


# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.
