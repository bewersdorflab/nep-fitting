"%PYTHON%" setup.py install

if errorlevel 1 exit 1

:: Add more build steps here, if they are necessary.

"%PYTHON%" install_plugin.py dist

:: See
:: http://docs.continuum.io/conda/build.html
:: for a list of environment variables that are set during the build process.
