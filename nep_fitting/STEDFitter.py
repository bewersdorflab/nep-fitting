"""
Entry point for STED fitting so that we can launch with all the right tools already loaded
"""

from PYME.DSView import dsviewer
from PYME.DSView import modules
import sys


def main():
    #monkey patch dsviewer modules so we can load nep-fitting on launch
    modules.modeModules['nep'] = modules.modeModules['lite'] + ['sted_psf_fitting',]
    
    dsviewer.main(sys.argv[1:] + ['-m', 'nep'])
    
if __name__ == '__main__':
    main()
