from PYME import config
import os
from distutils.dir_util import copy_tree

def main():
    this_dir = os.path.dirname(__file__)

    copy_tree(os.path.join(this_dir, 'etc', 'PYME'), config.user_config_dir)

if __name__ == '__main__':
    main()