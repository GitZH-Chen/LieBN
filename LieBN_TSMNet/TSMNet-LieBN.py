"""
@author: Ziheng Chen
Please cite the paper below if you use the code:

Ziheng Chen, Yue Song, Yunmei Liu, Nicu Sebe. A Lie Group Approach to Riemannian Batch Normalization. ICLR 2024

Copyright (C) 2024 Ziheng Chen
All rights reserved.
"""

import hydra
from omegaconf import DictConfig

import warnings
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning

from library.utils.hydra import hydra_helpers
from LieBN_utilities.Training import training

warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class Args:
    """ a Struct Class  """
    pass
args=Args()
args.config_name='LieBN.yaml'

@hydra_helpers
@hydra.main(config_path='./conf/', config_name=args.config_name, version_base='1.1')
# @hydra.main(config_path='./conf/', config_name=args.config_name)
def main(cfg: DictConfig):
    training(cfg,args)

if __name__ == '__main__':

    main()
