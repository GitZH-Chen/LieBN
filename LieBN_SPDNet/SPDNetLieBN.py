"""
@author: Ziheng Chen
Please cite the paper below if you use the code:

Ziheng Chen, Yue Song, Yunmei Liu, Nicu Sebe. A Lie Group Approach to Riemannian Batch Normalization. ICLR 2024

Copyright (C) 2024 Ziheng Chen
All rights reserved.
"""

import hydra
from omegaconf import DictConfig

from spd.training_script import training

class Args:
    """ a Struct Class  """
    pass
args=Args()
args.config_name='LieBN.yaml'


args.total_BN_model_types=['SPDNetLieBN', 'SPDNetBN']
args.total_LieBN_model_types=['SPDNetLieBN']

@hydra.main(config_path='./conf/', config_name=args.config_name, version_base='1.1')
def main(cfg: DictConfig):
    training(cfg,args)

if __name__ == "__main__":
    main()