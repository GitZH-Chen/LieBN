
from enum import Enum
import torch.nn as nn

class BatchNormDispersion(Enum):
    NONE = 'mean'
    SCALAR = 'scalar'
    VECTOR = 'vector'
class BatchNormTestStatsMode(Enum):
    BUFFER = 'buffer'
    REFIT = 'refit'
    ADAPT = 'adapt'
class BatchNormTestStatsInterface:
    def set_test_stats_mode(self, mode: BatchNormTestStatsMode):
        pass

class BaseBatchNorm(nn.Module, BatchNormTestStatsInterface):
    def __init__(self, eta=1.0, eta_test=0.1, test_stats_mode: BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER):
        super().__init__()
        self.eta = eta
        self.eta_test = eta_test
        self.test_stats_mode = test_stats_mode

    def set_test_stats_mode(self, mode: BatchNormTestStatsMode):
        self.test_stats_mode = mode

