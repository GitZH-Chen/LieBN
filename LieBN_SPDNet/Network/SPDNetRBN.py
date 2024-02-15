import torch.nn as nn
import cplx.nn as nn_cplx
import spd.nn as nn_spd
from spd.LieBN import LieBatchNormSPD

class SPDNetLieBN(nn.Module):
    def __init__(self,args):
        super(__class__, self).__init__()
        dims = [int(dim) for dim in args.architecture]
        self.feature = []
        if args.dataset == 'RADAR':
            self.feature.append(nn_cplx.SplitSignal_cplx(2, 20, 10))
            self.feature.append(nn_cplx.CovPool_cplx())
            self.feature.append(nn_spd.ReEig())

        for i in range(len(dims) - 2):
            self.feature.append(nn_spd.BiMap(1,1,dims[i], dims[i + 1]))
            self.feature.append(SPDBN(dims[i + 1],args))
            self.feature.append(nn_spd.ReEig())

        self.feature.append(nn_spd.BiMap(1,1,dims[-2], dims[-1]))
        self.feature.append(SPDBN(dims[-1],args))
        self.feature = nn.Sequential(*self.feature)

        self.classifier = LogEigMLR(dims[-1]**2,args.class_num)

    def forward(self, x):
        x_spd = self.feature(x)
        y = self.classifier(x_spd)
        return y

# LogEig MLR
class LogEigMLR(nn.Module):
    def __init__(self, input_dim, classnum):
        super(__class__, self).__init__()
        self.logeig = nn_spd.LogEig()
        self.linear = nn.Linear(input_dim, classnum).double()

    def forward(self, x):
        x_vec = self.logeig(x).view(x.shape[0], -1)
        y = self.linear(x_vec)
        return y

class SPDBN(nn.Module):
    def __init__(self, n,args,ddevice='cpu'):
        super(__class__, self).__init__()
        if args.BN_type == 'brooks':
            self.BN = nn_spd.BatchNormSPD(n,args.momentum)
        elif args.BN_type == 'LieBN':
            self.BN = LieBatchNormSPD(n,
                                      metric=args.metric,
                                      theta=args.theta, alpha=args.alpha, beta=args.beta,momentum=args.momentum)
        else:
            raise Exception('unknown BN {}'.format(args.BN_type))

    def forward(self, x):
        x_spd = self.BN(x)
        return x_spd

class SPDNet(nn.Module):
    def __init__(self,args):
        super(__class__, self).__init__()
        dims = [int(dim) for dim in args.architecture]
        self.feature = []
        if args.dataset == 'RADAR':
            self.feature.append(nn_cplx.SplitSignal_cplx(2, 20, 10))
            self.feature.append(nn_cplx.CovPool_cplx())
            self.feature.append(nn_spd.ReEig())

        for i in range(len(dims) - 2):
            self.feature.append(nn_spd.BiMap(1,1,dims[i], dims[i + 1]))
            self.feature.append(nn_spd.ReEig())

        self.feature.append(nn_spd.BiMap(1,1,dims[-2], dims[-1]))
        self.feature = nn.Sequential(*self.feature)

        self.classifier = LogEigMLR(dims[-1]**2,args.class_num)

    def forward(self, x):
        x_spd = self.feature(x)
        y = self.classifier(x_spd)
        return y



