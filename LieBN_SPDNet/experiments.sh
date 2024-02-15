#!/bin/bash

epochs=200
datasets=RADAR,HDM05

### Experiments on SPDNet,SPDNetBN
[ $? -eq 0 ] && python SPDNetLieBN.py -m dataset=$datasets nnet=SPDNetBN,SPDNet fit.epochs=$epochs

### Experiments on SPDNetLieBN-LEM
[ $? -eq 0 ] && python SPDNetLieBN.py -m dataset=$datasets nnet=SPDNetLieBN fit.epochs=$epochs nnet.model.metric=LEM

### Experiments on SPDNetLieBN-AIM
#RADAR
[ $? -eq 0 ] && python SPDNetLieBN.py -m dataset=RADAR nnet=SPDNetLieBN fit.epochs=$epochs nnet.model.metric=AIM nnet.model.theta=1.
#HDM05
[ $? -eq 0 ] && python SPDNetLieBN.py -m dataset=HDM05 nnet=SPDNetLieBN fit.epochs=$epochs nnet.model.metric=AIM nnet.model.theta=1,1.5

### Experiments on SPDNetLieBN-LCM
#RADAR
[ $? -eq 0 ] && python SPDNetLieBN.py -m dataset=RADAR nnet=SPDNetLieBN fit.epochs=$epochs nnet.model.metric=LCM nnet.model.theta=1.,-0.5
##HDM05
[ $? -eq 0 ] && python SPDNetLieBN.py -m dataset=HDM05 nnet=SPDNetLieBN fit.epochs=$epochs nnet.model.metric=LCM nnet.model.theta=1,0.5