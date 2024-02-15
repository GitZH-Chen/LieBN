#!/bin/bash
data_dir=/data #change this to your data folder

### Experiments on SPDDSMBN
[ $? -eq 0 ] && python TSMNet-LieBN.py -m data_dir=$data_dir dataset=hinss2021 evaluation=inter-subject+uda,inter-session+uda nnet=tsmnet_spddsmbn
### Experiments on DSMLieBN under standard metrics
[ $? -eq 0 ] && python TSMNet-LieBN.py -m data_dir=$data_dir dataset=hinss2021 evaluation=inter-subject+uda,inter-session+uda nnet=tsmnet_LieBN nnet.model.metric=LEM,LCM,AIM

### Experiments on DSMLieBN under 0.5-LCM for inter-session
[ $? -eq 0 ] && python TSMNet-LieBN.py -m data_dir=$data_dir dataset=hinss2021 evaluation=inter-session+uda nnet=tsmnet_LieBN nnet.model.metric=LCM nnet.model.theta=0.5
### Experiments on DSMLieBN under -0.5-AIM for inter-subject
[ $? -eq 0 ] && python TSMNet-LieBN.py -m data_dir=$data_dir dataset=hinss2021 evaluation=inter-subject+uda nnet=tsmnet_LieBN nnet.model.metric=AIM nnet.model.theta=-0.5