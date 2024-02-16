[<img src="https://img.shields.io/badge/OpenReview|forum-pp7onaiM4VB-8c1b13"></img>](https://openreview.net/forum?id=okYdj8Ysru)

# A Lie Group Approach to Riemannian Batch Normalization

This is the official code for our ICLR 2024 publication: *A Lie Group Approach to Riemannian Batch Normalization*. [[OpenReview](https://openreview.net/forum?id=okYdj8Ysru)].

If you find this project helpful, please consider citing us as follows:

```bib
@inproceedings{
chen2024a,
title={A Lie Group Approach to Riemannian Batch Normalization},
author={Ziheng Chen and Yue Song and Yunmei Liu and Nicu Sebe},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=okYdj8Ysru}
}
```

In case you have any problem, do not hesitate to contact me ziheng_ch@163.com.

## Requirements

Install necessary dependencies by `conda`:

```setup
conda env create --file environment.yaml
```
**Note** that the [hydra](https://hydra.cc/) package is used to manage configuration files.

## Experiments on the SPDNet

The code of experiments on SPDNet, SPDNetBN, and SPDNetLieBN is enclosed in the folder `./LieBN_SPDNet`

The implementation is based on the official code of *Riemannian batch normalization for SPD neural networks* [[Neurips 2019](https://papers.nips.cc/paper_files/paper/2019/hash/6e69ebbfad976d4637bb4b39de261bf7-Abstract.html)] [[code](https://papers.nips.cc/paper_files/paper/2019/file/6e69ebbfad976d4637bb4b39de261bf7-Supplemental.zip)].

### Dataset

The synthetic [Radar](https://www.dropbox.com/s/dfnlx2bnyh3kjwy/data.zip?e=1&dl=0) dataset is released by SPDNetBN. We further release our preprocessed [HDM05](https://www.dropbox.com/scl/fi/x2ouxjwqj3zrb1idgkg2g/HDM05.zip?rlkey=4f90ktgzfz28x3i2i4ylu6dvu&dl=0) dataset.

Please download the datasets and put them in your personal folder and change the `path` accordingly in `./LieBN_SPDNet/conf/dataset/RADAR.yaml` and `./LieBN_SPDNet/conf/dataset/HDM05.yaml`

### Running experiments

To run all the experiments on the Radar and HDM05 datasets, go to the folder `LieBN_SPDNet` and run this command:

```train
bash run_experiments.sh
```
This script contains the experiments on the Radar and HDM05 datasets shown in Tab. 4

## Experiments on the TSMNet

The code of experiments on TSMNet, TSMNet + SPDDSMBN, and TSMNet + DSMLieBN is enclosed in the folder `./LieBN_TSMNet`

The implementation is based on the official code of *SPD domain-specific batch normalization to crack interpretable unsupervised domain adaptation in EEG* [[Neurips 2022](https://openreview.net/forum?id=pp7onaiM4VB)] [[code](https://github.com/rkobler/TSMNet.git)].

### Dataset

The [Hinss2021](https://doi.org/10.5281/zenodo.5055046) dataset is publicly available. The [moabb](https://neurotechx.github.io/moabb/) and [mne](https://mne.tools) packages are used to download and preprocess these datasets. There is no need to manually download and preprocess the datasets. This is done automatically. If necessary, change the `data_dir` in `./LieBN_TSMNet/conf/LieBN.yaml` to your personal folder.

### Running experiments

To run all the experiments on the Radar and HDM05 datasets, go to the folder `LieBN_TSMNet` and run this command:

```train
bash run_experiments.sh
```
This script contains the experiments on the Hinss2021 datasets shown in Tab. 5

**Note:** You also can change the `data_dir` in `run_experiments.sh`, which will override the hydra config.



