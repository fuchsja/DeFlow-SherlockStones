<p align="center">
<h2 align="center">  This git is forked from: DeFlow: Self-supervised 3D Motion Estimation of Debris Flow :mountain:</h2>

<p align="center"><strong>ETH Zurich</strong></a>
  <h3 align="center"><a href="https://openaccess.thecvf.com/content/CVPR2023W/PCV/papers/Zhu_DeFlow_Self-Supervised_3D_Motion_Estimation_of_Debris_Flow_CVPRW_2023_paper.pdf">Paper</a> 
  | <a href="https://zhuliyuan.net/deflow">Website</a> | <a href="https://www.research-collection.ethz.ch/handle/20.500.11850/599948">Dataset</a> </h3> 
  <div align="center"></div>



<image src="misc/overview.png"/>
</p>

This repository is a fork of the official implementation of paper:
<b>DeFlow: Self-supervised 3D Motion Estimation of Debris Flow</b>, CVPRW 2023.

Existing work on scene flow estimation focuses on autonomous driving and mobile robotics, while automated solutions are lacking for motion in nature, such as that exhibited by debris flows. We propose \deflow, a model for 3D motion estimation of debris flows, together with a newly captured dataset. We adopt a novel multi-level sensor fusion architecture and self-supervision to incorporate the inductive biases of the scene. We further adopt a multi-frame temporal processing module to enable flow speed estimation over time. Our model achieves state-of-the-art optical flow and depth estimation on our dataset, and fully automates the motion estimation for debris flows.

This fork is part of a Bsc thesis project that builds up upon the DeFlow paper. 

## Installation :national_park:
First clone the repository:
```bash
git clone https://github.com/fuchsja/DeFlow-SherlockStones.git
cd DeFlow
```
Then add a folder for results:
```bash
mkdir results
```

You will need to install conda to build the environment.
```bash
conda create -n DeFlow python=3.9
conda activate DeFlow
pip install -r requirements.txt
```

## Dataset and pretrained model
The authors of the original DeFlow paper provide preprocessed debris flow dataset. The preprocessed dataset can be downloaded by running:
```shell
wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gsg/DeFlow/DeFlow_Dataset.zip
unzip DeFlow_Dataset.zip -d data
```
There is also the possiblity to download a pretrained version of the model here:
```shell
wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gsg/DeFlow/checkpoint.zip
unzip checkpoint.zip
```
You can also build your own dataset following the structure below
```Shell
├── Data
    ├── Cam1
        ├── 000001.jpg
        ├── 000002.jpg
        .
        .
        ├── 00000X.jpg
    ├── Cam2
        ├── 000001.jpg
        ├── 000002.jpg
        .
        .
        ├── 00000X.jpg
    ├── LiDAR
        ├── 000001.ply
        ├── 000002.ply
        .
        .
        ├── 00000X.ply
├── Transformations
        ├── cam_intrinxics.txt
        ├── LiCam_tranformations.txt
```
To evaluate the pretrained model on the given dataset, run:
```bash
python main.py --config_path configs/deflow_default.yaml
```
and you can change the mode to train in the config file in case you want to train the DeFlow model from scratch. 


## Citation
If you use DeFlow for any academic work, please cite the original paper.
```bibtex
@InProceedings{zhu2023DeFlow,
author = {Liyuan Zhu and Yuru Jia and Shengyu Huang and Nicholas Meyer and Andreas Wieser and Konrad Schindler, Jordan Aaron},
title = {DEFLOW: Self-supervised 3D Motion Estimation of Debris Flow},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2023}
}
```

Additionally, we thank the respective developers of the following open-source projects:
- [PCAccumulation](https://github.com/prs-eth/PCAccumulation) 
- [CamLiFlow](https://github.com/MCG-NJU/CamLiFlow) 
- [Self-mono-sf](https://github.com/visinf/self-mono-sf)
