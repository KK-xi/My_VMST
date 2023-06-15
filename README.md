# VMST-Net

This repository contains the implemtation code associated with Voxel-based Multi-scale Transformer Network for Event Stream Processing.

## Requirements
     Python 3.7 
     Pytorch 1.5.0
     cuda 10.2

## Installation
Clone this repository using:

     git clone https://github.com/KK-xi/My_VMST.git

Create a conda environment using the [environment.yml](environment.yml) file: 

     conda env create -f environment.yml
    
## Datasets
To generate the voxels, we refer to the code of [__**VoxelNet**__](https://github.com/skyhehe123/voxelnet-pytorch) and [__**TimoStoff**__](https://github.com/TimoStoff/events_contrast_maximization).

Take N-Caltech101 as an example:
    
    Training voxels are saved in './data/N-Caltech101/train' folder.
    
    Testing voxels are saved in './data/N-Caltech101/test' folder.

Each sample should contains feature and coords of voxels and label.

## Running examples
Take N-Caltech101 as an example:

    python main.py --train_dataset ./data/N-Caltech101/train/ --test_dataset ./data/N-Caltech101/test/ --arch_name VMST-Net_N-Cal --num_classes 101 --voxel_num 1024


## Citation
