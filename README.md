# Unsupervised-end-to-end-infrared-and-visible-image-fusion-network-using-learnable-fusion-strategy
This repository contains the codes for paper Unsupervised end-to-end infrared and visible image fusion network using learnable fusion strategy by Yili Chen, Minjie Wan*, Yunkai Xu, et al. (*Corresponding author).

The overall repository style is partially borrowed from PFNet (https://github.com/Junchao2018/Polarization-image-fusion). Thanks to Junchao Zhang.

The visible-thermal dataset TNO can be downloaded from https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029, the multi-focus dataset Lytro can be downloaded from https://mansournejati.ece.iut.ac.ir/content/lytro-multi-focus-dataset.
## Requirements
Python==3.7

Tensorflow==1.13.1

Cuda==10.0.13 and cudnn

h5py

opencv-python

other packages if needed
## Usage
1. Generating training and testing data using GenerateTrainingPatches_Tensorflow.m and GenerateTestingPatches_Tensorflow.m, put the generated mat into TrainingData and TestingData respectively.
2. Training your own model using backward.py, the relative parameters can be adjusted in the same file.
3. Generating fusion results with the trained model using test.py.
4. The default output format is '.mat', using Matlab to convert it to common picture format.
