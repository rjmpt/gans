# 671 Final Project: A Comparative Study of GAN Variants and their Efficacy at Image Generation on CIFAR-10
This repository contains all the code used in this project. Please refer to the brief descriptions in order to help navigate through the code.

• experiments.py: starting point of execution from which experiments can be run \
• Algorithms.py: contains the various models + hyperparameter combinations used in each experiment \
• Models/: contains the model definitions and objective function definitions for vanilla GAN, WGAN-GP, and fGAN 
  
  
• InceptionScore.py: script for computing inception score, adapted from https://github.com/taki0112/GAN_Metrics-Tensorflow/blob/master/main.py \
• DataLoaders/: contains the DataLoader class used to process CIFAR 10 train / test batches from .h5 file on disk \
• architectures.py: contains proto architectures for DCGAN based on different image input sizes \
• DNN.py: core library file which contains custom layer implementations and implementation of DNN class, which corresponds to a directed acylic graph (DAG) of layers


• optimization.py: core optimization functionality for training \
• visualizations.py: all code related to model visualizations \
• distributions.py, statistical_divergences.py, helper.py: various utilities containing the functionalities that their names imply
