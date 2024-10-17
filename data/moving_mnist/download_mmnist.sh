#!/usr/bin/env bash

# download mmnist and place them in `data/moving_mnist/`
wget http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

echo "Done"

# Make sure they are in the following structure:
# data
# ├── moving_mnist
# │   ├── mnist_cifar_test_seq.npy
# │   ├── mnist_test_seq.npy
# │   ├── train-images-idx3-ubyte.gz