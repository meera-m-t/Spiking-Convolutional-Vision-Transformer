# Spiking-Convolutional-Vision-Transformer
This repository presents the implementation of `SCvT` model from the paper Spiking Convolutional Vision Transformer (under review).
![model](model-1.png)

## Install
We used python 3.8 to run this code. To install all requirements via pip:
```bash
$ pip install -r requirements.txt
```

## Dataset:
We use Tiny-Imagenet-200 dataset to test our code starting from  small with subsets of 10 classes, then eventually expand to larger and larger subsets, making my way up to all 200 classes. To download this datset follow thes steps:

```bash
$ wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
$ unzip tiny-imagenet-200.zip
```

To resize the images in Tiny-Imagenet-200 dataset run:
```bash
$ python resize.py
```

## Train the model
To train the model, you can run: 

## Model Parallelism
To take advantage of multiple GPUs to train our larger model. We modified the library [`SpikeTorch`](https://github.com/miladmozafari/SpykeTorch/tree/master/SpykeTorch) to work for multbatch with multiple GPUs (see parrlell_SpykeTorch).



