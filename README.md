# Spiking-Convolutional-Vision-Transformer
This repository presents the implementation of `SCvT` model from the paper Spiking Convolutional Vision Transformer (under review).
![model](images/model-1.png)

## Install
To create an environment with  Python 3.8and download pytorch in CUDA Toolkit 11.3 run:

```bash
$conda create -n py3.8 python=3.8
$conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
To install all requirements via pip:
```bash
$ pip install -r requirements.txt
```

## Dataset:
We use Tiny-Imagenet-200 dataset to test our code starting from  small with subsets of 10 classes, then eventually expand to larger and larger subsets, making my way up to all 200 classes. To download this datset follow thes steps:

```bash
$ wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
$ unzip tiny-imagenet-200.zip
```

To resize the images in Tiny-Imagenet-200 dataset from `256X256` to `224X224`  run:
```bash
$ python resize.py
```

## Train the model

To train the model, you can run:
```bash
python run.py
``` 

## Model Parallelism
To take advantage of multiple GPUs to train our larger model. We modified the library [`SpikeTorch`](https://github.com/miladmozafari/SpykeTorch/tree/master/SpykeTorch) to work for multbatch with multiple GPUs (see [parrlell_SpykeTorch](https://github.com/meera-m-t/parallelSpykeTorch)).  `Note that in SpykeTorch’s modules and functionalities such as Convolution or Pooling, the mini-batch dimension is sacrificed for the time dimension. Therefore, it does not provide built-in batch processing at the moment.`



