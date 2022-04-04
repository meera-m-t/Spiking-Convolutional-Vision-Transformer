import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from ScVT import CvT
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
from sklearn.svm import LinearSVC
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = True
def get_performance(X, y, predictions):
    correct = 0
    silence = 0
    for i in range(len(predictions)):
        if X[i].sum() == 0:
            silence += 1
        else:
            if predictions[i] == y[i]:
                correct += 1
    return (correct/len(X), (len(X)-(correct+silence))/len(X), silence/len(X))


def train_unsupervise(network, data, layer_idx):
    network.train()
    for i in range(len(data)):
        data_in = data[i]
        if use_cuda:
            data_in = data_in.cuda()
        network(data_in, layer_idx)
        network.stdp(layer_idx)

def train_rl(network, data, target):
    network.train()
    perf = np.array([0,0,0]) # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in, 3)
        if d != -1:
            if d == target_in:
                perf[0]+=1
                network.reward()
            else:
                perf[1]+=1
                network.punish()
        else:
            perf[2]+=1
    return perf/len(data)


def test(network, data, target):
    network.eval()
    perf = np.array([0,0,0]) # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in, 1)
        if d != -1:
            if d == target_in:
                perf[0]+=1
            else:
                perf[1]+=1
        else:
            perf[2]+=1
    return perf/len(data)
        
        
class S1Transform:
    def __init__(self, filter, timesteps = 5):
        self.grayscale = transforms.Grayscale()                 
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.temporal_transform = utils.Intensity2Latency(timesteps)
        self.cnt = 0
    def __call__(self, image):
        self.cnt+=1
        image = self.grayscale(image)             
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0)
        image = self.filter(image)
        image = sf.local_normalization(image, 8)
        temporal_image = self.temporal_transform(image)   
        return temporal_image.sign().byte()

kernels = [ utils.DoGKernel(3,3/9,6/9),
            utils.DoGKernel(3,6/9,3/9),
            utils.DoGKernel(7,7/9,14/9),
            utils.DoGKernel(7,14/9,7/9),
            utils.DoGKernel(13,13/9,26/9),
            utils.DoGKernel(13,26/9,13/9)]
filter = utils.Filter(kernels, padding = 6, thresholds = 50)
s1c1 = S1Transform(filter)
trainset = utils.CacheDataset(torchvision.datasets.ImageFolder('tiny-224/train',  transform=s1c1))
ImageNet_trainLoader = DataLoader(trainset, batch_size=1,shuffle=True)
valset = utils.CacheDataset(torchvision.datasets.ImageFolder('tiny-224/val',  transform=s1c1))
ImageNet_valLoader = DataLoader(valset, batch_size=100, shuffle=False)
best_train = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch
best_test = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch
mozafari = CvT(6, 50, 10, (0.0001, -0.0035), (-0.0001, 0.00006), 0.5)
if use_cuda:
    mozafari.cuda()
#%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Training The First Layer
print("Training the first layer")
if os.path.isfile("saved_l1.net"):
    mozafari.load_state_dict(torch.load("saved_l1.net"))
else:
    for epoch in range(2):
        print("Epoch", epoch)           
        for i, (data,target) in enumerate(ImageNet_trainLoader):              
            train_unsupervise(mozafari, data, 1)  
                     
    torch.save(mozafari.state_dict(), "saved_l1.net")
#%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Training The Second Layer
print("Training the second layer")
if os.path.isfile("saved_l2.net"):
    mozafari.load_state_dict(torch.load("saved_l2.net"))
else:
    for epoch in range(4):
        print("Epoch", epoch)
        iter = 0
        for data,targets in ImageNet_trainLoader:
            train_unsupervise(mozafari, data, 2)        
           
    torch.save(mozafari.state_dict(), "saved_l2.net")

# initial adaptive learning rates
apr = mozafari.stdp3.learning_rate[0][0].item()
anr = mozafari.stdp3.learning_rate[0][1].item()
app = mozafari.anti_stdp3.learning_rate[0][1].item()
anp = mozafari.anti_stdp3.learning_rate[0][0].item()

adaptive_min = 0
adaptive_int = 1
apr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * apr
anr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * anr
app_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * app
anp_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * anp

# perf
best_train = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch
best_test = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch

# Training The Third Layer
print("Training the third layer")
for epoch in range(100):
    print("Epoch #:", epoch)
    perf_train = np.array([0.0,0.0,0.0])
    for data,targets in ImageNet_trainLoader:
        perf_train_batch = train_rl(mozafari, data, targets)
        print(perf_train_batch)
        #update adaptive learning rates
        apr_adapt = apr * (perf_train_batch[1] * adaptive_int + adaptive_min)
        anr_adapt = anr * (perf_train_batch[1] * adaptive_int + adaptive_min)
        app_adapt = app * (perf_train_batch[0] * adaptive_int + adaptive_min)
        anp_adapt = anp * (perf_train_batch[0] * adaptive_int + adaptive_min)
        mozafari.update_learning_rates(apr_adapt, anr_adapt, app_adapt, anp_adapt)
        perf_train += perf_train_batch
       

    perf_train /= len(ImageNet_trainLoader)
    if best_train[0] <= perf_train[0]:
        best_train = np.append(perf_train, epoch)
    print("Current Train:", perf_train)
    print("   Best Train:", best_train)

    # for data,targets in ImageNet_valLoader :

    #     perf_test = test(mozafari, data, targets)
    #     if best_test[0] <= perf_test[0]:
    #         best_test = np.append(perf_test, epoch)
    #         torch.save(mozafari.state_dict(), "saved.net")
    #     print(" Current Test:", perf_test)
    #     print("    Best Test:", best_test)

