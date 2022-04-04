import os
import torch
from SpykeTorch import functional as sf
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
import pytorch_spiking
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

use_cuda = True
def pair(t):
    return t if isinstance(t, tuple) else (t, t)
class SepConv2d(torch.nn.Module):
    def __init__(self,                               
                 dilation=1,):
        super(SepConv2d, self).__init__()        
        self.conv1 = snn.Convolution(50, 50, 5, 0.8, 0.05)       
        self.bn = torch.nn.BatchNorm2d(50)
        self.conv2 = snn.Convolution(50, 50, 2, 0.8, 0.05)

    def forward(self, x):        
        x = self.conv1(sf.pad(x, (0,2,2,0)))
        x = self.bn(x)
        x = self.conv2(sf.pad(x, (0,3,3,0)))     
        return x
        

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm([3136, 50])  
        self.norm2 = nn.LayerNorm([784, 50]) 
        self.norm3 = nn.LayerNorm([196, 50]) 
        self.fn = fn
    def forward(self, x, **kwargs):
        if x.shape[1] == 3136:
            return self.fn(self.norm1(x), **kwargs)
        if x.shape[1] == 784:
            return self.fn(self.norm2(x), **kwargs)           
        if x.shape[1] == 196:
            return self.fn(self.norm3(x), **kwargs)        


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.5):
        super().__init__()        
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.gelu = pytorch_spiking.SpikingActivation(nn.GELU())
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)
        

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x.to(device='cpu'))
        x = self.dropout1(x.to(device='cuda'))
        x = self.linear2(x.to(device='cuda'))
        x = self.dropout2(x)
        return x



class ConvAttention(nn.Module):
    def __init__(self, dim, img_size, heads, dim_head ,  dropout = 0.5):
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.heads = 1
        self.dim_head = dim_head     
        self.inner_dim = self.dim_head *  self.heads
        self.project_out = not (heads == 1 and dim_head == self.dim)
        self.scale = self.dim_head ** -0.5            
        self.to_q = SepConv2d()
        self.to_k = SepConv2d()
        self.to_v = SepConv2d()
        self.linear1 = nn.Linear(self.inner_dim, self.dim)
        self.drop1 = nn.Dropout(dropout)       
        self.identify = nn.Identity()       

    def forward(self, x):   
        b, n,_, h = *x.shape, self.heads
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)               
        v = self.to_v(x)        
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)      
        k = self.to_k(x)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        m = pytorch_spiking.SpikingActivation(nn.Softmax(dim=-1))
        dots1 = dots.view (-1,  dots.shape[2], dots.shape[3])   
        attn = m(dots1.to(device='cpu'))
        attn = attn.reshape(dots.shape[0], dots.shape[1], dots1.shape[1], dots1.shape[2])
        attn =attn.to(device='cuda')     
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        if self.project_out:
            out =  self.linear1(out)
        else:
            out = self.identify(out)
        return out
###########################################################################
class Transformer(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim,dropout):                        
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.dim = dim
        self.img_size = img_size  
        self.dim_head = dim_head          
        self.dropout = dropout       
        self.dim_head = dim_head  
        self.mlp_dim = mlp_dim     
        for _ in range(depth):      
            self.layers.append(nn.ModuleList([
                PreNorm(self.dim, ConvAttention(self.dim, self.img_size, self.heads, self.dim_head,self.dropout)), 
                PreNorm(self.dim, FeedForward(self.dim, self.mlp_dim, self.dropout))
            ]))

    def forward(self, x): 
        for attn, ff in self.layers:      
            x = attn(x) + x      
            x = ff(x) + x   
        return x
###########################################################################
class CvT(nn.Module):
    def __init__(self, input_channels, features_per_class, number_of_classes,stdp_lr, anti_stdp_lr,
            dropout=0.5, image_size=224, dim=50, out_channel=[50, 50, 50], kernel=[169, 29, 15],
                 heads=[1, 1, 1] , depth = [1, 2, 10],pool='cls',scale_dim=1):
        super(CvT, self).__init__()        
        self.features_per_class = features_per_class
        self.number_of_classes = number_of_classes
        self.number_of_features = features_per_class * number_of_classes  
        self.stdp_lr = stdp_lr
        self.anti_stdp_lr = anti_stdp_lr
        self.k1 = 5
        self.r1 = 3
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim
        ##### Stage 1 ################################################################
        self.conv1 = snn.Convolution(6, 50, kernel[0], 0.8, 0.05)   
        self.stdp1 = snn.STDP(self.conv1, (0.004, -0.003))                
        self.conv1_t = 15
        self.R1 = Rearrange('b c h w -> b (h w) c', h = image_size//4, w = image_size//4)
        self.norm1 = nn.LayerNorm([14*image_size, out_channel[0]])        
        self.stage1_transformer = Transformer(dim=self.dim, img_size=image_size//4,depth=depth[0], heads=heads[0], dim_head=self.dim,
                                              mlp_dim=self.dim * scale_dim, dropout=dropout)
        self.R11 = Rearrange('b (h w) c -> b c h w', h = image_size//4, w = image_size//4)

        # ###### Stage 2 ###################################################################
        scale = heads[1]//heads[0]
        self.dim2 = scale*self.dim
        self.conv2 = snn.Convolution(50, 50, kernel[1], 0.8, 0.05)
        self.conv2_t = 10
        self.k2 = 5
        self.r2 = 1      
        self.stdp2 = snn.STDP(self.conv2, (0.004, -0.003))  
        self.R2 = Rearrange('b c h w -> b (h w) c', h = image_size//8, w = image_size//8)
        self.norm2 = nn.LayerNorm([784, 50])        
        self.stage2_transformer =  Transformer(dim=self.dim , img_size=image_size//8, depth=depth[1], heads=heads[1], dim_head=self.dim,
                                              mlp_dim=self.dim  * scale_dim, dropout=dropout)                          
                                           
        self.R22 = Rearrange('b (h w) c -> b c h w', h = image_size//8, w = image_size//8) 
        # ###### Stage 3 ##################################################################
        input_channels = self.dim2
        scale = heads[2] // heads[1]
        self.dim3 = scale*self.dim2
        self.conv3 = snn.Convolution(50, 50, kernel[2], 0.8, 0.05)
        
        self.stdp3 = snn.STDP(self.conv3, (0.004, -0.003), False, 0.2, 0.8)
        self.anti_stdp3 = snn.STDP(self.conv3, (-0.004, 0.0005), False, 0.2, 0.8)
        self.R3 = Rearrange('b c h w -> b (h w) c', h = image_size//16, w = image_size//16)
        self.norm3 = nn.LayerNorm([196, 50])        
        self.stage3_transformer = Transformer(dim=self.dim, img_size=image_size//16, depth=depth[2], heads=heads[2], dim_head=self.dim,
                                              mlp_dim=self.dim* scale_dim, dropout=dropout)
        self.R33 = Rearrange('b (h w) c -> b c h w', h = image_size//16, w = image_size//16)
        self.max_ap = Parameter(torch.Tensor([0.15]))
        self.decision_map = []
        for i in range(10):
            self.decision_map.extend([i]*5)
        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0
    def forward(self, input, max_layer):
        input = input.float()            
        if self.training:
            pot = self.conv1(input)
            pot = self.R1(pot)        
            pot = self.norm1(pot)                      
            pot = self.stage1_transformer(pot)              
            pot = self.R11(pot)                       
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if max_layer == 1:
                self.spk_cnt1 += 1
                if self.spk_cnt1 >= 500:
                    self.spk_cnt1 = 0
                    ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp1.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
                self.save_data(input, pot, spk, winners)
                return spk, pot  
            spk_in =  spk    
            pot = self.conv2(spk_in)
            pot = self.R2(pot)
            pot = self.norm2(pot) 
            pot = self.stage2_transformer(pot)   
            pot = self.R22(pot) 
            spk, pot = sf.fire(pot, self.conv2_t, True)
            if max_layer == 2:
                self.spk_cnt2 += 1
                if self.spk_cnt2 >= 500:
                    self.spk_cnt2 = 0
                    ap = torch.tensor(self.stdp2.learning_rate[0][0].item(), device=self.stdp2.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp2.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k2, self.r2, spk)           
                self.save_data(spk_in, pot, spk, winners)
                return spk, pot
            spk_in = spk  
            pot = self.conv3(spk_in)       
            pot = self.R3(pot)
            pot = self.norm3(pot)
            pot = self.stage3_transformer(pot)
            pot = self.R33(pot)
            spk = sf.fire(pot)      
            winners = sf.get_k_winners(pot, 1, 0, spk)
            self.ctx["input_spikes"] = spk_in
            self.ctx["potentials"] = pot
            self.ctx["output_spikes"] = spk
            self.ctx["winners"] = winners
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
    #     else:
    #         pot = self.conv1(input)
    #         pot = self.R1(pot)
    #         pot = self.norm1(pot)
    #         pot= self.stage1_transformer(pot)            
    #         pot = self.R11(pot)
    #         spk, pot = sf.fire(pot, self.conv1_t, True)
    #         if max_layer == 1:
    #             return spk, pot
    #         pot = self.conv2(spk)
    #         pot = self.R2(pot)
    #         pot = self.norm2(pot)
    #         pot = self.stage2_transformer(pot) 
    #         pot = self.R22(pot)                           
    #         spk, pot = sf.fire(pot, self.conv2_t, True)
    #         if max_layer == 2:
    #             return spk, pot
    #         pot = self.conv3(spk)             
    #         pot = self.R3(pot)
    #         pot = self.norm3(pot)
    #         pot = self.stage3_transformer(pot)        
    #         pot = self.R33(pot)                     
    #         spk = sf.fire(pot)
    #         winners = sf.get_k_winners(pot, 1, 0, spk)
    #         output = -1
    #         if len(winners) != 0:
    #             output = self.decision_map[winners[0][0]]
    #         return output

    def save_data(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners
    def stdp(self, layer_idx):
        if layer_idx == 1:
            self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def update_learning_rates(self, stdp_ap, stdp_an, anti_stdp_ap, anti_stdp_an):
        self.stdp3.update_all_learning_rate(stdp_ap, stdp_an)
        self.anti_stdp3.update_all_learning_rate(anti_stdp_an, anti_stdp_ap)

    def reward(self):
        self.stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def punish(self):
        self.anti_stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

