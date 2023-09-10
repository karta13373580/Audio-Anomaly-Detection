from src.project_parameters import ProjectParameters
from DeepLearningTemplate.model import BaseModel, load_from_checkpoint
import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from .feature import Extractor
from os.path import isfile
from torchsummary import summary
from einops import rearrange
from timm.models.layers import trunc_normal_

class LightMutilHeadSelfAttention(nn.Module):
    """calculate the self attention with down sample the resolution for k, v, add the relative position bias before softmax
    Args:
        dim (int) : features map channels or dims 
        num_heads (int) : attention heads numbers
        relative_pos_embeeding (bool) : relative position embeeding 
        no_distance_pos_embeeding (bool): no_distance_pos_embeeding
        features_size (int) : features shape
        qkv_bias (bool) : if use the embeeding bias
        qk_scale (float) : qk scale if None use the default 
        attn_drop (float) : attention dropout rate
        proj_drop (float) : project linear dropout rate
        sr_ratio (float)  : k, v resolution downsample ratio
    Returns:
        x : LMSA attention result, the shape is (B, H, W, C) that is the same as inputs.
    """
    def __init__(self, dim, num_heads=8, features_size=56, 
                relative_pos_embeeding=False, no_distance_pos_embeeding=False, qkv_bias=False, qk_scale=None, 
                attn_drop=0., proj_drop=0., sr_ratio=1.):
        super(LightMutilHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}"
        self.dim = dim 
        self.num_heads = num_heads
        head_dim = dim // num_heads   # used for each attention heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_pos_embeeding = relative_pos_embeeding
        self.no_distance_pos_embeeding = no_distance_pos_embeeding

        self.features_size = features_size

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim) 
        
        if self.relative_pos_embeeding:
            self.relative_indices = generate_relative_distance(self.features_size)
            self.position_embeeding = nn.Parameter(torch.randn(2 * self.features_size - 1, 2 * self.features_size - 1))
        elif self.no_distance_pos_embeeding:
            self.position_embeeding = nn.Parameter(torch.randn(self.features_size ** 2, self.features_size ** 2))
        else:
            self.position_embeeding = None

        if self.position_embeeding is not None:
            trunc_normal_(self.position_embeeding, std=0.2)

    def forward(self, x):
        B_orign, C_orign, H_orign, W_orign = x.shape 
        # print('x1',x.shape) #[8, 1024, 64, 64]
        # x = self.downdim(x)
        # print('x2',x.shape) #[8, 256, 64, 64]
        B, C, H, W = x.shape 
        N = H*W #4096
        x_q = rearrange(x, 'B C H W -> B (H W) C')  # translate the B,C,H,W to B (H X W) C #[8, 4096, 256]
        q = self.q(x_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # B,N,H,DIM -> B,H,N,DIM #[8, 4, 4096, 64]


        # conv for down sample the x resoution for the k, v
        if self.sr_ratio > 1:
            x_reduce_resolution = self.sr(x)
            # print("x_reduce_resolution",x_reduce_resolution.shape) #[8, 256, 32, 32]
            x_kv = rearrange(x_reduce_resolution, 'B C H W -> B (H W) C ')
            # print("x_kv",x_kv.shape) #[8, 1024, 256]
            x_kv = self.norm(x_kv)
        else:
            x_kv = rearrange(x, 'B C H W -> B (H W) C ')
        
        kv_emb = rearrange(self.kv(x_kv), 'B N (dim h l ) -> l B h N dim', h=self.num_heads, l=2)         # 2 B H N DIM #[2, 8, 4, 1024, 64]
        k, v = kv_emb[0], kv_emb[1] #[8, 4, 1024, 64]

        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B H Nq DIM) @ (B H DIM Nk) -> (B H NQ NK) #[8, 4, 4096, 1024]
        
        # TODO: add the relation position bias, because the k_n != q_n, we need to split the position embeeding matrix
        q_n, k_n = q.shape[1], k.shape[2]
        
        if self.relative_pos_embeeding:
            attn = attn + self.position_embeeding[self.relative_indices[:, :, 0].type(torch.long), self.relative_indices[:, :, 1].type(torch.long)][:, :k_n]
        elif self.no_distance_pos_embeeding:
            attn = attn + self.position_embeeding[:, :k_n]
        attn = self.softmax(attn)
        attn = self.attn_drop(attn) #[8, 4, 4096, 1024]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B H NQ NK) @ (B H NK dim)  -> (B NQ H*DIM)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, 'B (H W) C -> B C H W ', H=H, W=W)
        
        return x 

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

#=====================================================cmt===============================================================
        dim = [1024, 256, 128]
        num_heads = [16, 4, 2]
        relative_pos_embeeding = 1
        no_distance_pos_embeeding = 0
        features_size = [32, 16]
        qkv_bias = 1
        qk_scale = None
        attn_drop = 0.1
        proj_drop = 0.1
        sr_ratio = [2, 4, 8]

        self.LMHSA1 = LightMutilHeadSelfAttention(
            dim = dim[2], 
            num_heads = num_heads[2],
            relative_pos_embeeding = relative_pos_embeeding,
            no_distance_pos_embeeding = no_distance_pos_embeeding,
            features_size = features_size[0],
            qkv_bias = qkv_bias,
            qk_scale = qk_scale,
            attn_drop = attn_drop,
            proj_drop = proj_drop,
            sr_ratio = sr_ratio[2])

        self.LMHSA2 = LightMutilHeadSelfAttention(
            dim = dim[2], 
            num_heads = num_heads[2],
            relative_pos_embeeding = relative_pos_embeeding,
            no_distance_pos_embeeding = no_distance_pos_embeeding,
            features_size = features_size[1],
            qkv_bias = qkv_bias,
            qk_scale = qk_scale,
            attn_drop = attn_drop,
            proj_drop = proj_drop,
            sr_ratio = sr_ratio[2])

#=======================================================================================================================

#==================================================self-attention=======================================================

        self.attn3 = Self_Attn( latent_dim*4,  'relu')
        self.attn4 = Self_Attn( latent_dim*5, 'relu') 
        self.attn5 = Self_Attn( latent_dim*6, 'relu') 

#=======================================================================================================================