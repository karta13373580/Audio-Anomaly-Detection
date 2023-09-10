import torch
from torch.nn import functional as F 
import numpy as np
from einops import rearrange, repeat

# block=2
# halo=1

#[1, 1, 8, 8]
# x = torch.Tensor([[[[  1,  2,  3,  4,  5, 6, 7, 8],
#    					[  9, 10, 11, 12, 13, 14, 15, 16],
#    					[ 17, 18, 19, 20, 21, 22, 23, 24],
#    					[ 25, 26, 27, 28, 29, 30, 31, 32],
#    					[ 33, 34, 35, 36, 37, 38, 39, 40],
#                     [ 41, 42, 43, 44, 45, 46, 47, 48],
#                     [ 49, 50, 51, 52, 53, 54, 55, 56],
#                     [ 57, 58, 59, 60, 61, 62, 63, 64]]]])

#[1, 1, 4, 4]
# x = torch.Tensor([[[[ 1, 2, 3, 4],
#    					[ 5, 6, 7, 8],
#    					[ 9, 10, 11, 12],
#                     [ 13, 14, 15, 16]]]])

# print(x.shape)
# b, c, h, w = x.shape
# torch.set_printoptions(profile="full")

# x = F.unfold(x, kernel_size = block + halo * 2, stride = block, padding = halo) 
# print(x)
# print("x",x.size())

# x_window = rearrange(x, 'b (c j) i -> (b i) j c', c = c)
# print("x_window",x_window.size())


#多channel版本
# x = torch.Tensor([[[[ 1, 2, 3, 4],
#    					[ 5, 6, 7, 8],
#    					[ 9, 10, 11, 12],
#                     [ 13, 14, 15, 16]],
					
# 				   [[ 17, 18, 19, 20],
#    					[ 21, 22, 23, 24],
#    					[ 25, 26, 27, 28],
#                     [ 29, 30, 31, 32]]]])
# print(x.shape)
# torch.set_printoptions(profile="full")

# x = F.unfold(x, kernel_size = 3, stride = 1, padding = 0) 
# # x = F.unfold(x, kernel_size = block + halo * 2, stride = block, padding = halo) 
# print(x)
# print("x",x.size())

# x_window = rearrange(x, 'b (c j) i -> (b i) j c', c = c)
# print("x_window",x_window.size())

#====================================================================================

q = [[[ 1,  2,  3],
      [ 4,  5,  6],
	  [ 7,  8,  9]],

     [[10, 11, 12],
      [13, 14, 15],
	  [16, 17, 18]]]
q = np.array(q)
print("q",q.shape)

k = [[[ 1,  2,  3],
      [ 4,  5,  6],
	  [ 7,  8,  9],
	  [10, 11, 12]],

     [[13, 14, 15],
	  [16, 17, 18],
	  [19, 20, 21],
	  [22, 23, 24]]]
k = np.array(k)
print("k",k.shape)

# q = torch.rand(2,3,3)
# k = torch.rand(2,4,3)
q = torch.from_numpy(q)
k = torch.from_numpy(k)
c = torch.einsum('b i d, b j d -> b i j', q, k)
#算法如下
# tensor([[[  1*1+2*2+3*3=14,   1*4+2*5+3*6=32,   1*7+2*8+3*9=50,   1*10+2*11+3*12=68],
#          [  4*1+5*2+6*3=32,   4*4+5*5+6*6=77,  4*7+5*8+6*9=122,  4*10+5*11+6*12=167],
#          [  50,  122,  194,  266]],

#         [[ 464,  563,  662,  761],
#          [ 590,  716,  842,  968],
#          [ 716,  869, 1022, 1175]]]

print('c',c.shape) #torch.Size([2, 3, 4])
print(c)
