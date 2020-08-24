import torch
import torch.nn.functional as F
import numpy as np
from skimage.io import imread, imsave
from math import pi, sqrt, exp
import warnings
warnings.simplefilter("ignore")

import models.networks as networks
from inverse_warp import *
from moviepy.editor import *

from skimage.restoration import denoise_bilateral

from os import listdir
from os.path import join

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return(b_0, b_1)




class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def gauss(s):
    sigma = 10
    n = 31
    r = range(-int(n/2),int(n/2)+1)
    layer = [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]    
    layer = [l/layer[15] for l in layer]
    layer[16:] = [0] * 15
    return layer
    
def warp(rgb, depth, intrinsic, pose1, pose2):

    intrinsic = torch.squeeze(intrinsic,1).type(torch.cuda.DoubleTensor)
    pose1 = torch.squeeze(pose1,1).type(torch.cuda.DoubleTensor)
    pose2 = torch.squeeze(pose2,1).type(torch.cuda.DoubleTensor)
    nlayer = depth.shape[1]
    depth_scale = 1/np.linspace(0.01,1,num=nlayer)
    depth = depth.type(torch.cuda.FloatTensor)
    rgb = rgb.type(torch.cuda.FloatTensor)
    
    for i in list(range(nlayer)):
        cur_depth = (depth_scale[i] * torch.ones((1,384,384))).type(torch.cuda.DoubleTensor)
        warp = inverse_warp(rgb[:,i,:,:,:].permute(0,3,1,2),cur_depth,pose2,pose1,intrinsic)
        mask = inverse_warp(torch.unsqueeze(depth[:,i-1,:,:],1).repeat(1,3,1,1),cur_depth,pose2,pose1,intrinsic)
        if i==0:
            total = warp
        else:
            rgb_by_alpha = warp * mask
            total = rgb_by_alpha + total * (1.0 - mask)

    return total
        
def warp_con(rgb_tensor, depth_tensor, intrinsic, pose1, pose2):

    depth_tensor = torch.from_numpy(depth_tensor)
    rgb_tensor = torch.from_numpy(rgb_tensor)
    depth_tensor = torch.unsqueeze(depth_tensor,0).type(torch.cuda.DoubleTensor)
    rgb_tensor = torch.unsqueeze(rgb_tensor,0).type(torch.cuda.DoubleTensor)
    out = warp(rgb_tensor, depth_tensor, intrinsic, pose1, pose2)
    out = (np.transpose(np.squeeze(out.cpu().numpy()),(1,2,0))*255.0).astype(np.uint8)

    return out
    
    
    
root = "../"

# load network 
#opt = Namespace(netG = 'spade', ngf = 64, num_upsampling_layers = 'normal', crop_size=384, aspect_ratio=1.0, use_vae=False, norm_G='spectralspadesyncbatch3x3', semantic_nc=1, nmpi=128, gpu_ids=[0], init_type='xavier', init_variance=0.02)
#net2 = networks.define_G(opt)
#net2.load_state_dict(torch.load("net_second.pth"))

label_folder = root+"inputs"
label_paths = [f for f in listdir(label_folder)]

net2 = torch.load("net_second.pth")
net2.eval()
net2.cuda()




for label_path in label_paths:
    # Read rgb/depth
    real_rgb = torch.unsqueeze(torch.cuda.FloatTensor(imread(root+'inters/'+label_path.replace('.png','_rgb.png'))/255.0 * 2 -1), 0).permute(0,3,1,2)
    real_depth = imread(root+'inters/'+label_path.replace('.png','_depth.png'))


    # Filter the depth
    noisy = real_depth
    result = denoise_bilateral(noisy, sigma_color=0.5, sigma_spatial=4, win_size=7,multichannel=False)
    b = np.percentile(noisy, list(range(100)) )
    a = np.percentile(result, list(range(100)) )
    x = estimate_coef(a, b)
    result = (result * x[1] + x[0])
    real_depth = result


    real_depth = torch.cuda.FloatTensor(real_depth)/255.0


    # Inference
    inputs = torch.cat((real_rgb, real_depth.unsqueeze(0).unsqueeze(0)) ,1)
    fake_image, fake_depth = net2(inputs)


    # Calculate alphas
    num_layer = fake_depth.shape[1]
    depth_scale = 1/np.linspace(0.01,1,num= num_layer)
    depth = 1/(real_depth*2 + 0.00001)

    alphas = []
    alpha = torch.squeeze(( (depth > depth_scale[0])))
    alphas.append(alpha)
    for i in range(1,len(depth_scale)):
        alpha = torch.squeeze(( (depth > depth_scale[i]) & (depth < depth_scale[i-1])))
        alphas.append(alpha)
    alphas = (torch.stack(alphas, 0)).type(torch.cuda.FloatTensor)

    kernel = torch.unsqueeze(torch.unsqueeze(torch.cuda.FloatTensor(gauss(1.0)), 1), 2)        
    alphas[0,:,:] = 0
    alphas = alphas.permute(1, 2, 0).view(384*384,1,num_layer)
    kernel = kernel.view(1, 1, -1)
    alphas = F.conv1d(alphas, kernel, padding = (31-1)//2)
    alphas = torch.unsqueeze(alphas.view(384,384,num_layer).permute(2,0,1), 0)


    # Calculate rgbs
    blend_weight = (fake_depth + 1.0) / 2.0
    blend_weight = blend_weight.repeat(3,1,1,1).permute(1,0,2,3)
    rgbs = blend_weight * real_rgb.repeat(128,1,1,1) + (1-blend_weight) * fake_image.repeat(128,1,1,1)
    rgbs = (rgbs +1)/2


    # save alphas and rgbs
    rgbs = np.clip(rgbs.cpu().detach().numpy(), 0, 1)
    alphas = np.clip(alphas.cpu().detach().numpy(), 0, 1)
    for i in range(128):
        rgb_layer = (np.transpose(rgbs[i,:,:,:], (1,2,0)) * 255.0).astype(np.uint8)
        depth_layer = (alphas[0,i,:,:] * 255.0).astype(np.uint8)

        imsave(root+'inters/'+label_path.replace('.png','_rgb%03d.png' % i),rgb_layer)
        imsave(root+'inters/'+label_path.replace('.png','_depth%03d.png' % i),depth_layer)




for label_path in label_paths:

    # load alphas and rgbs

    alphas = []
    for i in range(128):
        alphas.append(imread(root+'inters/'+label_path.replace('.png','_depth%03d.png' % i)))
    alphas = (np.stack(alphas)/255.0).astype(np.float32)

    rgbs = []
    for i in range(128):
        rgbs.append(imread(root+'inters/'+label_path.replace('.png','_rgb%03d.png' % i)))
    rgbs = (np.stack(rgbs)/255.0).astype(np.float32)




    # generate circular video
    dist = 0.2
    num = 30
    x =  np.sin(np.linspace(0, 2*np.pi * (num-2) / (num-1), num)) * dist
    y = np.cos(np.linspace(0, 2*np.pi * (num-2) / (num-1), num)) * dist
    z = np.linspace(0, -dist, num)
    xv = list(x) + list(x)
    yv = list(y) + list(y) 
    zv = list(z) + list(z)[::-1] 


    images = []
    for shift in range(len(xv)):
        
        intrinsic = np.reshape(np.array([192, 0, 192, 0, 192, 192, 0, 0, 1]), [1,3,3]).astype(np.float32)
        ref_pose = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]

        tgt_pose = ref_pose.copy()
        tgt_pose[3] += xv[shift]
        tgt_pose[7] += yv[shift]
        tgt_pose[11] += zv[shift]
        
        intrinsic = np.reshape(intrinsic,[1,3,3])
        ref_pose = np.reshape(ref_pose,[4,4])
        tgt_pose = np.reshape(tgt_pose,[4,4])
        pose1 = np.reshape(np.matmul(tgt_pose, np.linalg.inv(ref_pose)), [1,4,4]).astype(np.float32)
        pose2 = np.reshape(np.matmul(ref_pose, np.linalg.inv(tgt_pose)), [1,4,4]).astype(np.float32)
        pose1 = pose1[:,:3,:]
        pose2 = pose2[:,:3,:]        

        intrinsic = torch.from_numpy(intrinsic)
        pose1 = torch.from_numpy(pose1)
        pose2 = torch.from_numpy(pose2)

        out = warp_con(rgbs, alphas, intrinsic, pose1, pose2)
        images.append(out)


    clip = ImageSequenceClip(images, fps=25)
    clip.write_videofile(root+'outputs/'+label_path.replace('.png','_cir.mp4'), fps=25)





    # generate ken burns video
    dist = 0.3
    num = 15
    x = list(np.linspace(-dist, 0, num)) + list(np.linspace(0, dist, num) )
    y = [0] * 60
    z = list(np.linspace(-dist, 0, num)) + list(np.linspace(0, -dist, num))
    xv = list(x) + list(x)[::-1] 
    yv = list(y) + list(y)[::-1]  
    zv = list(z) + list(z)[::-1] 


    images = []
    for shift in range(len(xv)):    
        intrinsic = np.reshape(np.array([128, 0, 128, 0, 128, 128, 0, 0, 1]), [1,3,3]).astype(np.float32)
        ref_pose = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]

        tgt_pose = ref_pose.copy()
        tgt_pose[3] += xv[shift]
        tgt_pose[7] += yv[shift]
        tgt_pose[11] += zv[shift]
        intrinsic = np.reshape(intrinsic,[1,3,3])
        ref_pose = np.reshape(ref_pose,[4,4])
        tgt_pose = np.reshape(tgt_pose,[4,4])
        pose1 = np.reshape(np.matmul(tgt_pose, np.linalg.inv(ref_pose)), [1,4,4]).astype(np.float32)
        pose2 = np.reshape(np.matmul(ref_pose, np.linalg.inv(tgt_pose)), [1,4,4]).astype(np.float32)
        pose1 = pose1[:,:3,:]
        pose2 = pose2[:,:3,:]        

        intrinsic = torch.from_numpy(intrinsic)
        pose1 = torch.from_numpy(pose1)
        pose2 = torch.from_numpy(pose2)

        out = warp_con(rgbs, alphas, intrinsic, pose1, pose2)
        images.append(out)


    clip = ImageSequenceClip(images, fps=25)
    clip.write_videofile(root+'outputs/'+label_path.replace('.png','_ken.mp4'), fps=25)

