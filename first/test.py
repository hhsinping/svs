import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from skimage.io import imsave
from skimage.transform import resize
from os import listdir
from os.path import join

import warnings
warnings.simplefilter("ignore")

import models.networks as networks
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

root = "../"

# Load network
#opt = Namespace(netG = 'spade', ngf = 64, num_upsampling_layers='normal', crop_size=384, aspect_ratio=1.0, use_vae=False, semantic_nc=151, norm_G='spectralspadesyncbatch3x3', gpu_ids=[0], init_type='xavier', init_variance=0.02)
#net1 = networks.define_G(opt)
#net1.load_state_dict(torch.load("net_first.pth"))
net1 = torch.load("net_first.pth")
net1.eval()
net1.cuda()


transform_label = transforms.Compose([
    transforms.Resize(size=[256, 256], interpolation=Image.NEAREST),
    transforms.ToTensor()]
)



# Load label map

label_folder = root+"inputs"
label_paths = [f for f in listdir(label_folder)]

for label_path in label_paths:

    name = join(label_folder, label_path)
    label = Image.open(name)

    label_tensor = transform_label(label) * 255.0
    label_tensor = label_tensor+1
    label_tensor[label_tensor == 256] = 150
    label_tensor = torch.unsqueeze(label_tensor, 0).long().cuda()

    bs, _, h, w = label_tensor.size()
    input_label = torch.cuda.FloatTensor(bs, 151, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_tensor, 1.0)


    # Inference
    x = net1(input_semantics)
    x = (((x + 1) / 2)*255).cpu().detach().numpy().astype(np.uint8)
    depth = x[0,0,:,:]
    rgb = np.transpose(x[0,1:,:,:], (1, 2, 0))



    depth = (np.clip(resize(depth, (384, 384)), 0, 1) * 255).astype(np.uint8)
    rgb = (np.clip(resize(rgb, (384, 384)), 0, 1) *255).astype(np.uint8)

    # Save outputs
    imsave(root+'inters/'+label_path.replace('.png','_rgb.png'),rgb)
    imsave(root+'inters/'+label_path.replace('.png','_depth.png'),depth)





