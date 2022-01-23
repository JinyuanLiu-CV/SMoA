import cv2
import torch
from model import Encoder, Decoder
from os.path import join
from os import listdir
import PIL.Image as Image
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
from os.path import exists
from utils import get_train_images_auto, get_test_images
import genotypes
import torch.nn.functional as F
import glob

tensor_to_pil = transforms.ToPILImage()
model_dir = r'E:\已投论文\SPL\code\trainJT'
encoder1_path1 = join(model_dir, 'encoder1_epoch6.pt')
encoder2_path2 = join(model_dir, 'encoder2_epoch6.pt')
decoder_path = join(model_dir, 'decoder_epoch6.pt')

genotype_en1 = eval("genotypes.%s" % 'genotype_en1')
genotype_en2 = eval("genotypes.%s" % 'genotype_en2')

genotype2 = eval("genotypes.%s" % 'genotype_de')

encoder1 = Encoder(16, 2, genotype_en1).cuda()
encoder2 = Encoder(16, 2, genotype_en2).cuda()

decoder = Decoder(16, 2, genotype2).cuda()


params1 = torch.load(encoder1_path1)
params2 = torch.load(encoder2_path2)
params3 = torch.load(decoder_path)

encoder1.load_state_dict(params1)
encoder2.load_state_dict(params2)

decoder.load_state_dict(params3)

encoder1.eval()
encoder2.eval()

decoder.eval()

c = 1e-2

image_dir2 = r'C:\Users\ADMIN\Desktop\Test'

par = os.getcwd()
image_dir2 = par + '\\testImage'
save_dir = par + '\\result'

if not exists(save_dir):
    os.mkdir(save_dir)

def vsm2(tensor):
    his = tensor.histc(bins=256, min=0, max=255)
    sal = torch.zeros(256).to(torch.int64).cuda()
    for i in range(256):
        for j in range(256):
            sal[i] += abs(j-i)*his[j]
    sal = sal.div(sal.max())#.to(torch.float32)
    map = torch.zeros_like(tensor).cuda().to(torch.float32)
    for i in range(256):
        map[tensor == i] = sal[i]
    return map


def vsf3(tensor1, tensor2):
    t1 = (tensor1/tensor1.max()*255).to(torch.int)
    t2 = (tensor2/tensor2.max()*255).to(torch.int)
    weight1 = vsm2(t1)
    weight2 = vsm2(t2)
    F = (0.5 + 0.5 * (weight1 - weight2)) * tensor1 + (0.5 + 0.5 * (weight2 - weight1)) * tensor2
    return F

def fuse_L(ir_path, vis_path, save_dir, name):
    image_ir_path = join(ir_path)
    image_vis_path = join(vis_path)
    tensor_ir = get_test_images(image_ir_path).cuda()
    tensor_ir.requires_grad = False
    tensor_vis = get_test_images(image_vis_path).cuda()
    tensor_vis.requires_grad = False

    en11, en12 = encoder1(tensor_ir)
    en21, en22 = encoder2(tensor_vis)

    en1 = vsf3(en11, en21)
    en2 = vsf3(en12, en22)

    tensor_f = decoder(en1, en2).cpu()
    image_tensor = tensor_f.squeeze()

    image_array = np.asarray(image_tensor.detach())
    image_pil = Image.fromarray(image_array).convert('L')
    image_pil.save(os.path.join(save_dir, name.split('.')[0] + '.jpg'))


def fuse_RGB(ir_path, vis_path, save_dir, name):
    ir = cv2.imread(ir_path)
    vis = cv2.imread(vis_path)
    vis_ycrcb = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)

    tensor1 = torch.tensor(ir[:, :, 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    tensor2 = torch.tensor(vis_ycrcb[:, :, 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

    en11, en12 = encoder1(tensor1)
    en21, en22 = encoder2(tensor2)

    en1 = vsf3(en11, en21)
    en2 = vsf3(en12, en22)

    tensor_f = decoder(en1, en2).cpu()

    image_tensor = tensor_f.squeeze()
    image_tensor = torch.clamp(image_tensor, 0, 255)
    image_array = np.asarray(image_tensor.detach(), dtype=int)
    re = np.stack([image_array, vis_ycrcb[:, :, 1], vis_ycrcb[:, :, 2]], axis=2).astype(np.uint8)
    re = cv2.cvtColor(re, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(os.path.join(save_dir, name.split('.')[0] + '_RGB.jpg'), re)
    cv2.imwrite(os.path.join(save_dir, name.split('.')[0] + '_L.jpg'), image_array.astype(np.uint8))

def test():
    with torch.no_grad():
        namelist = os.listdir(os.path.join(image_dir2, 'ir'))
        for name in namelist:
            ir_path = os.path.join(image_dir2, 'ir', name)
            vis_path = os.path.join(image_dir2, 'vis', name)
            if name.startswith('A'):
                fuse_L(ir_path, vis_path, save_dir, name)
            else:
                fuse_RGB(ir_path, vis_path, save_dir, name)


if __name__ == '__main__':
    test()