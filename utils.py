import os
import random
from os import listdir
from os.path import join

import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

from imageio import imread, imsave


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    # random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    # print('BATCH SIZE %d.' % BATCH_SIZE)
    # print('Train images number %d.' % num_imgs)
    # print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]  # 多出来的不用处理？
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches

def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = imread(path, pilmode=mode)
        # image = Image.open(path, )
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')

    if height is not None and width is not None:
        #image = imresize(image, [height, width], interp='nearest')
        image = np.array(Image.fromarray(image).resize((height, width)))
    return image


'''
def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths] #加个[]？
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L': #什么模式
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])#灰度图片
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])#RGB
        images.append(image)

    images = np.stack(images, axis=0) #增加一维
    images = torch.from_numpy(images).float()
    return images
'''

pil_to_tensor = transforms.ToTensor()

def get_train_images_auto(paths):
    if isinstance(paths, str):
        paths = [paths]  # 加个[]？
    images = []
    for path in paths:
        image = Image.open(path)
        mode = image.mode
        if mode == 'RGB':  # 什么模式
            image = image.convert('L')
        image = np.reshape(image, [1, image.size[1], image.size[0]])  # 灰度图片
        # image = pil_to_tensor(image)
        # print(image.shape)
        images.append(image)
    images = np.stack(images, axis=0)  # 增加一维
    images = torch.from_numpy(images).float()
    # images = torch.stack(images)
    # print(images.size())
    return images

#0-255
# def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
#     if isinstance(paths, str):
#         paths = [paths] #加个[]？
#     images = []
#     for path in paths:
#         image = get_image(path, height, width, mode=mode)
#         if mode == 'L': #什么模式
#             image = np.reshape(image, [1, image.shape[0], image.shape[1]])#灰度图片
#         else:
#             image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])#RGB
#         images.append(image)
#
#     images = np.stack(images, axis=0) #增加一维
#     images = torch.from_numpy(images).float()
#     return images


def get_test_images(paths, height=None, width=None, mode='L'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:#不就一张图片吗
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            # test = ImageToTensor(image).numpy()
            # shape = ImageToTensor(image).size()
            image = ImageToTensor(image).float().numpy()*255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


'''
loader = transforms.Compose([
    transforms.ToTensor()])


def get_train_images_auto(paths):
    if isinstance(paths, str):
        paths = [paths]  # 加个[]？
    images = []
    for path in paths:
        image = Image.open(path)
        # print(image.size)
        mode = image.mode
        if mode == 'RGB':  # 什么模式
            image = image.convert('L')
        img = loader(image)
        images.append(img)
    images = torch.stack(images)  # 增加一维
    # print(images)
    return images
'''


def list_images(directory):  # 得到所有图片路径
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])  # names没有用到
    return images


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')

    torch.save(state, filename)

    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')

        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        print('x.size:', x.shape)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
