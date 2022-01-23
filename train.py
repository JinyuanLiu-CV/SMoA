import os
import random
import sys
import time
import glob
import numpy as np
import torch
from PIL import Image

import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import Encoder, Decoder
import pytorch_msssim
import torchvision.transforms as transforms
import genotypes
parser = argparse.ArgumentParser("untitled")

parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='init learning rate')  #0.025-->2e-4
# parser.add_argument('--learning_rate_min', type=float, default=1e-5, help='min learning rate')  #0.001-->1e-4
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
# parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
# parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')

parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
# parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--dataset1', type=str, default=r'C:\Users\ADMIN\Desktop\DATA\data128\crop_infrared', help='Infrared images for training')
parser.add_argument('--dataset2', type=str, default=r'C:\Users\ADMIN\Desktop\DATA\data128\crop_visible', help='Visible images for training')

args = parser.parse_args()
args.save = 'train{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True  # 加速
    torch.manual_seed(args.seed)  # 为CUP设置随机种子
    cudnn.enabled = True  # 使用非确定性算法优化运行
    torch.cuda.manual_seed(args.seed)  # 为GPU设置随机种子
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    mse_loss = torch.nn.MSELoss().cuda()
    ssim_loss = pytorch_msssim.msssim

    genotype_en1 = eval('genotypes.%s' % 'genotype_en1')
    genotype_en2 = eval('genotypes.%s' % 'genotype_en2')

    genotype2 = eval('genotypes.%s' % 'genotype_de')

    encoder1 = Encoder(args.init_channels, args.layers, genotype_en1).cuda()
    encoder2 = Encoder(args.init_channels, args.layers, genotype_en2).cuda()

    decoder = Decoder(args.init_channels, args.layers, genotype2).cuda()


    # logging.info("param size = %fMB", utils.count_parameters_in_MB(encoder1)*3)

    para1 = [{'params': encoder1.parameters(), 'lr': args.learning_rate},
              {'params': decoder.parameters(), 'lr': args.learning_rate}]
    para2 = [{'params': encoder2.parameters(), 'lr': args.learning_rate},
              {'params': decoder.parameters(), 'lr': args.learning_rate}]
    optimizer1 = torch.optim.Adam(para1, args.learning_rate)
    optimizer2 = torch.optim.Adam(para2, args.learning_rate)

    epochs = args.epochs
    Infrared_path_list = utils.list_images(args.dataset1)
    Visible_path_list = utils.list_images(args.dataset2)
    random.shuffle(Infrared_path_list)
    random.shuffle(Visible_path_list)
    train_num = 15000

    Infrared_path_list = Infrared_path_list[:train_num]
    Visible_path_list = Visible_path_list[:train_num]
    train_queue1, batches = utils.load_dataset(Infrared_path_list, args.batch_size)  # infrared train
    train_queue2, batches = utils.load_dataset(Visible_path_list, args.batch_size)  # infrared train
    train_queue12 = [train_queue1, train_queue2]
    optimizer12 = [optimizer1, optimizer2]
    encoder12 = [encoder1, encoder2]
    print("len of(infrared_train_queue):", len(train_queue1)*2)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer3, gamma=0.9)
    for epoch in range(epochs):
        # lr = scheduler.get_last_lr()
        # logging.info('epoch %d lr %e', epoch, lr[0])

        # training

        train(train_queue12, batches, args, encoder12, decoder, mse_loss, ssim_loss, optimizer12, epoch)

        if (epoch+1)%5==0:
            utils.save(encoder1, os.path.join(args.save, 'encoder1_epoch'+str(epoch+1)+'.pt'))
            utils.save(encoder2, os.path.join(args.save, 'encoder2_epoch'+str(epoch+1)+'.pt'))
            utils.save(decoder, os.path.join(args.save, 'decoder_epoch'+str(epoch+1)+'.pt'))


        # scheduler.step()


tensor_to_pil = transforms.ToPILImage()


def train(train_queue_IV, batches, args, encoder12, decoder, mse_loss, ssim_loss, optimizer12, epoch):
    encoder12[0].train()
    encoder12[1].train()
    decoder.train()
    for batch in range(batches):
        for i, train_queue, encoder, optimizer in zip(range(2), train_queue_IV, encoder12, optimizer12):

            image_paths_train = train_queue[batch * args.batch_size:(batch * args.batch_size + args.batch_size)]  # 训练一批

            inputs = utils.get_train_images_auto(image_paths_train).cuda()

            en1, en2 = encoder(inputs)
            outputs = decoder(en1, en2)


            optimizer.zero_grad()

            ssim_loss_value = 0.
            pixel_loss_value = 0.
            for output, input in zip(outputs, inputs):
                output, input = torch.unsqueeze(output, 0), torch.unsqueeze(input, 0)
                pixel_loss_temp = mse_loss(input, output)
                ssim_loss_temp = ssim_loss(input, output, normalize=True, val_range=255)
                ssim_loss_value += (1 - ssim_loss_temp)
                pixel_loss_value += pixel_loss_temp
            ssim_loss_value /= len(outputs)
            pixel_loss_value /= len(outputs)

            total_loss = pixel_loss_value + 100*ssim_loss_value  # 加权？
            # total_loss = torch.tensor(total_loss, dtype=torch.float)
            total_loss.backward()
            # nn.utils.clip_grad_norm_(model_former.parameters(), args.grad_clip)
            # nn.utils.clip_grad_value_(model_former.parameters(), args.grad_clip)
            # nn.utils.clip_grad_value_(model_latter.parameters(), args.grad_clip)
            optimizer.step()
            if i==0:
                logging.info("Infrared_epoch: %d batch: %d loss: %f", epoch, batch, total_loss)
            else:
                logging.info("Visible_epoch: %d batch: %d loss: %f", epoch, batch, total_loss)


if __name__ == '__main__':
    main()
