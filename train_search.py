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
from model_search import Encoder, Decoder
from architect import Architect
import pytorch_msssim
import torchvision.transforms as transforms

parser = argparse.ArgumentParser("untitled")

parser.add_argument('--batch_size', type=int, default=4, help='batch size')  # 64改成了4
parser.add_argument('--learning_rate', type=float, default=1e-5, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=1e-6, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=4e-4, help='learning rate for arch encoding')  # 3e-4
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')  # 1e-3
parser.add_argument('--dataset1', type=str, default=r'C:\Users\ADMIN\Desktop\search_U2F\data_vis_ir64_160\crop_infrared160', help='Infrared images for training')
parser.add_argument('--dataset2', type=str, default=r'C:\Users\ADMIN\Desktop\search_U2F\data_vis_ir64_160\crop_visible160', help='Visible images for training')
args = parser.parse_args()
args.save = 'search-C{}-B{}-{}'.format(args.init_channels, args.batch_size, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
os.mkdir(args.save+'/output')


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

    encoder1 = Encoder(args.init_channels, args.layers).cuda()
    encoder2 = Encoder(args.init_channels, args.layers).cuda()

    decoder = Decoder(args.init_channels, args.layers).cuda()

    # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    para1 = [{'params': encoder1.parameters(), 'lr': args.learning_rate},
            {'params': decoder.parameters(), 'lr': args.learning_rate}]
    para2 = [{'params': encoder2.parameters(), 'lr': args.learning_rate},
            {'params': decoder.parameters(), 'lr': args.learning_rate}]
    optimizer1 = torch.optim.SGD(
        para1,
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer2 = torch.optim.SGD(
        para2,
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    epochs = args.epochs
    # 加载数据集
    Infrared_path_list = utils.list_images(args.dataset1)
    Visible_path_list = utils.list_images(args.dataset2)
    # train_num = len(total_path_list)//2
    random.shuffle(Infrared_path_list)
    random.shuffle(Visible_path_list)
    # train_num = 15000
    train_num = 15000

    Infrared_path_list = Infrared_path_list[:train_num]
    Visible_path_list = Visible_path_list[:train_num]

    train_queue1, batches = utils.load_dataset(Infrared_path_list[:train_num//2], args.batch_size)
    valid_queue1, batches = utils.load_dataset(Infrared_path_list[train_num//2:train_num], args.batch_size)

    train_queue2, batches = utils.load_dataset(Visible_path_list[:train_num//2], args.batch_size)
    valid_queue2, batches = utils.load_dataset(Visible_path_list[train_num//2:train_num], args.batch_size)


    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, args.epochs, eta_min=args.learning_rate_min)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, args.epochs, eta_min=args.learning_rate_min)
    architect1 = Architect(encoder1, decoder, args, mse_loss, ssim_loss)
    architect2 = Architect(encoder2, decoder, args, mse_loss, ssim_loss)

    train_queue = [train_queue1, train_queue2]
    valid_queue = [valid_queue1, valid_queue2]
    encoder = [encoder1, encoder2]
    optimizer = [optimizer1, optimizer2]
    architect = [architect1, architect2]
    for epoch in range(epochs):
        # lr = scheduler.get_lr()[0]
        lr1 = scheduler1.get_last_lr()
        lr2 = scheduler2.get_last_lr()

        logging.info('epoch %d lr1 %e lr2 %e', epoch, lr1[0], lr2[0])

        genotype_en1 = encoder1.genotype()
        genotype_en2 = encoder2.genotype()
        genotype_de = decoder.genotype()
        logging.info('genotype_en1 = %s', genotype_en1)
        logging.info('genotype_en2 = %s', genotype_en2)
        logging.info('genotype_de = %s', genotype_de)

        print(F.softmax(encoder1.alphas, dim=-1))
        print(F.softmax(encoder2.alphas, dim=-1))
        print(F.softmax(decoder.alphas, dim=-1))

        logging.info('en1 = %s', F.softmax(encoder1.alphas, dim=-1))
        logging.info('en2 = %s', F.softmax(encoder2.alphas, dim=-1))
        logging.info('de = %s', F.softmax(decoder.alphas, dim=-1))

        # training

        train(train_queue, valid_queue, batches, encoder, decoder, architect, mse_loss, ssim_loss, optimizer, [lr1, lr2], epoch)
        if (epoch+1) % 5 == 0:
            utils.save(encoder1, os.path.join(args.save, 'encoder1_epoch'+str(epoch+1)+'.pt'))
            utils.save(encoder2, os.path.join(args.save, 'encoder2_epoch'+str(epoch+1)+'.pt'))
            utils.save(decoder, os.path.join(args.save, 'decoder_epoch'+str(epoch+1)+'.pt'))

        scheduler1.step()
        scheduler2.step()


tensor_to_pil = transforms.ToPILImage()


def train(train_queue12, valid_queue12, batches, encoder12, decoder, architect12, mse_loss, ssim, optimizer12, lr12, epoch):
    encoder12[0].train()
    encoder12[1].train()
    decoder.train()
    for batch in range(batches):
        for i, train_queue, valid_queue, architect, optimizer, encoder, lr in zip(range(2), train_queue12, valid_queue12, architect12, optimizer12, encoder12, lr12):
            image_paths_train = train_queue[batch * args.batch_size:(batch * args.batch_size + args.batch_size)]  # 训练一批
            inputs = utils.get_train_images_auto(image_paths_train).cuda()  # 取出一批图片并且变成张量

            image_paths_valid = valid_queue[batch * args.batch_size:(batch * args.batch_size + args.batch_size)]  # 训练一批
            inputs_search = utils.get_train_images_auto(image_paths_valid).cuda()  # 取出一批图片并且变成张量

            architect.step(inputs, inputs_search, lr, optimizer, unrolled=args.unrolled)

            print(F.softmax(encoder.alphas, dim=-1))
            print(F.softmax(decoder.alphas, dim=-1))

            en1, en2 = encoder(inputs)
            outputs = decoder(en1, en2)


            optimizer.zero_grad()

            ssim_loss_value = 0.
            pixel_loss_value = 0.
            for output, input in zip(outputs, inputs):
                output, input = torch.unsqueeze(output, 0), torch.unsqueeze(input, 0)
                pixel_loss_temp = mse_loss(input, output)
                ssim_loss_temp = ssim(input, output, normalize=True, val_range=255)
                ssim_loss_value += (1 - ssim_loss_temp)
                pixel_loss_value += pixel_loss_temp
            ssim_loss_value /= len(outputs)
            pixel_loss_value /= len(outputs)
            total_loss = pixel_loss_value + 100*ssim_loss_value
            total_loss.backward()
            nn.utils.clip_grad_value_(encoder.parameters(), args.grad_clip)
            nn.utils.clip_grad_value_(decoder.parameters(), args.grad_clip)
            optimizer.step()
            if i==0:
                logging.info("Infrared_epoch: %d batch: %d loss: %f", epoch, batch+1, total_loss)
            else:
                logging.info("Visible_epoch: %d batch: %d loss: %f", epoch, batch+1, total_loss)


if __name__ == '__main__':
    main()
