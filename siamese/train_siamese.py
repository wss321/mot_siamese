# -*- coding: utf-8 -*-

from __future__ import division, print_function

import argparse
import os
import time
from shutil import copyfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torchvision import transforms

# from reid_sampler import StratifiedSampler
from model import TripletSiamese
from random_erasing import RandomErasing
from tripletfolder import TripletFolder
from tensorboardX import SummaryWriter

matplotlib.use('agg')

version = torch.__version__

if __name__ == '__main__':

    ######################################################################
    # Options
    # --------
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name', default='siamese1',
                        type=str, help='output model name')
    parser.add_argument('--data_dir', default='E:/PyProjects/datasets/Market/pytorch', type=str,
                        help='training dir path')
    parser.add_argument('--train_all', action='store_true',
                        help='use all training data')
    parser.add_argument('--color_jitter', action='store_true',
                        help='use color jitter in training')
    parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
    parser.add_argument('--num_epoch', default=60, type=int, help='epoches')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='weight of bce loss')
    parser.add_argument('--erasing_p', default=0, type=float,
                        help='Random Erasing probability, in [0,1]')

    opt = parser.parse_args()
    data_dir = opt.data_dir
    name = opt.name
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
    # print(gpu_ids[0])

    ######################################################################
    # Load Data
    # ---------
    #

    transform_train_list = [
        # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256, 128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_val_list = [
        transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    if opt.erasing_p > 0:
        transform_train_list = transform_train_list + \
                               [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

    if opt.color_jitter:
        transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                       hue=0)] + transform_train_list

    # print(transform_train_list)
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }

    train_type = 'train'
    if opt.train_all:
        train_type = 'train_all'

    image_datasets = {}
    image_datasets['train'] = TripletFolder(os.path.join(data_dir, train_type),
                                            data_transforms['train'])
    image_datasets['val'] = TripletFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

    batch = {}

    class_names = image_datasets['train'].classes
    class_vector = [s[1] for s in image_datasets['train'].samples]
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=True, num_workers=8)
                   for x in ['train', 'val']}
    # for data in dataloaders["train"]:
    #         inputs, labels, pos, neg = data
    #         print(labels)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    use_gpu = torch.cuda.is_available()

    since = time.time()
    y_loss = {}  # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []


    def eval_model(model, criterion_ide, criterion_bce, summary_writer):
        with torch.no_grad():
            start = time.time()
            # Each epoch has a training and validation phase
            for phase in ['val']:
                if phase == 'train':
                    scheduler.step()
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_verif_loss = 0.0
                running_corrects = 0.0
                running_verif_corrects = 0.0
                # Iterate over data.
                global_iter = 0
                for data in dataloaders[phase]:
                    # get the inputs
                    inputs, labels_x, pos, neg, labels_n = data
                    now_batch_size, c, h, w = inputs.shape

                    if now_batch_size < opt.batchsize:  # next epoch
                        continue

                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        pos = Variable(pos.cuda())
                        neg = Variable(neg.cuda())
                        labels_x = Variable(labels_x.cuda())
                        labels_n = Variable(labels_n.cuda())
                    else:
                        inputs = Variable(inputs.cuda())
                        pos = Variable(pos)
                        neg = Variable(neg)
                        labels_x = Variable(labels_x)
                        labels_n = Variable(labels_n)
                    # zero the parameter gradients

                    # forward
                    ide_x, ide_p, ide_n, bce_px, bce_nx, bce_np = model(
                        inputs, pos, neg)
                    # ---------------------------------
                    labels_0 = torch.zeros(now_batch_size, 1).long()
                    labels_1 = torch.ones(now_batch_size, 1).long()
                    # one-hot
                    labels_0 = torch.zeros(
                        now_batch_size, 2).scatter_(1, labels_0, 1)
                    labels_1 = torch.zeros(
                        now_batch_size, 2).scatter_(1, labels_1, 1)

                    labels_0 = Variable(labels_0.cuda())
                    labels_1 = Variable(labels_1.cuda())

                    _, ide_preds_x = torch.max(ide_x.data, 1)
                    _, ide_preds_p = torch.max(ide_p.data, 1)
                    _, ide_preds_n = torch.max(ide_n.data, 1)

                    _, bce_preds_px = torch.max(bce_px.data, 1)
                    _, bce_preds_nx = torch.max(bce_nx.data, 1)
                    _, bce_preds_np = torch.max(bce_np.data, 1)

                    loss_ide_x = criterion_ide(ide_x, labels_x)
                    loss_ide_p = criterion_ide(ide_p, labels_x)
                    loss_ide_n = criterion_ide(ide_n, labels_n)
                    loss_ide = ((loss_ide_x + loss_ide_n) /
                                2 + loss_ide_p) / 2

                    loss_bce = (criterion_bce(bce_px, labels_1) +
                                criterion_bce(bce_nx, labels_0) +
                                criterion_bce(bce_np, labels_0)
                                ) / 3
                    loss = loss_ide + loss_bce * opt.alpha
                    # statistics
                    # for the new version like 0.4.0 and 0.5.0
                    if int(version[0]) > 0 or int(version[2]) > 3:
                        running_loss += loss.item()  # * opt.batchsize
                        running_verif_loss += loss_bce.item()  # * opt.batchsize
                    else:  # for the old version like 0.3.0 and 0.3.1
                        running_loss += loss.data[0]
                        running_verif_loss += loss_bce.data[0]
                    corrects_ide = float(
                        torch.sum(ide_preds_x == labels_x.data)

                    )
                    running_corrects += corrects_ide

                    corrects_bce = 0.5 * \
                                   (float(torch.sum(bce_preds_px == 1)) +
                                    float(torch.sum(bce_preds_nx == 0)))
                    running_verif_corrects += corrects_bce
                    # summary
                    global_iter += 1
                    if global_iter % 1e2 == 0:  # Every 100 iters, we record it.
                        summary_writer.add_scalar(
                            'Val Total Loss', loss.item(), global_iter)
                        summary_writer.add_scalar(
                            'Val IDE Loss', loss_ide.item(), global_iter)
                        summary_writer.add_scalar(
                            'Val BCE Loss', loss_bce.item(), global_iter)
                        summary_writer.add_scalar(
                            'Val ide acc', corrects_ide / labels_x.shape[0], global_iter)
                        summary_writer.add_scalar(
                            'Val bce acc', corrects_bce / labels_x.shape[0], global_iter)

                datasize = dataset_sizes['train'] // opt.batchsize * \
                           opt.batchsize
                epoch_loss = running_loss / datasize
                epoch_verif_loss = running_verif_loss / datasize
                epoch_acc = running_corrects / datasize
                epoch_verif_acc = running_verif_corrects / datasize

                end = time.time()

                print(
                    '{} Time cost {:.2f} val Loss: {:.4f} val Loss_verif: {:.4f}  val Acc: {:.4f} val Verif_Acc: {:.4f} '.format(
                        phase, end - start, epoch_loss, epoch_verif_loss, epoch_acc, epoch_verif_acc))


    def train_model(model, criterion_ide, criterion_bce, optimizer, scheduler, num_epochs=25):

        since = time.time()
        trainwriter = SummaryWriter(
            '{}/{}/{}'.format("./model", opt.name, 'train'))
        global_iter = 0
        for epoch in range(num_epochs):
            start = time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 30)

            # Each epoch has a training and validation phase
            for phase in ['train']:
                if phase == 'train':
                    scheduler.step()
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_verif_loss = 0.0
                running_corrects = 0.0
                running_verif_corrects = 0.0
                # Iterate over data.
                for data in dataloaders[phase]:
                    # get the inputs
                    inputs, labels_x, pos, neg, labels_n = data
                    now_batch_size, c, h, w = inputs.shape

                    if now_batch_size < opt.batchsize:  # next epoch
                        continue

                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        pos = Variable(pos.cuda())
                        neg = Variable(neg.cuda())
                        labels_x = Variable(labels_x.cuda())
                        labels_n = Variable(labels_n.cuda())
                    else:
                        inputs = Variable(inputs.cuda())
                        pos = Variable(pos)
                        neg = Variable(neg)
                        labels_x = Variable(labels_x)
                        labels_n = Variable(labels_n)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    ide_x, ide_p, ide_n, bce_px, bce_nx, bce_np = model(
                        inputs, pos, neg)
                    # ---------------------------------
                    labels_0 = torch.zeros(now_batch_size, 1).long()
                    labels_1 = torch.ones(now_batch_size, 1).long()
                    # one-hot
                    labels_0 = torch.zeros(
                        now_batch_size, 2).scatter_(1, labels_0, 1)
                    labels_1 = torch.zeros(
                        now_batch_size, 2).scatter_(1, labels_1, 1)

                    labels_0 = Variable(labels_0.cuda())
                    labels_1 = Variable(labels_1.cuda())

                    _, ide_preds_x = torch.max(ide_x.data, 1)
                    _, ide_preds_p = torch.max(ide_p.data, 1)
                    _, ide_preds_n = torch.max(ide_n.data, 1)

                    _, bce_preds_px = torch.max(bce_px.data, 1)
                    _, bce_preds_nx = torch.max(bce_nx.data, 1)
                    _, bce_preds_np = torch.max(bce_np.data, 1)

                    loss_ide_x = criterion_ide(ide_x, labels_x)
                    loss_ide_p = criterion_ide(ide_p, labels_x)
                    loss_ide_n = criterion_ide(ide_n, labels_n)
                    loss_ide = ((loss_ide_x + loss_ide_n) / 2 + loss_ide_p) / 2

                    loss_bce = (criterion_bce(bce_px, labels_1) +
                                criterion_bce(bce_nx, labels_0) +
                                criterion_bce(bce_np, labels_0)
                                ) / 3
                    loss = loss_ide + loss_bce * opt.alpha
                    # print(loss.data, loss_id.data, loss_ia, loss_verif.data)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # statistics
                    # for the new version like 0.4.0 and 0.5.0
                    if int(version[0]) > 0 or int(version[2]) > 3:
                        running_loss += loss.item()  # * opt.batchsize
                        running_verif_loss += loss_bce.item()  # * opt.batchsize
                    else:  # for the old version like 0.3.0 and 0.3.1
                        running_loss += loss.data[0]
                        running_verif_loss += loss_bce.data[0]
                    corrects_ide = 1 / 3 * float(
                        torch.sum(ide_preds_x == labels_x.data) +
                        torch.sum(ide_preds_p == labels_x.data) +
                        torch.sum(ide_preds_n == labels_n.data)
                    )
                    running_corrects += corrects_ide

                    corrects_bce = 0.5 * \
                                   (float(torch.sum(bce_preds_px == 1)) +
                                    float(torch.sum(bce_preds_nx == 0)))
                    running_verif_corrects += corrects_bce
                    # summary
                    global_iter += 1
                    if global_iter % 1e2 == 0:  # Every 100 iters, we record it.
                        print(global_iter)
                        # eval_model(model, criterion_ide, criterion_bce, trainwriter)
                        trainwriter.add_scalar(
                            'Total Loss', loss.item(), global_iter)
                        trainwriter.add_scalar(
                            'IDE Loss', loss_ide.item(), global_iter)
                        trainwriter.add_scalar(
                            'BCE Loss', loss_bce.item(), global_iter)
                        trainwriter.add_scalar(
                            'ide acc', corrects_ide / labels_x.shape[0], global_iter)
                        trainwriter.add_scalar(
                            'bce acc', corrects_bce / labels_x.shape[0], global_iter)

                datasize = dataset_sizes['train'] // opt.batchsize * \
                           opt.batchsize
                epoch_loss = running_loss / datasize
                epoch_verif_loss = running_verif_loss / datasize
                epoch_acc = running_corrects / datasize
                epoch_verif_acc = running_verif_corrects / datasize

                end = time.time()

                print('{} Time cost {:.2f} Loss: {:.4f} Loss_verif: {:.4f}  Acc: {:.4f} Verif_Acc: {:.4f} '.format(
                    phase, end - start, epoch_loss, epoch_verif_loss, epoch_acc, epoch_verif_acc))

                y_loss[phase].append(epoch_loss)
                y_err[phase].append(1.0 - epoch_acc)

                # eval_model(model, criterion_ide, criterion_bce, trainwriter)

                if epoch % 5 == 0:
                    save_network(model, "{}".format(epoch))

                draw_curve(epoch)
                last_model_wts = model.state_dict()

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(last_model_wts, strict=False)
        save_network(model, "last")

        return model


    def draw_curve(current_epoch):
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
        #    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
        ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
        #    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
        if current_epoch == 0:
            ax0.legend()
            ax1.legend()
        fig.savefig(os.path.join('./model', name, 'train.jpg'))


    ######################################################################
    # Save model
    # ---------------------------

    def save_network(network, epoch_label):
        save_filename = 'net_%s.pth' % epoch_label
        save_path = os.path.join('./model', name, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if torch.cuda.is_available:
            network.cuda(gpu_ids[0])


    ######################################################################
    # Draw Curve
    # ---------------------------
    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="err")

    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    # Load a pretrainied model and reset final fully connected layer.
    #
    model = TripletSiamese(len(class_names))

    # print(model)

    if use_gpu:
        model = model.cuda()

    criterion1 = nn.CrossEntropyLoss().cuda()
    criterion2 = nn.BCELoss().cuda()

    not_base_params = list(map(id, model.ide_classifier.parameters())) \
                      + list(map(id, model.bce_classifier.parameters()))

    base_params = filter(lambda p: id(
        p) not in not_base_params, model.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.1 * opt.lr},
        {'params': model.ide_classifier.parameters(), 'lr': opt.lr},
        {'params': model.bce_classifier.parameters(), 'lr': opt.lr},
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer_ft, milestones=[30, 50, 70], gamma=0.1)

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # It should take around 1-2 hours on GPU.
    #
    dir_name = os.path.join('./model', name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
        copyfile('./train_siamese.py', dir_name + '/train_old.py')
        copyfile('model.py', dir_name + '/model_old.py')
        copyfile('./tripletfolder.py', dir_name + '/tripletfolder.py')
    else:
        raise Exception("{} already exist.".format(dir_name))

    # save opts
    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)
    model = train_model(model, criterion1, criterion2, optimizer_ft, exp_lr_scheduler,
                        num_epochs=opt.num_epoch)
