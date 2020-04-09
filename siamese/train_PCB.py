# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
# from reid_sampler import StratifiedSampler
from model import PCB
from random_erasing import RandomErasing
from tripletfolder import TripletFolder
import yaml
from shutil import copyfile

version = torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir', default='../Market/pytorch', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--alpha', default=1.0, type=float, help='weight of bce loss')
parser.add_argument('--erasing_p', default=0.0, type=float, help='Random Erasing probability, in [0,1]')

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

# Load Data

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
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list

data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:
    train_all = '_all'
image_datasets = {}
image_datasets['train'] = TripletFolder(os.path.join(data_dir, 'train_all'),
                                        data_transforms['train'])
image_datasets['val'] = TripletFolder(os.path.join(data_dir, 'val'),
                                      data_transforms['val'])

batch = {}

class_names = image_datasets['train'].classes
class_vector = [s[1] for s in image_datasets['train'].samples]
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=8)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()

since = time.time()
# inputs, classes, pos, pos_classes = next(iter(dataloaders['train']))
print(time.time() - since)

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def train_model(model, criterion1, criterion2, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    last_margin = 0.0

    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 40)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_bce_loss = 0.0
            running_ide_corrects = 0.0
            running_bce_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels, pos, neg = data
                now_batch_size, c, h, w = inputs.shape

                if now_batch_size < opt.batchsize:  # next epoch
                    continue

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    pos = Variable(pos.cuda())
                    neg = Variable(neg.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                ide_score, bce_pscore = model(inputs, pos)
                _, bce_nscore = model(inputs, neg)
                ide_part = {}
                n_bce_part = {}
                p_bce_part = {}
                sm = nn.Softmax(dim=1)
                num_part = 6
                p_score = bce_pscore[0]
                n_score = bce_nscore[0]
                score = sm(ide_score[0])
                for i in range(num_part):
                    if i > 0:
                        p_score = bce_pscore[0]
                        p_score = bce_pscore[0]
                        score += sm(ide_score[i])
                    ide_part[i] = ide_score[i]
                    p_bce_part[i] = bce_pscore[i]
                    n_bce_part[i] = bce_nscore[i]
                # print(pf.requires_grad)
                # loss
                # ---------------------------------
                labels_0 = torch.zeros(now_batch_size, 1).long()
                labels_1 = torch.ones(now_batch_size, 1).long()
                # one-hot
                labels_0 = torch.zeros(now_batch_size, 2).scatter_(1, labels_0, 1)
                labels_1 = torch.zeros(now_batch_size, 2).scatter_(1, labels_1, 1)

                labels_0 = Variable(labels_0.cuda())
                labels_1 = Variable(labels_1.cuda())

                _, preds = torch.max(score.data, 1)
                _, p_preds = torch.max(p_score.data, 1)
                _, n_preds = torch.max(n_score.data, 1)
                loss_ide, loss_bce = 0, 0
                for i in range(num_part):
                    loss_ide += criterion1(ide_part[i], labels)
                    loss_bce += (criterion2(p_bce_part[i], labels_1) +
                                 criterion2(n_bce_part[i], labels_0)) * 0.5
                loss = loss_ide + opt.alpha * loss_bce

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0 and 0.5.0
                    running_loss += loss.item()  # * opt.batchsize
                    running_bce_loss += loss_bce.item()  # * opt.batchsize
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0]
                    running_bce_loss += loss_bce.data[0]
                running_ide_corrects += float(torch.sum(preds == labels.data))
                running_bce_corrects += float(torch.sum(p_preds == 1)) + float(torch.sum(n_preds == 0))

            datasize = dataset_sizes['train'] // opt.batchsize * opt.batchsize
            epoch_loss = running_loss / datasize
            epoch_bce_loss = running_bce_loss / datasize
            epoch_acc = running_ide_corrects / datasize
            epoch_bce_acc = running_bce_corrects / (2 * datasize)

            end = time.time()

            print('{} Time cost {:.4f} Loss: {:.4f} Loss_verif: {:.4f}  Acc: {:.4f} Verif_Acc: {:.4f} '.format(
                phase, end - start, epoch_loss, epoch_bce_loss, epoch_acc, epoch_bce_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model

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
ax1 = fig.add_subplot(122, title="top1err")

# Train
model = PCB(len(class_names))
print(model)

if use_gpu:
    model = model.cuda()

criterion1 = nn.CrossEntropyLoss().cuda()
criterion2 = nn.BCELoss().cuda()

not_base_params = (list(map(id, model.ide_classifier0.parameters()))
                   + list(map(id, model.ide_classifier1.parameters()))
                   + list(map(id, model.ide_classifier2.parameters()))
                   + list(map(id, model.ide_classifier3.parameters()))
                   + list(map(id, model.ide_classifier4.parameters()))
                   + list(map(id, model.ide_classifier5.parameters()))

                   + list(map(id, model.bce_classifier0.parameters()))
                   + list(map(id, model.bce_classifier1.parameters()))
                   + list(map(id, model.bce_classifier2.parameters()))
                   + list(map(id, model.bce_classifier3.parameters()))
                   + list(map(id, model.bce_classifier4.parameters()))
                   + list(map(id, model.bce_classifier5.parameters()))
                   )

base_params = filter(lambda p: id(p) not in not_base_params, model.parameters())
optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': 0.1 * opt.lr},
    {'params': model.ide_classifier0.parameters(), 'lr': opt.lr},
    {'params': model.ide_classifier1.parameters(), 'lr': opt.lr},
    {'params': model.ide_classifier2.parameters(), 'lr': opt.lr},
    {'params': model.ide_classifier3.parameters(), 'lr': opt.lr},
    {'params': model.ide_classifier4.parameters(), 'lr': opt.lr},
    {'params': model.ide_classifier5.parameters(), 'lr': opt.lr},
    {'params': model.bce_classifier0.parameters(), 'lr': opt.lr},
    {'params': model.bce_classifier1.parameters(), 'lr': opt.lr},
    {'params': model.bce_classifier2.parameters(), 'lr': opt.lr},
    {'params': model.bce_classifier3.parameters(), 'lr': opt.lr},
    {'params': model.bce_classifier4.parameters(), 'lr': opt.lr},
    {'params': model.bce_classifier5.parameters(), 'lr': opt.lr},
], weight_decay=5e-4, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[20, 40], gamma=0.1)

dir_name = os.path.join('./model', name)
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
    copyfile('./train.py', dir_name + '/train_old.py')
    copyfile('model.py', dir_name + '/model_old.py')
    copyfile('./tripletfolder.py', dir_name + '/tripletfolder.py')
else:
    raise Exception(print("{} already exist.".format(dir_name)))

# save opts
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

model = train_model(model, criterion1, criterion2, optimizer_ft, exp_lr_scheduler,
                    num_epochs=100)
