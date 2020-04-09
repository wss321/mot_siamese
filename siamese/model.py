import torch.nn as nn
from torch.nn import init
from torchvision import models
import torch
from torch.nn import functional as F
import random


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.0, num_bottleneck=512, relu=False, bn=True, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bn:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x


# Defines the new fc layer
# |--Linear--|--bn--|--relu--|
class fc(nn.Module):
    def __init__(self, input_dim, out_dim, bn=True, relu=True):
        super(fc, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, out_dim)]
        if bn:
            add_block += [nn.BatchNorm1d(out_dim)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.fc = add_block

    def forward(self, x):
        x = self.fc(x)
        return x


def step(a):
    for i, vi in enumerate(a):
        for j, vj in enumerate(vi):
            if vj > 0:
                a[i, j] = 1
            else:
                a[i, j] = 0
    return a


class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x


# Define the ResNet50-based Model
class Siamese_test(nn.Module):

    def __init__(self, class_num, num_bottleneck=512):
        super(Siamese_test, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        resnet50.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = resnet50
        self.ide_classifier = ClassBlock(
            2048, class_num, num_bottleneck=num_bottleneck, droprate=0, bn=True, relu=True)
        self.bce_classifier = ClassBlock(
            2048, 2, droprate=0, bn=False, relu=True)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

        self.batch_drop = BatchDrop(0.5, 0.5)
        # self.part_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        # self.reduction = nn.Sequential(
        #     nn.Linear(2048, 1024, 1),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU()
        # )
        # self.reduction.apply(weights_init_kaiming)
        # self.softmax = nn.Linear(1024, class_num)
        # self.softmax.apply(weights_init_kaiming)

        self.fmap_pool = {}
        self.fm = [None, None]
        self.f_grad = None
        self.grad_pool = {}
        self.handlers = []
        self.candidate_layers = ["layer4"]

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(
                    module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(
                    module.register_backward_hook(save_grads(name)))

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self):
        return self.f_grad

    def forward(self, x, q):
        f_i = self.get_feature(x)
        c = self.ide_classifier(f_i)
        # self.fm[0] = self.fmap_pool["layer4"]
        if q is None:
            return c, None

        f_q = self.get_feature(q)
        ff = f_q.mul(f_i)
        v = self.bce_classifier(ff)
        v = nn.Sigmoid()(v)
        # self.fm[1] = self.fmap_pool["layer4"]

        return c, v

    def get_feature(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = self.batch_drop(x)
        f_i = x.view(x.size(0), x.size(1))
        return f_i

    def verify(self, fx, fy):
        self.ff = fx.mul(fy)
        v = self.bce_classifier(self.ff)
        v = nn.Sigmoid()(v)
        return v


class TripletSiamese(nn.Module):

    def __init__(self, class_num, num_bottleneck=512):
        super(TripletSiamese, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        resnet50.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = resnet50
        self.ide_classifier = ClassBlock(
            2048, class_num, num_bottleneck=num_bottleneck,
            droprate=0, bn=True, relu=True)
        self.bce_classifier = ClassBlock(
            2048, 2, droprate=0, bn=False, relu=True)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x, p, n):
        f_x = self.get_feature(x)
        f_p = self.get_feature(p)
        f_n = self.get_feature(n)
        # 识别
        ide_x = self.ide_classifier(f_x)
        ide_p = self.ide_classifier(f_p)
        ide_n = self.ide_classifier(f_n)
        # 相似度
        merge_px = f_p.mul(f_x)
        merge_nx = f_n.mul(f_x)
        merge_np = f_n.mul(f_p)
        bce_px = self.bce_classifier(merge_px)
        bce_nx = self.bce_classifier(merge_nx)
        bce_np = self.bce_classifier(merge_np)
        bce_px = nn.Sigmoid()(bce_px)
        bce_nx = nn.Sigmoid()(bce_nx)
        bce_np = nn.Sigmoid()(bce_np)
        return ide_x, ide_p, ide_n, bce_px, bce_nx, bce_np

    def get_feature(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        f_i = x.view(x.size(0), x.size(1))
        return f_i

    def verify(self, fx, fy):
        self.ff = fx.mul(fy)
        v = self.bce_classifier(self.ff)
        v = nn.Sigmoid()(v)
        return v


class SiameseBce(nn.Module):

    def __init__(self, class_num, num_bottleneck=512):
        super(SiameseBce, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        resnet50.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = resnet50
        self.bce_classifier = ClassBlock(
            2048, 2, droprate=0, bn=False, relu=True)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x, p, n):
        f_x = self.get_feature(x)
        f_p = self.get_feature(p)
        f_n = self.get_feature(n)
        # 相似度
        merge_px = f_p.mul(f_x)
        merge_nx = f_n.mul(f_x)
        merge_np = f_n.mul(f_p)
        bce_px = self.bce_classifier(merge_px)
        bce_nx = self.bce_classifier(merge_nx)
        bce_np = self.bce_classifier(merge_np)
        bce_px = nn.Sigmoid()(bce_px)
        bce_nx = nn.Sigmoid()(bce_nx)
        bce_np = nn.Sigmoid()(bce_np)
        return bce_px, bce_nx, bce_np

    def get_feature(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        f_i = x.view(x.size(0), x.size(1))
        return f_i

    def verify(self, fx, fy):
        self.ff = fx.mul(fy)
        v = self.bce_classifier(self.ff)
        v = nn.Sigmoid()(v)
        return v


class Siamese(nn.Module):

    def __init__(self, class_num, num_bottleneck=512):
        super(Siamese, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.fc_cl = fc(2048, num_bottleneck, relu=True)
        self.id_classifier = fc(
            num_bottleneck, class_num, bn=False, relu=False)
        self.fc_verif = fc(2048, num_bottleneck, bn=False, relu=True)
        self.bce = nn.Linear(num_bottleneck, 2)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x, q):
        f_i = self.get_feature(x)
        x = self.fc_cl(f_i)
        c = self.id_classifier(x)

        f_q = self.get_feature(q)
        v = self.verify(f_q, f_i)

        return c, v

    def get_feature(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        f_i = x.view(x.size(0), x.size(1))
        return f_i

    def verify(self, fx, fy):
        v = self.fc_verif(fx.mul(fy))
        v = self.bce(v)
        v = nn.Sigmoid()(v)
        return v

# Part Model proposed in Yifan Sun etal. (2018)


class PCB(nn.Module):
    def __init__(self, class_num):
        super(PCB, self).__init__()

        self.part = 6  # We cut the pool5 to 6 parts
        self.model = models.resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'ide_classifier' + str(i)
            setattr(self,
                    name,
                    ClassBlock(2048,
                               class_num,
                               droprate=0.5,
                               relu=True,
                               bn=True,
                               num_bottleneck=512))
            name1 = 'bce_classifier' + str(i)
            setattr(self, name1,
                    nn.Sequential(
                        ClassBlock(2048,
                                   2,
                                   droprate=0.5,
                                   relu=True,
                                   bn=True,
                                   num_bottleneck=256),
                        nn.Sigmoid()))

    def get_part_feature(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        # x = self.dropout(x)
        part = {}
        y = []
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:, :, i])
            y.append(part[i])
        return y

    def forward(self, query, gallery):
        f_q = self.get_part_feature(query)
        f_g = self.get_part_feature(gallery)
        bce_pred = []
        class_pred = []
        for i in range(len(f_q)):
            name = 'bce_classifier' + str(i)
            bce_classifier = getattr(self, name)
            v = bce_classifier(f_q[i].mul(f_g[i]))
            bce_pred.append(v)

            name = 'ide_classifier' + str(i)
            c = getattr(self, name)
            class_pred.append(c(f_q[i]))
        return class_pred, bce_pred


class PCB_test(nn.Module):
    def __init__(self, model):
        super(PCB_test, self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y


if __name__ == '__main__':
    # debug model structure
    # net = PCB(751)
    net = Siamese(751)
    # print(net)
    inp = torch.randn(2, 3, 256, 128)
    c, v = net.forward(inp, inp)
