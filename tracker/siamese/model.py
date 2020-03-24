import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models


######################################################################
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


# Defines the new fc layer and classification layer
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


class BottleClassifier(nn.Module):

    def __init__(self, in_dim, out_dim, relu=True, dropout=True, bottle_dim=512):
        super(BottleClassifier, self).__init__()

        bottle = [nn.Linear(in_dim, bottle_dim)]
        bottle += [nn.BatchNorm1d(bottle_dim)]
        if relu:
            bottle += [nn.LeakyReLU(0.1)]
        if dropout:
            bottle += [nn.Dropout(p=0.5)]
        bottle = nn.Sequential(*bottle)
        bottle.apply(weights_init_kaiming)
        self.bottle = bottle

        classifier = [nn.Linear(bottle_dim, out_dim)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.bottle(x)
        x = self.classifier(x)
        return x


# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, num_bottleneck=512):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.fc_cl = fc(2048, num_bottleneck, relu=True)
        self.id_classifier = fc(num_bottleneck, class_num, bn=False, relu=False)
        self.fc_verif = fc(2048, num_bottleneck, bn=False, relu=True)
        self.bce = nn.Linear(num_bottleneck, 2)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x, q):
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
        x = self.fc_cl(f_i)
        c = self.id_classifier(x)

        q = self.model.conv1(q)
        q = self.model.bn1(q)
        q = self.model.relu(q)
        q = self.model.maxpool(q)
        q = self.model.layer1(q)
        q = self.model.layer2(q)
        q = self.model.layer3(q)
        q = self.model.layer4(q)
        q = self.model.avgpool(q)
        f_q = q.view(q.size(0), q.size(1))

        v = self.fc_verif(f_i * f_q)
        v = self.bce(v)
        v = nn.Sigmoid()(v)

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

    def getFap(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc_cl(x)
        return x

    def verify(self, fx, fy):
        v = self.fc_verif(fx * fy)
        v = self.bce(v)
        v = nn.Sigmoid()(v)
        return v


class PCB(nn.Module):
    def __init__(self, num_part=6, feats=256, num_classes=751, share_classifier=False, pool="avg"):
        super(PCB, self).__init__()

        self.part = num_part

        resnet = models.resnet50(pretrained=True)

        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((self.part, 1))
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d((self.part, 1))
        else:
            raise Exception

        self.Feature = nn.ModuleList()
        for i in range(self.part):
            Conv = nn.Sequential(
                nn.Conv2d(2048, feats, kernel_size=1),
                nn.BatchNorm2d(feats),
                nn.ReLU(inplace=True)
            )
            Conv.apply(weights_init_kaiming)

            self.Feature.append(Conv)

            if args.share_feature:
                break

        self.Classifer = nn.ModuleList()
        for i in range(self.part):
            if share_classifier:
                Classifer = nn.Sequential(
                    nn.Linear(feats * num_part, num_classes, bias=True)
                )
            else:
                Classifer = nn.Sequential(
                    nn.Linear(feats, num_classes, bias=True)
                )

            Classifer.apply(weights_init_classifier)

            self.Classifer.append(Classifer)

            if share_classifier:
                break

        self.ignored_params = list(map(id, self.Feature.parameters()))
        self.ignored_params.extend(list(map(id, self.Classifer.parameters())))
        self.base_params = list(map(id, self.backbone.parameters()))

        self.mode = 'train'

    def forward(self, x):
        f0 = self.pool(self.backbone(x))
        f = []
        for i in range(self.part):
            if len(self.Feature) == 1:
                f.append(self.Feature[0](f0[:, :, i:i + 1, :]).squeeze(2).squeeze(2))
            else:
                f.append(self.Feature[i](f0[:, :, i:i + 1, :]).squeeze(2).squeeze(2))

        if self.mode == 'test':
            return torch.cat(f, 1)

        y = []
        for i in range(self.part):
            if len(self.Classifer) == 1:
                y.append(self.Classifer[0](torch.cat(f, 1)))
                break
            else:
                y.append(self.Classifer[i](f[i]))

        return y


if __name__ == '__main__':
    # debug model structure
    net = ft_net(751)
    # net = models.resnet50()
    print(net)
    # net = ft_net(751)
    # print(net)
    # input = Variable(torch.FloatTensor(8, 3, 224, 224))
    # output,f = net(input)
    # print('net output size:')
    # print(f.shape)
