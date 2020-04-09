from torchvision import datasets
# import os
import numpy as np
import random


# import torch


class TripletFolder(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(TripletFolder, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples])
        self.targets = targets   
        # cams = []
        # for s in self.samples:
        #     cams.append(self._get_cam_id(s[0]))
        # self.cams = np.asarray(cams)

    # def _get_cam_id(self, path):
    #     camera_id = []
    #     filename = os.path.basename(path)
    #     camera_id = filename.split('c')[1][0]
    #     # camera_id = filename.split('_')[2][0:2]
    #     return int(camera_id) - 1

    def _get_pos_sample(self, target, index):
        pos_index = np.argwhere(self.targets == target)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        rand = random.randint(0, len(pos_index) - 1)
        return self.samples[pos_index[rand]]

    def _get_neg_sample(self, target):
        neg_index = np.argwhere(self.targets != target)
        neg_index = neg_index.flatten()
        rand = random.randint(0, len(neg_index) - 1)
        return self.samples[neg_index[rand]]

    def __getitem__(self, index):
        path, target = self.samples[index]
        # cam = self.cams[index]
        # pos_path, neg_path
        pos_path = self._get_pos_sample(target, index)
        neg_path = self._get_neg_sample(target)

        sample = self.loader(path)
        pos = self.loader(pos_path[0])
        neg = self.loader(neg_path[0])
        neg_label = neg_path[1]

        if self.transform is not None:
            sample = self.transform(sample)
            pos = self.transform(pos)
            neg = self.transform(neg)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, pos, neg, neg_label


if __name__ == "__main__":
    from torchvision import transforms
    import torch
    import os

    data_dir = 'E:/PyProjects/datasets/Market/pytorch'
    transform_list = [
        # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256, 128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    T = TripletFolder(os.path.join(data_dir, 'train_all'),
                      transforms.Compose(transform_list))
    D = torch.utils.data.DataLoader(T, batch_size=16,
                                    shuffle=True, num_workers=8)
    D = iter(D)
    inputs, labels, pos, neg, neg_label = next(D)
    print(neg_label)
