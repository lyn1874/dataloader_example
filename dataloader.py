#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17, 13:14
Prepares the dataset using PyTorch
@author: li
"""
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


class PrepareData(Dataset):
    def __init__(self, filename, label, transform, read_gray):
        """
        :param filename: [number of images], path to read the images
        :param label: [number of images], label information for each of the images, numpy array
        :param transform: transformation
        :param read_gray: bool variable, if true, then the output channel for the image is 1, otherwise 3 (rgb)
        """
        self.filename = filename
        self.label = label
        self.len = len(self.filename)
        self.transform = transform
        self.read_gray = read_gray

    def __getitem__(self, index):
        filename = self.filename[index]
        img = Image.open(filename)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.read_gray:
            img = img.mean(dim=0, keepdims=True)
        _label = self.label[index]
        return img, _label

    def __len__(self):
        return self.len


def give_transformation(img_size):
    """
    Give the transformation that is required
    Args:
        img_size: see the official document for transforms.Resize to define img_size
    Return:
        transformation that is going to be send to the Class PrepareData
    Ops:
        This transformation needs to be user-defined based on your own configuration
        Note: one thing needs to be careful is that some functions only accept tensor or PIL input, so if your input
            is a numpy array, you need to first convert it to a tensor, then apply some of the transformations.
    """
    # resize_crop = transforms.RandomResizedCrop(img_size, scale=(0.8, 1.2),
    #                                            ratio=(0.8, 1.2))
    # random_flip = transforms.RandomHorizontalFlip()
    resize = transforms.Resize(img_size)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans_group = [resize, transforms.ToTensor(), normalize]
    trans = transforms.Compose(trans_group)
    return trans


def test(show=False):
    """Here shows a standard pipeline for PyTorch data loading process
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 5
    num_workers = 4  # can be any value, check the official explanation

    tr_path = sorted([v for v in os.listdir("dataset/") if '.jpg' in v or '.JPG' in v])
    tr_path = np.array(["dataset/" + v for v in tr_path])
    val_path = tr_path[-batch_size * 2:]
    tr_path = tr_path[:batch_size * 2]

    tr_label = np.random.randint(0, 4, [len(tr_path), 218, 180])
    # Note, I don't have the label here, so I just consider there are four classes, and the label are randomly sampled
    val_label = np.random.randint(0, 4, [len(val_path), 218, 180])

    imh, imw = 218, 180
    tr_transformation = give_transformation([imh, imw])
    val_transformation = give_transformation([imh, imw])

    tr_data = PrepareData(tr_path, tr_label, tr_transformation, True)
    val_data = PrepareData(val_path, val_label, val_transformation, True)

    train_params = {"batch_size": batch_size,
                    "drop_last": True,
                    "num_workers": num_workers,
                    "pin_memory": True,
                    "worker_init_fn": lambda _: np.random.seed(),
                    }
    # "sampler": sampler,   # If you want to random sample to get a balanced dataset, then use this argument

    val_params = {"batch_size": batch_size,
                  "drop_last": True,
                  "num_workers": num_workers,
                  "pin_memory": True}

    tr_data_loader = DataLoader(tr_data, **train_params)
    val_data_loader = DataLoader(val_data, **val_params)

    print("The shape of the tr data loader", len(tr_data_loader), " should equal to number of images // batch_size:",
          len(tr_path) // batch_size)
    max_epoch = 1
    for epoch in range(max_epoch):
        # model.train()
        for iterr, (_img, _label) in enumerate(tr_data_loader):
            _img = _img.view(batch_size, 1, imh, imw).to(torch.float32).to(device)
            _label = _label.view(batch_size, imh * imw).type(torch.LongTensor).to(device)
            # need to convert it to longtensor for nn.CrossEntropy
            print("Image shape:", _img.shape)  # [batch_size, num_channel, imh, imw]
            print("Label shape:", _label.shape)  # [batch_size, imh * imw], largest number should be num_class - 1
            print("Image max", _img.max(), " Image min ", _img.min())
            print("Label max", _label.max(), " Label min ", _label.min())
            if show:
                _img = _img.permute(0, 2, 3, 1)
                fig = plt.figure(figsize=(10, 5))
                for j, _s_im in enumerate(_img.detach().cpu().numpy()):
                    ax = fig.add_subplot(1, batch_size, j+1)
                    ax.imshow(_s_im)
                plt.subplots_adjust(wspace=0.1, hspace=0.1)
            # -------------------From here is the training updating process --------------#
            # optimizer.zero_grad()
            # pred = model(_img)  # [batch_size, imh * imw, num_classes]
            # loss = nn.CrossEntropy(reduction='sum')(pred, _label)
            # loss.backward()
            # optimizer.step()
            # ----------------------------------------------------------------------------#
        # -----Here are the validation step, do the same thing here------#
    plt.show()












