"""
Script for training a wall-segmentor on the datasets
provided in the Repository.

This wall-segmentor is used to improve hold detection
by removing extraneous holds.

From the root of this repository run the following command:
$ python train_utils train_wall_segmentor.py

This will set up the wall segmentor model in the correct folder
to be used properly by other modules in this repository.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from matplotlib import pyplot as plt
import os
import time

from dataset_factory import *
from model_factory import *
from copy import deepcopy

BATCH_SIZE = 16
EPOCHS = 10

model = get_segmentation_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss() # for aux clfs
optimizer = torch.optim.Adam(model.parameters())

train_data = COCO_WallDataset('datasets/orig_wall_coco', 'train')
val_data = COCO_WallDataset('datasets/orig_wall_coco', 'valid')
test_data = COCO_WallDataset('datasets/orig_wall_coco', 'test')

USE_DICE=False

def collate_fn(batch):
    shapes = [elem.shape for elem in batch]
    print(shapes)
    return batch

tl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
vl = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
test_l = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

def dice_loss(out, target):
    n_class = out[0].shape[0]
    pred = nn.Softmax(dim=1)(out)

    # numer=0
    # denom=0
    dice_scores = torch.zeros(1, 1).to(out.device)
    for cls in range(n_class):
        numer = 2 * torch.sum((pred[:, cls, :, :] * (target == cls)), axis=(1, 2)).sum() + 1
        denom = torch.sum((pred[:, cls, :, :]), axis=(1, 2)).sum() + torch.sum(target == cls, axis=(1, 2)).sum() + 1

        class_dice = (numer / denom).reshape(1, 1)
        dice_scores = torch.cat([dice_scores, class_dice], dim=1)
    
    dice = torch.sum(dice_scores) / (n_class - 1)
    
    return 1 - dice


def train_epoch(loader, model, device, criterions, optimizer):
    model.train()
    epoch_loss = 0
    N = 0
    for i, batch in enumerate(loader):
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)
        dic = model(images)
        pred, aux = dic['out'], dic['aux'] # N x C x H x W

        optimizer.zero_grad()
        if not USE_DICE:
            loss = criterions[0](pred, masks) + 0.4 * criterions[1](aux, masks)
        else:
            loss = dice_loss(pred, masks) + 0.4 * dice_loss(aux, masks)
        loss.backward()
        optimizer.step()

        N += images.shape[0]
        epoch_loss += loss.item() * images.shape[0]

    return model, epoch_loss / N

def evaluate(loader, model, device, criterions):
    model.eval()
    epoch_loss = 0
    N = 0
    for i, batch in enumerate(loader):
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)
        dic = model(images)
        pred, aux = dic['out'], dic['aux'] # N x C x H x W

        if not USE_DICE:
            loss = criterions[0](pred, masks) + 0.4 * criterions[1](aux, masks)
        else:
            loss = dice_loss(pred, masks) + 0.4 * dice_loss(aux, masks)

        N += images.shape[0]
        epoch_loss += loss.item() * images.shape[0]

    return epoch_loss / N

train_losses = []
val_losses = []
best_model_state = None
best_val_loss = 1e50
for epoch in range(20):
    start_time = time.time()    
    model, train_loss = train_epoch(tl, model, device, [criterion1, criterion2], optimizer)
    val_loss = evaluate(vl, model, device, [criterion1, criterion2])

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = deepcopy(model.state_dict())

    print(f"Epoch {epoch} elapsed {time.time() - start_time}s")
    print(f"Train Loss: {train_loss} Val Loss: {val_loss}\n")

    train_losses.append(train_loss)
    val_losses.append(val_loss)

def plot_losses(tls, vls):
    fig = plt.figure()
    plt.plot(tls, label='train')
    plt.plot(vls, label='val')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(-2, 2)
    plt.savefig('plots/hold_detection_loss.png')


if not os.path.exists('./plots'):
    os.makedirs('./plots')

plot_losses(train_losses, val_losses)
if not os.path.exists('./models'):
    os.makedirs('./models')

if not USE_DICE:
    torch.save(best_model_state, os.path.join('models/wall_segmentor.pth'))
else:
    torch.save(best_model_state, os.path.join('models/wall_segmentor_dice.pth'))

