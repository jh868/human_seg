import glob
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import tqdm
import torch.nn as nn
import torch.nn.functional as F


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
import cv2

from model import MobileUNet
import matplotlib.pyplot as plt


class Human(Dataset):
    def __init__(self, path_to_img, path_to_anno, train=True, transfrom=None):
        self.images = sorted(glob.glob(path_to_img + '/*.jpg'))
        self.annotations = sorted(glob.glob(path_to_anno + '/*.png'))

        self.X_train = self.images[:]
        self.Y_train = self.annotations[:]

        self.train = train
        self.transform = transfrom
        # self.input_size = input_size

    def __len__(self):
        if self.train:
            return len(self.X_train)
        # else:
        # return len(self.X_test)

    def preprocessing_mask(self, mask):
        # mask = mask.resize(self.input_size)
        # mask = np.resize(mask, self.input_size)

        mask[mask < 255] = 0
        mask[mask == 255.0] = 1
        mask = mask.astype(np.float32)
        # mask[mask != 1.0] = 0.0
        # mask[mask == 1.0] = 1.0
        mask = self.transform(image=mask)['image']

        # mask[mask < 255] = 0
        # mask[mask == 255.0] = 1

        # mask = torch.tensor(mask)
        # mask = mask.squeeze()
        return mask

    def __getitem__(self, i):
        X_train = cv2.imread(self.X_train[i])
        X_train = X_train.astype(np.float32)
        X_train = self.transform(image=X_train)['image']
        # X_train = torch.tensor(X_train)

        Y_train = cv2.imread(self.Y_train[i], cv2.IMREAD_GRAYSCALE)
        Y_train = self.preprocessing_mask(Y_train)

        return X_train, Y_train


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = A.Compose([
    A.Resize(width=256, height=256),
    # A.HorizontalFlip(p=0.3),
    # A.Rotate(limit=30, p=0.2),
    ToTensorV2()
])

train_set = Human(path_to_img='D:seg_resize/image/',
                  path_to_anno='D:seg_resize/mask/',
                  transfrom=transform,
                  )

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

model = MobileUNet().to(device)

lr = 0.001

optim = Adam(params=model.parameters(), lr=lr)

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

dice =DiceBCELoss()

# 가중치만 불러오기
# model.load_state_dict(torch.load('./mobilenet_v2-7ebf99e0.pth'), strict=False)

# load checkpoint
checkpoint = torch.load('D:pt_file/Portrait_seg_pretrain_40.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optim.load_state_dict(checkpoint['optimizer_state_dict'])
num_epoch = checkpoint['epoch']

# train
for epoch in range(num_epoch, 20000):
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optim.zero_grad()

        preds = model(data.to(device))
        # loss = nn.BCEWithLogitsLoss()(preds, label.type(torch.FloatTensor).to(device))
        loss = dice(preds, label.type(torch.FloatTensor).to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f'epoch: {epoch + 1} loss: {loss.item()}')

    # save checkpoint
    if (epoch + 1) % 5 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'epoch': epoch
        }, f'D:/pt_file/Portrait_seg_pretrain_{epoch + 1}.pth')

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.state_dict(),
    'epoch': 200
}, 'Portrait_segmentation.pth')
