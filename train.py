import glob
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import tqdm
import torch.nn as nn

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
import cv2

from model import MobileUNet
import matplotlib.pyplot as plt


class Human(Dataset):
    def __init__(self, path_to_img, path_to_anno, train=True, transfrom=None, transform_mask=None):
        self.images = sorted(glob.glob(path_to_img + '/*.jpg'))
        self.annotations = sorted(glob.glob(path_to_anno + '/*.png'))

        self.X_train = self.images[:]
        self.Y_train = self.annotations[:]

        self.train = train
        self.transform = transfrom
        self.transform_mask = transform_mask
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
        mask = self.transform(image=mask)['image']

        # mask[mask < 255] = 0
        # mask[mask == 255.0] = 1

        # mask = torch.tensor(mask)
        mask = mask.squeeze()
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
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.3),
    ToTensorV2()
])

train_set = Human(path_to_img='D:seg/image/',
                  path_to_anno='D:seg/mask/',
                  transfrom=transform,
                  )

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

model = MobileUNet().to(device)

lr = 0.001

optim = Adam(params=model.parameters(), lr=lr)

# 가중치만 불러오기
# model.load_state_dict(torch.load('./Human_seg_full_50.pth'), strict=False)

# load checkpoint
# checkpoint = torch.load('./checkpoint.tar')
# model.load_state_dict(checkpoint['model_state_dict'])
# optim.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']

# train
for epoch in range(200):
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optim.zero_grad()

        preds = model(data.to(device))
        loss = nn.BCEWithLogitsLoss()(preds, label.type(torch.FloatTensor).to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f'epoch: {epoch + 1} loss: {loss.item()}')

    # save checkpoint
    if (epoch + 1) % 5 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'epoch': epoch
        }, f'Portrait_seg_{epoch + 1}.pth')

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.state_dict(),
    'epoch': 200
}, 'Portrait_segmentation.pth')
