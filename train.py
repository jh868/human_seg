import glob

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import tqdm
import torch.nn as nn

from torchvision.transforms import Compose
from torchvision.transforms import ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomCrop
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
import cv2

from model import MobileUNet


class Human(Dataset):
    def __init__(self, path_to_img, path_to_anno, train=True, transfrom=None, input_size=(256, 256)):
        self.images = sorted(glob.glob(path_to_img + '/*.jpg'))
        self.annotations = sorted(glob.glob(path_to_anno + '/*.png'))

        # self.X_train = self.images[:int(0.9 * len(self.images))]
        # self.X_test = self.images[int(0.9 * len(self.images)):]
        # self.Y_train = self.annotations[:int(0.9 * len(self.annotations))]
        # self.Y_test = self.annotations[int(0.9 * len(self.annotations)):]
        #
        self.X_train = self.images[:]
        self.Y_train = self.annotations[:]


        self.train = train
        self.transform = transfrom
        self.input_size = input_size

    def __len__(self):
        if self.train:
            return len(self.X_train)
        # else:
            # return len(self.X_test)

    def preprocessing_mask(self, mask):
        mask = mask.resize(self.input_size)
        mask = self.transform(mask)
        mask = np.array(mask).astype(np.float32)

        mask[mask < 255] = 0
        mask[mask == 255.0] = 1

        mask = torch.tensor(mask)
        mask = mask.squeeze()
        return mask

    def __getitem__(self, i):
        X_train = Image.open(self.X_train[i])
        X_train = self.transform(X_train)
        Y_train = Image.open(self.Y_train[i])
        Y_train = self.preprocessing_mask(Y_train)

        return X_train, Y_train



device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = Compose([
    Resize((256, 256)),
    RandomRotation(180),
    RandomVerticalFlip(),
    RandomHorizontalFlip(),
    ToTensor()
])

train_set = Human(path_to_img='D:seg/image/',
                  path_to_anno='D:seg/mask/',
                  transfrom=transform)
test_set = Human(path_to_img='D:seg/image/',
                 path_to_anno='D:seg/mask/',
                 transfrom=transform,
                 train=False)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set)

model = MobileUNet().to(device)

lr = 0.0001

optim = Adam(params=model.parameters(), lr=lr)

for epoch in range(200):
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optim.zero_grad()

        preds = model(data.to(device))
        loss = nn.BCEWithLogitsLoss()(preds, label.type(torch.FloatTensor).to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f'epoch: {epoch + 1} loss: {loss.item()}')

    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), f'Human_seg_transform_{epoch+1}.pth')

torch.save(model.state_dict(), 'Human_segmentation.pth')

model.load_state_dict(torch.load('Human_segmentation_120.pth', map_location='cpu'))

data, label = test_set[1]
pred = model(torch.unsqueeze(data.to(device), dim=0)) > 0.5

mask = pred.cpu().detach().numpy()  # tensor -> numpy

mask = np.where(mask > 0.5, 1, 0)

data = np.transpose(data, (1, 2, 0))  # (3, 256, 256) -> (256, 256, 3)

masked_img = np.multiply(data.cpu().detach().numpy(), np.repeat(mask[:, :, np.newaxis], 3, axis=2))

cv2.imshow('Masked Image', masked_img)
cv2.waitKey(0)

import matplotlib.pyplot as plt

# with torch.no_grad():
#     plt.subplot(1, 2, 1)
#     plt.title('predicted')
#     plt.imshow(pred.cpu())
#     plt.subplot(1, 2, 2)
#     plt.title('real')
#     plt.imshow(label)
#     plt.show()