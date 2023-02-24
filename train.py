import glob

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import tqdm
import torch.nn as nn

from torchvision.transforms import Compose
from torchvision.transforms import ToTensor, Resize
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader

from model import MobileUNet


class Human(Dataset):
    def __init__(self, path_to_img, path_to_anno, train=True, transfrom=None, input_size=(256, 256)):
        self.images = sorted(glob.glob(path_to_img + '/*.jpg'))
        self.annotations = sorted(glob.glob(path_to_anno + '/*.png'))

        self.X_train = self.images[:int(0.8 * len(self.images))]
        self.X_test = self.images[int(0.8 * len(self.images)):]
        self.Y_train = self.annotations[:int(0.8 * len(self.annotations))]
        self.Y_test = self.annotations[int(0.8 * len(self.annotations)):]

        self.train = train
        self.transform = transfrom
        self.input_size = input_size

    def __len__(self):
        if self.train:
            return len(self.X_train)
        else:
            return len(self.X_test)

    def preprocessing_mask(self, mask):
        mask = mask.resize(self.input_size)
        mask = np.array(mask).astype(np.float32)

        mask[mask < 255] = 0
        mask[mask == 255.0] = 1

        mask = torch.tensor(mask)
        return mask

    def __getitem__(self, i):
        if self.train:
            X_train = Image.open(self.X_train[i])
            X_train = self.transform(X_train)
            Y_train = Image.open(self.Y_train[i])
            Y_train = self.preprocessing_mask(Y_train)

            # print(X_train.shape, Y_train.shape)

            return X_train, Y_train

        else:
            X_test = Image.open(self.X_test[i])
            X_test = self.transform(X_test)
            Y_test = Image.open(self.Y_test[i])
            Y_test = self.preprocessing_mask(Y_test)


            return X_test, Y_test


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = Compose([
    Resize((256, 256)),
    ToTensor()
])

train_set = Human(path_to_img='D:seg/image/',
                  path_to_anno='D:seg/mask/',
                  transfrom=transform)
test_set = Human(path_to_img='D:seg/image/',
                 path_to_anno='D:seg/mask/',
                 transfrom=transform,
                 train=False)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set)

model = MobileUNet().to(device)

lr = 0.0001

optim = Adam(params=model.parameters(), lr=lr)

for epoch in range(300):
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optim.zero_grad()

        preds = model(data.to(device))
        loss = nn.BCEWithLogitsLoss()(preds, label.type(torch.FloatTensor).to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f'epoch: {epoch + 1} loss: {loss.item()}')

    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), f'Human_segmentation_{epoch+1}.pth')

torch.save(model.state_dict(), 'Human_segmentation.pth')
