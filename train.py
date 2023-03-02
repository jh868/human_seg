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
        mask = self.transform_mask(image=mask)['image']
        mask = mask.astype(np.float32)

        mask[mask < 255] = 0
        mask[mask == 255.0] = 1

        mask = torch.tensor(mask)
        mask = mask.squeeze()
        return mask

    def __getitem__(self, i):
        X_train = cv2.imread(self.X_train[i])
        X_train = X_train.astype(np.float32)
        X_train = self.transform(image=X_train)['image']
        Y_train = cv2.imread(self.Y_train[i], cv2.IMREAD_GRAYSCALE)
        Y_train = self.preprocessing_mask(Y_train)

        return X_train, Y_train


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = A.Compose([
    A.Resize(width=256, height=256),
    A.HorizontalFlip(),
    ToTensorV2()
])
transform_mask = A.Compose([
    A.Resize(width=256, height=256),
    A.HorizontalFlip()
])

train_set = Human(path_to_img='D:seg/image/',
                  path_to_anno='D:seg/mask/',
                  transfrom=transform,
                  transform_mask=transform_mask
                  )

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

model = MobileUNet().to(device)

lr = 0.0001

optim = Adam(params=model.parameters(), lr=lr)

# 가중치만 불러오기
# model.load_state_dict(torch.load('./Human_seg_full_50.pth'), strict=False)

# load checkpoint
# checkpoint = torch.load('./checkpoint.tar')
# model.load_state_dict(checkpoint['model_state_dict'])
# optim.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']

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
    if (epoch+1) % 10 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'epoch': epoch
        }, f'Portrait_seg_{epoch+1}.pth')

torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'epoch': 200
        }, 'Portrait_segmentation.pth')

model.load_state_dict(torch.load('Human_seg_full_50.pth', map_location='cpu'))

img_path = './seg/image/1803151818-00000003.jpg'

img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

img_copy = cv2.resize(img, (256, 256))

img_copy2 = np.transpose(img_copy, (2, 0, 1))
img_copy2 = torch.unsqueeze(torch.tensor(img_copy2), 0)

pred = model(img_copy2.float())
pred = pred.detach().numpy()
pred = pred.astype(np.uint8)
print("pred = ", pred.shape)

_, mask = cv2.threshold(pred, 200, 255, cv2.THRESH_BINARY)
mask = cv2.bitwise_not(mask)  # 반전

# cv2로 투명배경 멕이기 # 근데 생각해보니까 투명 배경할 이유가없음
mask = cv2.merge((mask, mask, mask))
result = cv2.addWeighted(img_copy, 1, mask, 1, 0)

b, g, r = cv2.split(img_copy)
result = cv2.merge([b, g, r, pred], 4)
print("result= ", result.shape)

cv2.imwrite('result.png', result)
cv2.imshow(' ', result)
cv2.waitKey(0)

# 배경관련
fg_h, fg_w, _ = img.shape

background = cv2.imread('background.jpg')

bg_h, bg_w, _ = background.shape

# fit to fg width
background = cv2.resize(background, dsize=(fg_w, int(fg_w * bg_h / bg_w)))

bg_h, bg_w, _ = background.shape

margin = (bg_h - fg_h) // 2

if margin > 0:
    background = background[margin:-margin, :, :]
else:
    background = cv2.copyMakeBorder(background, top=abs(margin), bottom=abs(margin), left=0, right=0,
                                    borderType=cv2.BORDER_REPLICATE)

# final resize
background = cv2.resize(background, dsize=(fg_w, fg_h))

cv2.imshow("", background)
cv2.waitKey(0)

# 넣기
mask = cv2.resize(mask, (fg_w, fg_h))

print(background.shape)
print(img.shape)
print(mask.shape)

cv2.copyTo(img, mask, background)

cv2.imshow("", background)
cv2.waitKey(0)
