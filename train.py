import glob

import numpy as np
import segmentation_models_pytorch.utils.train
import torch
from torch.utils.data.dataset import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
import cv2

import segmentation_models_pytorch as smp


# pip install segmentation-models-pytorch


class Human(Dataset):
    def __init__(self, path_to_img, path_to_mask, train=True, transfrom=None):
        self.images_path = sorted(glob.glob(path_to_img + '/*.jpg'))
        self.masks_path = sorted(glob.glob(path_to_mask + '/*.png'))

        self.images = self.images_path[:int(len(self.images_path) * 0.9)]
        self.masks = self.masks_path[:int(len(self.images_path) * 0.9)]
        self.valid_images = self.images_path[int(len(self.images_path) * 0.9):]
        self.valid_masks = self.masks_path[int(len(self.images_path) * 0.9):]

        self.train = train
        self.transform = transfrom

    def __len__(self):
        if self.train:
            return len(self.images)
        else:
            return len(self.valid_images)

    def __getitem__(self, i):
        if self.train:
            img = cv2.imread(self.images[i])
            # img = img.astype(np.float32)

            mask = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)
            mask[mask < 30] = 0
            mask[mask >= 30] = 1
            # mask = mask.astype(np.float32)
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

            # img2 = np.transpose(img, (1,2,0))
            # cv2.imshow('img', img2.detach().numpy())
            # print(img2)
            # cv2.waitKey(0)
            #
            # cv2.imshow('img', mask.detach().numpy())
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            mask = mask.unsqueeze(0)

            return img, mask
        else:
            valid_img = cv2.imread(self.valid_images[i])
            # valid_img = valid_img.astype(np.float32)

            valid_mask = cv2.imread(self.valid_masks[i], cv2.IMREAD_GRAYSCALE)
            valid_mask[valid_mask < 30] = 0
            valid_mask[valid_mask >= 30] = 1
            # valid_mask = valid_mask.astype(np.float32)
            augmented = self.transform(image=valid_img, mask=valid_mask)
            valid_img = augmented['image']
            valid_mask = augmented['mask']
            valid_mask = valid_mask.unsqueeze(0)

            return valid_img, valid_mask


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = A.Compose([
    A.Resize(width=256, height=256),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomShadow(),
    A.RandomRain(blur_value=2),
    A.RandomSunFlare(src_radius=100),
    A.RandomRotate90(),
    A.Rotate(limit=30),
    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3)),
    A.CoarseDropout(),
    A.Normalize(),
    ToTensorV2()
])

train_set = Human(path_to_img='D:seg_resize/image/',
                  path_to_mask='D:seg_resize/mask/',
                  transfrom=transform,
                  )
valid_set = Human(path_to_img='D:seg_resize/image/',
                  path_to_mask='D:seg_resize/mask/',
                  transfrom=transform,
                  train=False
                  )

train_loader = DataLoader(train_set, batch_size=180, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=180, shuffle=False)

model = smp.DeepLabV3Plus(
    encoder_name='timm-mobilenetv3_small_minimal_100',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
    activation='sigmoid'
)

model.to(device)

# 가중치만 불러오기
# model.load_state_dict(torch.load('./best_deeplabv3plus_resnet101_voc_os16.pth'), strict=False)

model.load_state_dict(torch.load('./DeepLabPlus/Best_DeepLabV3_2_small_minimal100_Aug7.pth'))

lr = 0.0001

optim = Adam(params=model.parameters(), lr=lr)

# loss = smp.utils.losses.DiceLoss()
loss = smp.utils.losses.JaccardLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optim,
    device=device,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=device,
    verbose=True,
)

max_score = 0

for i in range(1000):
    print(f'Epoch: {i + 1}')
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model.state_dict(), './DeepLabPlus/Best_DeepLabV3_2_small_minimal100_Aug8.pth')
        torch.save(model, './DeepLabPlus/Best_model.pt')
        print('Model saved!')
    if (i + 1) % 5 == 0:
        torch.save(model.state_dict(), f'./DeepLabPlus/epoch_{i+1}.pth')

# load checkpoint
# checkpoint = torch.load('D:fix_mask/Portrait_seg_15.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optim.load_state_dict(checkpoint['optimizer_state_dict'])
# num_epoch = checkpoint['epoch']

# best_loss = 9999
#
# train
# for epoch in range(200):
#     model.train()
#     iterator = tqdm.tqdm(train_loader)
#     running_loss = 0.0
#     for data, label in iterator:
#         optim.zero_grad()
#
#         preds = model(data.to(device))
#         loss = nn.BCEWithLogitsLoss()(preds['out'], label.type(torch.FloatTensor).to(device))
#         # loss = loss_func.DiceBCELoss()(preds['out'], label.type(torch.FloatTensor).to(device))
#
#         loss.backward()
#         optim.step()
#
#         running_loss += loss.item()
#
#         iterator.set_description(f'train: {epoch + 1} loss: {loss.item()}')
#     print('Train Loss:', running_loss / len(train_loader))
#
#     # model.eval()
#     # with torch.no_grad():
#     #     valid_running_loss = 0.0
#     #     valid_iterator = tqdm.tqdm(valid_loader)
#     #     for data, label in valid_iterator:
#     #         preds = model(data.to(device))
#     #         loss = nn.BCEWithLogitsLoss()(preds['out'], label.type(torch.FloatTensor).to(device))
#     #
#     #         valid_running_loss += loss.item()
#     #         valid_iterator.set_description(f'valid: {epoch + 1} loss: {loss.item()}')
#     #     print('Valid Loss:', valid_running_loss / len(valid_loader))
#
#     # save checkpoint
#     if (epoch + 1) % 5 == 0:
#         torch.save(model.state_dict(), f'deeplab_Aug_{epoch + 1}.pth')
#
#     if best_loss > running_loss / len(train_loader):
#         best_loss = running_loss / len(train_loader)
#         torch.save(model.state_dict(), 'best_deeplab_Aug.pth')
#
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optim.state_dict(),
#     'epoch': 200
# }, 'Portrait_segmentation.pth')
