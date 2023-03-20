import copy

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import torch
import cv2
import numpy as np
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import os
import glob


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = deeplabv3_mobilenet_v3_large()
model.classifier = DeepLabHead(960, 1)
print(model)

checkpoint = torch.load('./best_deeplab.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

transform = A.Compose([
    A.Resize(width=256, height=256),
    # A.HorizontalFlip(),
    # A.Rotate(limit=20),
    ToTensorV2()
])


# path = './test/image/00001.jpg'
img_path = glob.glob(os.path.join('./test/image/', '*.jpg'))

for path in img_path:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img_copy2 = copy.deepcopy(img)
    fg_h, fg_w, _ = img.shape
    img_copy = cv2.resize(img, (256, 256))
    img = img.astype(np.float32)
    img = transform(image=img)['image'].unsqueeze(0)
    # print(img.shape)

    model.eval()
    with torch.no_grad():
        output = model(img)['out']
    output = torch.sigmoid(output)

    output = output.detach().numpy()
    output = output.squeeze()

    # print(output)
    output = (output > 0.5).astype(np.uint8) * 255
    # print(output)
    # cv2.imshow('', output)
    # cv2.waitKey(0)

    _, mask = cv2.threshold(output, 30, 255, cv2.THRESH_BINARY)

    b, g, r = cv2.split(img_copy)
    result = cv2.merge([b, g, r, output], 4)

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

    # 넣기
    mask = cv2.resize(mask, (fg_w, fg_h))

    cv2.copyTo(img_copy2, mask, background)

    cv2.imshow("", background)
    cv2.waitKey(10)
