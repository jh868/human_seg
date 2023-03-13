from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import torch
import cv2
import numpy as np
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = deeplabv3_mobilenet_v3_large()
model.classifier = DeepLabHead(960, 1)

checkpoint = torch.load('./best_deeplab.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

transform = A.Compose([
    A.Resize(width=256, height=256),
    # A.HorizontalFlip(),
    # A.Rotate(limit=20),
    ToTensorV2()
])


path = './test/image/00001.jpg'

img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
print(img.shape)
img = img.astype(np.float32)
img = transform(image=img)['image'].unsqueeze(0)
# print(img.shape)

model.eval()
with torch.no_grad():
    output = model(img)['out']
output = torch.sigmoid(output)

print(output.shape)
output = output.detach().numpy()
print(output.shape)
cv2.imshow('', output.squeeze())
cv2.waitKey(0)


# import matplotlib.pyplot as plt
#
# plt.imshow(output.squeeze(), alpha=0.5)
# plt.show()


# img_copy = cv2.resize(img, (256, 256))
#
# img_copy2 = np.transpose(img_copy, (2, 0, 1))
# img_copy2 = torch.unsqueeze(torch.tensor(img_copy2), 0)
# # img_copy2 = torch.tensor(img_copy2)
# print(img_copy2.shape)
#
# pred = model(img_copy2.float())
# pred = pred.detach().numpy()
# pred = pred.astype(np.uint8)
#
# cv2.imshow('', pred)
