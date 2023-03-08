import os
import glob
import cv2
import tqdm

img_path = glob.glob(os.path.join('D:seg/image/', '*.jpg'))
mask_path = glob.glob(os.path.join('D:seg/mask/', '*.png'))

resized_img_path = 'D:seg_resize/image/'
resized_mask_path = 'D:seg_resize/mask/'

os.makedirs(resized_img_path, exist_ok=True)
os.makedirs(resized_mask_path, exist_ok=True)

for i in tqdm.tqdm(img_path):
    img = cv2.imread(i)
    img_name = i.split('\\')[-1]
    img = cv2.resize(img, (256, 256))
    cv2.imwrite(resized_img_path + img_name, img)

for i in tqdm.tqdm(mask_path):
    img = cv2.imread(i)
    mask_name = i.split('\\')[-1]
    img = cv2.resize(img, (256, 256))
    cv2.imwrite(resized_img_path + mask_name, img)
