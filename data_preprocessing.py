import os
import glob
import shutil
import cv2
import matplotlib.pyplot as plt

path = 'D://archive/matting_human_half/'
dir_img = 'D:seg/image/'
dir_label = 'D:seg/label/'

img_path = glob.glob(os.path.join(dir_img, '*.jpg'))
label_path = glob.glob(os.path.join(dir_label, '*.png'))

# mask 이미지
os.makedirs('D:seg/mask', exist_ok=True)

for i in label_path:
    name = i.split('\\')[-1].split('.')[0]
    rgba_image = cv2.imread(i, cv2.IMREAD_UNCHANGED)
    alpha = rgba_image[:, :, 3]
    cv2.imwrite(f'D:seg/mask/{name}.png', alpha)
    # print(rgba_image.shape)
    # plt.imshow(alpha)
    # plt.show()

# 누락 파일 제거
# img_paths = sorted(img_path)
# label_paths = sorted(label_path)
# cnt = 0
#
# for img, label in zip(img_paths, label_paths):
#     img_name = img.split('\\')[-1].split('.')[0]
#     label_name = label.split('\\')[-1].split('.')[0]
#
#     if label_name != img_name:
#         print(img_name, label_name)
#         break

# 데이터 이동
# image_path = glob.glob(os.path.join(path, '*', '*', '*', '*.jpg'))
# label_path = glob.glob(os.path.join(path, '*', '*', '*', '*.png'))
#
# os.makedirs(dir_img, exist_ok=True)
# os.makedirs(dir_label, exist_ok=True)
#
# for img in image_path:
#     img_name = img.split('\\')[-1]
#     shutil.copy(img, dir_img + img_name)
#
# for label in label_path:
#     label_name = label.split('\\')[-1]
#     shutil.copy(label, dir_label + label_name)
