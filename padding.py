import cv2
import os
import glob

def padding(img, set_size):
    try:
        h, w, c = img.shape
    except:
        print('파일을 다시 확인')
        raise

    if h < w:
        new_width = set_size
        new_height = int(new_width * (h / w))
    else:
        new_height = set_size
        new_width = int(new_height * (w / h))

    if max(h, w) < set_size:
        img = cv2.resize(img, (new_width, new_height), cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img, (new_width, new_height), cv2.INTER_AREA)

    try:
        h, w, c = img.shape
    except:
        print('파일을 다시 확인')
        raise

    delta_w = set_size - w
    delta_h = set_size - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return new_img

path = 'D:seg/'
save_path = 'D:seg_pad/mask/'

image_path = glob.glob(os.path.join(path, '*', 'masks_instances', '*', '*.png'))
os.makedirs(save_path, exist_ok=True)

for i in image_path:
    name = i.split('\\')[-2]
    folder = i.split('\\')[-4]
    # print(folder, name)
    img = cv2.imread(i)
    img = padding(img, 256)
    cv2.imwrite(save_path + folder + f'{name}.png', img)
