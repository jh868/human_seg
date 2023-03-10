import torch
import cv2
import numpy as np
from model import MobileUNet
import glob
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MobileUNet().to(device)

checkpoint = torch.load('./pt_file_3/Portrait_seg_pretrain_30.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

path = './test/image/'
# img_path = './test/image/00002.jpg'

img_path = glob.glob(os.path.join(path, '*.jpg'))

for i in img_path:

    img = cv2.imread(i, cv2.IMREAD_UNCHANGED)

    img_copy = cv2.resize(img, (256, 256))

    img_copy2 = np.transpose(img_copy, (2, 0, 1))
    img_copy2 = torch.unsqueeze(torch.tensor(img_copy2), 0)

    pred = model(img_copy2.float())
    pred = pred.detach().numpy()
    pred = pred.astype(np.uint8)
    # print("pred = ", pred.shape)

    _, mask = cv2.threshold(pred, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)  # 반전

    # cv2로 투명배경 멕이기 # 근데 생각해보니까 투명 배경할 이유가없음
    # mask = cv2.merge((mask, mask, mask))
    # result = cv2.addWeighted(img_copy, 1, mask, 1, 0)

    b, g, r = cv2.split(img_copy)
    result = cv2.merge([b, g, r, pred], 4)
    # print("result= ", result.shape)
    #
    # cv2.imwrite('result.png', result)
    # cv2.imshow(' ', result)
    # cv2.waitKey(0)

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

    # cv2.imshow("", background)
    # cv2.waitKey(0)

    # 넣기
    mask = cv2.resize(mask, (fg_w, fg_h))

    # print(background.shape)
    # print(img.shape)
    # print(mask.shape)

    cv2.copyTo(img, mask, background)

    cv2.imshow("", background)
    cv2.waitKey(10)
