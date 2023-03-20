import numpy as np
import gradio as gr
import torch
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = deeplabv3_mobilenet_v3_large()
model.classifier = DeepLabHead(960, 1)

checkpoint = torch.load('./best_deeplab.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])


def apply(image_input, image_background, select):
    img_transform = A.Compose([
        A.Resize(width=256, height=256),
        ToTensorV2()
    ])

    img = image_input.astype(np.float32)
    img = img_transform(image=img)['image'].unsqueeze(0)
    # print(img.shape)

    model.eval()
    with torch.no_grad():
        output = model(img)['out']
    output = torch.sigmoid(output)

    output = output.detach().numpy()
    output = output.squeeze()

    output = (output > 0.5).astype(np.uint8) * 255

    _, mask = cv2.threshold(output, 30, 255, cv2.THRESH_BINARY)
    # mask = cv2.bitwise_not(mask)  # 반전

    ### 가장자리 부드럽게 ###

    kernel = np.ones((3, 3), np.uint8)
    # 오프닝
    mask = cv2.erode(mask, kernel, iterations=5)
    mask = cv2.dilate(mask, kernel, iterations=5)
    # 가우시안 블러
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    ### 필터링
    for i in select:
        if i == "AdvancedBlur":
            transform = A.Compose([
                A.AdvancedBlur(p=1),
            ])
            image_background = transform(image=image_background)['image']
        elif i == "CLAHE":
            transform = A.Compose([
                A.CLAHE(p=1),
            ])
            image_background = transform(image=image_background)['image']
        elif i == "Defocus":
            transform = A.Compose([
                A.Defocus(p=1),
            ])
            image_background = transform(image=image_background)['image']
        elif i == "BrightnessContrast":
            transform = A.Compose([
                A.Defocus(p=1),
            ])
            image_background = transform(image=image_background)['image']
        elif i == "Fog":
            transform = A.Compose([
                A.RandomFog(p=1),
            ])
            image_background = transform(image=image_background)['image']
        elif i == "Gamma":
            transform = A.Compose([
                A.RandomGamma(p=1),
            ])
            image_background = transform(image=image_background)['image']
        elif i == "Gravel":
            transform = A.Compose([
                A.RandomGravel(p=1),
            ])
            image_background = transform(image=image_background)['image']
        elif i == "Rain":
            transform = A.Compose([
                A.RandomRain(p=1),
            ])
            image_background = transform(image=image_background)['image']
        elif i == "Shadow":
            transform = A.Compose([
                A.RandomShadow(p=1),
            ])
            image_background = transform(image=image_background)['image']
        elif i == "Snow":
            transform = A.Compose([
                A.RandomSnow(p=1),
            ])
            image_background = transform(image=image_background)['image']
        elif i == "SunFlare":
            transform = A.Compose([
                A.RandomSunFlare(p=1),
            ])
            image_background = transform(image=image_background)['image']
        elif i == "ToneCurve":
            transform = A.Compose([
                A.RandomToneCurve(p=1),
            ])
            image_background = transform(image=image_background)['image']
        elif i == "Gray":
            transform = A.Compose([
                A.ToGray(p=1),
            ])
            image_background = transform(image=image_background)['image']

    #### background 
    fg_h, fg_w, _ = image_input.shape

    bg_h, bg_w, _ = image_background.shape

    # fit to fg width
    image_background = cv2.resize(image_background, dsize=(fg_w, int(fg_w * bg_h / bg_w)))

    bg_h, bg_w, _ = image_background.shape

    margin = (bg_h - fg_h) // 2

    if margin > 0:
        image_background = image_background[margin:-margin, :, :]
    else:
        image_background = cv2.copyMakeBorder(image_background, top=abs(margin), bottom=abs(margin), left=0, right=0,
                                              borderType=cv2.BORDER_REPLICATE)

    # final resize
    image_background = cv2.resize(image_background, dsize=(fg_w, fg_h))

    mask = cv2.resize(mask, (fg_w, fg_h))
    cv2.copyTo(image_input, mask, image_background)

    return image_background


def stop(inp):
    return inp


# 구체적 화면 코드 

with gr.Blocks() as demo:
    gr.Markdown("# 배경 변경 프로그램")
    gr.Markdown("배경 이미지 변경")
    gr.HTML("""<div style="display: inline-block;  float: right;">Made By 5 Team : 이성규, 김민정, 이승현, 이주형, 민안세</div>""")

    # 1 번탭
    with gr.Tab("Image Upload"):
        with gr.Row():
            image_input = gr.Image(label="Upload IMG")
            image_background = gr.Image(label="Upload background Image")
        select = gr.Dropdown(
            ["AdvancedBlur", "CLAHE", "Defocus", "BrightnessContrast", "Fog", "Gamma", "Gravel", "Rain", "Shadow",
             "Snow", "SunFlare", "ToneCurve", "Gray"], label="Background effect", value=["Rain", "Gray"],
            multiselect=True)
        image_button = gr.Button("TransForm Image")
        image_output = gr.Image(label="Output IMG")

        image_button.click(apply, inputs=[image_input, image_background, select], outputs=image_output)

        gr.Examples(
            label="Image",
            examples=["./1803151818-00000003.jpg", "./1803151818-00000004.jpg", "./1803151818-00000006.jpg"],
            inputs=image_input,
        )
        gr.Examples(
            label="background Image",
            examples=["./background.jpg"],
            inputs=image_background,
        )

    # 2번 탭
    with gr.Tab("Using WebCam"):
        with gr.Row():
            image_web = gr.Image(source="webcam", streaming=True, label="Web Cam")
            image_input = gr.Image(label="IMG")
            image_background = gr.Image(label="Upload background Image")
        select = gr.Dropdown(
            ["AdvancedBlur", "CLAHE", "Defocus", "BrightnessContrast", "Fog", "Gamma", "Gravel", "Rain", "Shadow",
             "Snow", "SunFlare", "ToneCurve", "Gray"], label="Background effect", value=["Rain", "Gray"],
            multiselect=True)
        image_button = gr.Button("TransForm Image")
        image_output = gr.Image(label="Output IMG")

        image_button.click(stop, inputs=image_web, outputs=image_input)
        image_button.click(apply, inputs=[image_input, image_background, select], outputs=image_output)

        gr.Examples(
            label="background Image",
            examples=["./background.jpg"],
            inputs=image_background,
        )

demo.launch(share=True)
