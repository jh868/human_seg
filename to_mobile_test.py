import torch
import torchvision
import segmentation_models_pytorch as smp
import torch.onnx
import onnx
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


model = deeplabv3_mobilenet_v3_large()
model.classifier = DeepLabHead(960, 1)

model.load_state_dict(torch.load('./best_deeplab.pth', map_location='cpu')['model_state_dict'])

model = torch.quantization.convert(model)

scripted_model = torch.jit.script(model)

opt_model = optimize_for_mobile(scripted_model)
opt_model._save_for_lite_interpreter("deeplabv3_scripted_optimized.ptl")

# torch.jit.save(opt_model, 'mobile.pt')

# model.eval()
#
# x = torch.randn(batch_size, 3, 256, 256, requires_grad=True)
# torch_out = model(x)
#
# # 모델 변환
# torch.onnx.export(model,  # 실행될 모델
#                   x,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
#                   "mobile.onnx",  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
#                   export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
#                   opset_version=11,  # 모델을 변환할 때 사용할 ONNX 버전
#                   do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
#                   input_names=['input'],  # 모델의 입력값을 가리키는 이름
#                   output_names=['output'],  # 모델의 출력값을 가리키는 이름
#                   dynamic_axes={'input': {0: 'batch_size'},  # 가변적인 길이를 가진 차원
#                                 'output': {0: 'batch_size'}})
#
# onnx_model = onnx.load("mobile.onnx")
# onnx.checker.check_model(onnx_model)
