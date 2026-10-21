# dx_engine 최소 예제: ResNet50으로 이미지 1장 분류
#   python3 classify.py <모델>.dxnn <이미지>
import sys
import cv2
import numpy as np
from dx_engine import InferenceEngine

model_path, image_path = sys.argv[1], sys.argv[2]
ie = InferenceEngine(model_path)

info = ie.get_input_tensors_info()[0]
print("input :", info)
_, h, w, _ = info["shape"]                      # NHWC, UINT8

img = cv2.imread(image_path)
img = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB)
outputs = ie.run([img.astype(np.uint8)])        # 출력 텐서 목록

scores = outputs[0].reshape(-1)
top5 = np.argsort(scores)[::-1][:5]
print("output:", outputs[0].shape, outputs[0].dtype)
print("top-5 :", [(int(i), round(float(scores[i]), 3)) for i in top5])
print(f"NPU {ie.get_npu_inference_time() / 1000:.2f} ms, latency {ie.get_latency() / 1000:.2f} ms")
