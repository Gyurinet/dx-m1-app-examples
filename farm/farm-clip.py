#!/usr/bin/env python3
"""농장 대상 이름 판정(CLIP) - DX-M1 NPU

범용 탐지 모델은 COCO 80개 클래스만 알아서 돼지를 cow/sheep 등으로 표시한다.
이 스크립트는 CLIP으로 '농장에서 쓰는 이름'과 이미지를 직접 비교한다.

  1) 사진 전체 판정
     farm-clip.py --image tree.jpg --labels "익은 귤=a photo of ripe orange mandarins" \
                                            "덜 익은 귤=a photo of unripe green mandarins" --out result.jpg
  2) 탐지 결과 다시 이름 붙이기(탐지 예제의 --show-log 출력 사용)
     ./bin/yolov11x_sync ... --show-log > det.log
     farm-clip.py --image pen.jpg --det-log det.log --labels "어미 돼지=..." "새끼 돼지=..." --out result.jpg

이미지 인코더는 NPU(.dxnn), 텍스트 인코더는 CPU(ONNX Runtime)에서 실행한다.
"""
import argparse
import re
import sys

import cv2
import numpy as np
import onnxruntime
import open_clip
from dx_engine import InferenceEngine
from PIL import Image, ImageDraw, ImageFont

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)  # *-transform.txt 의 Normalize 값
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
FONT = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
PALETTE = [(255, 194, 26), (64, 196, 255), (255, 99, 132), (124, 220, 120), (200, 140, 255)]
DET_LINE = re.compile(r"^\[DET\]\s+(\S+)\s+([\d.]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)")


def parse_labels(items):
    names, prompts = [], []
    for item in items:
        if "=" not in item:
            sys.exit(f"--labels 형식은 '표시이름=영어 설명' 입니다: {item}")
        name, prompt = item.split("=", 1)
        names.append(name.strip()); prompts.append(prompt.strip())
    return names, prompts


def encode_text(onnx_path, model_name, prompts):
    tokens = open_clip.get_tokenizer(model_name)(prompts).numpy().astype(np.int64)
    sess = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    # 배포된 텍스트 인코더 ONNX는 배치 크기 1로 고정되어 있어 문장을 하나씩 넣는다
    feats = np.concatenate([sess.run(None, {name: tokens[i:i + 1]})[0] for i in range(len(tokens))])
    return feats / np.linalg.norm(feats, axis=-1, keepdims=True)


def preprocess(rgb, size=224):
    # Resize(짧은 변 224, bicubic) -> CenterCrop(224) -> Normalize, NCHW float32
    h, w = rgb.shape[:2]
    s = size / min(h, w)
    img = cv2.resize(rgb, (max(size, round(w * s)), max(size, round(h * s))), interpolation=cv2.INTER_CUBIC)
    y, x = (img.shape[0] - size) // 2, (img.shape[1] - size) // 2
    img = img[y:y + size, x:x + size].astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])


def encode_image(engine, rgb):
    out = engine.run([preprocess(rgb)])
    feat = np.asarray(out[0] if isinstance(out, list) else out, dtype=np.float32).reshape(-1)
    return feat / np.linalg.norm(feat)


def softmax(x, scale):
    z = np.exp((x - x.max()) * scale)
    return z / z.sum()


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def draw_label(draw, xy, text, color, fnt):
    # 상자 왼쪽 위 바깥에 라벨을 붙이고, 화면 위로 넘치면 상자 안쪽에 붙인다
    x, y = xy
    l, t, r, b = draw.textbbox((0, 0), text, font=fnt)
    pad = max(2, fnt.size // 6)
    top = y - (b - t) - 2 * pad
    if top < 0:
        top = y
    draw.rectangle([x, top, x + (r - l) + 2 * pad, top + (b - t) + 2 * pad], fill=color)
    draw.text((x + pad - l, top + pad - t), text, font=fnt, fill=(20, 20, 20))


def main():
    ap = argparse.ArgumentParser(description="농장 대상 이름 판정(CLIP, DX-M1 NPU)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--labels", nargs="+", required=True, help="'표시이름=영어 설명' 을 2개 이상")
    ap.add_argument("--det-log", help="탐지 예제 --show-log 출력 파일. 주면 상자마다 판정")
    ap.add_argument("--min-score", type=float, default=0.5, help="--det-log 에서 사용할 탐지 신뢰도 하한")
    ap.add_argument("--out", help="결과 이미지 저장 경로")
    ap.add_argument("--image-encoder", default="dxnn/ViT-L-14-quickgelu-dfn2b.dxnn")
    ap.add_argument("--text-encoder", default="onnx/ViT-L-14-quickgelu-dfn2b-text.onnx")
    ap.add_argument("--model-name", default="ViT-L-14-quickgelu")
    ap.add_argument("--scale", type=float, default=100.0, help="CLIP 로짓 배율")
    a = ap.parse_args()

    names, prompts = parse_labels(a.labels)
    dets = []
    if a.det_log:
        # 탐지 결과가 없으면 0마리로 보고하지 않고 멈춘다(탐지 명령이 실행되지 않은 경우가 대부분)
        for line in open(a.det_log, encoding="utf-8", errors="ignore"):
            m = DET_LINE.match(re.sub(r"\x1b\[[0-9;]*m", "", line).strip())
            if m:
                dets.append((m.group(1), float(m.group(2)), *[float(v) for v in m.groups()[2:]]))
        if not dets:
            sys.exit(f"{a.det_log}에 [DET] 줄이 없습니다. 탐지 명령이 --show-log로 실행되어 이 파일에 출력했는지 확인하세요.")
    text = encode_text(a.text_encoder, a.model_name, prompts)
    engine = InferenceEngine(a.image_encoder)
    bgr = cv2.imread(a.image)
    if bgr is None:
        sys.exit(f"이미지를 읽을 수 없습니다: {a.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    fnt = font(max(16, W // 40))

    if not a.det_log:
        prob = softmax(text @ encode_image(engine, rgb), a.scale)
        order = np.argsort(-prob)
        print("rank  score   label")
        for r, i in enumerate(order, 1):
            print(f"{r:>4}  {prob[i]:.3f}  {names[i]}")
        # 왼쪽 위에 판정 막대
        bar_w, row = W // 3, fnt.size + 14
        draw.rectangle([10, 10, 30 + bar_w, 20 + row * len(order)], fill=(20, 20, 20))
        for r, i in enumerate(order):
            y = 18 + r * row
            draw.rectangle([20, y, 20 + int(bar_w * prob[i]), y + row - 8], fill=PALETTE[i % len(PALETTE)])
            draw.text((26, y + 1), f"{names[i]} {prob[i] * 100:.0f}%", font=fnt, fill=(255, 255, 255))
    else:
        boxes = [d for d in dets if d[1] >= a.min_score]
        counts = {n: 0 for n in names}
        print("box  det(class score)      ->  label (score)")
        for k, (cls, sc, x1, y1, x2, y2) in enumerate(boxes, 1):
            mx, my = (x2 - x1) * 0.1, (y2 - y1) * 0.1   # 상자 둘레 10%를 함께 잘라 문맥을 준다
            crop = rgb[int(max(0, y1 - my)):int(min(H, y2 + my)), int(max(0, x1 - mx)):int(min(W, x2 + mx))]
            if crop.size == 0:
                continue
            prob = softmax(text @ encode_image(engine, crop), a.scale)
            i = int(prob.argmax()); counts[names[i]] += 1
            print(f"{k:>3}  {cls:>8} {sc:.2f}  ->  {names[i]} ({prob[i]:.2f})")
            color = PALETTE[i % len(PALETTE)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(3, W // 300))
            draw_label(draw, (x1, y1), f"{names[i]} {prob[i] * 100:.0f}%", color, fnt)
        summary = "  ".join(f"{n} {c}" for n, c in counts.items())
        print("count:", summary)
        l, t, r, b = draw.textbbox((0, 0), summary, font=fnt)
        draw.rectangle([10, H - (b - t) - 30, 30 + (r - l), H - 10], fill=(20, 20, 20))
        draw.text((20, H - (b - t) - 22 - t), summary, font=fnt, fill=(255, 255, 255))

    if a.out:
        canvas.save(a.out, quality=92)
        print("saved:", a.out)


if __name__ == "__main__":
    main()
