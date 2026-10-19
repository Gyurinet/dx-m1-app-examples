# 동영상(또는 카메라)에서 객체를 검출한다. --depth 1이면 동기, 2 이상이면 비동기로 겹쳐 실행
import argparse, collections, time
import cv2
import numpy as np
from dx_engine import InferenceEngine
from yolo_common import letterbox, postprocess

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("source", help="동영상 파일, 카메라 번호(0) 또는 rtsp:// 주소")
ap.add_argument("--depth", type=int, default=4, help="동시에 NPU에 넣어 둘 프레임 수")
ap.add_argument("--save", help="결과 동영상 파일")
ap.add_argument("--frames", type=int, default=0, help="처리할 프레임 수(0은 끝까지)")
a = ap.parse_args()

ie = InferenceEngine(a.model)
cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
out = None
pending = collections.deque()                    # (작업 번호, 원본 프레임, 비율, 여백)
count, t0 = 0, time.perf_counter()

def finish(job, frame, r, off):
    global out, count
    dets = postprocess(ie.wait(job)[0][0], r, off)
    if a.save:
        for x1, y1, x2, y2, s, c in dets:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (29, 168, 255), 2)
        if out is None:
            out = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), 30, frame.shape[1::-1])
        out.write(frame)
    count += 1

while True:
    ok, frame = cap.read()
    if not ok or (a.frames and count + len(pending) >= a.frames):
        break
    img, r, off = letterbox(frame)
    job = ie.run_async([cv2.cvtColor(img, cv2.COLOR_BGR2RGB)])
    pending.append((job, frame, r, off))
    if len(pending) >= a.depth:                  # 가장 오래된 프레임부터 결과를 받는다
        finish(*pending.popleft())
while pending:
    finish(*pending.popleft())

dt = time.perf_counter() - t0
print(f"{count} frames, {count / dt:.1f} FPS (depth {a.depth})")
if out: out.release()
