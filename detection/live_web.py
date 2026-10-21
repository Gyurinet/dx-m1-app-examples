# 카메라·동영상·RTSP를 NPU로 실시간 검출하고, 결과 화면을 PC 브라우저로 보낸다(MJPEG)
import argparse, collections, os, queue, socket, statistics, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
os.environ.setdefault("DXRT_DYNAMIC_CPU_THREAD", "ON")   # DX-RT 권장: 모델의 CPU 작업 스레드를 부하에 맞춰 늘린다
import cv2
from dx_engine import InferenceEngine
from yolo_common import letterbox, postprocess

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
        "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
        "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
        "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
        "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
        "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier "
        "toothbrush").split()

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("source", help="카메라 번호(0), 동영상 파일 또는 rtsp:// 주소")
ap.add_argument("--port", type=int, default=8080, help="브라우저로 열 포트")
ap.add_argument("--conf", type=float, default=0.4, help="표시할 검출 신뢰도 하한")
ap.add_argument("--depth", type=int, default=4, help="동시에 NPU에 넣어 둘 프레임 수")
ap.add_argument("--width", type=int, default=1280, help="카메라에 요청할 가로 해상도")
ap.add_argument("--view-width", type=int, default=1280, help="브라우저로 보낼 화면의 최대 가로 크기")
ap.add_argument("--max-speed", action="store_true", help="동영상을 원래 속도에 맞추지 않고 최대 속도로 처리")
a = ap.parse_args()

latest = {"jpg": None, "no": 0, "viewers": 0}
cond = threading.Condition()

def publish(frame):
    if not latest["viewers"]:                            # 보는 사람이 없으면 JPEG 인코딩을 쉰다
        return
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    with cond:
        latest["jpg"], latest["no"] = buf.tobytes(), latest["no"] + 1
        cond.notify_all()

def draw(frame, dets, info):
    k = min(1.0, a.view_width / frame.shape[1])           # 보낼 크기로 줄이고 좌표도 같은 비율로
    if k < 1:
        frame = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    for x1, y1, x2, y2, s, c in dets:
        p1, p2 = (int(x1 * k), int(y1 * k)), (int(x2 * k), int(y2 * k))
        cv2.rectangle(frame, p1, p2, (29, 168, 255), 2)
        cv2.putText(frame, f"{COCO[int(c)]} {s:.2f}", (p1[0], max(16, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (29, 168, 255), 2)
    cv2.rectangle(frame, (0, 0), (430, 34), (20, 20, 20), -1)
    cv2.putText(frame, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

def run():
    ie = InferenceEngine(a.model)
    camera = a.source.isdigit()
    is_file = not camera and "://" not in a.source
    cap = cv2.VideoCapture(int(a.source) if camera else a.source)
    if camera:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # USB 카메라는 MJPG가 더 빠르다
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.width * 9 // 16)
    if not cap.isOpened():
        print(f"입력을 열 수 없습니다: {a.source}", flush=True)
        os._exit(1)
    period = 1 / (cap.get(cv2.CAP_PROP_FPS) or 30) if is_file and not a.max_speed else 0   # 동영상은 원래 속도로 재생
    pending, stamps = collections.deque(), collections.deque(maxlen=30)
    npu_ms, delay_ms = collections.deque(maxlen=30), collections.deque(maxlen=30)
    next_t = report_t = time.perf_counter()

    def finish(job, frame, r, off, t_in):
        nonlocal report_t
        dets = postprocess(ie.wait(job)[0][0], r, off, conf=a.conf)
        now = time.perf_counter()
        stamps.append(now)
        npu_ms.append(ie.get_npu_inference_time() / 1000)   # NPU가 실제로 계산한 시간
        delay_ms.append((now - t_in) * 1000)                # 프레임을 읽은 뒤 결과가 나오기까지
        fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0
        npu, delay = statistics.median(npu_ms), statistics.median(delay_ms)
        publish(draw(frame, dets, f"DX-M1 NPU {npu:.1f} ms | {fps:.1f} FPS | {len(dets)} obj"))
        if now - report_t >= 5:
            print(f"처리 속도 {fps:.1f} FPS, NPU 계산 {npu:.1f} ms, 지연 {delay:.0f} ms", flush=True)
            report_t = now

    # 읽기(디코딩)는 따로 돌려 추론과 겹친다. 카메라·RTSP는 가장 최근 프레임만 남겨 지연이 쌓이지 않게 한다
    frames = queue.Queue(maxsize=a.depth if is_file else 1)

    def reader():
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_file:                              # 동영상은 끝나면 처음부터 다시
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frames.put(None)
                return
            item = (frame, time.perf_counter())
            if is_file:
                frames.put(item)
                continue
            try:
                frames.get_nowait()                      # 처리하지 못한 이전 프레임은 버린다
            except queue.Empty:
                pass
            frames.put(item)

    threading.Thread(target=reader, daemon=True).start()
    while True:
        item = frames.get()
        if item is None:
            break
        frame, t_in = item
        if is_file:                                      # 동영상은 미리 읽어 둔 대기 시간을 지연에 넣지 않는다
            t_in = time.perf_counter()
        img, r, off = letterbox(frame)
        pending.append((ie.run_async([cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]), frame, r, off, t_in))
        if len(pending) >= a.depth:
            finish(*pending.popleft())
        if period:
            next_t = max(next_t + period, time.perf_counter() - period)
            time.sleep(max(0, next_t - time.perf_counter()))
    while pending:
        finish(*pending.popleft())
    print("입력이 끝났습니다.", flush=True)

PAGE = ("<!doctype html><meta charset='utf-8'><title>DX-M1 실시간 검출</title>"
        "<body style='margin:0;background:#111;color:#eee;font-family:sans-serif'>"
        "<p style='margin:8px'>DX-M1 실시간 검출</p><img src='/stream' style='max-width:100%'>").encode()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            no = 0
            with cond:
                latest["viewers"] += 1
            try:
                while True:
                    with cond:
                        cond.wait_for(lambda: latest["no"] != no, timeout=5)
                        no, jpg = latest["no"], latest["jpg"]
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with cond:
                    latest["viewers"] -= 1
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass

def my_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))             # 실제로 보내지 않고 나가는 주소만 확인
            return s.getsockname()[0]
        except OSError:
            return "<제어기 IP>"

threading.Thread(target=run, daemon=True).start()
print(f"PC 브라우저에서 http://{my_ip()}:{a.port}/ 를 엽니다. 끝내려면 Ctrl+C", flush=True)
try:
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()
except KeyboardInterrupt:
    pass
