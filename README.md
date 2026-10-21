# dx-m1-app-examples

통합제어기(CM5 기반)와 DEEPX DX-M1 NPU로 만드는 추론 응용 프로그램 예제입니다. 통합제어기에서 실행합니다.

| 폴더 | 내용 | DX-M1 개발 입문서 |
|---|---|---|
| `detection/` | YOLO 검출 전처리(레터박스)·후처리(NMS) `yolo_common.py`, 동영상·카메라 비동기 검출 `detect_video.py` | 8.2절, 8.4절 |
| `cpp/` | DX-RT C++ API로 이미지 1장을 분류하는 최소 예제(CMake) | 8.3절 |
| `farm/` | CLIP으로 농장 대상(익은 귤, 돼지 등) 이름을 판정하는 `farm-clip.py`, 감귤 검출 설정 `conf50.json`, 예제 사진 `samples/`(출처 `samples/SOURCES.md`) | 8.3절, 9장 |

## 준비

DX-RT(`libdxrt-bin`)와 `dx_app` 예제 환경이 설치되어 있어야 합니다(입문서 2장). Python 예제는 `dx_app`의 가상환경(`dx_engine`, OpenCV 포함)에서 실행합니다.

```sh
git clone https://github.com/Gyurinet/dx-m1-app-examples.git ~/dx-m1-app-examples
source ~/sample/dx_app/venv-dxnn/bin/activate
```

## 사용 예

```sh
# 동영상 검출: --depth 1은 동기, 2 이상은 비동기로 겹쳐 실행
cd ~/dx-m1-app-examples/detection
python detect_video.py <모델>.dxnn <동영상 | 카메라 번호 | rtsp://주소> --depth 4
python detect_video.py <모델>.dxnn <동영상> --frames 90 --save out.mp4

# C++ 분류 예제
cd ~/dx-m1-app-examples/cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/classify <분류 모델>.dxnn ../farm/samples/citrus-tree.jpg

# 농장 이름 판정(CLIP): dx-clip-starter 폴더에 복사해 그 가상환경으로 실행
F=~/dx-m1-app-examples/farm; S=$F/samples
cp $F/farm-clip.py ~/sample/dx-clip-starter/
cd ~/sample/dx-clip-starter
.venv-clip/bin/python farm-clip.py --image $S/citrus-tree.jpg \
  --labels "익은 귤=a photo of ripe orange mandarins on a tree" "덜 익은 귤=a photo of unripe green mandarins on a tree"

# 탐지 상자마다 판정: dx_app 검출 예제의 --show-log 출력을 넘긴다
~/sample/dx_app/bin/yolov11x_sync -m ~/workspace/res/models/yolo11-x_640x640.dxnn -i $S/sow-piglets.jpg --no-display --show-log > det.log
.venv-clip/bin/python farm-clip.py --image $S/sow-piglets.jpg --det-log det.log --min-score 0.3 \
  --labels "돼지=a photo of a pig" "소=a photo of a cow" "양=a photo of a sheep" "개=a photo of a dog"

# 감귤 개수 세기: dx_app의 YOLOv7 예제에 신뢰도 기준 0.5 설정을 적용
cd ~/sample/dx_app
./bin/yolov7_sync -m assets/models/yolov7_640x640.dxnn -i $S/citrus-tree.jpg --no-display --save --show-log --config $F/conf50.json
```

검출 예제는 출력이 `[1, 4 + 클래스 수, 후보 수]`인 YOLO(v8·11) 모델에 맞춰져 있습니다. `dx-m1-model-convert`로 변환한 `yolo11n.dxnn`, `yolo11n_topk.dxnn`으로 확인했습니다.

## 확인한 환경

- 통합제어기: Raspberry Pi Compute Module 5, Debian 13(trixie), PCIe Gen3 x1
- DEEPX: DX-RT 3.4.2, `dx_engine` 3.4.2(Python 3.13), 펌웨어 v2.7.4, `dx_app` v3.2.2
- 빌드: CMake 3.31.6, OpenCV(`dx_app` 설치 때 함께 설치)
- 동영상 처리 속도는 동영상 파일로 측정했습니다. 카메라·RTSP 입력은 입력 장치에 따라 달라질 수 있습니다.
- `farm/samples/`의 사진으로 가이드 9장 그림의 판정 결과를 재현할 수 있습니다.

## 라이선스

규리넷 주식회사의 독점 라이선스입니다. 제품을 공급받은 고객은 제품의 개발·시험·운영 목적으로 사용·수정할 수 있으며, 서면 허가 없이 제3자에게 배포할 수 없습니다. 자세한 조건은 `LICENSE`를 참고합니다. `farm/samples/`의 사진은 원래 조건(퍼블릭 도메인·CC0)을 따릅니다.
