// DX-RT C++ 최소 예제: 분류 모델로 이미지 1장 추론
#include <dxrt/dxrt_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>

int main(int argc, char* argv[])
{
    if (argc < 3) { std::cerr << "usage: classify <model.dxnn> <image>\n"; return 1; }
    dxrt::InferenceEngine ie(argv[1]);

    // 입력: uint8 NHWC(RGB). 정규화는 모델 안에서 처리된다
    auto in = ie.GetInputs().front();
    int h = static_cast<int>(in.shape()[1]), w = static_cast<int>(in.shape()[2]);
    cv::Mat img = cv::imread(argv[2]), rgb;
    cv::resize(img, rgb, cv::Size(w, h));
    cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);

    auto outputs = ie.Run(rgb.data);                   // 동기 실행
    auto& out = outputs.front();
    const float* score = static_cast<const float*>(out->data());
    size_t n = static_cast<size_t>(out->shape().back());

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b) { return score[a] > score[b]; });
    for (int i = 0; i < 5; ++i)
        std::cout << idx[i] << "\t" << score[idx[i]] << "\n";
    std::cout << "NPU " << ie.GetNpuInferenceTime() / 1000.0 << " ms, latency "
              << ie.GetLatency() / 1000.0 << " ms\n";
    return 0;
}
