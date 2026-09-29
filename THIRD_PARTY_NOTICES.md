# Third-party notices

Navbea RVM 1.1.1-local is distributed under GPL-3.0-only. The following components
are included in source or binary distributions. Exact package versions are
also recorded in `SBOM.spdx.json`.

## Robust Video Matting and model

Robust Video Matting was developed by Shanchuan Lin, Linjie Yang, Imran Saleemi
and Soumyadip Sengupta and is distributed under GPL-3.0.

- Upstream source: https://github.com/PeterL1n/RobustVideoMatting
- Pinned release: https://github.com/PeterL1n/RobustVideoMatting/releases/tag/v1.0.0
- Model: `rvm_mobilenetv3_fp32.onnx`
- Model SHA-256: `88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828`
- Full license: `LICENSE`
- Changes made around the model: `CHANGES.md`

The optional MobileNetV3 FP16 and ResNet50 FP16/FP32 ONNX models are downloaded
only on request from the same official v1.0.0 release. Their exact sizes and
SHA-256 checksums are recorded in `models/catalog.json`. These unmodified models
are also GPL-3.0; they are not included in the default installer payload, except
`rvm_resnet50_fp32.onnx` (SHA-256
`25db300fcb6ee27f941a1b52c97856e8d1f13c7f35817f81a612f89af0e8a85c`), which ships
as the optional photo matting model "RVM ResNet50" listed in `models/still-catalog.json`.

## BiRefNet photo matting models (optional)

BiRefNet (Bilateral Reference for High-Resolution Dichotomous Image
Segmentation) was developed by Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu,
Jorma Laaksonen, Wanli Ouyang and Nicu Sebe and is distributed under the MIT
license.

- Upstream source: https://github.com/ZhengPeng7/BiRefNet
- ONNX exports: https://huggingface.co/onnx-community/BiRefNet_lite-ONNX and
  https://huggingface.co/onnx-community/BiRefNet-portrait-ONNX (MIT)

The models are not bundled. A technician downloads one only on request from a
pinned repository revision; its size and SHA-256 are verified against
`models/still-catalog.json` before use. They matte captured photographs only;
the live preview always uses RVM. Foreground colour estimation follows
Forte & Pitié, "Approximate Fast Foreground Colour Estimation" (ICIP 2021),
reimplemented in `worker/still_matte.py`.

## Subject scoring models (score mode)

The "score" subject-selection mode ranks people by closeness, facing the camera
and height, so people walking behind the guests are removed from the matte.
Both models run locally on the CPU; no image or result is stored.

- Depth Anything V2 Small (Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao,
  Xiaogang Xu, Jiashi Feng, Hengshuang Zhao), Apache-2.0. Unmodified FP16 ONNX
  export from https://huggingface.co/onnx-community/depth-anything-v2-small
  (`onnx/model_fp16.onnx`), shipped as `models/depth-anything-v2-small-fp16.onnx`.
- YuNet face detector (Shiqi Yu et al., OpenCV Zoo), MIT. Unmodified
  `face_detection_yunet_2023mar.onnx` from
  https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet.

SHA-256 checksums are recorded in `models/scoring.json` and verified before use.

## Bundled diagnostics UI

- React 19.2.0, ReactDOM 19.2.0 and Scheduler 0.27.0 — MIT. License:
  `THIRD_PARTY_LICENSES/React-ReactDOM-Scheduler-MIT.txt`.
- @phosphor-icons/react 2.1.10 — MIT. License:
  `THIRD_PARTY_LICENSES/Phosphor-Icons-MIT.txt`.
- Electron 43.4.0 — MIT, with Chromium and other notices. Packaged Windows
  distributions include `LICENSE` and `LICENSES.chromium.html` beside the
  executable.

## Local-media protocol

The vendored `@navbea/local-media-protocol` implementation is MIT licensed.
Its license is at `local-media-protocol/LICENSE`.

## Hermetic Python runtime

The package preserves license and metadata files from every installed wheel.
The principal components are:

- Python 3.12 — PSF-2.0; `runtime/LICENSE.txt`.
- ONNX Runtime / ONNX Runtime DirectML 1.24.4 — MIT;
  `runtime/Lib/site-packages/onnxruntime/LICENSE` and
  `runtime/Lib/site-packages/onnxruntime/ThirdPartyNotices.txt`.
- NumPy 2.5.2 — BSD-3-Clause and bundled compatible notices;
  `runtime/Lib/site-packages/numpy-2.5.2.dist-info/licenses/`.
- Pillow 12.3.0 — MIT-CMU;
  `runtime/Lib/site-packages/pillow-12.3.0.dist-info/licenses/`.
- FlatBuffers 25.12.19 — Apache-2.0.
- Packaging 26.3 — Apache-2.0 OR BSD-2-Clause.
- Protobuf 7.36.1 — BSD-3-Clause.
- SymPy 1.14.0 and mpmath 1.3.0 — BSD-style licenses.
- pip 25.0.1 — MIT.

These components are unmodified dependencies. Their original license and
notice files remain inside the hermetic runtime distribution.
