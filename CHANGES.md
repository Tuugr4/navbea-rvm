# Changes from upstream Robust Video Matting

Navbea RVM 1.1.1-local uses the unmodified official
`rvm_mobilenetv3_fp32.onnx` model published with Robust Video Matting v1.0.0.
The model SHA-256 is recorded in `models/manifest.json`.

The following independently maintained components were added around the
upstream model:

- a hermetic Python inference worker that reads versioned NVF1 frames from
  standard input and writes sequence-aligned GRAY8 masks to standard output;
- a Node supervisor with local-only control and stream sockets, bounded queues,
  capability tokens, worker restart handling and native-resolution still matte;
- CPU, CUDA and DirectML execution-provider selection with safe CPU fallback;
- an Electron diagnostics application and Windows/Linux service definitions;
- release verification, model checksum validation and corresponding-source
  packaging.
- optional official FP16 and ResNet50 model downloads with checksum validation,
  explicit activation, live/native inference checks and rollback;
- FP16-aware tensor input handling and actual execution-provider reporting;
- a model-management interface with hardware memory guidance and a separate
  administrator-only mutation channel. The bundled FP32 default is unchanged.
- an opt-in background-removed camera preview using sequence/timestamp-matched
  JPEG and alpha frames, expiring viewer leases and no extra inference pass.

The upstream model architecture, training code and official weights are not
represented as original Navbea work. Their authors and source are identified in
`THIRD_PARTY_NOTICES.md`.
