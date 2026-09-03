# RVM packaging

Release CI produces signed Windows x64 and Ubuntu 24.04 x64 packages for CPU
and CUDA variants. Each package contains:

- Electron diagnostics UI;
- Node supervisor and local-media protocol runtime;
- hermetic Python environment;
- pinned RVM TorchScript model matching `models/manifest.json`;
- complete GPL-3.0 text, notices and corresponding source offer.

Run `npm run release:verify` against the assembled package before signing. The
runtime must not invoke pip, download weights or contact a cloud service.
