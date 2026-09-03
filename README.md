# Navbea RVM

Independent GPL-3.0 local Robust Video Matting application. It contains no
cloud or platform credentials and does not own capture hardware. The service
receives a short-lived read-only local-media grant, consumes the versioned
NVF1 binary stream and publishes sequence-aligned `GRAY8` alpha masks.

## Components

- `service/`: Node OS-service supervisor and local control/stream endpoints.
- `worker/`: hermetic Python/TorchScript RVM inference worker.
- `electron/` and `src/`: technician diagnostics UI.
- `models/manifest.json`: release-pinned model metadata and checksum.

Production packages must include the complete GPL-3.0 text, corresponding
source archive, upstream copyright notices and the exact model checksum.

See `SOURCE_OFFER.md` for corresponding-source access, `CHANGES.md` for the
modifications surrounding upstream RVM, and `THIRD_PARTY_NOTICES.md` plus
`SBOM.spdx.json` for the dependency inventory.
