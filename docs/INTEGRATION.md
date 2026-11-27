# Integrating with Navbea RVM

This document describes how a kiosk application and a Camera Service connect
to the local live background-removal service, in the order the calls happen
during a guest session. Everything stays on the machine: there is no TCP
listener and no cloud call.

```
Camera Service ──NVF1 MJPEG──▶ RVM service ──NVF1 GRAY8 masks──▶ kiosk
      ▲                          │   ▲                              │
      └──── native still JPEG ◀──┘   └──── POST /v1/still-mattes ◀───┘
```

## Endpoints and credentials

| | Windows | Linux |
|---|---|---|
| RVM control (HTTP) | `\\.\pipe\navbea-rvm-control-v1` | `/run/navbea/rvm-control-v1.sock` |
| RVM mask stream | `\\.\pipe\navbea-rvm-stream-v1` | `/run/navbea/rvm-stream-v1.sock` |
| Client token | `%ProgramData%\Navbea\RVM\client-token.txt` | `/var/lib/navbea/rvm/client-token.txt` |

`NAVBEA_RUNTIME_DIR` and `NAVBEA_RVM_DATA_DIR` move the Linux sockets and the
data folder. Every control request except `GET /v1/health` carries the token
in the `x-navbea-client-token` header; a wrong token returns `401`. Model and
processing-unit changes need the administrator token (`admin-token.txt`,
header `x-navbea-admin-token`) and are meant for the diagnostics window only.
Errors are JSON `{ "error": "...", "code": "..." }`, usually with status `503`.

## Guest flow

1. **Check readiness** – `GET /v1/health`. Use `status` (`starting`, `ready`,
   `unavailable`), `errorCode` and `readiness.canRetryOpenSession`.
   `GET /v1/capabilities` states the input profile (`matting-720p24`) and the
   output contract (`gray8`, sequence aligned).
2. **Open a session** – `POST /v1/sessions` with `{}` (optional
   `resetState: true` to clear the recurrent state for a new guest, optional
   `subjectSelection` policy, see below). The first session opens the Camera
   source on demand and starts the model if it is not already warm. Response
   `201`:

   ```json
   { "id": "…", "sourceStreamId": "…", "maskStreamId": "…",
     "maskEndpoint": "<mask stream endpoint>", "capability": "…", "expiresAt": "…" }
   ```

   Up to 16 sessions may share the pipeline. The capability expires after
   120 s; a session nobody connects to is closed after 125 s.
3. **Read live masks** – connect to `maskEndpoint`, write one line
   `{"sessionId":"…","capability":"…"}\n` and wait for `{"status":"ready"}\n`.
   An invalid capability closes the socket without an answer. NVF1 frames
   follow (see below): `GRAY8` masks whose `sequence` and `monotonicNs` equal
   the camera frame they belong to, so the kiosk composites each mask with
   the matching camera frame. A slow reader only ever gets the latest mask.
4. **Take the photo** – the Camera Service captures a native still and hands
   the kiosk a camera artifact. Send it to RVM:

   ```http
   POST /v1/still-mattes
   { "cameraArtifact": { "id": "…", "captureId": "…", "width": 1920, "height": 1080,
                         "rvmReadCapability": "…" } }
   ```

   RVM downloads the JPEG from the Camera Service and returns `201` with a mask
   artifact (`id`, `width`, `height`, `byteLength`, `readCapability`,
   `timings`) and, when available, `foregroundArtifact` (a JPEG with the
   background removed). Only one photo is processed at a time; a second
   request while one runs returns `RVM native capture is busy`.
5. **Read the results** – `GET /v1/artifacts/<id>` with the client token and
   `Authorization: Bearer <readCapability>`. Masks are raw `GRAY8`
   (`application/x-navbea-gray8`, size in `x-image-width`/`x-image-height`);
   the foreground is `image/jpeg`. Each artifact can be read **once** and
   expires after 120 s.
6. **Finish** – close the mask stream or call `DELETE /v1/sessions/<id>`
   (answers `deleted` or `already-absent`). When the last session ends RVM
   resets the model state and releases the Camera session; the model stays
   warm for five minutes.

Ordering rules:

- Request the photo while the mask stream is open, or after it closed: a
  still requested right after the last stream closed waits for the reset
  instead of failing.
- Do not change models or the processing unit during a session; those calls
  are refused until sessions and photos finish.

## Subject selection (optional)

Passing `subjectSelection` when opening a session removes people who are not
the guests (for example someone walking behind them). It requires
`models/yolov5n-seg.onnx` (`health.subjectSelection` shows availability) and
must be the only session. Policy fields: `mode` (`area`, `tracking`, `all`,
`score`), `zone` (normalized standing area), `maxPeople` (1–12), `stableMs`
(300–3000) and, in `score` mode, `threshold` (50–95). Call
`POST /v1/sessions/<id>/subjects/lock` before the photo and pass the returned
`lockId` with `sessionId` in the still-matte request; `…/unlock` releases it.
Masks are then preceded by a `SUBJECT_STATE` JSON message.

## NVF1 frames

Each message is a 64-byte big-endian header followed by the payload
(implementation: `local-media-protocol/index.cjs`).

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `NVF1` |
| 4 | 2 | version (1) |
| 6 | 2 | type: 1 frame, 2 dropped, 3 end, 4 error, 6 subject state |
| 8 | 4 | flags |
| 12 | 8 | sequence (camera frame number) |
| 20 | 8 | monotonic capture time, ns |
| 28 | 4 | width |
| 32 | 4 | height |
| 36 | 4 | payload length (≤ 128 MiB) |
| 40 | 2 | codec: 1 MJPEG, 2 GRAY8, 3 JPEG, 4 JSON |
| 44 | 16 | stream UUID |
| 60 | 4 | reserved |

## What RVM expects from the Camera Service

RVM is a client of the Camera Service (`navbea-camera-*` endpoints, token in
the camera data folder or `NAVBEA_CAMERA_CLIENT_TOKEN`):

- `GET /v1/health` → `status: "ready"`, `apiVersion: "1.x"`, `nativeStill`
  `{ width, height }`; `operationState` while a capture or test runs.
- `POST /v1/sessions` with `{ "profile": "matting-720p24" }` →
  `{ id, streamId, capability }`; `DELETE /v1/sessions/<id>`.
- Stream endpoint: same one-line handshake, answer `{"status":"ready"}\n`,
  then MJPEG/JPEG frames at about 24 fps. Frames older than 500 ms are skipped.
- `GET /v1/artifacts/<id>` with `Authorization: Bearer <rvmReadCapability>`
  → the native still JPEG.

## Developing without hardware

`scripts/dev-camera.cjs` implements the contract above with synthetic frames
(rendered by the RVM Python runtime) or your own JPEGs.

```sh
npm run camera:dev      # development camera on .runtime/run
npm run service:dev     # RVM in fixture mode against it
npm run flow:check                 # full guest flow with the real model
npm run flow:check -- --fixture    # same flow without a model
```

`flow:check` needs `models/rvm_mobilenetv3_fp32.onnx` (checksum in
`models/manifest.json`) and the Python runtime from `worker/requirements-*.txt`.
On Windows the development camera uses the production pipe names, so stop a
real Camera Service first.
