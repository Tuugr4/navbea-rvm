// End-to-end check of the live background-removal flow, without hardware.
//
// Starts a development camera (scripts/dev-camera.cjs) and the real RVM
// service in a throwaway data folder, then drives the service the way the
// kiosk does: open session -> read live masks -> still matte while live ->
// still matte right after the live stream closed -> release everything.
//
//   npm run flow:check              real model (models/rvm_mobilenetv3_fp32.onnx)
//   npm run flow:check -- --fixture service fixture mode, no model needed
//
// Options: --frames N (live masks to read, default 12), --still-size WxH
// (default 1280x720), --keep (keep the temporary folder and logs; a failed run
// always keeps them).
const crypto = require("node:crypto");
const fs = require("node:fs");
const http = require("node:http");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { Codec, FrameDecoder, MessageType, endpointPaths } = require("../local-media-protocol/index.cjs");
const { DevCamera, renderTestFrames } = require("./dev-camera.cjs");

const root = path.resolve(__dirname, "..");
const args = process.argv.slice(2);
const flag = name => args.includes(name);
const option = (name, fallback) => { const index = args.indexOf(name); return index >= 0 && args[index + 1] ? args[index + 1] : fallback; };
const fixture = flag("--fixture"), keep = flag("--keep");
const frameCount = Number(option("--frames", "12"));
const stillSize = option("--still-size", "1280x720");
const [stillW, stillH] = stillSize.split("x").map(Number);
const usageError = !Number.isInteger(frameCount) || frameCount < 2 || frameCount > 500 ? "--frames must be a whole number between 2 and 500"
  : !/^\d+x\d+$/.test(stillSize) || stillW < 64 || stillH < 64 || stillW > 8192 || stillH > 8192 ? "--still-size must look like 1920x1080 (64-8192 per side)" : null;
if (usageError) { console.error(usageError); process.exit(2); }

const results = [];
function check(name, condition, detail = "") {
  results.push({ name, ok: Boolean(condition) });
  console.log(`${condition ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
  if (!condition) throw new Error(`Check failed: ${name}`);
}
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function request(endpoint, token, method, pathname, { body, authorization, timeout = 180_000 } = {}) {
  const payload = body === undefined ? null : Buffer.from(JSON.stringify(body));
  return new Promise((resolve, reject) => {
    const req = http.request({ socketPath: endpoint, method, path: pathname, timeout, headers: {
      ...(token ? { "x-navbea-client-token": token } : {}), ...(authorization ? { authorization } : {}),
      ...(payload ? { "content-type": "application/json", "content-length": payload.length } : {}),
    } }, res => {
      const chunks = []; res.on("data", chunk => chunks.push(chunk));
      res.on("end", () => { const data = Buffer.concat(chunks); let json = null; if (/json/.test(res.headers["content-type"] || "")) try { json = JSON.parse(data.toString()); } catch {} resolve({ status: res.statusCode, headers: res.headers, data, json }); });
      res.on("error", reject);
    });
    req.on("timeout", () => req.destroy(new Error(`${method} ${pathname} timed out`)));
    req.on("error", reject); if (payload) req.write(payload); req.end();
  });
}

/** Opens the mask stream; resolves with an object to read masks from and close it. */
function openMaskStream(endpoint, sessionId, capability) {
  return new Promise((resolve, reject) => {
    const socket = net.createConnection(endpoint), decoder = new FrameDecoder();
    const masks = [], waiters = []; let ready = false, preamble = Buffer.alloc(0), failure = null;
    const notify = () => { for (const waiter of waiters.splice(0)) waiter(); };
    const timer = setTimeout(() => socket.destroy(new Error("Mask stream handshake timed out")), 15_000);
    socket.on("connect", () => socket.write(`${JSON.stringify({ sessionId, capability })}\n`));
    socket.on("data", chunk => {
      try {
        if (!ready) {
          preamble = Buffer.concat([preamble, chunk]); const end = preamble.indexOf(10); if (end < 0) return;
          if (JSON.parse(preamble.subarray(0, end)).status !== "ready") throw new Error("Unexpected mask handshake");
          ready = true; clearTimeout(timer); chunk = preamble.subarray(end + 1);
          resolve({
            masks,
            async take(count, timeoutMs) {
              const deadline = Date.now() + timeoutMs;
              while (masks.length < count && !failure && !socket.destroyed && Date.now() < deadline) await new Promise(next => { waiters.push(next); setTimeout(next, 250); });
              if (failure) throw failure;
              return masks.slice(0, count);
            },
            close() { socket.destroy(); return new Promise(done => socket.destroyed ? done() : socket.once("close", done)); },
          });
        }
        for (const frame of decoder.push(chunk)) if (frame.type === MessageType.FRAME) masks.push(frame);
        notify();
      } catch (error) { failure = error; socket.destroy(error); }
    });
    socket.on("error", error => { failure = error; clearTimeout(timer); if (!ready) reject(error); notify(); });
    socket.on("close", () => { clearTimeout(timer); if (!ready) reject(new Error("Mask stream closed during handshake")); notify(); });
  });
}

/** True when the stream socket closes without answering the handshake. */
function rejectsHandshake(endpoint, message) {
  return new Promise(resolve => {
    const socket = net.createConnection(endpoint); let answered = false;
    socket.on("connect", () => socket.write(`${JSON.stringify(message)}\n`));
    socket.on("data", () => { answered = true; socket.destroy(); });
    socket.on("error", () => undefined);
    socket.on("close", () => resolve(!answered));
    setTimeout(() => socket.destroy(), 5_000).unref();
  });
}

function regionMean(mask, width, height, [x0, y0, x1, y1]) {
  let sum = 0, count = 0;
  for (let y = Math.floor(y0 * height); y < Math.floor(y1 * height); y++) for (let x = Math.floor(x0 * width); x < Math.floor(x1 * width); x++) { sum += mask[y * width + x]; count++; }
  return Math.round(sum / count);
}
// The synthetic figure stands in the middle; the outer fifths are background.
function checkMatte(name, mask, width, height) {
  const person = regionMean(mask, width, height, [0.46, 0.65, 0.54, 0.95]);
  const background = Math.max(regionMean(mask, width, height, [0, 0, 0.2, 1]), regionMean(mask, width, height, [0.8, 0, 1, 1]));
  check(`${name}: person kept, background removed`, person >= 128 && background < 128, `person ${person}/255, background ${background}/255`);
}

async function stillMatte(rvm, token, camera) {
  const capture = await request(camera.paths.control, camera.token, "POST", "/v1/dev/captures");
  const started = Date.now();
  const result = await request(rvm.control, token, "POST", "/v1/still-mattes", { body: { cameraArtifact: capture.json } });
  return { capture: capture.json, result, ms: Date.now() - started };
}

async function main() {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "navbea-rvm-flow-"));
  const env = { ...process.env, NAVBEA_RVM_DEV: "1", NAVBEA_RUNTIME_DIR: path.join(temp, "run"), NAVBEA_RVM_DATA_DIR: path.join(temp, "data"), NAVBEA_CAMERA_DATA_DIR: path.join(temp, "camera-data"), NAVBEA_CAMERA_CLIENT_TOKEN: crypto.randomBytes(36).toString("base64url") };
  delete env.NAVBEA_RVM_FIXTURE; if (fixture) env.NAVBEA_RVM_FIXTURE = "1";
  const logFile = path.join(temp, "service.log");
  let camera = null, service = null, exitInfo = null, passed = false;
  try {
    console.log(`Mode: ${fixture ? "fixture (no model)" : "real model"} · temp ${temp}`);
    if (!fixture) {
      const manifest = require("../models/manifest.json");
      const model = process.env.RVM_MODEL_PATH || path.join(root, "models", manifest.model);
      if (!fs.existsSync(model)) throw new Error(`Model not found: ${model}\nDownload ${manifest.model} from ${manifest.upstream} into models/, or run with --fixture.`);
    }
    // On Windows the endpoints are fixed pipe names shared with an installed
    // service; never run the check against a production RVM by accident.
    const rvm = endpointPaths("rvm", process.platform, env);
    if (await request(rvm.control, null, "GET", "/v1/health", { timeout: 2_000 }).then(() => true, () => false)) throw new Error(`An RVM service is already running on ${rvm.control}. Stop it before running the flow check.`);
    const frames = renderTestFrames({ live: "1280x720", still: stillSize, directory: path.join(temp, "frames") });
    camera = await new DevCamera({ token: env.NAVBEA_CAMERA_CLIENT_TOKEN, livePath: frames.live, stillPath: frames.still, env }).start();

    service = spawn(process.execPath, [path.join(root, "service", "main.cjs")], { cwd: root, env, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    const log = fs.createWriteStream(logFile); service.stdout.pipe(log); service.stderr.pipe(log);
    service.once("exit", (code, signal) => { exitInfo = { code, signal }; });
    const tokenFile = path.join(env.NAVBEA_RVM_DATA_DIR, "client-token.txt");
    let token = null, health = null; const bootDeadline = Date.now() + 240_000;
    while (Date.now() < bootDeadline && !exitInfo) {
      try { token ||= fs.readFileSync(tokenFile, "utf8").trim(); health = (await request(rvm.control, null, "GET", "/v1/health", { timeout: 5_000 })).json; if (health?.status === "ready") break; } catch {}
      await sleep(500);
    }
    if (exitInfo) throw new Error(`RVM service exited during startup (${JSON.stringify(exitInfo)}); see ${logFile}`);
    check("service starts and reports ready", health?.status === "ready", health ? `worker ${health.workerState}, ${health.metrics?.device}` : "no health");

    const unauthorized = await request(rvm.control, "wrong-token", "GET", "/v1/capabilities");
    check("control API rejects a wrong client token", unauthorized.status === 401);
    const capabilities = await request(rvm.control, token, "GET", "/v1/capabilities");
    check("capabilities advertise sequence-aligned GRAY8 masks", capabilities.status === 200 && capabilities.json.output?.codec === "gray8" && capabilities.json.output?.sequenceAligned === true);

    let started = Date.now();
    const opened = await request(rvm.control, token, "POST", "/v1/sessions", { body: {} });
    const session = opened.json;
    check("session opens", opened.status === 201 && session?.id && session.capability && session.maskEndpoint, `${Date.now() - started} ms, ${opened.json?.error || "ok"}`);
    check("camera source session opened by RVM", camera.state().openSessions === 1);

    check("mask stream rejects a forged capability", await rejectsHandshake(session.maskEndpoint, { sessionId: session.id, capability: "forged" }));
    started = Date.now();
    const stream = await openMaskStream(session.maskEndpoint, session.id, session.capability);
    const masks = await stream.take(frameCount, 120_000);
    const seconds = (Date.now() - started) / 1000;
    check(`receives ${frameCount} live masks`, masks.length === frameCount, `${(masks.length / seconds).toFixed(1)} masks/s on this machine`);
    check("masks are GRAY8 with full-size payloads", masks.every(mask => mask.codec === Codec.GRAY8 && mask.width > 0 && mask.payload.length === mask.width * mask.height), `${masks[0].width}x${masks[0].height}`);
    check("masks belong to this session's stream", masks.every(mask => mask.streamId === session.maskStreamId));
    check("mask sequence numbers follow the camera frames", masks.every((mask, index) => index === 0 || mask.sequence > masks[index - 1].sequence), `first ${masks[0].sequence}, last ${masks.at(-1).sequence}`);
    if (!fixture) checkMatte("live mask", masks.at(-1).payload, masks.at(-1).width, masks.at(-1).height);

    const live = await stillMatte(rvm, token, camera);
    check("still matte while live", live.result.status === 201 && live.result.json.width === stillW && live.result.json.height === stillH, `${live.ms} ms, ${live.result.json?.error || `${live.result.json?.width}x${live.result.json?.height}`}`);
    const artifact = await request(rvm.control, token, "GET", `/v1/artifacts/${live.result.json?.id}`, { authorization: `Bearer ${live.result.json.readCapability}` });
    check("still mask artifact is readable", artifact.status === 200 && artifact.data.length === stillW * stillH && artifact.headers["content-type"] === "application/x-navbea-gray8");
    if (!fixture) checkMatte("still mask", artifact.data, stillW, stillH);
    const reread = await request(rvm.control, token, "GET", `/v1/artifacts/${live.result.json.id}`, { authorization: `Bearer ${live.result.json.readCapability}` });
    check("still mask artifact is single-use", reread.status === 404);
    if (live.result.json.foregroundArtifact) {
      const foreground = await request(rvm.control, token, "GET", `/v1/artifacts/${live.result.json.foregroundArtifact.id}`, { authorization: `Bearer ${live.result.json.foregroundArtifact.readCapability}` });
      check("foreground (background removed) JPEG is readable", foreground.status === 200 && foreground.headers["content-type"] === "image/jpeg" && foreground.data[0] === 0xff && foreground.data[1] === 0xd8, `${foreground.data.length} bytes`);
    }

    await stream.close();
    const afterClose = await stillMatte(rvm, token, camera);
    check("still matte right after the live stream closed", afterClose.result.status === 201, `${afterClose.ms} ms, ${afterClose.result.json?.error || "ok"}`);

    const removed = await request(rvm.control, token, "DELETE", `/v1/sessions/${session.id}`);
    check("session release is idempotent", removed.status === 200 && ["deleted", "already-absent"].includes(removed.json?.status), removed.json?.status);
    const releaseDeadline = Date.now() + 10_000;
    while (camera.state().openSessions && Date.now() < releaseDeadline) await sleep(100);
    check("camera source session released", camera.state().openSessions === 0);
    const finalHealth = (await request(rvm.control, null, "GET", "/v1/health")).json;
    check("service healthy after the flow", finalHealth.status === "ready" && !finalHealth.errorCode, `${finalHealth.metrics.frames} masks, ${finalHealth.metrics.dropped} frames skipped`);
    console.log(`\nAll ${results.length} checks passed.`);
    passed = true;
  } finally {
    if (service && !exitInfo) { service.kill(); await new Promise(resolve => { const timer = setTimeout(() => { service.kill("SIGKILL"); resolve(); }, 10_000); service.once("exit", () => { clearTimeout(timer); resolve(); }); }); }
    await camera?.stop();
    if (keep || !passed) console.log(`Kept ${temp} (service log: ${logFile})`); else fs.rmSync(temp, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
  }
}

main().catch(error => { console.error(`\n${error.message}`); if (!results.length || results.at(-1).ok) console.error("(failed before a check completed)"); process.exitCode = 1; });
