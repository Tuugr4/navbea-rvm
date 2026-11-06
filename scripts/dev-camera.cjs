// Development stand-in for the Navbea Camera Service.
//
// It speaks the same local contract the RVM service uses in production
// (control endpoint + NVF1 stream endpoint) and replays a JPEG as a live
// camera, so the whole RVM flow can be exercised without capture hardware.
// Never install or run it next to a real Camera Service: on Windows both use
// the same named pipes.
//
//   node scripts/dev-camera.cjs [--live frame.jpg] [--still photo.jpg] [--fps 24]
//                               [--live-size 1280x720] [--still-size 1280x720]
//
// Without --live/--still it renders synthetic frames of the given sizes with
// the RVM Python runtime (scripts/make-test-frames.py). NAVBEA_CAMERA_CLIENT_TOKEN must match
// the token the RVM service is started with.
const crypto = require("node:crypto");
const fs = require("node:fs");
const http = require("node:http");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { Codec, MessageType, encodeFrame, endpointPaths } = require("../local-media-protocol/index.cjs");

const MAX_BODY = 64 * 1024;

function jpegSize(buffer) {
  if (buffer.length < 4 || buffer[0] !== 0xff || buffer[1] !== 0xd8) throw new Error("Not a JPEG file");
  let offset = 2;
  while (offset + 9 < buffer.length) {
    if (buffer[offset] !== 0xff) { offset += 1; continue; }
    const marker = buffer[offset + 1];
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) { offset += 2; continue; }
    const length = buffer.readUInt16BE(offset + 2);
    if ([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf].includes(marker)) {
      return { width: buffer.readUInt16BE(offset + 7), height: buffer.readUInt16BE(offset + 5) };
    }
    offset += 2 + length;
  }
  throw new Error("JPEG size marker not found");
}

function findPython() {
  const { RvmService } = require("../service/rvm-service.cjs");
  const python = new RvmService({ env: process.env }).workerOptions().python;
  if (!python || !fs.existsSync(python)) throw new Error(`RVM Python runtime not found (${python || "none"}). Create .venv with worker/requirements-cpu.txt or set RVM_PYTHON_PATH.`);
  return python;
}

/** Renders synthetic live/still frames into a new temporary folder. */
function renderTestFrames({ live = "1280x720", still = "1280x720", directory } = {}) {
  const output = directory || fs.mkdtempSync(path.join(os.tmpdir(), "navbea-dev-camera-"));
  execFileSync(findPython(), ["-I", path.join(__dirname, "make-test-frames.py"), output, live, still], { stdio: ["ignore", "ignore", "pipe"], windowsHide: true });
  return { live: path.join(output, "live.jpg"), still: path.join(output, "still.jpg"), directory: output };
}

class DevCamera {
  constructor({ token, livePath, stillPath, fps = 24, platform = process.platform, env = process.env, log = () => undefined }) {
    if (!token || token.length < 32) throw new Error("NAVBEA_CAMERA_CLIENT_TOKEN must be at least 32 characters");
    if (!Number.isFinite(fps) || fps < 1 || fps > 60) throw new Error("fps must be between 1 and 60");
    this.token = token; this.fps = fps; this.platform = platform; this.log = log;
    this.paths = endpointPaths("camera", platform, env);
    this.live = fs.readFileSync(livePath); this.liveSize = jpegSize(this.live);
    this.still = fs.readFileSync(stillPath); this.stillSize = jpegSize(this.still);
    this.sessions = new Map(); this.captures = new Map(); this.sockets = new Set();
    this.sequence = 0n; this.framesSent = 0; this.control = null; this.stream = null;
  }
  state() { return { openSessions: this.sessions.size, streams: this.sockets.size, framesSent: this.framesSent, captures: this.captures.size }; }
  async start() {
    this.control = http.createServer((req, res) => this.handle(req, res).catch(error => this.reply(res, 500, { error: error.message })));
    this.stream = net.createServer(socket => this.handleStream(socket));
    await this.listen(this.control, this.paths.control);
    await this.listen(this.stream, this.paths.stream);
    this.log(`control ${this.paths.control}`); this.log(`stream  ${this.paths.stream}`);
    return this;
  }
  async listen(server, target) {
    if (this.platform !== "win32") {
      fs.mkdirSync(path.dirname(target), { recursive: true });
      try { fs.unlinkSync(target); } catch (error) { if (error.code !== "ENOENT") throw error; }
    }
    await new Promise((resolve, reject) => {
      server.once("error", error => reject(error.code === "EADDRINUSE" ? new Error(`${target} is already in use (is a real Camera Service running?)`) : error));
      server.listen(target, resolve);
    });
  }
  reply(res, status, payload) {
    if (res.headersSent || res.destroyed) return;
    const body = Buffer.from(JSON.stringify(payload));
    res.writeHead(status, { "content-type": "application/json", "content-length": body.length, "cache-control": "no-store" });
    res.end(body);
  }
  async readJson(req) {
    const chunks = []; let length = 0;
    for await (const chunk of req) { length += chunk.length; if (length > MAX_BODY) throw new Error("Body too large"); chunks.push(chunk); }
    return length ? JSON.parse(Buffer.concat(chunks).toString()) : {};
  }
  authorized(req) {
    const given = Buffer.from(String(req.headers["x-navbea-client-token"] || "")), expected = Buffer.from(this.token);
    return given.length === expected.length && crypto.timingSafeEqual(given, expected);
  }
  async handle(req, res) {
    if (!this.authorized(req)) return this.reply(res, 401, { error: "Unauthorized" });
    const url = new URL(req.url, "http://local");
    this.log(`${req.method} ${url.pathname}`);
    if (req.method === "GET" && url.pathname === "/v1/health") {
      return this.reply(res, 200, { status: "ready", service: "camera", apiVersion: "1.0", development: true, nativeStill: this.stillSize, readiness: { cameraValidated: true, canRetryOpenSession: true } });
    }
    if (req.method === "GET" && url.pathname === "/v1/dev/state") return this.reply(res, 200, this.state());
    if (req.method === "POST" && url.pathname === "/v1/sessions") {
      const body = await this.readJson(req);
      const session = { id: crypto.randomUUID(), streamId: crypto.randomUUID(), capability: crypto.randomBytes(32).toString("base64url"), profile: body.profile || null };
      this.sessions.set(session.id, session);
      return this.reply(res, 201, session);
    }
    const sessionMatch = url.pathname.match(/^\/v1\/sessions\/([a-f0-9-]{36})$/);
    if (req.method === "DELETE" && sessionMatch) {
      const existed = this.sessions.delete(sessionMatch[1]);
      for (const socket of this.sockets) if (socket.sessionId === sessionMatch[1]) socket.destroy();
      return this.reply(res, 200, { status: existed ? "deleted" : "already-absent" });
    }
    // Development-only: simulate the guest pressing the shutter. The real
    // Camera Service creates this artifact from its native still capture.
    if (req.method === "POST" && url.pathname === "/v1/dev/captures") {
      const capture = { id: crypto.randomUUID(), captureId: crypto.randomUUID(), ...this.stillSize, rvmReadCapability: crypto.randomBytes(32).toString("base64url") };
      this.captures.set(capture.id, capture);
      setTimeout(() => this.captures.delete(capture.id), 120_000).unref();
      return this.reply(res, 201, capture);
    }
    const artifactMatch = url.pathname.match(/^\/v1\/artifacts\/([a-f0-9-]{36})$/);
    if (req.method === "GET" && artifactMatch) {
      const capture = this.captures.get(artifactMatch[1]);
      if (!capture) return this.reply(res, 404, { error: "Artifact not found" });
      if (String(req.headers.authorization || "") !== `Bearer ${capture.rvmReadCapability}`) return this.reply(res, 401, { error: "Artifact capability mismatch" });
      res.writeHead(200, { "content-type": "image/jpeg", "content-length": this.still.length, "x-image-width": capture.width, "x-image-height": capture.height, "cache-control": "no-store" });
      return res.end(this.still);
    }
    return this.reply(res, 404, { error: "Not found" });
  }
  handleStream(socket) {
    let handshake = Buffer.alloc(0), timer = null;
    const stop = () => { clearInterval(timer); timer = null; this.sockets.delete(socket); };
    socket.on("close", stop); socket.on("error", stop);
    socket.on("data", chunk => {
      if (timer) return;
      handshake = Buffer.concat([handshake, chunk]);
      if (handshake.length > 8192) return socket.destroy();
      const end = handshake.indexOf(10); if (end < 0) return;
      let message; try { message = JSON.parse(handshake.subarray(0, end).toString()); } catch { return socket.destroy(); }
      const session = this.sessions.get(message.sessionId);
      if (!session || session.capability !== message.capability) { this.log("rejected stream handshake"); return socket.destroy(); }
      socket.sessionId = session.id; this.sockets.add(socket);
      socket.write('{"status":"ready"}\n');
      this.log(`stream started for ${session.id}`);
      timer = setInterval(() => {
        if (socket.destroyed || !this.sessions.has(session.id)) return socket.destroy();
        // Latest-frame semantics: skip a frame instead of queueing behind a slow reader.
        if (socket.writableLength > this.live.length * 2) return;
        this.sequence += 1n; this.framesSent += 1;
        socket.write(encodeFrame({ type: MessageType.FRAME, codec: Codec.MJPEG, streamId: session.streamId, sequence: this.sequence, monotonicNs: process.hrtime.bigint(), ...this.liveSize, payload: this.live }));
      }, Math.round(1000 / this.fps));
    });
  }
  async stop() {
    for (const socket of this.sockets) socket.destroy();
    await Promise.all([this.control, this.stream].filter(server => server?.listening).map(server => new Promise(resolve => server.close(resolve))));
    if (this.platform !== "win32") for (const target of [this.paths.control, this.paths.stream]) try { fs.unlinkSync(target); } catch {}
  }
}

function argument(name) { const index = process.argv.indexOf(name); return index > 0 ? process.argv[index + 1] : undefined; }

async function main() {
  const token = process.env.NAVBEA_CAMERA_CLIENT_TOKEN;
  let livePath = argument("--live"), stillPath = argument("--still"), rendered = null;
  if (!livePath || !stillPath) {
    rendered = renderTestFrames({ live: argument("--live-size") || "1280x720", still: argument("--still-size") || "1280x720" });
    livePath ||= rendered.live; stillPath ||= rendered.still;
  }
  const camera = await new DevCamera({ token, livePath, stillPath, fps: Number(argument("--fps") || 24), log: line => console.log(`[dev-camera] ${line}`) }).start();
  const shutdown = async () => { await camera.stop(); if (rendered) fs.rmSync(rendered.directory, { recursive: true, force: true }); process.exit(0); };
  for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, shutdown);
  if (process.send) process.send({ ready: true });
}

if (require.main === module) main().catch(error => { console.error(`[dev-camera] ${error.message}`); process.exit(1); });

module.exports = { DevCamera, jpegSize, renderTestFrames };
