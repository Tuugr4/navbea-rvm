const crypto = require("node:crypto");
const fs = require("node:fs");
const fsp = require("node:fs/promises");
const http = require("node:http");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { resolveCpuProfile } = require("./cpu-profile.cjs");

function loadProtocol() {
  const candidates = [
    process.env.NAVBEA_PROTOCOL_PATH,
    path.resolve(__dirname, "../local-media-protocol/index.cjs"),
  ].filter(Boolean);
  for (const candidate of candidates) try { return require(candidate); } catch { /* next */ }
  throw new Error("@navbea/local-media-protocol could not be located");
}
const { Codec, FrameDecoder, MessageType, encodeFrame, endpointPaths, signCapability, verifyCapability } = loadProtocol();

const VERSION = "1.0.5-local";
const API_VERSION = "1.1";
const MAX_CONTROL_BODY = 64 * 1024;

function writeLatestBounded(subscriber, encoded, onBackpressure = () => undefined) {
  if (subscriber.closed) return;
  if (subscriber.blocked) { subscriber.pending = encoded; onBackpressure(); return; }
  const flush = () => {
    if (subscriber.closed) { subscriber.pending = null; subscriber.blocked = false; return; }
    const pending = subscriber.pending; subscriber.pending = null;
    if (!pending) { subscriber.blocked = false; return; }
    if (subscriber.socket.write(pending)) { subscriber.blocked = false; return; }
    subscriber.blocked = true;
    subscriber.socket.once("drain", flush);
  };
  if (!subscriber.socket.write(encoded)) {
    subscriber.blocked = true;
    subscriber.socket.once("drain", flush);
  }
}

function defaultDataRoot(platform = process.platform, env = process.env) {
  if (env.NAVBEA_RVM_DATA_DIR) return path.resolve(env.NAVBEA_RVM_DATA_DIR);
  if (platform === "win32") return path.join(env.PROGRAMDATA || "C:\\ProgramData", "Navbea", "RVM");
  return "/var/lib/navbea/rvm";
}
function cameraDataRoot(platform, env) {
  if (env.NAVBEA_CAMERA_DATA_DIR) return path.resolve(env.NAVBEA_CAMERA_DATA_DIR);
  if (platform === "win32") return path.join(env.PROGRAMDATA || "C:\\ProgramData", "Navbea", "Camera");
  return "/var/lib/navbea/camera";
}
function json(res, status, payload) { const body = Buffer.from(JSON.stringify(payload)); res.writeHead(status, { "content-type": "application/json", "content-length": body.length, "cache-control": "no-store" }); res.end(body); }
async function readJson(req) { const chunks = []; let length = 0; for await (const chunk of req) { length += chunk.length; if (length > MAX_CONTROL_BODY) throw new Error("Control body too large"); chunks.push(chunk); } return length ? JSON.parse(Buffer.concat(chunks).toString()) : {}; }

class PythonWorker {
  constructor(options) { this.options = options; this.child = null; this.decoder = new FrameDecoder(); this.pending = new Map(); this.onMask = null; this.onExit = null; this.onFailure = null; }
  failPending(error) {
    for (const item of this.pending.values()) item.reject(error);
    this.pending.clear();
  }
  start() {
    const startupId = crypto.randomUUID();
    const args = [this.options.script, "--model", this.options.model, "--device", this.options.device || "auto"];
    args.push("--startup-id", startupId);
    args.push("--downsample-ratio", String(this.options.liveRatio ?? 0.375), "--still-ratio", String(this.options.stillRatio ?? 0.5), "--threads", String(this.options.threads ?? 0));
    this.child = spawn(this.options.python, args, { windowsHide: true, shell: false, stdio: ["pipe", "pipe", "pipe"], env: { ...process.env, PYTHONUNBUFFERED: "1" } });
    this.child.stdout.on("data", chunk => {
      try { for (const frame of this.decoder.push(chunk)) { const key = frame.sequence.toString(); const pending = this.pending.get(key); pending?.resolve(frame); this.pending.delete(key); if (pending && !pending.native) this.onMask?.(frame); } }
      catch (error) { this.failPending(error); this.onFailure?.(error); this.child?.kill(); }
    });
    this.child.stderr.on("data", chunk => { const line = `${new Date().toISOString()} ${chunk.toString()}\n`; if (this.options.logPath) try { fs.appendFileSync(this.options.logPath, line); } catch {} });
    this.child.stdin.on("error", error => { this.failPending(error); this.onFailure?.(error); });
    this.child.stdout.on("error", error => { this.failPending(error); this.onFailure?.(error); });
    this.child.stderr.on("error", error => { this.onFailure?.(error); });
    return new Promise((resolve, reject) => {
      let settled = false;
      let diagnostic = "";
      const timer = setTimeout(() => { settled = true; reject(new Error("RVM model readiness timed out")); this.child?.kill(); }, this.options.startupTimeoutMs || 120_000);
      this.child.stderr.setEncoding("utf8");
      this.child.stderr.on("data", chunk => {
        if (settled) return;
        diagnostic += chunk;
        if (diagnostic.length > 64 * 1024) { diagnostic = ""; return; }
        let newline;
        while ((newline = diagnostic.indexOf("\n")) >= 0) {
          const line = diagnostic.slice(0, newline); diagnostic = diagnostic.slice(newline + 1);
          try {
            const event = JSON.parse(line);
            if (event.event === "rvm-ready" && event.protocol === 1 && event.startupId === startupId && event.inferenceVerified === true) {
              settled = true; clearTimeout(timer); resolve();
            }
          } catch { /* Non-protocol Python diagnostics are logged separately. */ }
        }
      });
      this.child.once("error", error => { clearTimeout(timer); this.failPending(error); this.onFailure?.(error); if (!settled) { settled = true; reject(error); } });
      this.child.once("exit", code => {
        clearTimeout(timer);
        const error = new Error(`RVM worker exited (${code})`);
        this.failPending(error);
        this.onExit?.(code);
        if (!settled) { settled = true; reject(error); }
      });
    });
  }
  process(frame) {
    const child = this.child;
    if (!child || child.killed || child.stdin.destroyed || child.stdin.writableEnded) return Promise.reject(new Error("RVM worker is not running"));
    if (this.pending.size >= 2) return Promise.reject(new Error("RVM worker backpressure"));
    const key = frame.sequence.toString();
    return new Promise((resolve, reject) => {
      const fail = error => { const pending = this.pending.get(key); if (!pending) return; this.pending.delete(key); pending.reject(error); };
      const timeout = setTimeout(() => { const error = new Error("RVM inference timed out"); fail(error); this.onFailure?.(error); child.kill(); }, frame.flags & 1 ? 60_000 : 10_000);
      const settle = callback => value => { clearTimeout(timeout); callback(value); };
      this.pending.set(key, { resolve: settle(resolve), reject: settle(reject), native: Boolean(frame.flags & 1), started: process.hrtime.bigint() });
      const encoded = encodeFrame({ ...frame, codec: Codec.MJPEG, type: MessageType.FRAME });
      if (this.options.debugDir && this.pending.size <= 2) try { fs.mkdirSync(this.options.debugDir, { recursive: true }); fs.writeFileSync(path.join(this.options.debugDir, `input-${key}.nvf1`), encoded); } catch {}
      try { child.stdin.write(encoded, error => { if (error) fail(error); }); }
      catch (error) { fail(error); }
    });
  }
  stop() { this.onExit = null; this.onFailure = null; if (this.child && !this.child.killed) this.child.kill(); this.child = null; this.failPending(new Error("RVM worker stopped")); }
}

class RvmService {
  constructor(options = {}) {
    this.platform = options.platform || process.platform; this.env = options.env || process.env;
    this.paths = options.paths || endpointPaths("rvm", this.platform, this.env); this.cameraPaths = options.cameraPaths || endpointPaths("camera", this.platform, this.env);
    this.dataRoot = options.dataRoot || defaultDataRoot(this.platform, this.env); this.cameraDataRoot = options.cameraDataRoot || cameraDataRoot(this.platform, this.env);
    this.fixture = options.fixture === true || this.env.NAVBEA_RVM_FIXTURE === "1";
    this.clientSecret = options.clientSecret || null; this.capabilitySecret = options.capabilitySecret || null; this.cameraClientToken = options.cameraClientToken || this.env.NAVBEA_CAMERA_CLIENT_TOKEN || null;
    this.controlServer = null; this.streamServer = null; this.worker = null; this.cameraSocket = null;
    this.sessions = new Map(); this.subscribers = new Set(); this.artifacts = new Map(); this.startedAt = Date.now();
    this.status = "starting"; this.errorCode = null; this.error = null; this.sourceSession = null;
    this.metrics = { frames: 0, dropped: 0, inferenceMs: null, maskFps: 0, device: this.env.RVM_DEVICE || "auto" }; this.stopping = false; this.workerRestartTimer = null; this.stillCaptureBusy = false;
    this.sequenceTimes = new Map(); this.latestSourceFrame = null;
    this.workerRestarts = 0; this.ownedEndpoints = new Set();
  }
  async ensureStorage() {
    await fsp.mkdir(this.dataRoot, { recursive: true, mode: 0o755 });
    const load = async (target, value) => { if (value) return value; const mode = this.platform !== "win32" && path.basename(target) === "client-token.txt" ? 0o644 : 0o600; try { const existing = (await fsp.readFile(target, "utf8")).trim(); if (existing.length < 32) throw new Error("Invalid local service credential"); if (this.platform !== "win32") await fsp.chmod(target, mode); return existing; } catch (error) { if (error.code !== "ENOENT") throw error; } const created = crypto.randomBytes(48).toString("base64url"); await fsp.writeFile(target, `${created}\n`, { mode, flag: "wx" }); return created; };
    this.clientSecret = await load(path.join(this.dataRoot, "client-token.txt"), this.clientSecret);
    this.capabilitySecret = await load(path.join(this.dataRoot, "capability-secret.txt"), this.capabilitySecret);
    const deadline = Date.now() + Number(this.env.NAVBEA_CAMERA_STARTUP_WAIT_MS || 30_000);
    while (!this.cameraClientToken) {
      try { this.cameraClientToken = (await fsp.readFile(path.join(this.cameraDataRoot, "client-token.txt"), "utf8")).trim(); }
      catch (error) { if (error.code !== "ENOENT" || Date.now() >= deadline || this.stopping) throw error; await new Promise(resolve => setTimeout(resolve, 250)); }
      if (this.cameraClientToken && this.cameraClientToken.length < 32) throw new Error("Invalid Camera client credential");
    }
  }
  workerOptions() {
    const packagedRoot = this.env.NAVBEA_RVM_ROOT || path.resolve(__dirname, "..");
    const first = values => values.find(value => value && fs.existsSync(value)) || values.filter(Boolean)[0];
    const python = this.env.RVM_PYTHON_PATH || first(this.platform === "win32" ? [path.join(packagedRoot, "runtime", "python.exe"), path.join(packagedRoot, "runtime", "Scripts", "python.exe"), path.join(packagedRoot, ".runtime-build", "Scripts", "python.exe"), path.join(packagedRoot, ".venv", "Scripts", "python.exe")] : [path.join(packagedRoot, "runtime", "bin", "python"), path.join(packagedRoot, ".runtime-build", "bin", "python"), path.join(packagedRoot, ".venv", "bin", "python")]);
    const model = this.env.RVM_MODEL_PATH || first([path.join(packagedRoot, "models", "rvm_mobilenetv3_fp32.onnx"), path.join(packagedRoot, "rvm_mobilenetv3_fp32.onnx"), path.join(packagedRoot, "models", "rvm_mobilenetv3_fp32.torchscript"), path.join(packagedRoot, "rvm_mobilenetv3_fp32.torchscript")]);
    return { python, script: path.join(packagedRoot, "worker", "rvm_worker.py"), model, device: this.env.RVM_DEVICE || "cpu", liveRatio: Number(this.env.RVM_LIVE_RATIO || 0.375), stillRatio: Number(this.env.RVM_STILL_RATIO || 0.5), threads: this.env.RVM_CPU_THREADS || "auto", logPath: path.join(this.dataRoot, "worker.log"), debugDir: this.env.RVM_DEBUG_FRAME_DIR || "" };
  }
  async startWorker() {
    if (this.stopping) return;
    if (this.fixture) { this.status = "ready"; this.metrics.device = "fixture"; return; }
    const options = this.workerOptions();
    for (const target of [options.python, options.script, options.model]) await fsp.access(target);
    this.status = "starting";
    this.metrics.threadMode = "calibrating";
    const profile = await resolveCpuProfile(options, this.dataRoot);
    if (this.stopping) return;
    options.threads = profile.threads;
    this.metrics.threads = profile.threads; this.metrics.threadMode = profile.mode;
    this.metrics.availableProcessors = profile.cpu.available; this.metrics.physicalCores = profile.cpu.physical;
    this.metrics.device = options.device;
    this.worker = new PythonWorker(options);
    const worker = this.worker;
    worker.onMask = frame => this.acceptMask(frame);
    worker.onFailure = error => { this.status = "unavailable"; this.errorCode = "RVM_WORKER_PIPE_FAILED"; this.error = error.message; };
    const startedAt = Date.now();
    worker.onExit = code => { if (this.worker !== worker) return; if (Date.now() - startedAt > 120_000) this.workerRestarts = 0; this.worker = null; this.scheduleWorkerRestart(new Error(`RVM worker exited (${code})`)); };
    try { await worker.start(); }
    catch (error) { worker.stop(); if (this.worker === worker) this.worker = null; throw error; }
    if (this.stopping) { worker.stop(); this.worker = null; return; }
    this.status = "ready"; this.errorCode = null; this.error = null;
  }
  cameraRequest(method, pathname, body, authorization) {
    const payload = body === undefined ? null : Buffer.from(JSON.stringify(body));
    return new Promise((resolve, reject) => {
      const req = http.request({ socketPath: this.cameraPaths.control, method, path: pathname, headers: { "x-navbea-client-token": this.cameraClientToken, ...(authorization ? { authorization } : {}), ...(payload ? { "content-type": "application/json", "content-length": payload.length } : {}) } }, res => {
        const chunks = []; res.on("data", chunk => chunks.push(chunk)); res.on("end", () => { const data = Buffer.concat(chunks); if ((res.statusCode || 500) >= 400) { let message; try { message = JSON.parse(data.toString()).error; } catch {} reject(new Error(message || `Camera request failed (${res.statusCode})`)); } else resolve({ headers: res.headers, data }); });
      }); req.on("error", reject); if (payload) req.write(payload); req.end();
    });
  }
  async ensureCameraSource() {
    if (this.cameraSocket && !this.cameraSocket.destroyed) return;
    const healthResponse = await this.cameraRequest("GET", "/v1/health");
    const cameraHealth = JSON.parse(healthResponse.data.toString());
    if (cameraHealth.status !== "ready" || String(cameraHealth.apiVersion || "").split(".")[0] !== "1") throw new Error(`Camera Service is incompatible or unavailable (${cameraHealth.apiVersion || "unknown"})`);
    const response = await this.cameraRequest("POST", "/v1/sessions", { profile: "matting-720p24" });
    this.sourceSession = JSON.parse(response.data.toString());
    const decoder = new FrameDecoder(); let acknowledged = false; let preamble = Buffer.alloc(0);
    const socket = net.createConnection(this.cameraPaths.stream);
    this.cameraSocket = socket;
    socket.once("connect", () => { if (!socket.destroyed) socket.write(`${JSON.stringify({ sessionId: this.sourceSession.id, capability: this.sourceSession.capability })}\n`, error => { if (error && this.cameraSocket === socket) { this.error = error.message; this.status = "unavailable"; this.errorCode = "CAMERA_STREAM_FAILED"; } }); });
    socket.on("data", chunk => {
      if (!acknowledged) { preamble = Buffer.concat([preamble, chunk]); const end = preamble.indexOf(10); if (end < 0) return; acknowledged = true; chunk = preamble.subarray(end + 1); }
      if (!chunk.length) return;
      for (const frame of decoder.push(chunk)) void this.processSourceFrame(frame);
    });
    socket.on("close", () => { if (this.cameraSocket !== socket) return; this.cameraSocket = null; this.sourceSession = null; this.status = "unavailable"; this.errorCode = "CAMERA_STREAM_LOST"; });
    socket.on("error", error => { if (this.cameraSocket !== socket) return; this.error = error.message; this.status = "unavailable"; this.errorCode = "CAMERA_STREAM_FAILED"; });
  }
  async processSourceFrame(frame) {
    // Admit only one live inference. A second queued frame adds a full CPU
    // inference interval to mask latency without increasing throughput.
    if (this.stillCaptureBusy) { this.metrics.dropped += 1; return; }
    if (this.sequenceTimes.size >= 1) { if (this.latestSourceFrame) this.metrics.dropped += 1; this.latestSourceFrame = frame; return; }
    const started = process.hrtime.bigint(); this.sequenceTimes.set(frame.sequence.toString(), started);
    if (this.fixture) {
      const width = 1280, height = 720;
      this.acceptMask({ ...frame, width, height, codec: Codec.GRAY8, payload: Buffer.alloc(width * height, 255) });
      return;
    }
    try { if (!this.worker) throw new Error("RVM worker is not running"); await this.worker.process(frame); } catch (error) { this.sequenceTimes.delete(frame.sequence.toString()); this.metrics.dropped += 1; this.error = error.message; if (!this.worker || /not running|EPIPE|EOF|closed/i.test(error.message)) { this.status = "unavailable"; this.errorCode = "RVM_WORKER_PIPE_FAILED"; } }
    finally { const latest = this.latestSourceFrame; this.latestSourceFrame = null; if (latest && !this.stillCaptureBusy && !this.stopping && this.cameraSocket) void this.processSourceFrame(latest); }
  }
  acceptMask(frame) {
    const key = frame.sequence.toString(); const started = this.sequenceTimes.get(key); this.sequenceTimes.delete(key);
    if (started) this.metrics.inferenceMs = Number(process.hrtime.bigint() - started) / 1e6;
    this.metrics.frames += 1; this.status = "ready"; this.errorCode = null; this.error = null;
    const now = Date.now(); this.metrics.windowStarted ||= now; this.metrics.windowFrames = (this.metrics.windowFrames || 0) + 1;
    if (now - this.metrics.windowStarted >= 1000) { this.metrics.maskFps = Math.round(this.metrics.windowFrames * 1000 / (now - this.metrics.windowStarted)); this.metrics.windowStarted = now; this.metrics.windowFrames = 0; }
    for (const subscriber of this.subscribers) this.sendMask(subscriber, frame);
  }
  sendMask(subscriber, frame) {
    if (!subscriber.authorized || subscriber.closed) return;
    const encoded = encodeFrame({ type: MessageType.FRAME, codec: Codec.GRAY8, streamId: subscriber.streamId, sequence: frame.sequence, monotonicNs: frame.monotonicNs, width: frame.width, height: frame.height, payload: frame.payload });
    writeLatestBounded(subscriber, encoded, () => { this.metrics.dropped += 1; });
  }
  health() { return { status: this.status, service: "rvm", version: VERSION, apiVersion: API_VERSION, uptimeSeconds: Math.round((Date.now() - this.startedAt) / 1000), errorCode: this.errorCode, error: this.error, sourceSession: this.sourceSession?.id || null, metrics: { frames: this.metrics.frames, dropped: this.metrics.dropped, maskFps: this.metrics.maskFps, inferenceMs: this.metrics.inferenceMs === null ? null : Number(this.metrics.inferenceMs.toFixed(2)), device: this.metrics.device, threads: this.metrics.threads ?? null, threadMode: this.metrics.threadMode ?? null, availableProcessors: this.metrics.availableProcessors ?? null, physicalCores: this.metrics.physicalCores ?? null } }; }
  authorized(req) { const a = Buffer.from(String(req.headers["x-navbea-client-token"] || "")); const b = Buffer.from(String(this.clientSecret)); return a.length === b.length && a.length > 0 && crypto.timingSafeEqual(a, b); }
  async createSession() {
    if (this.status !== "ready") throw new Error("RVM is not ready"); await this.ensureCameraSource();
    const id = crypto.randomUUID(), maskStreamId = crypto.randomUUID(), exp = Math.floor(Date.now() / 1000) + 120;
    const capability = signCapability({ aud: "rvm", scope: "stream:read", session: id, streamId: maskStreamId, exp }, this.capabilitySecret);
    const item = { id, sourceStreamId: this.sourceSession.streamId, maskStreamId, maskEndpoint: this.paths.stream, capability, expiresAt: new Date(exp * 1000).toISOString() };
    this.sessions.set(id, item); setTimeout(() => this.sessions.delete(id), 125_000).unref?.(); return item;
  }
  async resetPipeline() {
    const cameraSocket = this.cameraSocket; this.cameraSocket = null; this.sourceSession = null; cameraSocket?.destroy(); this.sequenceTimes.clear(); this.latestSourceFrame = null;
    if (!this.fixture) { this.worker?.stop(); this.worker = null; await this.startWorker(); }
  }
  async stillMatte(cameraArtifact) {
    const response = await this.cameraRequest("GET", `/v1/artifacts/${cameraArtifact.id}`, undefined, `Bearer ${cameraArtifact.rvmReadCapability}`);
    const sequence = BigInt(Date.now()); const streamId = crypto.randomUUID();
    const source = { type: MessageType.FRAME, flags: 3, codec: Codec.JPEG, streamId, sequence, monotonicNs: process.hrtime.bigint(), width: cameraArtifact.width || 0, height: cameraArtifact.height || 0, payload: response.data };
    let mask;
    this.stillCaptureBusy = true;
    this.latestSourceFrame = null;
    try {
      if (this.fixture) mask = { ...source, width: cameraArtifact.width || 1920, height: cameraArtifact.height || 1080, payload: Buffer.alloc((cameraArtifact.width || 1920) * (cameraArtifact.height || 1080), 255) };
      else {
        const deadline = Date.now() + 5000;
        while (this.worker.pending.size && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 10));
        if (this.worker.pending.size) throw new Error("RVM live queue did not drain for native capture");
        mask = await this.worker.process(source);
      }
    } finally { this.stillCaptureBusy = false; }
    let foregroundArtifact;
    if (mask.codec === 5) {
      const maskLength = mask.payload.readUInt32BE(0);
      if (maskLength !== mask.width * mask.height || mask.payload.length <= 4 + maskLength) throw new Error("Invalid native foreground response");
      const payload = mask.payload.subarray(4 + maskLength);
      const id = crypto.randomUUID(), exp = Math.floor(Date.now() / 1000) + 120;
      foregroundArtifact = { id, width: mask.width, height: mask.height, byteLength: payload.length, mimeType: "image/jpeg", expiresAt: new Date(exp * 1000).toISOString(), readCapability: signCapability({ aud: "rvm", scope: "artifact:read", artifact: id, exp }, this.capabilitySecret) };
      this.artifacts.set(id, { ...foregroundArtifact, payload });
      setTimeout(() => this.artifacts.delete(id), 125_000).unref?.();
      mask = { ...mask, codec: Codec.GRAY8, payload: mask.payload.subarray(4, 4 + maskLength) };
    }
    if (mask.payload.length !== mask.width * mask.height) throw new Error("Invalid native mask length");
    const id = crypto.randomUUID(), exp = Math.floor(Date.now() / 1000) + 120;
    const artifact = { id, payload: mask.payload, width: mask.width, height: mask.height, byteLength: mask.payload.length, mimeType: "application/x-navbea-gray8", expiresAt: new Date(exp * 1000).toISOString() };
    artifact.readCapability = signCapability({ aud: "rvm", scope: "artifact:read", artifact: id, exp }, this.capabilitySecret);
    this.artifacts.set(id, artifact); setTimeout(() => this.artifacts.delete(id), 125_000).unref?.(); const { payload: _payload, ...publicItem } = artifact; return { ...publicItem, foregroundArtifact };
  }
  async handleControl(req, res) {
    const url = new URL(req.url, "http://local");
    try {
      if (req.method === "GET" && url.pathname === "/v1/health") return json(res, 200, this.health());
      if (!this.authorized(req)) return json(res, 401, { error: "Unauthorized" });
      if (req.method === "GET" && url.pathname === "/v1/capabilities") return json(res, 200, { apiVersion: API_VERSION, input: { profile: "matting-720p24", codec: "mjpeg", minimumFps: 24 }, output: { codec: "gray8", sequenceAligned: true, maximumLagMs: 150 }, nativeStill: true });
      if (req.method === "POST" && url.pathname === "/v1/sessions") return json(res, 201, await this.createSession());
      const sessionMatch = url.pathname.match(/^\/v1\/sessions\/([a-f0-9-]+)$/);
      if (req.method === "DELETE" && sessionMatch) { const existed = this.sessions.delete(sessionMatch[1]); if (existed && !this.sessions.size) await this.resetPipeline(); return json(res, 200, { status: existed ? "deleted" : "already-absent" }); }
      if (req.method === "POST" && url.pathname === "/v1/still-mattes") { const body = await readJson(req); return json(res, 201, await this.stillMatte(body.cameraArtifact)); }
      const match = url.pathname.match(/^\/v1\/artifacts\/([a-f0-9-]+)$/);
      if (req.method === "GET" && match) {
        const item = this.artifacts.get(match[1]); if (!item) return json(res, 404, { error: "Artifact not found" });
        const claims = verifyCapability(String(req.headers.authorization || "").replace(/^Bearer\s+/i, ""), this.capabilitySecret, { audience: "rvm", scope: "artifact:read" });
        if (claims.artifact !== item.id) return json(res, 401, { error: "Artifact mismatch" }); this.artifacts.delete(item.id);
        res.writeHead(200, { "content-type": item.mimeType, "content-length": item.payload.length, "x-image-width": item.width, "x-image-height": item.height, "cache-control": "no-store" }); return res.end(item.payload);
      }
      return json(res, 404, { error: "Not found" });
    } catch (error) { return json(res, 503, { error: error.message, code: "RVM_REQUEST_FAILED" }); }
  }
  handleStream(socket) {
    const subscriber = { socket, authorized: false, closed: false, blocked: false, pending: null, streamId: null }; let handshake = Buffer.alloc(0);
    socket.on("data", chunk => { if (subscriber.authorized) return; handshake = Buffer.concat([handshake, chunk]); if (handshake.length > 8192) return socket.destroy(); const end = handshake.indexOf(10); if (end < 0) return; try { const message = JSON.parse(handshake.subarray(0, end).toString()); const claims = verifyCapability(message.capability, this.capabilitySecret, { audience: "rvm", scope: "stream:read", session: message.sessionId }); const item = this.sessions.get(message.sessionId); if (!item || item.maskStreamId !== claims.streamId) throw new Error("Unknown RVM session"); subscriber.authorized = true; subscriber.streamId = item.maskStreamId; this.subscribers.add(subscriber); socket.write('{"status":"ready"}\n'); } catch { socket.destroy(); } });
    socket.on("close", () => { subscriber.closed = true; this.subscribers.delete(subscriber); }); socket.on("error", () => { subscriber.closed = true; this.subscribers.delete(subscriber); });
  }
  async listen(server, socketPath) {
    if (this.platform !== "win32") {
      await fsp.mkdir(path.dirname(socketPath), { recursive: true, mode: 0o755 });
      const alive = await new Promise(resolve => { const socket = net.createConnection(socketPath); socket.setTimeout(1000, () => { socket.destroy(); resolve(true); }); socket.once("connect", () => { socket.destroy(); resolve(true); }); socket.once("error", error => resolve(!["ENOENT","ECONNREFUSED"].includes(error.code))); });
      if (alive) throw new Error("RVM endpoint is already active");
      await fsp.unlink(socketPath).catch(error => { if (error.code !== "ENOENT") throw error; });
    }
    const shared = this.env.NAVBEA_PIPE_READABLE_ALL === "1";
    await new Promise((resolve, reject) => { server.once("error", reject); server.listen(this.platform === "win32" ? { path: socketPath, readableAll: shared, writableAll: shared } : socketPath, resolve); });
    this.ownedEndpoints.add(socketPath);
    if (this.platform !== "win32") await fsp.chmod(socketPath, 0o666);
  }
  scheduleWorkerRestart(error) {
    this.status = "unavailable"; this.errorCode = "RVM_RUNTIME_UNAVAILABLE"; this.error = error.message;
    if (this.stopping || this.workerRestartTimer || this.workerRestarts >= 5) return;
    const delay = Math.min(30_000, 1000 * 2 ** this.workerRestarts++);
    this.workerRestartTimer = setTimeout(() => { this.workerRestartTimer = null; void this.startWorker().catch(error => this.scheduleWorkerRestart(error)); }, delay);
  }
  async start() {
    try {
      await this.ensureStorage();
      this.controlServer = http.createServer((req, res) => void this.handleControl(req, res)); this.streamServer = net.createServer(socket => this.handleStream(socket));
      await this.listen(this.controlServer, this.paths.control); await this.listen(this.streamServer, this.paths.stream);
      await this.startWorker().catch(error => this.scheduleWorkerRestart(error));
      return this;
    } catch (error) { await this.stop(); throw error; }
  }
  async stop() {
    this.stopping = true; if (this.workerRestartTimer) clearTimeout(this.workerRestartTimer);
    this.cameraSocket?.destroy(); this.worker?.stop();
    for (const item of this.subscribers) item.socket.destroy();
    await Promise.all([this.controlServer, this.streamServer].filter(server => server?.listening).map(server => new Promise(resolve => server.close(resolve))));
    if (this.platform !== "win32") for (const value of this.ownedEndpoints) await fsp.unlink(value).catch(() => {});
  }
}

module.exports = { RvmService, PythonWorker, defaultDataRoot, VERSION, API_VERSION, writeLatestBounded };
