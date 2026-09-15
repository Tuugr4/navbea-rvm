const buildInfo = require("./build-info.cjs");
const crypto = require("node:crypto");
const fs = require("node:fs");
const fsp = require("node:fs/promises");
const http = require("node:http");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const { spawn, execFile } = require("node:child_process");
const { resolveCpuProfile } = require("./cpu-profile.cjs");
const { ModelManager } = require('./model-manager.cjs');
const { runModelProbe } = require('./model-probe.cjs');
const { LivePreview } = require('./live-preview.cjs');

function loadProtocol() {
  const candidates = [
    process.env.NAVBEA_PROTOCOL_PATH,
    path.resolve(__dirname, "../local-media-protocol/index.cjs"),
  ].filter(Boolean);
  for (const candidate of candidates) try { return require(candidate); } catch { /* next */ }
  throw new Error("@navbea/local-media-protocol could not be located");
}
const { Codec, FrameDecoder, MessageType, encodeFrame, endpointPaths, signCapability, verifyCapability, normalizeSubjectPolicy } = loadProtocol();

const VERSION = "1.1.1-local";
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
    args.push("--still-max-edge", String(this.options.stillMaxEdge ?? 1024));
    args.push("--still-threads", String(this.options.stillThreads ?? 0));
    if (this.options.subjectModel) args.push("--subject-model", this.options.subjectModel);
    this.child = spawn(this.options.python, args, { windowsHide: true, shell: false, stdio: ["pipe", "pipe", "pipe"], env: { ...process.env, PYTHONUNBUFFERED: "1" } });
    this.child.stdout.on("data", chunk => {
      try { for (let frame of this.decoder.push(chunk)) {
        const key = `${frame.streamId}:${frame.sequence}`; const pending = this.pending.get(key); this.pending.delete(key);
        if (frame.type === MessageType.ERROR) { const error = JSON.parse(frame.payload.toString()); pending?.reject(Object.assign(new Error(error.message), { code: error.code })); continue; }
        if (frame.codec === 6) {
          if (frame.payload.length < 8) throw new Error("Invalid subject response");
          const metaLength = frame.payload.readUInt32BE(0), maskLength = frame.payload.readUInt32BE(4);
          if (metaLength > 16384 || maskLength !== frame.width * frame.height || frame.payload.length < 8 + metaLength + maskLength) throw new Error("Invalid subject response lengths");
          const subjectState = JSON.parse(frame.payload.subarray(8, 8 + metaLength).toString());
          const mask = frame.payload.subarray(8 + metaLength, 8 + metaLength + maskLength), foreground = frame.payload.subarray(8 + metaLength + maskLength);
          if (foreground.length) frame = { ...frame, codec: Codec.GRAY8, subjectState, payload: mask, foregroundPayload:foreground };
          else frame = { ...frame, codec: Codec.GRAY8, subjectState, payload: mask };
        }
        pending?.resolve(frame); if (pending && !pending.native) this.onMask?.(frame);
      } }
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
        diagnostic += chunk;
        if (diagnostic.length > 64 * 1024) { diagnostic = ""; return; }
        let newline;
        while ((newline = diagnostic.indexOf("\n")) >= 0) {
          const line = diagnostic.slice(0, newline); diagnostic = diagnostic.slice(newline + 1);
          try {
            const event = JSON.parse(line);
            if(event.event==='provider-change'&&typeof event.provider==='string'){this.actualProvider=event.provider;this.onProvider?.(event.provider,event.reason);}
            if (!settled && event.event === "rvm-ready" && event.protocol === 1 && event.startupId === startupId && event.inferenceVerified === true) {
              this.actualProvider=event.provider;this.onProvider?.(event.provider);
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
    const key = `${frame.streamId}:${frame.sequence}`;
    if (this.pending.has(key)) return Promise.reject(new Error("Duplicate worker request"));
    return new Promise((resolve, reject) => {
      const fail = error => { const pending = this.pending.get(key); if (!pending) return; this.pending.delete(key); pending.reject(error); };
      const timeout = setTimeout(() => { const error = new Error("RVM inference timed out"); fail(error); this.onFailure?.(error); child.kill(); }, frame.flags & 1 ? 60_000 : 10_000);
      const settle = callback => value => { clearTimeout(timeout); callback(value); };
      this.pending.set(key, { resolve: settle(resolve), reject: settle(reject), native: Boolean(frame.flags & 1) || [5, 7].includes(frame.type), started: process.hrtime.bigint() });
      const encoded = encodeFrame({ ...frame, codec: frame.codec ?? Codec.MJPEG, type: frame.type ?? MessageType.FRAME });
      if (this.options.debugDir && !this.debugSampleWritten) try { fs.mkdirSync(this.options.debugDir, { recursive: true }); fs.writeFileSync(path.join(this.options.debugDir, "input-sample.nvf1"), encoded); this.debugSampleWritten = true; } catch {}
      try { child.stdin.write(encoded, error => { if (error) fail(error); }); }
      catch (error) { fail(error); }
    });
  }
  async reset() {
    const deadline = Date.now() + 65_000;
    while (this.pending.size && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 10));
    if (this.pending.size) throw new Error("RVM reset could not drain pending work");
    const result = await this.process({ type: 5, codec: Codec.JSON, streamId: crypto.randomUUID(), sequence: 0n, payload: Buffer.alloc(0) });
    if (result.type !== 5 || result.codec !== Codec.JSON || result.payload.length) throw new Error("RVM reset acknowledgement is invalid");
  }
  async subjectControl(action, policy) {
    const deadline = Date.now() + 65_000;
    while (this.pending.size && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 10));
    if (this.pending.size) throw new Error("Subject control queue did not drain");
    const result = await this.process({ type: 7, codec: Codec.JSON, streamId: crypto.randomUUID(), sequence: 0n, payload: Buffer.from(JSON.stringify({ action, policy })) });
    if (result.type !== 7 || result.codec !== Codec.JSON) throw new Error("Invalid subject control acknowledgement");
    return JSON.parse(result.payload.toString());
  }
  stop() { this.onExit = null; this.onFailure = null; if (this.child && !this.child.killed) this.child.kill(); this.child = null; this.failPending(new Error("RVM worker stopped")); }
}

class RvmService {
  constructor(options = {}) {
    this.platform = options.platform || process.platform; this.env = options.env || process.env;
    this.paths = options.paths || endpointPaths("rvm", this.platform, this.env); this.cameraPaths = options.cameraPaths || endpointPaths("camera", this.platform, this.env);
    this.dataRoot = options.dataRoot || defaultDataRoot(this.platform, this.env); this.cameraDataRoot = options.cameraDataRoot || cameraDataRoot(this.platform, this.env);
    this.fixture = options.fixture === true || this.env.NAVBEA_RVM_FIXTURE === "1";
    this.clientSecret = options.clientSecret || null; this.adminSecret=options.adminSecret||null; this.capabilitySecret = options.capabilitySecret || null; this.cameraClientToken = options.cameraClientToken || this.env.NAVBEA_CAMERA_CLIENT_TOKEN || null;
    this.controlServer = null; this.streamServer = null; this.worker = null; this.cameraSocket = null;
    this.sessions = new Map(); this.subscribers = new Set(); this.artifacts = new Map(); this.startedAt = Date.now();
    this.status = "starting"; this.errorCode = null; this.error = null; this.sourceSession = null;
    this.metrics = { frames: 0, dropped: 0, inferenceMs: null, maskFps: 0, device: this.env.RVM_DEVICE || "auto" }; this.stopping = false; this.workerRestartTimer = null; this.stillCaptureBusy = false;
    this.sequenceTimes = new Map(); this.latestSourceFrame = null;
    this.workerRestarts = 0; this.ownedEndpoints = new Set();
    this.resetPromise = null; this.stillTask = null; this.workerStartPromise = null; this.idleTimer = null;
    this.modelAbort=new AbortController();
    this.preview = new LivePreview(this);
  }
  async ensureStorage() {
    await fsp.mkdir(this.dataRoot, { recursive: true, mode: 0o755 });
    const load = async (target, value) => { if (value) return value; const mode = this.platform !== "win32" && path.basename(target) === "client-token.txt" ? 0o644 : 0o600; try { const existing = (await fsp.readFile(target, "utf8")).trim(); if (existing.length < 32) throw new Error("Invalid local service credential"); if (this.platform !== "win32") await fsp.chmod(target, mode); return existing; } catch (error) { if (error.code !== "ENOENT") throw error; } const created = crypto.randomBytes(48).toString("base64url"); await fsp.writeFile(target, `${created}\n`, { mode, flag: "wx" }); return created; };
    this.clientSecret = await load(path.join(this.dataRoot, "client-token.txt"), this.clientSecret);
    const adminFile=path.join(this.dataRoot,'admin-token.txt');
    this.adminSecret = await load(adminFile,this.adminSecret);
    if(this.platform==='win32'&&!this.fixture&&this.env.NAVBEA_RVM_DEV!=='1')await new Promise((resolve,reject)=>execFile(path.join(this.env.SystemRoot||'C:\\Windows','System32','icacls.exe'),[adminFile,'/inheritance:r','/grant:r','*S-1-5-18:(F)','*S-1-5-32-544:(F)'],{windowsHide:true,timeout:10000},error=>error?reject(error):resolve()));
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
    const selected=this.models?.active;
    const model = selected ? this.models.file(this.models.entry(selected.id)) : this.env.RVM_MODEL_PATH || first([path.join(packagedRoot, "models", "rvm_mobilenetv3_fp32.onnx"), path.join(packagedRoot, "rvm_mobilenetv3_fp32.onnx"), path.join(packagedRoot, "models", "rvm_mobilenetv3_fp32.torchscript"), path.join(packagedRoot, "rvm_mobilenetv3_fp32.torchscript")]);
    return { python, script: path.join(packagedRoot, "worker", "rvm_worker.py"), model, device: selected&&selected.id!==this.models.catalogue.defaultModel?selected.device:this.env.RVM_DEVICE || "cpu", liveRatio: Number(this.env.RVM_LIVE_RATIO || 0.375), stillRatio: Number(this.env.RVM_STILL_RATIO || 0.5), stillMaxEdge: Number(this.env.RVM_STILL_MAX_EDGE || 1024), threads: this.env.RVM_CPU_THREADS || "auto", cpuBudget: this.env.RVM_CPU_BUDGET || "auto", logPath: path.join(this.dataRoot, "worker.log"), debugDir: this.env.RVM_DEBUG_FRAME_DIR || "" };
  }
  ensureWorker() {
    if (this.workerStartPromise) return this.workerStartPromise;
    if (this.worker || this.fixture) return Promise.resolve();
    const work = this.startWorker();
    this.workerStartPromise = work;
    return work.finally(() => { if (this.workerStartPromise === work) this.workerStartPromise = null; });
  }
  async startWorker() {
    if (this.stopping) return;
    if (this.fixture) { this.status = "ready"; this.metrics.device = "fixture"; return; }
    if(this.modelManagerError)throw new Error(this.modelManagerError);
    const options = this.workerOptions();
    const subjectModel = path.resolve(path.dirname(options.script), "../models/yolov5n-seg.onnx");
    if (fs.existsSync(subjectModel)) {
      const manifest = require("../models/subjects.json");
      const hash = crypto.createHash("sha256").update(await fsp.readFile(subjectModel)).digest("hex");
      if (hash !== manifest.sha256) throw new Error("Subject model integrity check failed");
      options.subjectModel = subjectModel;
    }
    this.subjectModelReady = Boolean(options.subjectModel);
    for (const target of [options.python, options.script, options.model]) await fsp.access(target);
    this.status = "starting";
    this.metrics.threadMode = "calibrating";
    const profile = this.workerProfile || await resolveCpuProfile(options, this.dataRoot);
    this.workerProfile = profile;
    if (this.stopping) return;
    options.threads = profile.threads;
    options.stillThreads = Number(this.env.RVM_STILL_CPU_THREADS || profile.threads);
    if (!Number.isInteger(options.stillThreads) || options.stillThreads < 1 || options.stillThreads > profile.budget) throw new Error("RVM_STILL_CPU_THREADS must fit the configured CPU budget");
    this.metrics.stillThreads = options.stillThreads;
    this.metrics.threads = profile.threads; this.metrics.threadMode = profile.mode;
    this.metrics.availableProcessors = profile.cpu.available; this.metrics.physicalCores = profile.cpu.physical;
    this.metrics.device = options.device;
    this.worker = new PythonWorker(options);
    const worker = this.worker;
    worker.onProvider=(provider,reason)=>{if(this.worker!==worker)return;this.metrics.device=provider;this.metrics.providerFallback=reason||null;};
    worker.onMask = frame => this.acceptMask(frame);
    worker.onFailure = error => { this.status = "unavailable"; this.errorCode = "RVM_WORKER_PIPE_FAILED"; this.error = error.message; };
    const startedAt = Date.now();
    worker.onExit = code => { if (this.worker !== worker) return; if (Date.now() - startedAt > 120_000) this.workerRestarts = 0; this.worker = null; this.scheduleWorkerRestart(new Error(`RVM worker exited (${code})`)); };
    try { await worker.start(); }
    catch (error) { worker.stop(); if (this.worker === worker) this.worker = null; throw error; }
    if (this.stopping) { worker.stop(); this.worker = null; return; }
    this.loadedModelId=this.models?.active.id;
    this.status = "ready"; this.errorCode = null; this.error = null;
  }
  cameraRequest(method, pathname, body, authorization) {
    const payload = body === undefined ? null : Buffer.from(JSON.stringify(body));
    return new Promise((resolve, reject) => {
      const req = http.request({ socketPath: this.cameraPaths.control, method, path: pathname, timeout: 15_000, headers: { "x-navbea-client-token": this.cameraClientToken, ...(authorization ? { authorization } : {}), ...(payload ? { "content-type": "application/json", "content-length": payload.length } : {}) } }, res => {
        const chunks = []; let length = 0;
        const limit = pathname.startsWith("/v1/artifacts/") ? 64 * 1024 * 1024 : 1024 * 1024;
        res.on("data", chunk => { length += chunk.length; if (length > limit) req.destroy(new Error("Camera response exceeds its size limit")); else chunks.push(chunk); });
        res.on("error", reject); res.on("aborted", () => reject(new Error("Camera response was interrupted")));
        res.on("end", () => { const data = Buffer.concat(chunks); if ((res.statusCode || 500) >= 400) { let result; try { result = JSON.parse(data.toString()); } catch {} reject(Object.assign(new Error(result?.error || `Camera request failed (${res.statusCode})`), { code: result?.code })); } else resolve({ headers: res.headers, data }); });
      }); req.on("timeout", () => req.destroy(new Error("Camera request timed out"))); req.on("error", reject); if (payload) req.write(payload); req.end();
    });
  }
  async ensureCameraSource() {
    if (this.sourceStartPromise) return this.sourceStartPromise;
    if (this.cameraSocket && !this.cameraSocket.destroyed) return;
    const work = this.openCameraSource();
    this.sourceStartPromise = work;
    try { await work; } finally { if (this.sourceStartPromise === work) this.sourceStartPromise = null; }
  }
  async openCameraSource() {
    const healthResponse = await this.cameraRequest("GET", "/v1/health");
    const cameraHealth = JSON.parse(healthResponse.data.toString());
    if (cameraHealth.status !== "ready" || String(cameraHealth.apiVersion || "").split(".")[0] !== "1") throw new Error(`Camera Service is incompatible or unavailable (${cameraHealth.apiVersion || "unknown"})`);
    const response = await this.cameraRequest("POST", "/v1/sessions", { profile: "matting-720p24" });
    this.sourceSession = JSON.parse(response.data.toString());
    if (this.stopping) { const id = this.sourceSession.id; this.sourceSession = null; await this.cameraRequest("DELETE", `/v1/sessions/${id}`).catch(() => undefined); throw new Error("RVM service is stopping"); }
    const decoder = new FrameDecoder(); let acknowledged = false; let preamble = Buffer.alloc(0);
    const socket = net.createConnection(this.cameraPaths.stream);
    this.cameraSocket = socket;
    socket.once("connect", () => { if (!socket.destroyed) socket.write(`${JSON.stringify({ sessionId: this.sourceSession.id, capability: this.sourceSession.capability })}\n`, error => { if (error && this.cameraSocket === socket) { this.error = error.message; this.status = "unavailable"; this.errorCode = "CAMERA_STREAM_FAILED"; } }); });
    socket.on("data", chunk => {
      try {
      if (!acknowledged) { preamble = Buffer.concat([preamble, chunk]); const end = preamble.indexOf(10); if (end < 0) { if (preamble.length > 8192) throw new Error("Camera handshake is too large"); return; } if (end > 8192 || JSON.parse(preamble.subarray(0, end)).status !== "ready") throw new Error("Invalid Camera stream handshake"); acknowledged = true; chunk = preamble.subarray(end + 1); preamble = Buffer.alloc(0); }
      if (!chunk.length) return;
      for (const frame of decoder.push(chunk)) void this.processSourceFrame(frame);
      } catch (error) { socket.destroy(error); }
    });
    socket.on("close", () => {
      if (this.cameraSocket !== socket) return;
      const source = this.sourceSession;
      this.cameraSocket = null; this.sourceSession = null; this.status = "unavailable"; this.errorCode = "CAMERA_STREAM_LOST";
      if (source) void this.cameraRequest("DELETE", `/v1/sessions/${source.id}`).catch(() => undefined);
    });
    socket.on("error", error => { if (this.cameraSocket !== socket) return; this.error = error.message; this.status = "unavailable"; this.errorCode = "CAMERA_STREAM_FAILED"; });
  }
  async processSourceFrame(frame) {
    if (frame.type !== MessageType.FRAME || ![Codec.MJPEG, Codec.JPEG].includes(frame.codec)) return;
    if (frame.monotonicNs && Number(process.hrtime.bigint() - frame.monotonicNs) / 1e6 > 500) { this.metrics.dropped++; return; }
    // Admit only one live inference. A second queued frame adds a full CPU
    // inference interval to mask latency without increasing throughput.
    if (this.stillCaptureBusy || this.resetPromise || !this.cameraSocket) { this.metrics.dropped += 1; return; }
    if (this.sequenceTimes.size >= 1) { if (this.latestSourceFrame) this.metrics.dropped += 1; this.latestSourceFrame = frame; return; }
    const started = process.hrtime.bigint(); this.sequenceTimes.set(frame.sequence.toString(), started);
    this.preview.offerSource(frame);
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
    if (this.resetPromise || this.stopping) return;
    this.preview.acceptMask(frame);
    if (started) this.metrics.inferenceMs = Number(process.hrtime.bigint() - started) / 1e6;
    this.metrics.frames += 1; this.status = "ready"; this.errorCode = null; this.error = null;
    const now = Date.now(); this.metrics.windowStarted ||= now; this.metrics.windowFrames = (this.metrics.windowFrames || 0) + 1;
    if (now - this.metrics.windowStarted >= 1000) { this.metrics.maskFps = Math.round(this.metrics.windowFrames * 1000 / (now - this.metrics.windowStarted)); this.metrics.windowStarted = now; this.metrics.windowFrames = 0; }
    for (const subscriber of this.subscribers) this.sendMask(subscriber, frame);
  }
  sendMask(subscriber, frame) {
    if (!subscriber.authorized || subscriber.closed) return;
    const mask = encodeFrame({ type: MessageType.FRAME, codec: Codec.GRAY8, streamId: subscriber.streamId, sequence: frame.sequence, monotonicNs: frame.monotonicNs, width: frame.width, height: frame.height, payload: frame.payload });
    const encoded = frame.subjectState ? Buffer.concat([encodeFrame({ type: MessageType.SUBJECT_STATE, codec: Codec.JSON, streamId: subscriber.streamId, sequence: frame.sequence, monotonicNs: frame.monotonicNs, payload: Buffer.from(JSON.stringify(frame.subjectState)) }), mask]) : mask;
    writeLatestBounded(subscriber, encoded, () => { this.metrics.dropped += 1; });
  }
  health() { return { status: this.status, service: "rvm", sourceTimestamps: "camera-monotonic", version: VERSION, build: buildInfo, apiVersion: API_VERSION, uptimeSeconds: Math.round((Date.now() - this.startedAt) / 1000), errorCode: this.errorCode, error: this.error, sourceSession: this.sourceSession?.id || null, workerState: this.worker ? (this.stillCaptureBusy ? "native" : this.resetPromise ? "resetting" : "warm") : this.fixture ? "fixture" : "idle", modelSha256: this.workerProfile?.modelHash || null, metrics: { frames: this.metrics.frames, dropped: this.metrics.dropped, maskFps: this.metrics.maskFps, inferenceMs: this.metrics.inferenceMs === null ? null : Number(this.metrics.inferenceMs.toFixed(2)), device: this.metrics.device, threads: this.metrics.threads ?? null, threadMode: this.metrics.threadMode ?? null, availableProcessors: this.metrics.availableProcessors ?? null, physicalCores: this.metrics.physicalCores ?? null, cpuBudget: this.workerProfile?.budget ?? null, stillThreads: this.metrics.stillThreads ?? null, stillMaxEdge: Number(this.env.RVM_STILL_MAX_EDGE || 1024), stillRatio: Number(this.env.RVM_STILL_RATIO || .5), lastStill: this.metrics.lastStill || null } }; }
  authorized(req,role='client') { const a = Buffer.from(String(req.headers[role==='admin'?'x-navbea-admin-token':"x-navbea-client-token"] || "")); const secret=role==='admin'?this.adminSecret:this.clientSecret; if(!secret)return false;const b = Buffer.from(String(secret)); return a.length === b.length && a.length > 0 && crypto.timingSafeEqual(a, b); }
  modelsStatus(){if(this.modelsInitializing)return{available:false,error:'Modeller kontrol ediliyor.'};if(!this.models)return{available:false,error:'Model yönetimi henüz hazır değil.'};const status=this.models.snapshot();const activeId=this.loadedModelId||status.active.id;return {...status,available:true,active:{...status.active,id:activeId},models:status.models.map(item=>({...item,active:item.id===activeId})),switching:Boolean(this.modelChange)||status.switching,activeSessions:this.sessions.size,actualProvider:this.metrics.device};}
  changeModel(id){return this.lifecycle(async()=>{
    if(!this.models)throw Error('Model yönetimi hazır değil.');
    if(this.preview.active)throw Error('Model değiştirmek için canlı önizlemeyi kapatın.');
    if(this.sessions.size||this.stillTask||this.resetPromise||this.workerStartPromise)throw Error('Model değiştirmek için aktif çekim oturumunun bitmesini bekleyin.');
    const previous={...this.models.active};this.modelChange=true;clearTimeout(this.idleTimer);clearTimeout(this.workerRestartTimer);this.workerRestartTimer=null;
    try{
      const cameraHealth=await this.cameraRequest('GET','/v1/health').catch(()=>null);const native=cameraHealth?.nativeStill;
      if(cameraHealth?.operationState)throw Error('Kamera bir işlem yürütüyor. Model değişimi için kamera testinin veya çekimin bitmesini bekleyin.');
      if(native?.width*native?.height>this.models.limits(id).maxNativePixels)throw Error('Bu model seçili kameranın fotoğraf çözünürlüğü için doğrulanmış sınırı aşıyor. Mevcut modeli kullanın.');
      await this.models.activate(id);if(previous.id===this.models.active.id)return this.modelsStatus();
      this.worker?.stop();this.worker=null;this.workerProfile=null;this.workerRestarts=0;
      try{await this.startWorker();}
      catch(error){await this.models.restore(previous,'Yeni model başlatılamadı; önceki model geri yüklendi. '+error.message);this.worker?.stop();this.worker=null;this.workerProfile=null;await this.startWorker();throw error;}
      return this.modelsStatus();
    }finally{this.modelChange=false;}
  });}
  lifecycle(action) {
    const work = (this.lifecycleTail || Promise.resolve()).catch(() => {}).then(action);
    this.lifecycleTail = work;
    return work;
  }
  createSession(options = {}) { if(this.modelsInitializing)return Promise.reject(new Error('RVM modelleri kontrol ediliyor.'));if(this.modelChange)return Promise.reject(new Error('RVM modeli değiştiriliyor; işlem tamamlandıktan sonra tekrar deneyin.'));return this.lifecycle(() => this.openSession(options)); }
  async openSession(options) {
    if(this.modelChange)throw Error('RVM modeli değiştiriliyor; işlem tamamlandıktan sonra tekrar deneyin.');
    const policy = options.subjectSelection === undefined ? null : normalizeSubjectPolicy(options.subjectSelection);
    if (this.subjectOwner) throw new Error("A subject selection session already owns the pipeline");
    if (policy && this.sessions.size) throw new Error("Close existing RVM sessions before selecting subjects");
    clearTimeout(this.idleTimer);
    if (this.resetPromise) await this.resetPromise;
    await this.ensureWorker();
    if (policy && (!this.subjectModelReady || this.fixture)) throw new Error("Subject instance model is unavailable");
    if (this.stopping) throw new Error("RVM service is stopping");
    if (this.sessions.size >= 16) throw new Error("Too many RVM sessions");
    if (options.resetState && !this.fixture && this.worker) {
      const work = (async () => { if (this.stillTask) await this.stillTask.catch(() => undefined); await this.worker.reset(); this.sequenceTimes.clear(); this.latestSourceFrame = null; })();
      this.resetPromise = work;
      try { await work; } finally { if (this.resetPromise === work) this.resetPromise = null; }
    }
    if (this.status !== "ready") throw new Error("RVM is not ready"); await this.ensureCameraSource();
    const id = crypto.randomUUID(), maskStreamId = crypto.randomUUID(), exp = Math.floor(Date.now() / 1000) + 120;
    const capability = signCapability({ aud: "rvm", scope: "stream:read", session: id, streamId: maskStreamId, exp }, this.capabilitySecret);
    const item = { id, sourceStreamId: this.sourceSession.streamId, maskStreamId, maskEndpoint: this.paths.stream, capability, expiresAt: new Date(exp * 1000).toISOString() };
    this.sessions.set(id, item);
    if (policy) {
      this.subjectOwner = id;
      try { await this.controlSubjects(id, "configure", policy); }
      catch (error) { this.subjectOwner = null; this.sessions.delete(id); await this.resetPipeline(); throw error; }
    }
    setTimeout(() => { if (![...this.subscribers].some(subscriber => subscriber.sessionId === id)) void this.deleteSession(id).catch(error => this.scheduleWorkerRestart(error)); }, 125_000).unref?.();
    return item;
  }
  deleteSession(id) { return this.lifecycle(() => this.closeSession(id)); }
  async controlSubjects(id, action, policy) {
    if (!id || id !== this.subjectOwner || !this.sessions.has(id)) throw new Error("Subject session mismatch");
    if (this.stillTask || this.resetPromise) throw new Error("Subject selection is busy");
    const work = this.worker.subjectControl(action, policy);
    this.resetPromise = work;
    this.latestSourceFrame = null;
    try { const state = await work; this.subjectLock = state.lockId || null; return state; }
    finally { if (this.resetPromise === work) this.resetPromise = null; }
  }
  async closeSession(id) {
    const existed = this.sessions.delete(id);
    if (!existed) return false;
    if (this.subjectOwner === id) { this.subjectOwner = null; this.subjectLock = null; }
    for (const subscriber of this.subscribers) if (subscriber.sessionId === id) subscriber.socket.destroy();
    if (!this.sessions.size) { await this.resetPipeline(); if(this.preview.active&&!this.stopping) { await this.ensureWorker(); await this.ensureCameraSource(); } }
    return true;
  }
  resetPipeline() {
    if (this.resetPromise) return this.resetPromise;
    this.preview.clear();
    const cameraSocket = this.cameraSocket, sourceSession = this.sourceSession;
    this.cameraSocket = null; this.sourceSession = null; cameraSocket?.destroy(); this.latestSourceFrame = null;
    const work = (async () => {
      if (this.stillTask) await this.stillTask.catch(() => undefined);
      if (!this.fixture && this.worker) await this.worker.reset();
      this.sequenceTimes.clear(); this.metrics.maskFps = 0;
      if (sourceSession) await this.cameraRequest("DELETE", `/v1/sessions/${sourceSession.id}`).catch(() => undefined);
      // Drop native allocator memory after a quiet interval, not every person.
      clearTimeout(this.idleTimer);
      this.idleTimer = setTimeout(() => {
        if (this.sessions.size || this.stillTask || this.cameraSocket || this.stopping) return;
        this.worker?.stop(); this.worker = null;
      }, 5 * 60_000);
      this.idleTimer.unref?.();
    })();
    this.resetPromise = work;
    return work.catch(error => {
      this.worker?.stop(); this.worker = null;
      this.scheduleWorkerRestart(error);
      throw error;
    }).finally(() => { if (this.resetPromise === work) this.resetPromise = null; });
  }
  stillMatte(cameraArtifact, selection) {
    if (this.subjectOwner && (selection?.sessionId !== this.subjectOwner || !this.subjectLock || selection?.lockId !== this.subjectLock)) return Promise.reject(new Error("Subject lock is required for native verification"));
    if (this.stillTask || this.resetPromise) return Promise.reject(new Error("RVM native capture is busy"));
    const retainedBytes = [...this.artifacts.values()].reduce((sum, item) => sum + item.payload.length, 0);
    if (this.artifacts.size > 6 || retainedBytes > 192 * 1024 * 1024) return Promise.reject(new Error("Read or expire previous native artifacts before requesting another capture"));
    clearTimeout(this.idleTimer);
    const work = this.performStillMatte(cameraArtifact);
    this.stillTask = work;
    return work.finally(() => { if (this.stillTask === work) this.stillTask = null; });
  }
  async performStillMatte(cameraArtifact) {
    if(this.models&&cameraArtifact.width*cameraArtifact.height>this.models.limits().maxNativePixels)throw Error('Seçili modelin fotoğraf çözünürlüğü sınırı aşıldı. Varsayılan MobileNetV3 FP32 modeline dönün.');
    const startedAt = performance.now();
    this.stillCaptureBusy = true;
    this.latestSourceFrame = null;
    try {
    await this.ensureWorker();
    const response = await this.cameraRequest("GET", `/v1/artifacts/${cameraArtifact.id}`, undefined, `Bearer ${cameraArtifact.rvmReadCapability}`);
    const sourceReadMs = performance.now() - startedAt;
    const sequence = BigInt(Date.now()); const streamId = crypto.randomUUID();
    const source = { type: MessageType.FRAME, flags: 3, codec: Codec.JPEG, streamId, sequence, monotonicNs: process.hrtime.bigint(), width: cameraArtifact.width || 0, height: cameraArtifact.height || 0, payload: response.data };
    let mask;
    let queueMs = 0;
    const queueStartedAt = performance.now();
      if (this.fixture) mask = { ...source, width: cameraArtifact.width || 1920, height: cameraArtifact.height || 1080, payload: Buffer.alloc((cameraArtifact.width || 1920) * (cameraArtifact.height || 1080), 255) };
      else {
        const deadline = Date.now() + 5000;
        while (this.worker.pending.size && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 10));
        if (this.worker.pending.size) throw new Error("RVM live queue did not drain for native capture");
        queueMs = performance.now() - queueStartedAt;
        mask = await this.worker.process(source);
      }
    if (this.subjectOwner && (!mask.subjectState?.verifiedNative || mask.subjectState.lockId !== this.subjectLock)) throw new Error("Native subject verification is missing");
    let foregroundArtifact;
    if (mask.foregroundPayload || mask.codec === 5) {
      const maskLength = mask.foregroundPayload ? mask.payload.length : mask.payload.readUInt32BE(0);
      if (maskLength !== mask.width * mask.height || (!mask.foregroundPayload&&mask.payload.length <= 4 + maskLength)) throw new Error("Invalid native foreground response");
      const payload = mask.foregroundPayload || mask.payload.subarray(4 + maskLength);
      const id = crypto.randomUUID(), exp = Math.floor(Date.now() / 1000) + 120;
      foregroundArtifact = { id, width: mask.width, height: mask.height, byteLength: payload.length, mimeType: "image/jpeg", expiresAt: new Date(exp * 1000).toISOString(), readCapability: signCapability({ aud: "rvm", scope: "artifact:read", artifact: id, exp }, this.capabilitySecret) };
      this.artifacts.set(id, { ...foregroundArtifact, payload });
      setTimeout(() => this.artifacts.delete(id), 125_000).unref?.();
      mask = { ...mask, codec: Codec.GRAY8, payload: mask.foregroundPayload?mask.payload:mask.payload.subarray(4, 4 + maskLength),foregroundPayload:undefined };
    }
    if (mask.payload.length !== mask.width * mask.height) throw new Error("Invalid native mask length");
    const timings = { sourceReadMs: Math.round(sourceReadMs), queueMs: Math.round(queueMs), workerAndDecodeMs: Math.round(performance.now() - queueStartedAt - queueMs), totalMs: Math.round(performance.now() - startedAt), inputBytes: source.payload.length, maskBytes: mask.payload.length, foregroundBytes: foregroundArtifact?.byteLength || 0, serviceRssBytes: process.memoryUsage().rss };
    this.metrics.lastStill = { captureId: cameraArtifact.captureId || cameraArtifact.id, ...timings };
    const id = crypto.randomUUID(), exp = Math.floor(Date.now() / 1000) + 120;
    const artifact = { id, payload: mask.payload, width: mask.width, height: mask.height, byteLength: mask.payload.length, mimeType: "application/x-navbea-gray8", expiresAt: new Date(exp * 1000).toISOString() };
    artifact.readCapability = signCapability({ aud: "rvm", scope: "artifact:read", artifact: id, exp }, this.capabilitySecret);
    this.artifacts.set(id, artifact); setTimeout(() => this.artifacts.delete(id), 125_000).unref?.(); const { payload: _payload, ...publicItem } = artifact; return { ...publicItem, foregroundArtifact, subjectVerification: mask.subjectState, captureId: cameraArtifact.captureId || cameraArtifact.id, timings };
    } finally { this.stillCaptureBusy = false; }
  }
  async handleControl(req, res) {
    const url = new URL(req.url, "http://local");
    try {
      if (req.method === "GET" && url.pathname === "/v1/health") return json(res, 200, { ...this.health(), ...(this.modelChange?{status:'starting',errorCode:'RVM_MODEL_SWITCHING',error:'RVM modeli değiştiriliyor.'}:{}), model:this.models?{id:this.loadedModelId||this.models.active.id,...this.models.limits(this.loadedModelId||this.models.active.id)}:null, subjectSelection: this.subjectModelReady ? { version: 1, modes: ["area", "tracking", "all"], nativeVerification: true, occlusionGuard: true } : null });
      if (!this.authorized(req)&&!this.authorized(req,'admin')) return json(res, 401, { error: "Unauthorized" });
      if(req.method==='POST'&&url.pathname==='/v1/preview/start')return json(res,200,await this.preview.start());
      const previewMatch=url.pathname.match(/^\/v1\/preview\/([a-f0-9-]{36})(\/stop)?$/);
      if(previewMatch&&req.method==='POST'&&previewMatch[2])return json(res,200,await this.preview.stop(previewMatch[1]));
      if(previewMatch&&req.method==='GET'&&!previewMatch[2]){
        const packet=this.preview.read(previewMatch[1]);
        if(!packet){res.writeHead(204,{'cache-control':'no-store'});return res.end();}
        res.writeHead(200,{'content-type':'application/x-navbea-preview','content-length':packet.length,'cache-control':'no-store'});return res.end(packet);
      }
      if(req.method==='GET'&&url.pathname==='/v1/models')return json(res,200,this.modelsStatus());
      if(req.method==='POST'&&['/v1/models/download','/v1/models/cancel','/v1/models/activate'].includes(url.pathname)){
        if(!this.authorized(req,'admin'))return json(res,403,{error:'Model yönetimi için yönetici izni gerekiyor.'});
        if(!this.models||this.modelsInitializing)throw Error('Model yönetimi hazır değil.');
        const body=await readJson(req);
        return json(res,200,url.pathname.endsWith('/download')?this.models.startDownload(body.id):url.pathname.endsWith('/cancel')?this.models.cancelDownload():await this.changeModel(body.id));
      }
      if (req.method === "GET" && url.pathname === "/v1/capabilities") return json(res, 200, { apiVersion: API_VERSION, input: { profile: "matting-720p24", codec: "mjpeg", minimumFps: 24 }, output: { codec: "gray8", sequenceAligned: true, maximumLagMs: 150 }, nativeStill: true });
      if (req.method === "POST" && url.pathname === "/v1/sessions") return json(res, 201, await this.createSession(await readJson(req)));
      const sessionMatch = url.pathname.match(/^\/v1\/sessions\/([a-f0-9-]+)$/);
      if (req.method === "DELETE" && sessionMatch) return json(res, 200, { status: await this.deleteSession(sessionMatch[1]) ? "deleted" : "already-absent" });
      const subjectMatch = url.pathname.match(/^\/v1\/sessions\/([a-f0-9-]+)\/subjects\/(lock|unlock)$/);
      if (req.method === "POST" && subjectMatch) return json(res, 200, await this.lifecycle(() => this.controlSubjects(subjectMatch[1], subjectMatch[2])));
      if (req.method === "POST" && url.pathname === "/v1/still-mattes") { const body = await readJson(req); return json(res, 201, await this.stillMatte(body.cameraArtifact, body)); }
      const match = url.pathname.match(/^\/v1\/artifacts\/([a-f0-9-]+)$/);
      if (req.method === "GET" && match) {
        const item = this.artifacts.get(match[1]); if (!item) return json(res, 404, { error: "Artifact not found" });
        const claims = verifyCapability(String(req.headers.authorization || "").replace(/^Bearer\s+/i, ""), this.capabilitySecret, { audience: "rvm", scope: "artifact:read" });
        if (claims.artifact !== item.id) return json(res, 401, { error: "Artifact mismatch" }); this.artifacts.delete(item.id);
        res.writeHead(200, { "content-type": item.mimeType, "content-length": item.payload.length, "x-image-width": item.width, "x-image-height": item.height, "cache-control": "no-store" }); return res.end(item.payload);
      }
      return json(res, 404, { error: "Not found" });
    } catch (error) { return json(res, 503, { error: error.message, code: error.code || "RVM_REQUEST_FAILED" }); }
  }
  handleStream(socket) {
    const subscriber = { socket, authorized: false, closed: false, blocked: false, pending: null, streamId: null }; let handshake = Buffer.alloc(0);
    socket.on("data", chunk => { if (subscriber.authorized) return; handshake = Buffer.concat([handshake, chunk]); if (handshake.length > 8192) return socket.destroy(); const end = handshake.indexOf(10); if (end < 0) return; try { const message = JSON.parse(handshake.subarray(0, end).toString()); const claims = verifyCapability(message.capability, this.capabilitySecret, { audience: "rvm", scope: "stream:read", session: message.sessionId }); const item = this.sessions.get(message.sessionId); if (!item || item.maskStreamId !== claims.streamId) throw new Error("Unknown RVM session"); subscriber.authorized = true; subscriber.sessionId = item.id; subscriber.streamId = item.maskStreamId; this.subscribers.add(subscriber); socket.write('{"status":"ready"}\n'); } catch { socket.destroy(); } });
    socket.on("close", () => { subscriber.closed = true; this.subscribers.delete(subscriber); if (!this.stopping && subscriber.sessionId && ![...this.subscribers].some(item => item.sessionId === subscriber.sessionId)) void this.deleteSession(subscriber.sessionId).catch(error => this.scheduleWorkerRestart(error)); }); socket.on("error", () => { subscriber.closed = true; this.subscribers.delete(subscriber); });
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
    this.workerRestartTimer = setTimeout(() => { this.workerRestartTimer = null; void this.ensureWorker().catch(error => this.scheduleWorkerRestart(error)); }, delay);
  }
  async start() {
    try {
      await this.ensureStorage();
      this.controlServer = http.createServer((req, res) => void this.handleControl(req, res)); this.streamServer = net.createServer(socket => this.handleStream(socket));
      await this.listen(this.controlServer, this.paths.control); await this.listen(this.streamServer, this.paths.stream);
      this.modelsInitializing=true;
      try {
      if(!this.fixture){const options=this.workerOptions();this.models=new ModelManager({dataRoot:this.dataRoot,bundledRoot:path.dirname(options.model),probe:(model,device)=>runModelProbe(this.workerOptions(),model,device,this.modelAbort.signal)});try{await this.models.initialize();}catch(error){this.modelManagerError=error.message;this.models=null;}}
      }finally{this.modelsInitializing=false;}
      await this.startWorker().catch(async error => {if(this.models&&this.models.active.id!==this.models.catalogue.defaultModel){await this.models.restore({id:this.models.catalogue.defaultModel,device:'cpu'},'Seçilen model başlatılamadı; varsayılan modele dönüldü.');this.workerProfile=null;await this.startWorker().catch(error=>this.scheduleWorkerRestart(error));}else this.scheduleWorkerRestart(error);});
      return this;
    } catch (error) { await this.stop(); throw error; }
  }
  async stop() {
    this.preview.dispose();
    this.stopping = true; if (this.workerRestartTimer) clearTimeout(this.workerRestartTimer);
    this.modelAbort.abort();this.models?.cancelDownload();await this.models?.downloadPromise;
    clearTimeout(this.idleTimer);
    const source = this.sourceSession, socket = this.cameraSocket;
    this.sourceSession = null; this.cameraSocket = null; socket?.destroy(); this.worker?.stop();
    if (source) await this.cameraRequest("DELETE", `/v1/sessions/${source.id}`).catch(() => undefined);
    this.artifacts.clear();
    for (const item of this.subscribers) item.socket.destroy();
    await this.lifecycleTail?.catch(() => undefined);
    await Promise.all([this.controlServer, this.streamServer].filter(server => server?.listening).map(server => new Promise(resolve => server.close(resolve))));
    if (this.platform !== "win32") for (const value of this.ownedEndpoints) await fsp.unlink(value).catch(() => {});
  }
}

module.exports = { RvmService, PythonWorker, defaultDataRoot, VERSION, API_VERSION, writeLatestBounded };
