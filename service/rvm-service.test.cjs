const test = require("node:test");
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs/promises");
const http = require("node:http");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const protocol = require("../local-media-protocol/index.cjs");
const { RvmService, PythonWorker } = require("./rvm-service.cjs");

class FakeCameraService {
  constructor(paths, secret) { this.paths = paths; this.secret = secret; this.sessions = new Map(); this.sockets = new Set(); this.sequence = 0n; }
  async start() {
    this.control = http.createServer((req, res) => {
      const send = (status, value) => { const body = Buffer.from(JSON.stringify(value)); res.writeHead(status, { "content-type": "application/json", "content-length": body.length }); res.end(body); };
      if (req.url === "/v1/health") return send(200, { status: "ready", service: "camera", version: "test", apiVersion: "1.0" });
      if (req.headers["x-navbea-client-token"] !== this.secret) return send(401, { error: "Unauthorized" });
      if (req.method === "POST" && req.url === "/v1/sessions") {
        const id = crypto.randomUUID(), streamId = crypto.randomUUID(), exp = Math.floor(Date.now() / 1000) + 60;
        const capability = protocol.signCapability({ aud: "camera", scope: "stream:read", session: id, streamId, exp }, this.secret);
        const item = { id, streamId, capability, streamEndpoint: this.paths.stream, expiresAt: new Date(exp * 1000).toISOString() };
        this.sessions.set(id, item); return send(201, item);
      }
      return send(404, { error: "Not found" });
    });
    this.stream = net.createServer(socket => {
      this.sockets.add(socket); let handshake = Buffer.alloc(0), timer = null;
      socket.on("data", chunk => {
        if (timer) return; handshake = Buffer.concat([handshake, chunk]); const end = handshake.indexOf(10); if (end < 0) return;
        try {
          const value = JSON.parse(handshake.subarray(0, end).toString()); const item = this.sessions.get(value.sessionId);
          const claims = protocol.verifyCapability(value.capability, this.secret, { audience: "camera", scope: "stream:read", session: value.sessionId });
          if (!item || claims.streamId !== item.streamId) throw new Error("Unknown session");
          socket.write('{"status":"ready"}\n');
          timer = setInterval(() => { if (socket.destroyed) return; this.sequence += 1n; socket.write(protocol.encodeFrame({ type: protocol.MessageType.FRAME, codec: protocol.Codec.MJPEG, streamId: item.streamId, sequence: this.sequence, monotonicNs: process.hrtime.bigint(), width: 1280, height: 720, payload: Buffer.from("jpeg") })); }, 10);
        } catch { socket.destroy(); }
      });
      socket.on("error", () => undefined); socket.on("close", () => { if (timer) clearInterval(timer); this.sockets.delete(socket); });
    });
    await Promise.all([new Promise(resolve => this.control.listen(this.paths.control, resolve)), new Promise(resolve => this.stream.listen(this.paths.stream, resolve))]);
  }
  async stop() {
    for (const socket of this.sockets) socket.destroy();
    await Promise.all([this.control, this.stream].filter(Boolean).map(server => new Promise(resolve => server.close(resolve))));
  }
}

function request(socketPath, method, pathname, token, body) {
  return new Promise((resolve, reject) => {
    const payload = body ? Buffer.from(JSON.stringify(body)) : null;
    const req = http.request({ socketPath, method, path: pathname, headers: { "x-navbea-client-token": token, ...(payload ? { "content-type": "application/json", "content-length": payload.length } : {}) } }, res => {
      const chunks = []; res.on("data", chunk => chunks.push(chunk)); res.on("end", () => resolve({ status: res.statusCode, payload: JSON.parse(Buffer.concat(chunks).toString()) }));
    }); req.on("error", reject); if (payload) req.write(payload); req.end();
  });
}

test("consumes Camera directly and publishes sequence-aligned GRAY8 masks", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "navbea-rvm-test-")); const id = crypto.randomUUID(); const shared = crypto.randomBytes(48).toString("base64url");
  const cameraPaths = { control: `\\\\.\\pipe\\camera-${id}-control`, stream: `\\\\.\\pipe\\camera-${id}-stream` };
  const rvmPaths = { control: `\\\\.\\pipe\\rvm-${id}-control`, stream: `\\\\.\\pipe\\rvm-${id}-stream` };
  const camera = new FakeCameraService(cameraPaths, shared);
  const rvm = new RvmService({ platform: "win32", paths: rvmPaths, cameraPaths, dataRoot: path.join(root, "rvm"), cameraDataRoot: path.join(root, "camera"), fixture: true, clientSecret: shared, capabilitySecret: shared, cameraClientToken: shared });
  await camera.start(); await rvm.start();
  try {
    const created = await request(rvmPaths.control, "POST", "/v1/sessions", shared, {}); assert.equal(created.status, 201);
    const frame = await new Promise((resolve, reject) => {
      const socket = net.createConnection(rvmPaths.stream); const decoder = new protocol.FrameDecoder(); let ack = false, preamble = Buffer.alloc(0);
      socket.once("connect", () => socket.write(`${JSON.stringify({ sessionId: created.payload.id, capability: created.payload.capability })}\n`));
      socket.on("data", chunk => { if (!ack) { preamble = Buffer.concat([preamble, chunk]); const end = preamble.indexOf(10); if (end < 0) return; ack = true; chunk = preamble.subarray(end + 1); } for (const item of decoder.push(chunk)) { socket.destroy(); resolve(item); } }); socket.on("error", reject);
    });
    assert.equal(frame.codec, protocol.Codec.GRAY8); assert.equal(frame.streamId, created.payload.maskStreamId); assert.equal(frame.payload.length, 1280 * 720);
  } finally { await rvm.stop(); await camera.stop(); await fs.rm(root, { recursive: true, force: true }); }
});

test("fails closed when the RVM runtime is unavailable", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "navbea-rvm-unavailable-")); const id = crypto.randomUUID(); const shared = crypto.randomBytes(48).toString("base64url");
  const service = new RvmService({ platform: "win32", paths: { control: `\\\\.\\pipe\\rvm-${id}-control`, stream: `\\\\.\\pipe\\rvm-${id}-stream` }, cameraPaths: { control: `\\\\.\\pipe\\missing-${id}`, stream: `\\\\.\\pipe\\missing-${id}-stream` }, dataRoot: root, cameraDataRoot: root, clientSecret: shared, capabilitySecret: shared, cameraClientToken: shared, env: { RVM_PYTHON_PATH: "missing", RVM_MODEL_PATH: "missing" } });
  await service.start();
  try { assert.equal(service.health().status, "unavailable"); assert.equal(service.health().errorCode, "RVM_RUNTIME_UNAVAILABLE"); }
  finally { await service.stop(); await fs.rm(root, { recursive: true, force: true }); }
});

test("a stale session delete cannot reset a newer empty pipeline", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "navbea-rvm-stale-delete-")); const id = crypto.randomUUID(); const shared = crypto.randomBytes(48).toString("base64url");
  const paths = { control: `\\\\.\\pipe\\rvm-stale-${id}-control`, stream: `\\\\.\\pipe\\rvm-stale-${id}-stream` };
  const service = new RvmService({ platform: "win32", paths, cameraPaths: { control: `\\\\.\\pipe\\missing-${id}`, stream: `\\\\.\\pipe\\missing-${id}-stream` }, dataRoot: root, cameraDataRoot: root, fixture: true, clientSecret: shared, capabilitySecret: shared, cameraClientToken: shared });
  await service.start();
  let resets = 0;
  service.resetPipeline = async () => { resets += 1; };
  try {
    const removed = await request(paths.control, "DELETE", `/v1/sessions/${crypto.randomUUID()}`, shared);
    assert.equal(removed.status, 200);
    assert.equal(removed.payload.status, "already-absent");
    assert.equal(resets, 0);
  } finally { await service.stop(); await fs.rm(root, { recursive: true, force: true }); }
});

test("worker stdin closure rejects the frame without an uncaught write EOF", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "navbea-rvm-pipe-test-"));
  const script = path.join(root, "exit-on-frame.cjs");
  await fs.writeFile(script, 'process.stdin.once("data", () => process.exit(17)); process.stdin.resume();\n');
  const worker = new PythonWorker({ python: process.execPath, script, model: "unused", device: "cpu", logPath: path.join(root, "worker.log") });
  try {
    await worker.start();
    await assert.rejects(worker.process({
      type: protocol.MessageType.FRAME,
      flags: 0,
      codec: protocol.Codec.MJPEG,
      streamId: crypto.randomUUID(),
      sequence: 1n,
      monotonicNs: 1n,
      width: 1,
      height: 1,
      payload: Buffer.from("jpeg"),
    }), /RVM worker exited|EPIPE|EOF|closed/i);
  } finally {
    worker.stop();
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("native foreground and alpha are separate one-use artifacts", async () => {
  const secret = crypto.randomBytes(48).toString("base64url");
  const service = new RvmService({ clientSecret: secret, capabilitySecret: secret, cameraClientToken: secret });
  service.cameraRequest = async () => ({ data: Buffer.from("source-jpeg") });
  const alpha = Buffer.from([0, 127, 255, 255]);
  const foreground = Buffer.from([255, 216, 255, 217]);
  const size = Buffer.alloc(4); size.writeUInt32BE(alpha.length);
  service.worker = { pending: new Map(), process: async source => {
    assert.equal(source.flags, 3);
    return { width: 2, height: 2, codec: 5, payload: Buffer.concat([size, alpha, foreground]) };
  } };
  const result = await service.stillMatte({ id: "raw", rvmReadCapability: "read", width: 2, height: 2 });
  assert.deepEqual(service.artifacts.get(result.id).payload, alpha);
  assert.deepEqual(service.artifacts.get(result.foregroundArtifact.id).payload, foreground);
  assert.notEqual(result.readCapability, result.foregroundArtifact.readCapability);
  assert.equal(service.health().metrics.frames, 0, "native inference must not count as a live mask");
  const response = { writeHead() {}, end(payload) { this.payload = payload; } };
  const req = { method: "GET", url: `/v1/artifacts/${result.foregroundArtifact.id}`, headers: { "x-navbea-client-token": secret, authorization: `Bearer ${result.foregroundArtifact.readCapability}` } };
  await service.handleControl(req, response);
  assert.deepEqual(response.payload, foreground);
  assert.equal(service.artifacts.has(result.foregroundArtifact.id), false);
});

test("slow CPU retains the newest waiting frame instead of an old inference queue", async () => {
  const service = new RvmService();
  service.cameraSocket = {};
  const calls = [], completions = [];
  service.worker = { process: frame => new Promise(resolve => { calls.push(frame.sequence); completions.push(() => { service.acceptMask({ ...frame, payload: Buffer.from([255]), width: 1, height: 1 }); resolve(); }); }) };
  const first = service.processSourceFrame({ sequence: 1n });
  await service.processSourceFrame({ sequence: 2n });
  await service.processSourceFrame({ sequence: 3n });
  assert.deepEqual(calls, [1n]);
  completions[0](); await first;
  assert.deepEqual(calls, [1n, 3n]);
  assert.equal(service.metrics.dropped, 1);
  service.stopping = true; completions[1]();
});
