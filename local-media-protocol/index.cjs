const crypto = require("node:crypto");
const path = require("node:path");

const MAGIC = Buffer.from("NVF1");
const PROTOCOL_VERSION = 1;
const HEADER_BYTES = 64;
const MAX_PAYLOAD_BYTES = 128 * 1024 * 1024;
const MessageType = Object.freeze({ FRAME: 1, DROPPED: 2, END: 3, ERROR: 4 });
const Codec = Object.freeze({ MJPEG: 1, GRAY8: 2, JPEG: 3, JSON: 4 });

function endpointPaths(service, platform = process.platform, env = process.env) {
  if (!/^[a-z][a-z0-9-]{1,31}$/.test(service)) throw new Error("Invalid service name");
  if (platform === "win32") return {
    control: `\\\\.\\pipe\\navbea-${service}-control-v1`,
    stream: `\\\\.\\pipe\\navbea-${service}-stream-v1`,
  };
  const runtimeRoot = env.NAVBEA_RUNTIME_DIR || "/run/navbea";
  return {
    control: path.posix.join(runtimeRoot.replaceAll("\\", "/"), `${service}-control-v1.sock`),
    stream: path.posix.join(runtimeRoot.replaceAll("\\", "/"), `${service}-stream-v1.sock`),
  };
}

function uuidBytes(value) {
  const compact = String(value || "").replaceAll("-", "");
  if (!/^[a-f0-9]{32}$/i.test(compact)) throw new Error("Invalid stream UUID");
  return Buffer.from(compact, "hex");
}

function uuidString(value) {
  const hex = value.toString("hex");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function encodeHeader(frame) {
  const payloadLength = Number(frame.payloadLength ?? frame.payload?.length ?? 0);
  if (!Number.isSafeInteger(payloadLength) || payloadLength < 0 || payloadLength > MAX_PAYLOAD_BYTES) throw new Error("Invalid frame payload length");
  const header = Buffer.alloc(HEADER_BYTES);
  MAGIC.copy(header, 0);
  header.writeUInt16BE(frame.version ?? PROTOCOL_VERSION, 4);
  header.writeUInt16BE(frame.type ?? MessageType.FRAME, 6);
  header.writeUInt32BE(frame.flags ?? 0, 8);
  header.writeBigUInt64BE(BigInt(frame.sequence ?? 0), 12);
  header.writeBigUInt64BE(BigInt(frame.monotonicNs ?? process.hrtime.bigint()), 20);
  header.writeUInt32BE(frame.width ?? 0, 28);
  header.writeUInt32BE(frame.height ?? 0, 32);
  header.writeUInt32BE(payloadLength, 36);
  header.writeUInt16BE(frame.codec ?? Codec.MJPEG, 40);
  uuidBytes(frame.streamId).copy(header, 44);
  return header;
}

function decodeHeader(header) {
  if (!Buffer.isBuffer(header) || header.length < HEADER_BYTES) throw new Error("Incomplete frame header");
  if (!header.subarray(0, 4).equals(MAGIC)) throw new Error("Invalid frame magic");
  const version = header.readUInt16BE(4);
  if (version !== PROTOCOL_VERSION) throw new Error(`Unsupported protocol version ${version}`);
  const payloadLength = header.readUInt32BE(36);
  if (payloadLength > MAX_PAYLOAD_BYTES) throw new Error("Frame payload exceeds protocol limit");
  return {
    version, type: header.readUInt16BE(6), flags: header.readUInt32BE(8),
    sequence: header.readBigUInt64BE(12), monotonicNs: header.readBigUInt64BE(20),
    width: header.readUInt32BE(28), height: header.readUInt32BE(32), payloadLength,
    codec: header.readUInt16BE(40), streamId: uuidString(header.subarray(44, 60)),
  };
}

function encodeFrame(frame) {
  const payload = Buffer.isBuffer(frame.payload) ? frame.payload : Buffer.from(frame.payload || []);
  return Buffer.concat([encodeHeader({ ...frame, payloadLength: payload.length }), payload]);
}

class FrameDecoder {
  constructor(options = {}) {
    this.maxPayloadBytes = options.maxPayloadBytes ?? MAX_PAYLOAD_BYTES;
    if (!Number.isSafeInteger(this.maxPayloadBytes) || this.maxPayloadBytes < 0 || this.maxPayloadBytes > MAX_PAYLOAD_BYTES) {
      throw new Error("Invalid decoder payload limit");
    }
    this.headerBytes = Buffer.allocUnsafe(HEADER_BYTES);
    this.headerOffset = 0;
    this.header = null;
    this.payload = null;
    this.payloadOffset = 0;
    this.error = null;
  }

  push(chunk) {
    if (this.error) throw this.error;
    if (!Buffer.isBuffer(chunk)) chunk = Buffer.from(chunk);
    const frames = [];
    let offset = 0;
    try {
      while (offset < chunk.length) {
        if (!this.header) {
          const count = Math.min(HEADER_BYTES - this.headerOffset, chunk.length - offset);
          chunk.copy(this.headerBytes, this.headerOffset, offset, offset + count);
          this.headerOffset += count;
          offset += count;
          if (this.headerOffset < HEADER_BYTES) break;
          this.header = decodeHeader(this.headerBytes);
          if (this.header.payloadLength > this.maxPayloadBytes) throw new Error("Frame payload exceeds decoder limit");
          // One bounded allocation after validation; fragmented native masks are
          // copied once instead of repeatedly copying the accumulated payload.
          this.payload = Buffer.allocUnsafe(this.header.payloadLength);
        }
        const count = Math.min(this.header.payloadLength - this.payloadOffset, chunk.length - offset);
        chunk.copy(this.payload, this.payloadOffset, offset, offset + count);
        this.payloadOffset += count;
        offset += count;
        if (this.payloadOffset < this.header.payloadLength) break;
        frames.push({ ...this.header, payload: this.payload });
        // Emitted buffers remain owned by the caller and are never reused.
        this.header = null;
        this.headerOffset = 0;
        this.payload = null;
        this.payloadOffset = 0;
      }
    } catch (error) {
      this.error = error;
      this.header = null;
      this.payload = null;
      throw error;
    }
    return frames;
  }
}

function signCapability(claims, secret) {
  if (!secret || Buffer.byteLength(secret) < 32) throw new Error("Capability secret must contain at least 32 bytes");
  const now = Math.floor(Date.now() / 1000);
  const payload = { iat: now, ...claims };
  if (!payload.exp || payload.exp <= now) throw new Error("Capability expiry must be in the future");
  const body = Buffer.from(JSON.stringify(payload)).toString("base64url");
  const signature = crypto.createHmac("sha256", secret).update(body).digest("base64url");
  return `${body}.${signature}`;
}

function verifyCapability(token, secret, expected = {}) {
  const [body, signature, extra] = String(token || "").split(".");
  if (!body || !signature || extra) throw new Error("Malformed capability token");
  const wanted = crypto.createHmac("sha256", secret).update(body).digest();
  const supplied = Buffer.from(signature, "base64url");
  if (wanted.length !== supplied.length || !crypto.timingSafeEqual(wanted, supplied)) throw new Error("Invalid capability signature");
  const claims = JSON.parse(Buffer.from(body, "base64url").toString("utf8"));
  const now = Math.floor(Date.now() / 1000);
  if (!Number.isFinite(claims.exp) || claims.exp <= now) throw new Error("Capability expired");
  if (expected.audience && claims.aud !== expected.audience) throw new Error("Capability audience mismatch");
  if (expected.scope && !String(claims.scope || "").split(" ").includes(expected.scope)) throw new Error("Capability scope mismatch");
  if (expected.session && claims.session !== expected.session) throw new Error("Capability session mismatch");
  return claims;
}

module.exports = {
  MAGIC, PROTOCOL_VERSION, HEADER_BYTES, MAX_PAYLOAD_BYTES, MessageType, Codec,
  endpointPaths, encodeHeader, decodeHeader, encodeFrame, FrameDecoder,
  signCapability, verifyCapability,
};
