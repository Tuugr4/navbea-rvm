const fs = require('node:fs/promises');
const http = require('node:http');
const path = require('node:path');
const controller = require('./model-control.cjs');

function decodePreview(buffer) {
  if (buffer.length < 4) throw Error('Geçersiz önizleme paketi.');
  const length = buffer.readUInt32BE(0);
  if (length > 4096 || buffer.length < length + 4) throw Error('Geçersiz önizleme başlığı.');
  const meta = JSON.parse(buffer.subarray(4, 4 + length).toString());
  if (!Number.isInteger(meta.width) || !Number.isInteger(meta.height) || meta.width <= 0 || meta.height <= 0 || meta.width * meta.height > 1280 * 720 || meta.maskBytes !== meta.width * meta.height || !Number.isInteger(meta.jpegBytes) || meta.jpegBytes <= 0 || meta.jpegBytes > 3 * 1024 * 1024 || buffer.length !== 4 + length + meta.jpegBytes + meta.maskBytes || !Number.isFinite(meta.sourceAtEpochMs) || !/^\d{1,20}$/.test(meta.sequence)) throw Error('Önizleme boyutları doğrulanamadı.');
  return { status: 'frame', ...meta, jpeg: Uint8Array.from(buffer.subarray(4 + length, 4 + length + meta.jpegBytes)), mask: Uint8Array.from(buffer.subarray(4 + length + meta.jpegBytes)) };
}
class PreviewClient {
  constructor(options) { this.options = options; this.lease = null; this.generation = 0; }
  start() {
    if (this.starting) return this.starting;
    if (this.lease) return Promise.resolve({ started: true });
    const generation = ++this.generation;
    this.starting = controller.request(this.options, '/v1/preview/start', {}).then(async result => {
      if (!/^[a-f0-9-]{36}$/.test(result.lease)) throw Error('Geçersiz önizleme oturumu.');
      if (generation !== this.generation) { await controller.request(this.options, '/v1/preview/' + result.lease + '/stop', {}).catch(() => {}); return { started: false }; }
      this.lease = result.lease; return { started: true };
    }).finally(() => { this.starting = null; });
    return this.starting;
  }
  async stop() {
    ++this.generation; const lease = this.lease; this.lease = null;
    if (lease) await controller.request(this.options, '/v1/preview/' + lease + '/stop', {});
    return { stopped: true };
  }
  frame() {
    if (!this.lease) return Promise.resolve({ status: 'waiting' });
    if (this.reading) return this.reading;
    const lease = this.lease;
    this.reading = fs.readFile(path.join(this.options.dataRoot, 'client-token.txt'), 'utf8').then(token => new Promise((resolve, reject) => {
      const req = http.request({ socketPath: this.options.control, path: '/v1/preview/' + lease, headers: { 'x-navbea-client-token': token.trim() }, timeout: 3000 }, res => {
        const chunks = []; let length = 0;
        res.on('data', chunk => { length += chunk.length; if (length > 4 * 1024 * 1024) req.destroy(Error('Önizleme paketi çok büyük.')); else chunks.push(chunk); });
        res.on('error', reject); res.on('aborted', () => reject(Error('Önizleme bağlantısı kesildi.')));
        res.on('end', () => {
          try {
            if (this.lease !== lease || res.statusCode === 204) return resolve({ status: 'waiting' });
            const bytes = Buffer.concat(chunks);
            if (res.statusCode !== 200) throw Error(JSON.parse(bytes.toString()).error || 'Önizleme alınamadı.');
            resolve(decodePreview(bytes));
          } catch (error) { reject(error); }
        });
      });
      req.on('error', reject); req.on('timeout', () => req.destroy(Error('Önizleme zaman aşımına uğradı.'))); req.end();
    })).finally(() => { this.reading = null; });
    return this.reading;
  }
}
module.exports = { PreviewClient, decodePreview };
