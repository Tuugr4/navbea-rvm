const crypto = require('node:crypto');

// A diagnostic viewer observes the existing inference, without owning the
// subject-selection session or adding another model invocation.
class LivePreview {
  constructor(service, { now = Date.now, monotonic = () => process.hrtime.bigint() } = {}) {
    this.service = service; this.now = now; this.monotonic = monotonic;
    this.leases = new Map(); this.source = null; this.pair = null;
    this.timer = setInterval(() => void this.expire(), 1000); this.timer.unref?.();
  }
  get active() { return [...this.leases.values()].some(expires => expires > this.now()); }
  clear() { this.source = null; this.pair = null; }
  async start() {
    return this.service.lifecycle(async () => {
      const s = this.service;
      if (s.stopping || s.modelChange || s.modelsInitializing) throw Error('RVM hazırlanıyor. Biraz sonra tekrar deneyin.');
      if (this.leases.size >= 2) throw Error('Açık RVM önizlemesini önce kapatın.');
      clearTimeout(s.idleTimer);
      await s.ensureWorker();
      await s.ensureCameraSource();
      const lease = crypto.randomUUID(); this.leases.set(lease, this.now() + 6000);
      return { lease };
    });
  }
  offerSource(frame) { this.source = this.active ? frame : null; }
  acceptMask(frame) {
    const source = this.source; this.source = null;
    if (!this.active || !source || source.sequence !== frame.sequence || source.streamId !== frame.streamId || source.monotonicNs !== frame.monotonicNs) return;
    if (!frame.width || !frame.height || frame.width * frame.height > 1280 * 720 || frame.payload.length !== frame.width * frame.height || source.payload.length > 3 * 1024 * 1024) return;
    const age = Number(this.monotonic() - frame.monotonicNs) / 1e6;
    if (age < 0 || age > 500) return;
    this.pair = { source, mask: frame };
  }
  read(lease) {
    const expires = this.leases.get(lease);
    if (!expires || expires <= this.now()) throw Error('Önizleme süresi doldu. Yeniden açın.');
    this.leases.set(lease, this.now() + 6000);
    const pair = this.pair;
    if (!pair || this.service.stillCaptureBusy || this.service.resetPromise || this.service.status !== 'ready') return null;
    const age = Number(this.monotonic() - pair.mask.monotonicNs) / 1e6;
    if (age < 0 || age > 500) { this.pair = null; return null; }
    const metadata = Buffer.from(JSON.stringify({ sequence: pair.mask.sequence.toString(), width: pair.mask.width, height: pair.mask.height, sourceAtEpochMs: this.now() - age, jpegBytes: pair.source.payload.length, maskBytes: pair.mask.payload.length }));
    const header = Buffer.alloc(4); header.writeUInt32BE(metadata.length);
    return Buffer.concat([header, metadata, pair.source.payload, pair.mask.payload]);
  }
  stop(lease) {
    return this.service.lifecycle(async () => {
      if (!this.leases.delete(lease)) return { stopped: true };
      if (!this.active) {
        this.clear();
        if (!this.service.sessions.size && !this.service.stopping) await this.service.resetPipeline();
      }
      return { stopped: true };
    });
  }
  async expire() {
    if (this.expiring) return; this.expiring = true;
    try { for (const [id, until] of this.leases) if (until <= this.now()) await this.stop(id); }
    catch { /* Health already reports a failed source/worker; never throw from the timer. */ }
    finally { this.expiring = false; }
  }
  dispose() { clearInterval(this.timer); this.leases.clear(); this.clear(); }
}
module.exports = { LivePreview };
