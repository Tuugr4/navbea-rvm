const test = require("node:test");
const assert = require("node:assert/strict");
const { threadCandidates, chooseProfile } = require("./cpu-profile.cjs");

test("CPU candidates include all available processors without a four-thread ceiling", () => {
  assert.deepEqual(threadCandidates(6, 12), [1, 2, 3, 4, 6, 12]);
  assert.deepEqual(threadCandidates(20, 20), [1, 2, 4, 10, 20]);
  assert.deepEqual(threadCandidates(1, 1), [1]);
  assert.deepEqual(threadCandidates(64, 2), [1, 2]);
});
test("calibration can select full CPU or fewer threads based on measured latency", () => {
  assert.equal(chooseProfile([{ threads: 4, p95Ms: 90 }, { threads: 12, p95Ms: 60 }]).threads, 12);
  assert.equal(chooseProfile([{ threads: 4, p95Ms: 62 }, { threads: 12, p95Ms: 60 }]).threads, 4);
  assert.equal(chooseProfile([{ threads: 4, p95Ms: 65, schedulerP95Ms: 1 }, { threads: 12, p95Ms: 60, schedulerP95Ms: 20 }]).threads, 4);
  assert.throws(() => chooseProfile([{ threads: 12, p95Ms: NaN }]), /no usable/);
});
