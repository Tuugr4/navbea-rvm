// SPDX-License-Identifier: MIT
const DEFAULT_SUBJECT_POLICY = Object.freeze({ schemaVersion: 1, mode: "area", zone: Object.freeze({ x: .15, y: .55, width: .7, height: .42 }), maxPeople: 6, stableMs: 800 });
const DEFAULT_SUBJECT_THRESHOLD = 75;
function normalizeSubjectPolicy(value) {
  if (value === undefined || value === null) value = {};
  if (typeof value !== "object" || Array.isArray(value) || Object.keys(value).some(key => !["schemaVersion", "mode", "zone", "maxPeople", "stableMs", "threshold"].includes(key))) throw new Error("Invalid subject selection policy");
  const result = { ...DEFAULT_SUBJECT_POLICY, ...value, zone: { ...(value.zone || DEFAULT_SUBJECT_POLICY.zone) } };
  if (result.schemaVersion !== 1 || !["area", "tracking", "all", "score"].includes(result.mode)) throw new Error("Unsupported subject selection mode");
  if (!Number.isInteger(result.maxPeople) || result.maxPeople < 1 || result.maxPeople > 12) throw new Error("Subject group size must be between 1 and 12");
  if (!Number.isInteger(result.stableMs) || result.stableMs < 300 || result.stableMs > 3000) throw new Error("Subject stability must be between 300 and 3000 ms");
  // The score threshold only exists in score mode, so older RVM builds never see the key.
  if (result.mode === "score") {
    result.threshold = value.threshold ?? DEFAULT_SUBJECT_THRESHOLD;
    if (!Number.isInteger(result.threshold) || result.threshold < 50 || result.threshold > 95) throw new Error("Subject score threshold must be between 50 and 95");
  } else delete result.threshold;
  const z = result.zone;
  if (Object.keys(z).length !== 4 || ![z.x, z.y, z.width, z.height].every(Number.isFinite) || z.x < 0 || z.y < 0 || z.width < .1 || z.height < .05 || z.x + z.width > 1.000001 || z.y + z.height > 1.000001) throw new Error("Subject standing area must be a valid normalized rectangle");
  return result;
}
module.exports = { DEFAULT_SUBJECT_POLICY, DEFAULT_SUBJECT_THRESHOLD, normalizeSubjectPolicy };