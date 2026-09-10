const fs = require("node:fs");
const path = require("node:path");
let info = { buildId: null, sourceRevision: null, uncommitted: true };
try {
  const value = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "build-info.json"), "utf8"));
  if (value.schemaVersion === 1 && /^[a-f0-9]{64}$/.test(value.buildId)) info = {
    buildId: value.buildId,
    sourceRevision: /^[a-f0-9]{40,64}$/.test(value.sourceRevision || "") ? value.sourceRevision : null,
    uncommitted: value.uncommitted === true,
  };
} catch { /* Source checkouts have no packaged build identity. */ }
module.exports = Object.freeze(info);
