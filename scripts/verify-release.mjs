import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";

const root = path.resolve(process.argv[2] || ".");
const manifest = JSON.parse(await readFile(path.join(root, "models", "manifest.json"), "utf8"));
const license = await readFile(path.join(root, "LICENSE"), "utf8");
if (license.length < 30_000 || !license.includes("GNU GENERAL PUBLIC LICENSE") || !license.includes("END OF TERMS AND CONDITIONS")) throw new Error("Complete GPL-3.0 license text is required");
if (!/^[a-f0-9]{64}$/.test(manifest.sha256)) throw new Error("Pinned model SHA-256 is required");
const modelPath = path.join(root, "models", manifest.model); await stat(modelPath);
const modelHash = createHash("sha256").update(await readFile(modelPath)).digest("hex");
if (modelHash !== manifest.sha256) throw new Error("RVM model checksum mismatch");
for (const required of ["README.md", "THIRD_PARTY_NOTICES.md", "worker/rvm_worker.py"]) await stat(path.join(root, required));
console.log(JSON.stringify({ status: "release-ready", model: manifest.model, sha256: modelHash, license: manifest.license }, null, 2));
