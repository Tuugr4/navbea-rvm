import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";

const root = path.resolve(process.argv[2] || ".");
const packageJson = JSON.parse(await readFile(path.join(root, "package.json"), "utf8").catch(async () => readFile(path.join(root, "app", "package.json"), "utf8")));
const manifest = JSON.parse(await readFile(path.join(root, "models", "manifest.json"), "utf8"));
const license = await readFile(path.join(root, "LICENSE"), "utf8");
if (license.length < 30_000 || !license.includes("GNU GENERAL PUBLIC LICENSE") || !license.includes("END OF TERMS AND CONDITIONS")) throw new Error("Complete GPL-3.0 license text is required");
if (!/^[a-f0-9]{64}$/.test(manifest.sha256)) throw new Error("Pinned model SHA-256 is required");
const modelPath = path.join(root, "models", manifest.model); await stat(modelPath);
const modelHash = createHash("sha256").update(await readFile(modelPath)).digest("hex");
if (modelHash !== manifest.sha256) throw new Error("RVM model checksum mismatch");
const required = [
  "README.md", "SOURCE_OFFER.md", "CHANGES.md", "THIRD_PARTY_NOTICES.md", "SBOM.spdx.json",
  "THIRD_PARTY_LICENSES/React-ReactDOM-Scheduler-MIT.txt", "THIRD_PARTY_LICENSES/Phosphor-Icons-MIT.txt",
  "local-media-protocol/LICENSE", "worker/rvm_worker.py",
];
for (const item of required) await stat(path.join(root, item));
const offer = await readFile(path.join(root, "SOURCE_OFFER.md"), "utf8");
const local = packageJson.version.endsWith("-local");
const localReceipt = local ? JSON.parse(await readFile(path.join(root,"RVM_RELEASE_MANIFEST.json"),"utf8")) : null;
if (!offer.includes(`Navbea RVM ${packageJson.version}`) || (!local && !offer.includes(`/releases/tag/v${packageJson.version}`))) throw new Error("Source offer does not match the package version");
if (local && (localReceipt.kind !== "local-working-tree" || localReceipt.version !== packageJson.version || localReceipt.sourceOffer !== `source/navbea-rvm-${packageJson.version}-corresponding-source.zip`)) throw new Error("Local corresponding-source receipt is missing or inconsistent");
const notices = await readFile(path.join(root, "THIRD_PARTY_NOTICES.md"), "utf8");
for (const name of ["Robust Video Matting", "ReactDOM", "Scheduler", "Phosphor", "Electron", "ONNX Runtime", "NumPy", "Pillow"]) if (!notices.includes(name)) throw new Error(`Third-party notice is missing ${name}`);
const sbom = JSON.parse(await readFile(path.join(root, "SBOM.spdx.json"), "utf8"));
if (sbom.spdxVersion !== "SPDX-2.3" || !sbom.packages?.some(item => item.name === "Navbea RVM" && item.versionInfo === packageJson.version)) throw new Error("SPDX SBOM does not match the package version");
for (const runtimeLicense of [
  "runtime/LICENSE.txt",
  "runtime/Lib/site-packages/onnxruntime/LICENSE",
  "runtime/Lib/site-packages/onnxruntime/ThirdPartyNotices.txt",
]) await stat(path.join(root, runtimeLicense));
const sourceArchive = path.join(root, "source", `navbea-rvm-${packageJson.version}-corresponding-source.zip`);
const sourceInfo = await stat(sourceArchive);
if (sourceInfo.size < 15_000_000) throw new Error("Corresponding-source archive is missing upstream source/checkpoint material");
const sourceHash = createHash("sha256").update(await readFile(sourceArchive)).digest("hex");
if (local && (localReceipt.sourceSha256 !== sourceHash || localReceipt.modelSha256 !== modelHash)) throw new Error("Local source/model hash mismatch");
console.log(JSON.stringify({ status: local ? "local-source-verified" : "release-ready", version: packageJson.version, model: manifest.model, modelSha256: modelHash, sourceSha256: sourceHash, license: manifest.license }, null, 2));
