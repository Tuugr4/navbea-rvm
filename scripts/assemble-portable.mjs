import { access, cp, mkdir, rename, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const projectRoot = path.resolve(".");
const output = path.resolve(process.argv[2] || path.join(os.tmpdir(), "navbea-rvm-portable"));
const runtime = path.resolve(process.argv[3] || ".runtime-build");
const electronCandidates = [path.resolve("node_modules/electron/dist")];
let electronDist;
for (const candidate of electronCandidates) try { await access(path.join(candidate, "electron.exe")); electronDist = candidate; break; } catch {}
if (!electronDist) throw new Error("Electron runtime is unavailable");
const allowedRoots = [path.resolve(os.tmpdir()), path.join(projectRoot, "out-installer")];
if (!allowedRoots.some(root => output === root || output.startsWith(`${root}${path.sep}`))) throw new Error("Portable output must stay in TEMP or out-installer");

await rm(output, { recursive: true, force: true });
await cp(electronDist, output, { recursive: true });
await rename(path.join(output, "electron.exe"), path.join(output, "navbea-rvm.exe"));
const resources = path.join(output, "resources");
const appRoot = path.join(resources, "app");
await mkdir(appRoot, { recursive: true });
await writeFile(path.join(appRoot, "package.json"), `${JSON.stringify({ name: "navbea-rvm", version: "1.0.3", main: "electron/main.cjs" }, null, 2)}\n`);
for (const name of ["dist", "electron"]) await cp(path.join(projectRoot, name), path.join(appRoot, name), { recursive: true });
for (const name of ["service", "worker", "models", "deploy"]) await cp(path.join(projectRoot, name), path.join(resources, name), { recursive: true });
for (const name of ["LICENSE", "THIRD_PARTY_NOTICES.md"]) await cp(path.join(projectRoot, name), path.join(resources, name));
await cp(path.resolve(projectRoot, "local-media-protocol"), path.join(resources, "local-media-protocol"), { recursive: true });
await cp(runtime, path.join(resources, ".runtime-build"), { recursive: true });
console.log(output);
