import { spawn } from "node:child_process";
import { access } from "node:fs/promises";
import path from "node:path";

const candidates = [
  process.env.RVM_PYTHON_PATH,
  path.resolve(".venv/Scripts/python.exe"),
  "python3",
  "python",
].filter(Boolean);

for (const candidate of candidates) {
  if (path.isAbsolute(candidate)) try { await access(candidate); } catch { continue; }
  const code = await new Promise(resolve => { const child = spawn(candidate, ["worker/rvm_worker.py", "--self-test"], { stdio: "inherit", windowsHide: true }); child.once("error", () => resolve(1)); child.once("exit", value => resolve(value ?? 1)); });
  if (code === 0) process.exit(0);
}
throw new Error("No compatible Python interpreter could run the RVM worker self-test");
