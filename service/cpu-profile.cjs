const os = require("node:os");
const fs = require("node:fs/promises");
const path = require("node:path");
const crypto = require("node:crypto");
const { execFile, spawn } = require("node:child_process");
const { promisify } = require("node:util");
const exec = promisify(execFile);

function threadCandidates(physical, available) {
  return [...new Set([1, 2, 4, Math.ceil(physical / 2), physical, available].map(n => Math.max(1, Math.min(n, available))))].sort((a, b) => a - b);
}
function chooseProfile(results) {
  const valid = results.filter(r => Number.isInteger(r.threads) && r.threads > 0 && Number.isFinite(r.p95Ms) && r.p95Ms > 0);
  if (!valid.length) throw new Error("CPU calibration returned no usable measurements");
  const score = r => r.p95Ms + (r.schedulerP95Ms || 0);
  const best = Math.min(...valid.map(score));
  // Near-identical results favor less contention, without imposing a core cap.
  return valid.filter(r => score(r) <= best * 1.05).sort((a, b) => a.threads - b.threads)[0];
}
async function topology() {
  const available = os.availableParallelism?.() || os.cpus().length || 1;
  let physical = available;
  try {
    if (process.platform === "win32") {
      const result = await exec("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"], { windowsHide: true, timeout: 8000 });
      const n = Number(result.stdout.trim()); if (Number.isInteger(n) && n > 0) physical = Math.min(n, available);
    } else if (process.platform === "linux") {
      const text = await fs.readFile("/proc/cpuinfo", "utf8");
      const cores = new Set(text.split(/\n\s*\n/).map(block => {
        const socket = block.match(/^physical id\s*:\s*(\d+)/m), core = block.match(/^core id\s*:\s*(\d+)/m);
        return core && socket ? `${socket[1]}:${core[1]}` : null;
      }).filter(Boolean));
      if (cores.size) physical = Math.min(cores.size, available);
    }
  } catch { /* All available processors remain a candidate if topology is unavailable. */ }
  return { available, physical, models: [...new Set(os.cpus().map(cpu => cpu.model))], platform: process.platform, arch: process.arch };
}
function percentile(values, fraction) { const sorted = [...values].sort((a, b) => a - b); return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))] || 0; }
function benchmark(options, candidates) {
  return new Promise((resolve, reject) => {
    const child = spawn(options.python, [path.join(path.dirname(options.script), "cpu_benchmark.py"), "--model", options.model, "--ratio", String(options.liveRatio), "--candidates", candidates.join(",")], { windowsHide: true, shell: false, stdio: ["ignore", "pipe", "pipe"] });
    let pending = "", stderr = "", current = null, gaps = [], previous = performance.now(); const results = [];
    const tick = setInterval(() => { const now = performance.now(); if (current) gaps.push(Math.max(0, now - previous - 16)); previous = now; }, 16);
    const deadline = setTimeout(() => child.kill(), 90_000);
    const clean = () => { clearInterval(tick); clearTimeout(deadline); };
    child.stderr.on("data", chunk => { stderr = (stderr + chunk).slice(-1500); });
    child.stdout.on("data", chunk => {
      pending += chunk;
      while (pending.includes("\n")) {
        const index = pending.indexOf("\n"), line = pending.slice(0, index); pending = pending.slice(index + 1);
        try {
          const event = JSON.parse(line);
          if (event.event === "candidate") { current = event.threads; gaps = []; previous = performance.now(); }
          if (event.event === "result" && event.threads === current) { results.push({ ...event, schedulerP95Ms: Math.round(percentile(gaps, .95) * 100) / 100 }); current = null; }
        } catch { /* Ignore non-protocol diagnostic lines. */ }
      }
    });
    child.once("error", error => { clean(); reject(error); });
    child.once("exit", code => { clean(); if (code === 0 && results.length === candidates.length) resolve(results); else reject(new Error(`CPU calibration failed (${code}): ${stderr.trim()}`)); });
  });
}
async function resolveCpuProfile(options, dataRoot) {
  const cpu = await topology();
  const budget = options.cpuBudget === undefined || options.cpuBudget === "auto" ? Math.max(1, cpu.available - Math.ceil(cpu.available / 4)) : Number(options.cpuBudget);
  if (!Number.isInteger(budget) || budget < 1 || budget > cpu.available) throw new Error(`RVM_CPU_BUDGET must be auto or between 1 and ${cpu.available}`);
  const modelHash = crypto.createHash("sha256").update(await fs.readFile(options.model)).digest("hex");
  const engineHash = crypto.createHash("sha256").update(await fs.readFile(options.script)).digest("hex");
  if (options.threads !== "auto" && options.threads !== undefined) {
    const threads = Number(options.threads);
    if (!Number.isInteger(threads) || threads < 1 || threads > budget) throw new Error(`RVM_CPU_THREADS must fit the configured CPU budget (${budget})`);
    return { threads, mode: "manual", cpu, budget, modelHash, engineHash };
  }
  const benchmarkHash = crypto.createHash("sha256").update(await fs.readFile(path.join(path.dirname(options.script), "cpu_benchmark.py"))).digest("hex");
  const runtime = await exec(options.python, ["-B", "-c", "import onnxruntime; print(onnxruntime.__version__)"], { windowsHide: true, timeout: 15_000 });
  const key = crypto.createHash("sha256").update(JSON.stringify({ schema: 2, cpu, budget, modelHash, engineHash, benchmarkHash, ort: runtime.stdout.trim(), ratio: options.liveRatio })).digest("hex");
  const file = path.join(dataRoot, "cpu-profile.json");
  try {
    const saved = JSON.parse(await fs.readFile(file, "utf8"));
    if (saved.key === key && saved.threads >= 1 && saved.threads <= budget && Number.isInteger(saved.threads)) return { ...saved, cached: true };
  } catch { /* First run, changed hardware/model, or invalid cache: measure again. */ }
  const results = await benchmark(options, threadCandidates(cpu.physical, budget));
  const selected = chooseProfile(results);
  const profile = { key, threads: selected.threads, mode: "auto", cpu, budget, modelHash, engineHash, results, measuredAt: new Date().toISOString() };
  await fs.mkdir(dataRoot, { recursive: true });
  const temporary = `${file}.${crypto.randomUUID()}.tmp`;
  await fs.writeFile(temporary, JSON.stringify(profile, null, 2), { mode: 0o600 });
  await fs.rename(temporary, file);
  return { ...profile, cached: false };
}
module.exports = { threadCandidates, chooseProfile, resolveCpuProfile };
