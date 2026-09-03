const { app, BrowserWindow, ipcMain, shell } = require("electron");
const fs = require("node:fs/promises");
const http = require("node:http");
const path = require("node:path");

// Scheduled-task service mode can outlive its transient launcher. A closed
// inherited stdout/stderr pipe is not an application failure.
for (const output of [process.stdout, process.stderr]) output?.on?.("error", () => undefined);
function loadProtocol() {
  const candidates = [process.env.NAVBEA_PROTOCOL_PATH, app.isPackaged ? path.join(process.resourcesPath, "local-media-protocol", "index.cjs") : null, path.resolve(__dirname, "..", "local-media-protocol", "index.cjs")].filter(Boolean);
  for (const candidate of candidates) try { return require(candidate); } catch { /* next */ }
  throw new Error("Local media protocol is missing");
}
const { endpointPaths } = loadProtocol();
const serviceMode = process.argv.includes("--service");
if (serviceMode) {
  const machineRoot = process.env.PROGRAMDATA || "C:\\ProgramData";
  process.env.NAVBEA_PROTOCOL_PATH ||= path.join(process.resourcesPath, "local-media-protocol", "index.cjs");
  process.env.NAVBEA_RVM_ROOT ||= process.resourcesPath;
  process.env.NAVBEA_RVM_DATA_DIR ||= path.join(machineRoot, "Navbea", "RVM");
  process.env.NAVBEA_CAMERA_DATA_DIR ||= path.join(machineRoot, "Navbea", "Camera");
  // DirectML requires an interactive session. The machine-wide task runs at
  // higher integrity than Kiosk; HMAC tokens remain mandatory on the shared pipe.
  process.env.NAVBEA_PIPE_READABLE_ALL ||= "1";
  process.env.RVM_DEVICE ||= "auto";
  const { RvmService } = require(path.join(process.resourcesPath, "service", "rvm-service.cjs"));
  const rvmService = new RvmService();
  rvmService.start().catch(error => { console.error(JSON.stringify({ level: "fatal", service: "rvm", error: error.message })); process.exitCode = 1; });
  for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => rvmService.stop().finally(() => process.exit(0)));
} else {
if (process.platform === "win32") app.setAppUserModelId("com.navbea.RVM");
app.setPath("userData", path.join(app.getPath("appData"), "Navbea", "RVM"));
function root() { return process.env.NAVBEA_RVM_DATA_DIR || (process.platform === "win32" ? path.join(process.env.PROGRAMDATA || "C:\\ProgramData", "Navbea", "RVM") : "/var/lib/navbea/rvm"); }
function resource(name) { return app.isPackaged ? path.join(process.resourcesPath, name) : path.join(__dirname, "..", name); }
async function health() {
  const token = (await fs.readFile(path.join(root(), "client-token.txt"), "utf8")).trim();
  return new Promise((resolve, reject) => { const req = http.request({ socketPath: endpointPaths("rvm", process.platform, process.env).control, method: "GET", path: "/v1/health", headers: { "x-navbea-client-token": token } }, res => { const chunks = []; res.on("data", chunk => chunks.push(chunk)); res.on("end", () => resolve(JSON.parse(Buffer.concat(chunks).toString()))); }); req.on("error", reject); req.end(); });
}
function createWindow() { const win = new BrowserWindow({ width: 1100, height: 760, minWidth: 860, minHeight: 600, show: false, backgroundColor: "#061426", autoHideMenuBar: true, webPreferences: { preload: path.join(__dirname, "preload.cjs"), nodeIntegration: false, contextIsolation: true, sandbox: true } }); win.once("ready-to-show", () => win.show()); if (process.env.VITE_DEV_SERVER_URL) win.loadURL(process.env.VITE_DEV_SERVER_URL); else win.loadFile(path.join(__dirname, "..", "dist", "index.html")); }
app.whenReady().then(createWindow); app.on("window-all-closed", () => app.quit());
ipcMain.handle("rvm:health", health);
ipcMain.handle("rvm:open-source", () => shell.openExternal("https://github.com/Tuugr4/navbea-rvm/releases/tag/v1.0.4"));
ipcMain.handle("rvm:open-upstream", () => shell.openExternal("https://github.com/PeterL1n/RobustVideoMatting"));
ipcMain.handle("rvm:open-license", () => shell.openPath(resource("LICENSE")));
ipcMain.handle("rvm:open-notices", () => shell.openPath(resource("THIRD_PARTY_NOTICES.md")));
ipcMain.handle("rvm:open-source-offer", () => shell.openPath(resource("SOURCE_OFFER.md")));
ipcMain.handle("rvm:open-data", () => shell.openPath(root()));
}
