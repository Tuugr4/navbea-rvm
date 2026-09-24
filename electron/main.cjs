const { app, BrowserWindow, ipcMain, shell, screen } = require("electron");
const fs = require("node:fs/promises");
const http = require("node:http");
const path = require("node:path");

// Scheduled-task service mode can outlive its transient launcher. A closed
// inherited stdout/stderr pipe is not an application failure.
for (const output of [process.stdout, process.stderr]) output?.on?.("error", () => undefined);
function loadProtocol() {
  const candidates = [!app.isPackaged ? process.env.NAVBEA_PROTOCOL_PATH : null, app.isPackaged ? path.join(process.resourcesPath, "local-media-protocol", "index.cjs") : null, path.resolve(__dirname, "..", "local-media-protocol", "index.cjs")].filter(Boolean);
  for (const candidate of candidates) try { return require(candidate); } catch { /* next */ }
  throw new Error("Local media protocol is missing");
}
const { endpointPaths } = loadProtocol();
const serviceMode = process.argv.includes("--service");
const modelAdmin=process.argv.includes('--model-admin');
if(modelAdmin){
  const controller=require('./model-control.cjs');const {kind,action,id}=controller.adminArguments(process.argv.slice(process.argv.indexOf('--model-admin')+1));
  Promise.resolve().then(()=>{if(!app.isPackaged||process.platform!=='win32')throw Error('Installed Windows application required');controller.check(action,id,kind);return controller.request({dataRoot:path.join(process.env.PROGRAMDATA||'C:\\ProgramData','Navbea','RVM'),control:endpointPaths('rvm','win32',{}).control,role:'admin'},controller.prefixes[kind]+'/'+action,{id});}).then(()=>app.exit(0),error=>{console.error(error.message);app.exit(1);});
} else if (serviceMode) {
  const machineRoot = process.env.PROGRAMDATA || "C:\\ProgramData";
  process.env.NAVBEA_PROTOCOL_PATH ||= path.join(process.resourcesPath, "local-media-protocol", "index.cjs");
  process.env.NAVBEA_RVM_ROOT ||= process.resourcesPath;
  process.env.NAVBEA_RVM_DATA_DIR ||= path.join(machineRoot, "Navbea", "RVM");
  process.env.NAVBEA_CAMERA_DATA_DIR ||= path.join(machineRoot, "Navbea", "Camera");
  // DirectML requires an interactive session. The machine-wide task runs at
  // higher integrity than Kiosk; HMAC tokens remain mandatory on the shared pipe.
  process.env.NAVBEA_PIPE_READABLE_ALL ||= "1";
  process.env.RVM_DEVICE ||= "cpu";
  const { RvmService } = require(path.join(process.resourcesPath, "service", "rvm-service.cjs"));
  const rvmService = new RvmService();
  rvmService.start().catch(error => { console.error(JSON.stringify({ level: "fatal", service: "rvm", error: error.message })); process.exitCode = 1; });
  for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => rvmService.stop().finally(() => process.exit(0)));
} else {
if (process.platform === "win32") app.setAppUserModelId("com.navbea.RVM");
app.setPath("userData", path.join(app.getPath("appData"), "Navbea", "RVM"));
function root() { return (!app.isPackaged&&process.env.NAVBEA_RVM_DATA_DIR) || (process.platform === "win32" ? path.join(process.env.PROGRAMDATA || "C:\\ProgramData", "Navbea", "RVM") : "/var/lib/navbea/rvm"); }
function resource(name) { return app.isPackaged ? path.join(process.resourcesPath, name) : path.join(__dirname, "..", name); }
async function health() { return modelController.request(modelOptions(),'/v1/health'); }
let modelWindow;
function createWindow() { const win = new BrowserWindow({ ...require("./window-layout.cjs").windowBoundsForWorkArea(screen.getDisplayNearestPoint(screen.getCursorScreenPoint()).workArea), show: false, backgroundColor: "#f4f4f2", autoHideMenuBar: true, webPreferences: { preload: path.join(__dirname, "preload.cjs"), nodeIntegration: false, contextIsolation: true, sandbox: true } }); modelWindow=win;win.webContents.setWindowOpenHandler(()=>({action:'deny'}));win.webContents.on('will-navigate',event=>event.preventDefault());win.once("ready-to-show", () => win.show()); if (process.env.VITE_DEV_SERVER_URL) win.loadURL(process.env.VITE_DEV_SERVER_URL); else win.loadFile(path.join(__dirname, "..", "dist", "index.html")); }
app.whenReady().then(createWindow); app.on("window-all-closed", () => app.quit());
ipcMain.handle("rvm:health", health);
const modelController=require('./model-control.cjs');
const modelOptions=()=>({dataRoot:root(),control:endpointPaths('rvm',process.platform,process.env).control,platform:process.platform,executable:process.execPath,systemRoot:process.env.SystemRoot||'C:\\Windows'});
const previewClient=new (require('./preview-client.cjs').PreviewClient)(modelOptions());
app.on('browser-window-created',(_event,win)=>win.once('closed',()=>void previewClient.stop().catch(()=>{})));
function requireModelWindow(event){if(!modelWindow||modelWindow.isDestroyed()||event.sender!==modelWindow.webContents||event.senderFrame!==event.sender.mainFrame)throw Error('Model yönetimi yalnız RVM penceresinden yapılabilir.');}
ipcMain.handle('rvm:models',event=>{requireModelWindow(event);return modelController.request(modelOptions(),'/v1/models');});
ipcMain.handle('rvm:preview-start',event=>{requireModelWindow(event);return previewClient.start();});
ipcMain.handle('rvm:preview-frame',event=>{requireModelWindow(event);return previewClient.frame();});
ipcMain.handle('rvm:preview-stop',event=>{requireModelWindow(event);return previewClient.stop();});
ipcMain.handle('rvm:still-models',event=>{requireModelWindow(event);return modelController.request(modelOptions(),'/v1/still-models');});
let modelAction=null;
const runModelAction=(event,action,id,kind)=>{requireModelWindow(event);modelController.check(action,id,kind);if(modelAction)throw Error('Model işlemi sürüyor.');modelAction=modelController.administer(modelOptions(),action,id,kind).finally(()=>{modelAction=null;});return modelAction;};
ipcMain.handle('rvm:model-action',(event,action,id)=>runModelAction(event,action,id,'live'));
ipcMain.handle('rvm:still-model-action',(event,action,id)=>runModelAction(event,action,id,'still'));
ipcMain.handle("rvm:open-source", () => app.getVersion().includes("-local") && app.isPackaged ? shell.openPath(resource("source")) : shell.openExternal("https://github.com/Tuugr4/navbea-rvm/releases/tag/v1.0.4"));
ipcMain.handle("rvm:open-upstream", () => shell.openExternal("https://github.com/PeterL1n/RobustVideoMatting"));
ipcMain.handle("rvm:open-license", () => shell.openPath(resource("LICENSE")));
ipcMain.handle("rvm:open-notices", () => shell.openPath(resource("THIRD_PARTY_NOTICES.md")));
ipcMain.handle("rvm:open-source-offer", () => shell.openPath(resource("SOURCE_OFFER.md")));
ipcMain.handle("rvm:open-data", () => shell.openPath(root()));
}
