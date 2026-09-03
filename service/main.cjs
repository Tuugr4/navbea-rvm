const os = require("node:os");
const path = require("node:path");
if (process.platform !== "win32" && !process.env.NAVBEA_RUNTIME_DIR && process.env.XDG_RUNTIME_DIR) process.env.NAVBEA_RUNTIME_DIR = path.join(process.env.XDG_RUNTIME_DIR, "navbea");
if (!process.env.NAVBEA_RVM_DATA_DIR && process.env.NAVBEA_RVM_DEV === "1") process.env.NAVBEA_RVM_DATA_DIR = path.join(os.tmpdir(), "navbea-rvm-dev");
const { RvmService } = require("./rvm-service.cjs");
const service = new RvmService();
service.start().catch(error => { console.error(JSON.stringify({ level: "fatal", service: "rvm", error: error.message })); process.exitCode = 1; });
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => service.stop().finally(() => process.exit(0)));
