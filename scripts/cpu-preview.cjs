const path = require("node:path");
const { RvmService } = require("../service/rvm-service.cjs");
const env = { ...process.env, RVM_DEVICE: "cpu", RVM_CPU_THREADS: "auto", RVM_LIVE_RATIO: process.env.RVM_LIVE_RATIO || "0.375", RVM_STILL_RATIO: "0.5" };
const service = new RvmService({
  env,
  dataRoot: path.resolve(__dirname, "../.runtime/cpu-preview"),
  paths: { control: "\\\\.\\pipe\\navbea-rvm-cpu-preview-control-v1", stream: "\\\\.\\pipe\\navbea-rvm-cpu-preview-stream-v1" },
});
service.start().then(() => console.log(JSON.stringify(service.health()))).catch(error => { console.error(error.message); process.exitCode = 1; });
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => service.stop().finally(() => process.exit(0)));
