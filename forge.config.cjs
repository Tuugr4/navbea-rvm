const path = require("node:path");
const runtimeSource = process.env.RVM_RUNTIME_SOURCE ? path.resolve(process.env.RVM_RUNTIME_SOURCE) : null;
const modelSource = process.env.RVM_MODEL_SOURCE ? path.resolve(process.env.RVM_MODEL_SOURCE) : null;
module.exports = {
  outDir: process.env.NAVBEA_RVM_FORGE_OUT || "out",
  packagerConfig: {
    asar: true, name: "Navbea RVM", executableName: "navbea-rvm",
    extraResource: ["service", "worker", "models", "deploy", "local-media-protocol", "LICENSE", "THIRD_PARTY_NOTICES.md", runtimeSource, modelSource].filter(Boolean),
  },
  makers: [
    { name: "@electron-forge/maker-squirrel", platforms: ["win32"], config: { name: "NavbeaRVM", setupExe: "Navbea-RVM-Setup.exe" } },
    { name: "@electron-forge/maker-zip", platforms: ["win32", "linux"] },
    { name: "@electron-forge/maker-deb", config: { options: { name: "navbea-rvm", productName: "Navbea RVM", maintainer: "Navbea" } } },
  ],
};
