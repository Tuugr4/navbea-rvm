const path = require("node:path");
const runtimeSource = process.env.RVM_RUNTIME_SOURCE ? path.resolve(process.env.RVM_RUNTIME_SOURCE) : null;
const modelSource = process.env.RVM_MODEL_SOURCE ? path.resolve(process.env.RVM_MODEL_SOURCE) : null;
module.exports = {
  outDir: process.env.NAVBEA_RVM_FORGE_OUT || "out",
  packagerConfig: {
    asar: true, name: "Navbea RVM", executableName: "navbea-rvm",
    extraResource: ["service", "worker", "models", "deploy", "local-media-protocol", "THIRD_PARTY_LICENSES", "LICENSE", "THIRD_PARTY_NOTICES.md", "SOURCE_OFFER.md", "CHANGES.md", "SBOM.spdx.json", modelSource].filter(Boolean),
  },
  hooks: { postPackage: async (_config, result) => {
    const fs = require("node:fs/promises");
    const { createHash } = require("node:crypto");
    if (!runtimeSource) throw new Error("RVM_RUNTIME_SOURCE must point to a complete relocatable runtime");
    try { await fs.access(path.join(runtimeSource,"pyvenv.cfg")); throw new Error("A virtualenv cannot be shipped as a relocatable runtime"); } catch (error) { if(error.code !== "ENOENT") throw error; }
    for (const output of result.outputPaths) {
      const manifest = JSON.parse(await fs.readFile(path.join(output, "resources/models/subjects.json"), "utf8"));
      const bytes = await fs.readFile(path.join(output, "resources/models", manifest.model));
      if (createHash("sha256").update(bytes).digest("hex") !== manifest.sha256) throw new Error("Pinned subject instance model is required in every RVM package");
      const python = result.platform === "win32" ? "python.exe" : "bin/python";
      await fs.access(path.join(runtimeSource,python));
      await fs.cp(runtimeSource,path.join(output,"resources/runtime"),{recursive:true});
    }
  } },
  makers: [
    { name: "@electron-forge/maker-squirrel", platforms: ["win32"], config: { name: "NavbeaRVM", setupExe: "Navbea-RVM-Setup.exe" } },
    { name: "@electron-forge/maker-zip", platforms: ["win32", "linux"] },
    { name: "@electron-forge/maker-deb", config: { options: { name: "navbea-rvm", bin: "navbea-rvm", productName: "Navbea RVM", maintainer: "Navbea", depends: ["systemd", "libgomp1", "libstdc++6", "libasound2t64"], scripts: Object.fromEntries(["postinst","prerm","postrm"].map(name => [name,path.join(__dirname,"deploy/linux",name)])) } } },
  ],
};
