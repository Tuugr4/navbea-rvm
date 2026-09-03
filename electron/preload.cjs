const { contextBridge, ipcRenderer } = require("electron");
contextBridge.exposeInMainWorld("rvm", { health: () => ipcRenderer.invoke("rvm:health"), openSource: () => ipcRenderer.invoke("rvm:open-source"), openData: () => ipcRenderer.invoke("rvm:open-data") });
