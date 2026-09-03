const { contextBridge, ipcRenderer } = require("electron");
contextBridge.exposeInMainWorld("rvm", {
  health: () => ipcRenderer.invoke("rvm:health"),
  openSource: () => ipcRenderer.invoke("rvm:open-source"),
  openUpstream: () => ipcRenderer.invoke("rvm:open-upstream"),
  openLicense: () => ipcRenderer.invoke("rvm:open-license"),
  openNotices: () => ipcRenderer.invoke("rvm:open-notices"),
  openSourceOffer: () => ipcRenderer.invoke("rvm:open-source-offer"),
  openData: () => ipcRenderer.invoke("rvm:open-data"),
});
