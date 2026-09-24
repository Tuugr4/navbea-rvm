const { contextBridge, ipcRenderer } = require("electron");
contextBridge.exposeInMainWorld("rvm", {
  health: () => ipcRenderer.invoke("rvm:health"),
  models: () => ipcRenderer.invoke('rvm:models'),
  modelAction: (action,id) => ipcRenderer.invoke('rvm:model-action',action,id),
  stillModels: () => ipcRenderer.invoke('rvm:still-models'),
  stillModelAction: (action,id) => ipcRenderer.invoke('rvm:still-model-action',action,id),
  previewStart: () => ipcRenderer.invoke('rvm:preview-start'),
  previewFrame: () => ipcRenderer.invoke('rvm:preview-frame'),
  previewStop: () => ipcRenderer.invoke('rvm:preview-stop'),
  openSource: () => ipcRenderer.invoke("rvm:open-source"),
  openUpstream: () => ipcRenderer.invoke("rvm:open-upstream"),
  openLicense: () => ipcRenderer.invoke("rvm:open-license"),
  openNotices: () => ipcRenderer.invoke("rvm:open-notices"),
  openSourceOffer: () => ipcRenderer.invoke("rvm:open-source-offer"),
  openData: () => ipcRenderer.invoke("rvm:open-data"),
});
