// electron/preload.ts
// For now, this can be empty or have basic setup if you need it later for IPC.
// Example using contextBridge (safer):
// import { contextBridge, ipcRenderer } from 'electron';
// contextBridge.exposeInMainWorld('myAPI', {
//   doSomething: () => ipcRenderer.invoke('do-something')
// });
console.log("Preload script loaded.");
