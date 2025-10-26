import { app, BrowserWindow } from "electron";
import path from "node:path";
import { fileURLToPath } from "node:url";
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
process.env.DIST = path.join(__dirname, "../dist");
const publicPath = app.isPackaged ? path.join(process.env.DIST, "../public") : path.join(__dirname, "../public");
let win;
const VITE_DEV_SERVER_URL = process.env["VITE_DEV_SERVER_URL"];
function createWindow() {
  console.log("Icon path:", path.join(publicPath, "bluescale.ico"));
  win = new BrowserWindow({
    icon: path.join(publicPath, "bluescale.ico"),
    // icon: path.join(process.env.VITE_PUBLIC, "electron-vite.svg"),
    webPreferences: {
      preload: path.join(__dirname, "preload.js")
      // If you need a preload script
      // sandbox: false, // Use with caution in untrusted environments
      // nodeIntegration: true, // Be careful with this, consider contextBridge
      // contextIsolation: false, // Set to true with a preload script using contextBridge
    }
  });
  win.webContents.on("did-frame-finish-load", () => {
    win?.webContents.openDevTools();
  });
  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL);
  } else {
    win.loadFile(path.join(process.env.DIST, "index.html"));
  }
}
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
    win = null;
  }
});
app.whenReady().then(createWindow);
