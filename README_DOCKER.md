# Running the Project in Docker

This project supports running the interactive 3D viewer directly from inside a Docker container. Follow the instructions below for your operating system to set up the GUI window forwarding correctly.

---

## 🪟 Windows (Easiest Method)

To see the MuJoCo viewer pop up on your Windows desktop from Docker, you need an X-server running.

1. **Install VcXsrv** (a free Windows X-server) from here: [VcXsrv Download](https://sourceforge.net/projects/vcxsrv/)
2. Once installed, **double-click the `windows_gui.xlaunch`** file included in this repository. 
   *(This launches VcXsrv with "Disable access control" pre-checked, which is required for Docker).*
3. Open your Command Prompt or PowerShell and run:
   ```bash
   docker compose run --rm locomotion
   ```
4. The interactive menu will appear in your terminal, and the simulation windows will pop up on your screen when you select a task!

> **Note for WSL2 Users:** If you run the command directly from inside a WSL terminal (e.g., Ubuntu app), WSLg will automatically intercept the window without needing VcXsrv!

---

## 🍎 macOS

macOS also requires an X-server to render Linux GUI windows.

1. **Install XQuartz** via Homebrew or their website:
   ```bash
   brew install --cask xquartz
   ```
2. Open **XQuartz** (it will appear in your Dock).
3. Go to **XQuartz > Preferences > Security** and check **"Allow connections from network clients"**.
4. Restart XQuartz to apply the settings.
5. In your terminal, run this command to whitelist your local machine:
   ```bash
   xhost + 127.0.0.1
   ```
6. Now you can run the container:
   ```bash
   docker compose run --rm locomotion
   ```

---

## 🐧 Linux

Linux users have it the easiest because the X11 server is already running natively. The `docker-compose.yml` is pre-configured to mount your local `/tmp/.X11-unix` socket.

1. You may need to grant local access to the X server:
   ```bash
   xhost +local:root
   ```
2. Run the container:
   ```bash
   docker compose run --rm locomotion
   ```
