# Unity Android client

This optional client visualises CSI predictions, replays saved experiment results, plans routes on the 15-point grid, and provides an AR alignment workflow. The Python experiments and backend can be reproduced without Unity.

## Requirements

- Unity Editor `6000.6.3f1`
- Android Build Support, Android SDK/NDK and OpenJDK installed through Unity Hub
- An ARCore-capable Android device for camera and AR functions

Package versions are pinned in `Packages/manifest.json`. Do not commit Unity-generated `Library`, `Temp`, `Logs`, `Builds` or `QA` directories.

## Project layout

- `Assets/CSI/`: application, positioning, networking, AR and editor scripts
- `Assets/Resources/Replay/`: fixed D1 replay used by Developer mode
- `Assets/Resources/Maps/`: the apartment map used by the client
- `Assets/Resources/AR/`: the reference image and runtime image library
- `Packages/` and `ProjectSettings/`: reproducible Unity project configuration
- `Tests/`: Unity test sources

## Open and verify

Open the `unity-app` directory as a Unity project. On Windows, the repository scripts can run the editor checks and Android build:

```powershell
$env:UNITY_EDITOR_PATH = "C:\Program Files\Unity\Hub\Editor\6000.6.3f1\Editor\Unity.exe"
.\tools\check_frontend_ui.ps1
.\tools\build_meeting_demo.ps1
```

The build script calls `CsiApp.Editor.ArPrototypeBuild.BuildAndroidWithModel`. Generated logs and APKs remain local.

## Connect to the backend

1. Start the Python service as described in [`backend/README.md`](../backend/README.md).
2. Put the phone and backend computer on the same network.
3. In the app, open **Settings → Connection settings** and enter `http://<computer-ip>:8765`.
4. Use **Developer → Live** to inspect the current service, or **Developer → Replay** to inspect the bundled D1 results without a server.

The implemented client uses `/api/meta`, `/api/status` and `/api/predict/latest`. The server does not implement WebSocket streaming; the client polls the HTTP endpoints.

## Positioning and AR

In **Positioning**, select different start and target labels, choose a candidate route, and confirm it before starting. The 2D view uses the selected model's received labels. The AR view additionally requires the fixed reference image to be scanned, its measured pose to be locked, and floor points 0, 4 and 10 to be checked at the site.

AR alignment is a manual site procedure, not an independent position measurement. Replay data is historical experiment data and must not be presented as a live run. Real-time use requires the live backend, current-session calibration and the physical layout used by the experiment.

Further details:

- [`docs/FRONTEND_BACKEND_INTEGRATION.md`](../docs/FRONTEND_BACKEND_INTEGRATION.md)
- [`docs/FRONTEND_BACKEND_USER_GUIDE.md`](../docs/FRONTEND_BACKEND_USER_GUIDE.md)
- [`backend/API_CONTRACT.md`](../backend/API_CONTRACT.md)
- [`docs/ar-reference/README.md`](../docs/ar-reference/README.md)
