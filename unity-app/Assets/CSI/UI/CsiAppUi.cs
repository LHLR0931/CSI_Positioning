using System;
using System.IO;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace CsiApp
{
    public sealed partial class CsiAppUi : MonoBehaviour
    {
        private static CsiAppUi activeInstance;
        private readonly NavigationState state = new NavigationState();
        private RectTransform canvasRect, safe, page;
        private Font font;
        private Sprite rounded;
        private ApartmentSceneView apartmentView;
        private AndroidPresentationBackHandler mobileBack;
        private bool apartmentReady;
        private string drawer = "", formError = "";
        private Rect lastSafe;
        private Vector2 lastScreen;
        private RectTransform formDrawer, formBottom;
        private Vector2 formDrawerOrigin;
        private bool appPaused;
#if UNITY_ANDROID && !UNITY_EDITOR
        private float nextImeQuery;
        private volatile float androidImeFraction = -1;
        private volatile bool imeQueryPending, uiDestroyed;
        private volatile int imeQueryGeneration;
#endif
        private static readonly Color Background = Hex("F3F5EF"), Ink = Hex("193C35"), Muted = Hex("697C74"),
            Teal = Hex("007D68"), Pale = Hex("E0EDE4"), Orange = Hex("A26325");

        private void Awake()
        {
            if (activeInstance != null && activeInstance != this)
            {
                Debug.LogWarning("Multiple CSI demo scenes are open. Only one UI instance is enabled.");
                enabled = false; return;
            }
            activeInstance = this;
            Screen.orientation = ScreenOrientation.Portrait;
            if (UnityEngine.Object.FindFirstObjectByType<Camera>() == null)
            {
                var cameraObject = new GameObject("UI Background Camera", typeof(Camera));
                cameraObject.transform.SetParent(transform, false);
                var camera = cameraObject.GetComponent<Camera>();
                camera.clearFlags = CameraClearFlags.SolidColor; camera.backgroundColor = Background;
                camera.cullingMask = 0;
            }
            font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            rounded = CreateRoundedSprite();
            var c = new GameObject("CSI Canvas", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            c.transform.SetParent(transform, false); canvasRect = c.GetComponent<RectTransform>();
            c.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = c.GetComponent<CanvasScaler>(); scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(390, 844); scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
            scaler.matchWidthOrHeight = .5f;
            if (UnityEngine.Object.FindFirstObjectByType<EventSystem>() == null)
            {
                var events = new GameObject("CSI Event System", typeof(EventSystem)); events.transform.SetParent(transform);
#if ENABLE_INPUT_SYSTEM
                var inputType = Type.GetType("UnityEngine.InputSystem.UI.InputSystemUIInputModule, Unity.InputSystem");
                if (inputType != null) events.AddComponent(inputType);
                else events.AddComponent<StandaloneInputModule>();
#else
                events.AddComponent<StandaloneInputModule>();
#endif
            }
            safe = new GameObject("Safe Area", typeof(RectTransform)).GetComponent<RectTransform>(); safe.SetParent(canvasRect, false);
            var sceneRoot = new GameObject("Confirmed Apartment Preview", typeof(ApartmentSceneView));
            sceneRoot.transform.SetParent(transform, false);
            apartmentView = sceneRoot.GetComponent<ApartmentSceneView>();
            try
            {
                apartmentView.Initialize(840, 720);
                apartmentReady = apartmentView.Texture != null;
            }
            catch (Exception e)
            {
                Debug.LogError("Apartment map preview could not initialize: " + e);
                state.Notice = "Apartment map unavailable. Grid-only fallback; see device log.";
            }
            InitializeIntegration(); ApplySafeArea(); Render();
        }
        private void Update()
        {
            if (lastSafe != Screen.safeArea || lastScreen != new Vector2(Screen.width, Screen.height)) ApplySafeArea();
            if (appPaused) return;
            UpdateIntegration();
            bool back = mobileBack != null && mobileBack.ConsumeBackRequest();
#if ENABLE_LEGACY_INPUT_MANAGER
            back |= UnityEngine.Input.GetKeyDown(KeyCode.Escape);
#endif
            if (back && (!string.IsNullOrEmpty(drawer) || state.Page != "Home")) HandleBack();
        }
        private void HandleBack()
        {
            if (!string.IsNullOrEmpty(drawer)) { drawer = ""; formError = ""; Render(); return; }
            if (state.Page == "Home") return;
            if (state.Page == "Developer" && IsRecording) { Open("Leave test"); return; }
            if (state.Page == "Developer") { if (LeaveDeveloper()) Render(); return; }
            if (state.Navigating) { Open("Leave navigation"); return; }
            state.GroundTruth = -1; state.Page = "Home"; Render();
            Debug.Log("CSI_UI_BACK: Home");
        }
        private void SyncBackPolicy()
        {
            // Home retains Android's normal exit policy. Only an open drawer or
            // an internal page owns Back, and only this scene owns this handler.
            bool needsBack = isActiveAndEnabled && (state.Page != "Home" || !string.IsNullOrEmpty(drawer));
            if (needsBack && mobileBack == null) mobileBack = new AndroidPresentationBackHandler();
            else if (!needsBack) ReleaseMobileBack();
        }
        private void ReleaseMobileBack()
        {
            if (mobileBack == null) return;
            mobileBack.Dispose(); mobileBack = null;
        }
        private void OnEnable() { if (activeInstance == this && page != null) SyncBackPolicy(); }
        private void OnDisable() { ReleaseMobileBack(); }
        private void LateUpdate()
        {
            if (appPaused || formDrawer == null || formBottom == null) return;
            float keyboardTop = KeyboardOcclusion();
            // A submit press can dismiss the IME before pointer-up. Keep the
            // pressed button in place until the gesture finishes.
#if ENABLE_LEGACY_INPUT_MANAGER
            if (keyboardTop <= 0 && (UnityEngine.Input.touchCount > 0 || UnityEngine.Input.GetMouseButton(0))) return;
#endif
            // Always calculate from the original layout. Never rebuild InputFields here:
            // keyboard animation, focus changes and validation must preserve their drafts.
            formDrawer.anchoredPosition = formDrawerOrigin;
            formDrawer.localScale = Vector3.one;
            if (keyboardTop <= 0) return;
            Vector2 keyboardLocal;
            if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(page,
                new Vector2(Screen.width * .5f, keyboardTop), null, out keyboardLocal)) return;
            const float margin = 12;
            float depth = -formBottom.anchoredPosition.y + formBottom.rect.height;
            float upper = page.rect.yMax - margin;
            float available = Mathf.Max(1, upper - keyboardLocal.y - margin);
            // Normally just lift the drawer. For a taller IME keep title, fields,
            // submit and error together inside the safe page without overlap.
            float scale = Mathf.Min(1, available / depth);
            float originalTop = page.rect.yMax + formDrawerOrigin.y;
            float top = Mathf.Min(upper, Mathf.Max(originalTop, keyboardLocal.y + margin + depth * scale));
            formDrawer.localScale = Vector3.one * scale;
            formDrawer.anchoredPosition = new Vector2((page.rect.width - formDrawer.rect.width * scale) * .5f,
                top - page.rect.yMax);
        }
        private float KeyboardOcclusion()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (!imeQueryPending && Time.realtimeSinceStartup >= nextImeQuery)
            {
                nextImeQuery = Time.realtimeSinceStartup + .1f;
                int generation = imeQueryGeneration;
                imeQueryPending = true;
                try
                {
                    using (var player = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
                    using (var activity = player.GetStatic<AndroidJavaObject>("currentActivity"))
                        activity.Call("runOnUiThread", new AndroidJavaRunnable(() => ReadAndroidIme(generation)));
                }
                catch (AndroidJavaException) { imeQueryPending = false; androidImeFraction = -1; }
            }
            if (androidImeFraction >= 0) return androidImeFraction * Screen.height;
#endif
            if (!TouchScreenKeyboard.visible) return 0;
            Rect area = TouchScreenKeyboard.area;
            if (area.height <= 0) return 0;
#if UNITY_ANDROID && !UNITY_EDITOR
            // Android reports its rectangle from the top-left; Unity UI uses bottom-left.
            return Mathf.Clamp(Screen.height - area.y, 0, Screen.height);
#else
            return Mathf.Clamp(area.yMax, 0, Screen.height);
#endif
        }
#if UNITY_ANDROID && !UNITY_EDITOR
        private void ReadAndroidIme(int generation)
        {
            float fraction = -1;
            try
            {
                if (uiDestroyed || generation != imeQueryGeneration) return;
                // Obtain callback-owned JNI references on the Android UI thread;
                // never capture the disposed activity used to schedule this callback.
                using (var version = new AndroidJavaClass("android.os.Build$VERSION"))
                {
                    if (version.GetStatic<int>("SDK_INT") < 30) return;
                }
                using (var player = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
                using (var activity = player.GetStatic<AndroidJavaObject>("currentActivity"))
                using (var window = activity.Call<AndroidJavaObject>("getWindow"))
                using (var decor = window.Call<AndroidJavaObject>("getDecorView"))
                using (var root = decor.Call<AndroidJavaObject>("getRootView"))
                using (var insets = root.Call<AndroidJavaObject>("getRootWindowInsets"))
                using (var types = new AndroidJavaClass("android.view.WindowInsets$Type"))
                {
                    if (insets == null) return;
                    int ime = types.CallStatic<int>("ime");
                    fraction = 0;
                    if (!insets.Call<bool>("isVisible", ime)) return;
                    using (var inset = insets.Call<AndroidJavaObject>("getInsets", ime))
                    using (var content = activity.Call<AndroidJavaObject>("findViewById", 16908290)) // android.R.id.content
                    using (var bounds = new AndroidJavaObject("android.graphics.Rect"))
                    {
                        int rootHeight = root.Call<int>("getHeight");
                        if (content == null || !content.Call<bool>("getGlobalVisibleRect", bounds)) return;
                        int top = bounds.Get<int>("top"), bottom = bounds.Get<int>("bottom");
                        int imeTop = rootHeight - inset.Get<int>("bottom");
                        // Insets are native-window pixels. Measure only overlap with
                        // the content view, so adjustResize and render scaling do not
                        // apply the keyboard height twice.
                        if (bottom > top) fraction = Math.Max(0, Math.Min(1, (float)(bottom - imeTop) / (bottom - top)));
                    }
                }
            }
            catch (AndroidJavaException) { fraction = -1; }
            finally
            {
                if (!uiDestroyed && generation == imeQueryGeneration) androidImeFraction = fraction;
                imeQueryPending = false;
            }
        }
#endif
        private void OnApplicationPause(bool paused)
        {
            appPaused = paused;
            if (paused) PauseIntegration();
            if (paused && mobileBack != null) mobileBack.DiscardPending();
#if UNITY_ANDROID && !UNITY_EDITOR
            imeQueryGeneration++; androidImeFraction = -1; nextImeQuery = 0;
#endif
            if (paused && state.Navigating) { state.Paused = true; state.Stale = true; state.Notice = "App paused. Restore demo data, then resume explicitly."; }
            if (!paused && page != null && formDrawer == null) Render();
        }
        private void ApplySafeArea()
        {
            lastSafe = Screen.safeArea; lastScreen = new Vector2(Screen.width, Screen.height);
            safe.anchorMin = new Vector2(lastSafe.xMin / Screen.width, lastSafe.yMin / Screen.height);
            safe.anchorMax = new Vector2(lastSafe.xMax / Screen.width, lastSafe.yMax / Screen.height);
            safe.offsetMin = safe.offsetMax = Vector2.zero;
            Canvas.ForceUpdateCanvases();
            if (page != null) FitPage();
        }
        private void FitPage()
        {
            float scale = Mathf.Min(safe.rect.width / 390, safe.rect.height / 844);
            page.localScale = Vector3.one * scale;
        }
        private void Run(Action action) { action(); Render(); }
        private void Open(string name) { drawer = name; formError = ""; Render(); }
        private void Render()
        {
            formDrawer = formBottom = null;
#if UNITY_ANDROID && !UNITY_EDITOR
            imeQueryGeneration++; androidImeFraction = -1; nextImeQuery = 0;
#endif
            if (page != null) { page.gameObject.SetActive(false); Destroy(page.gameObject); }
            page = new GameObject("UI / " + state.Page, typeof(RectTransform)).GetComponent<RectTransform>();
            page.SetParent(safe, false); page.anchorMin = page.anchorMax = page.pivot = new Vector2(.5f, .5f);
            page.sizeDelta = new Vector2(390, 844); FitPage();
            Panel(page, "Background", 0, 0, 390, 844, Background, false);
            if (state.Page == "Home") Home(); else Workspace();
            if (!string.IsNullOrEmpty(drawer)) DrawDrawer();
            CsiLocalization.ApplyNow();
            SyncBackPolicy();
        }
        private void Home()
        {
            Text(page, "CSI / INDOOR", 24, 32, 260, 24, 14, Ink, true);
            Button(page, "Settings", 270, 23, 96, 44, () => Open("Home settings"), Pale, Ink);
            Text(page, "Find your\nnext step.", 24, 105, 335, 136, 44, Ink, true);
            Panel(page, "Wayfinding card", 24, 256, 342, 184, Teal);
            Text(page, "01   POSITIONING", 44, 276, 300, 22, 12, Color.white, true);
            Text(page, "Find your destination.", 44, 310, 298, 36, 24, Color.white, true);
            Text(page, "Live position / map / calibrated AR", 44, 349, 298, 22, 13, Color.white);
            Button(page, "Open positioning  >", 44, 382, 295, 44,
                () => OpenScene("LivePositioning"), Color.white, Teal);
            Panel(page, "Developer card", 24, 460, 342, 184, Color.white);
            Text(page, "02   DEVELOPER", 44, 480, 300, 22, 12, Muted, true);
            Text(page, "Compare positioning.", 44, 514, 298, 36, 24, Ink, true);
            Text(page, "Prediction, ground truth and error", 44, 553, 298, 22, 13, Muted);
            Button(page, "Open developer  >", 44, 586, 295, 44,
                EnterDeveloper, Pale, Teal);
            Text(page, "15 reference points / 0.8 m spacing", 24, 682, 342, 25, 13, Muted);
            Text(page, "Designed by Yuhao & Hailan", 24, 774, 342, 22, 11, Muted, false, TextAnchor.MiddleCenter);
        }
        private void OpenScene(string sceneName)
        {
            if (!Application.CanStreamedLevelBeLoaded(sceneName))
            { state.Notice = "This tool is not included in this build."; Open("Action feedback"); return; }
            if (!StopAndSave()) { Open("Action feedback"); return; }
            PauseIntegration();
            if (sceneName == "LivePositioning" && sharedConnection != null) sharedConnection.SetPredictionPolling(false);
            ReleaseMobileBack();
            SceneManager.LoadScene(sceneName, LoadSceneMode.Single);
        }
        private void Workspace()
        {
            bool dev = state.Page == "Developer";
            Button(page, "<", 16, 20, 44, 44, HandleBack, Color.white, Ink);
            Text(page, dev ? "Developer" : "Grid debug", 72, 28, 214, 32, 23, Ink, true);
            Button(page, dev ? "More" : (state.TopView ? "3D view" : "Top view"), 278, 20, 96, 44,
                () => { if (dev) Open("Test"); else Run(() => state.TopView = !state.TopView); }, Pale, Ink);
            if (dev)
                Text(page, SourceSummary(), 24, 79, 342, 23, 11, Orange);
            else
            {
                Pill(page, "DEMO", 20, 82, 76);
                Text(page, state.Outside ? "Outside area (simulated)" : state.Stale ? "CSI stale / last estimate" : state.TrackingLostDemo ? "Tracking loss (simulated)" : "Simulated CSI: " + GridMap.Index(state.Prediction), 106, 88, 264, 25, 13, state.Stale || state.Outside || state.TrackingLostDemo ? Orange : Muted);
            }
            if (dev) DrawDeveloperUnified(); else Wayfinding();
            bool developerLive = dev && developerMode == DeveloperMode.Live;
            if (!developerLive)
                Text(page, dev ? DeveloperFooterNotice() : state.Notice, 24, dev ? 758 : 743, 342, dev ? 26 : 42, dev ? 10 : 12, Muted);
            Button(page, dev ? "Source, truth & recording   ^" : "Route details & demo controls   ^", 20, developerLive ? 725 : 790, 350, 44, () => Open(dev ? "Test" : "Route"), Color.white, Ink);
        }
        private void Wayfinding()
        {
            string title = !state.DestinationAvailable ? "Destination unavailable" : !state.Navigating ? "Preview your grid route" : state.Paused ? "Demo navigation paused" : state.Path.Count > 1 ? "Follow the highlighted route" : "Confirm your demo arrival";
            Panel(page, "Instruction", 20, 130, 350, 93, Color.white);
            Text(page, title, 36, 147, 318, 34, 20, Ink, true);
            Text(page, !state.DestinationAvailable ? "T01 and balcony exit label are pending" : "Virtual map / use buttons to advance demo", 36, 188, 318, 23, 12, Muted);
            ApartmentPreview(20, 240, 350, 300);
            Text(page, apartmentReady ? "VIRTUAL MAP PREVIEW / DEMO / " + (state.TopView ? "TOP" : "3D") : "MAP UNAVAILABLE / GRID-ONLY FALLBACK", 28, 551, 332, 25, 11, apartmentReady ? Muted : Orange, true);
            Text(page, "Blue: position   Amber: target   Green: route", 28, 576, 332, 16, 11, Muted);
            Panel(page, "Destination", 24, 592, 342, 74, Color.white);
            Text(page, "DESTINATION", 40, 603, 245, 20, 10, Muted, true);
            Text(page, state.Destination, 40, 628, 242, 26, 19, Ink, true);
            Button(page, "Edit", 291, 608, 59, 44, () => Open("Destination"), Pale, Teal);
            if (state.Navigating)
            {
                Button(page, "Next demo step", 24, 681, 165, 48, () => Run(state.DemoSample), Teal, Color.white);
                Button(page, state.Paused ? "Resume" : "Pause", 201, 681, 79, 48, () => Run(state.PauseOrResume), Pale, Teal);
                Button(page, "End", 292, 681, 74, 48, () => Open("End"), Color.white, Ink);
            }
            else Button(page, state.DestinationAvailable ? "Start grid demo" : "Unavailable / endpoint pending", 24, 681, 342, 48,
                () => Run(() => state.Start()), Teal, Color.white).interactable = state.DestinationAvailable;
        }
        private void ApartmentPreview(float x, float y, float w, float h)
        {
            if (!apartmentReady) { Map(x, y, w, h, true); return; }
            apartmentView.SetTopView(state.TopView);
            apartmentView.SetRoute(state.Path, state.Prediction, state.DestinationAvailable ? state.Target : -1);
            var go = new GameObject("Virtual apartment map / no AR camera", typeof(RectTransform), typeof(RawImage));
            Place(go, page, x, y, w, h);
            var preview = go.GetComponent<RawImage>();
            preview.texture = apartmentView.Texture; preview.raycastTarget = false;
        }
        private void Map(float x, float y, float w, float h, bool route)
        {
            Panel(page, "Grid diagram / 0.8 m spacing", x, y, w, h, Hex("EAF0E8"));
            var go = new GameObject("15 point grid", typeof(RectTransform), typeof(GridGraphic));
            var rt = Place(go, page, x, y, w, h); var grid = go.GetComponent<GridGraphic>();
            grid.State = state; grid.ShowRoute = route; grid.raycastTarget = false;
            if (!route && state.Page == "Developer")
            {
                grid.UseResult = true; grid.ResultPrediction = DisplayResult == null ? -1 : DisplayResult.Prediction.label; grid.ResultTruth = session.GroundTruth;
            }
            for (int n = 0; n < 15; n++)
            {
                Vector2 p = grid.Point(n);
                Text(rt, n.ToString(), p.x - rt.rect.xMin - 13, rt.rect.yMax - p.y - 12, 26, 24, 12, Ink, true, TextAnchor.MiddleCenter);
            }
        }
        private void DrawDrawer()
        {
            Button(page, "", 0, 0, 390, 844, () => Open(""), new Color(0, .08f, .06f, .42f), Color.clear, false);
            var sheet = Panel(page, "Drawer / " + drawer, 0, 274, 390, 570, Background);
            if (drawer == "Settings" || drawer == "Destination")
            { formDrawer = sheet; formDrawerOrigin = sheet.anchoredPosition; }
            sheet.GetComponent<Image>().raycastTarget = true;
            Panel(sheet, "Handle", 169, 11, 52, 4, Hex("C2CDC5"));
            Text(sheet, DrawerTitle(), 24, 37, 290, 38, 26, Ink, true);
            Button(sheet, "X", 322, 30, 44, 44, () => Open(""), Color.white, Ink);
            switch (drawer)
            {
                case "Home settings": HomeSettings(sheet); break;
                case "Language settings": LanguageSettings(sheet); break;
                case "Action feedback":
                    Text(sheet, state.Notice, 24, 112, 342, 164, 16, Orange);
                    Button(sheet, "Close", 24, 310, 342, 48, () => Open(""), Pale, Teal); break;
                case "Settings": ConnectionSettings(sheet); break;
                case "Destination": Destination(sheet); break;
                case "Tracking demo": TrackingDemo(sheet); break;
                case "Test": IntegrationControls(sheet); break;
                case "Ground truth": IntegrationTruth(sheet); break;
                case "Results": IntegrationResults(sheet); break;
                case "Replay controls": ReplayControls(sheet); break;
                case "Link status": LinkStatus(sheet); break;
                case "Route": Route(sheet); break;
                case "End": case "Leave navigation": End(sheet); break;
                case "Leave test":
                    Text(sheet, "Stop recording and save this run before\nreturning home?", 24, 100, 342, 82, 17, Ink);
                    Button(sheet, "Save & go home", 24, 205, 342, 50, () => { if (LeaveDeveloper()) Render(); }, Teal, Color.white);
                    Button(sheet, "Keep recording", 24, 275, 342, 50, () => Open(""), Pale, Teal); break;
            }
        }
        private string DrawerTitle()
        {
            switch (drawer) { case "Home settings": return "Settings"; case "Settings": return "Connection settings"; case "Action feedback": return "Unable to continue"; case "Test": return "Developer controls"; case "Route": return "Route & demo tools"; case "End": case "Leave navigation": return "End navigation?"; case "Leave test": return "Leave this test?"; default: return drawer; }
        }
        private void HomeSettings(RectTransform p)
        {
            Button(p, "Connection settings", 24, 118, 342, 56, () => Open("Settings"), Teal, Color.white);
            Button(p, "Language settings", 24, 196, 342, 56, () => Open("Language settings"), Pale, Teal);
        }
        private void LanguageSettings(RectTransform p)
        {
            bool chinese = CsiLocalization.IsChinese;
            Text(p, "APP LANGUAGE", 24, 104, 342, 22, 12, Muted, true);
            Button(p, "English", 24, 156, 342, 56, () => Run(() => CsiLocalization.SetLanguage(false)),
                chinese ? Color.white : Teal, chinese ? Ink : Color.white);
            Button(p, "简体中文", 24, 228, 342, 56, () => Run(() => CsiLocalization.SetLanguage(true)),
                chinese ? Teal : Color.white, chinese ? Color.white : Ink);
            Text(p, "Selected", 266, chinese ? 228 : 156, 84, 56, 11, Color.white, false, TextAnchor.MiddleRight);
            Text(p, "Your language choice is saved and applies throughout the app.", 24, 328, 342, 70, 14, Muted);
        }
        private void Destination(RectTransform p)
        {
            Button(p, "Balcony Entrance / Unavailable", 24, 98, 342, 60, () => { }, Pale, Muted).interactable = false;
            Text(p, "Door mapped; T01 and exit label are still pending.\nChoose a grid target for this demo.", 24, 171, 342, 42, 13, Muted);
            Text(p, "GRID COORDINATE", 24, 233, 342, 23, 12, Muted, true);
            Text(p, "Column (0-4)", 24, 256, 165, 18, 11, Muted);
            Text(p, "Row (0-2)", 201, 256, 165, 18, 11, Muted);
            var col = Input(p, state.DestinationAvailable ? (state.Target % 5).ToString() : "4", "Column: 0-4", 24, 275, 165, true);
            var row = Input(p, state.DestinationAvailable ? (state.Target / 5).ToString() : "2", "Row: 0-2", 201, 275, 165, true);
            var coordinateError = Text(p, formError, 24, 409, 342, 55, 14, Orange);
            formBottom = coordinateError.rectTransform;
            col.onValueChanged.AddListener(value => { formError = ""; coordinateError.text = ""; });
            row.onValueChanged.AddListener(value => { formError = ""; coordinateError.text = ""; });
            Button(p, "Preview grid route", 24, 344, 342, 50, () => {
                int i, j;
                if (!int.TryParse(col.text, out i) || !int.TryParse(row.text, out j) || i < 0 || i > 4 || j < 0 || j > 2) { formError = "Out of range. Use integer column 0-4, row 0-2."; coordinateError.text = formError; return; }
                Run(() => { state.SetTarget(false, GridMap.Label(i, j)); drawer = ""; });
            }, Teal, Color.white);
            Text(p, "Changing a target creates a new preview.\nStart navigation again when ready.", 24, 480, 342, 60, 13, Muted);
        }
        private void TrackingDemo(RectTransform p)
        {
            Text(p, "FAILURE-STATE SIMULATION", 24, 104, 342, 24, 12, Teal, true);
            Text(p, "This navigation demo shows a virtual model.\nIts tracking fault controls do not run an AR session.", 24, 160, 342, 92, 18, Ink);
            Text(p, "Tracking loss is a manual demo fault. Restore it,\ncheck for fresh demo data, then tap Resume.\nSwitching between top and 3D views does not\nclear a fault or establish AR alignment.", 24, 277, 342, 113, 15, Muted);
            Button(p, "Restore simulated tracking", 24, 427, 342, 52, () => Run(() => { state.RestoreTrackingDemo(); drawer = ""; }), Teal, Color.white);
            Text(p, "Use Positioning from Home for measured AR alignment.", 24, 500, 342, 40, 12, Muted);
        }
        private void Route(RectTransform p)
        {
            Text(p, state.Destination, 24, 100, 342, 35, 22, Ink, true);
            Text(p, "Grid path: " + (state.Path.Count == 0 ? "Unavailable" : string.Join(" > ", state.Path.ConvertAll(n => n.ToString()).ToArray())) + "\nInitial grid length: " + (double.IsNaN(state.PlannedGridLength) ? "N/A" : state.PlannedGridLength.ToString("F1") + " m") + "\n" + (state.Balcony ? "T01 and exit label pending; no door connector" : "Map coordinates are measured; positions are demo"), 24, 148, 342, 85, 13, Muted);
            Button(p, "Next demo position", 24, 251, 342, 44, () => Run(state.DemoSample), Teal, Color.white);
            Button(p, state.Stale ? "Restore demo data" : "Simulate stale CSI", 24, 306, 165, 44, () => Run(() => { if (state.Stale && !state.Outside) state.Accept(new PositionSample { Label = state.Prediction, Sequence = state.Sequence + 1, ReceivedUtc = DateTime.UtcNow }); else state.SetStale(true); }), Pale, Teal);
            Button(p, state.TrackingLostDemo ? "Restore tracking demo" : "Simulate tracking loss", 201, 306, 165, 44, () => Run(() => { if (state.TrackingLostDemo) state.RestoreTrackingDemo(); else state.InvalidateAr(); }), Pale, Teal);
            Button(p, "Tracking demo / recovery info", 24, 361, 342, 44, () => Open("Tracking demo"), Color.white, Ink);
            Button(p, state.Outside ? "Simulate re-entry" : "Simulate outside area", 24, 416, 342, 44, () => Run(() => state.SetOutside(!state.Outside)), Color.white, Ink);
            Text(p, "Outside/AR continuation is a future integration.\nThis skeleton freezes progress; it does not invent a position.", 24, 482, 342, 65, 13, Muted);
        }
        private void End(RectTransform p)
        {
            Text(p, "Confirm arrival at the selected grid target.\nThis is a demo; physical arrival is not detected.", 24, 103, 342, 85, 18, Ink);
            Button(p, "Confirm demo arrival / end", 24, 231, 342, 52, () => Finish(true), Teal, Color.white);
            Button(p, "End early", 24, 302, 342, 52, () => Finish(false), Pale, Teal);
            Button(p, "Continue navigation", 24, 373, 342, 52, () => Open(""), Color.white, Ink);
            Text(p, "Demo only. No CSI backend is connected.", 24, 474, 342, 57, 14, Muted);
        }
        private void Finish(bool arrived) { Run(() => { bool home = drawer == "Leave navigation"; state.End(arrived); if (home) state.Page = "Home"; drawer = ""; }); }

        private RectTransform Place(GameObject go, Transform parent, float x, float y, float w, float h)
        {
            var rt = go.GetComponent<RectTransform>(); rt.SetParent(parent, false);
            rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(0, 1); rt.anchoredPosition = new Vector2(x, -y); rt.sizeDelta = new Vector2(w, h); return rt;
        }
        private RectTransform Panel(Transform parent, string name, float x, float y, float w, float h, Color color, bool round = true)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Image)); var rt = Place(go, parent, x, y, w, h);
            var img = go.GetComponent<Image>(); img.color = color; img.raycastTarget = false;
            if (round) { img.sprite = rounded; img.type = Image.Type.Sliced; } return rt;
        }
        private Text Text(Transform parent, string value, float x, float y, float w, float h, int size, Color color, bool bold = false, TextAnchor align = TextAnchor.UpperLeft)
        {
            var go = new GameObject("Text", typeof(RectTransform), typeof(Text)); Place(go, parent, x, y, w, h);
            var text = go.GetComponent<Text>(); text.text = value; text.font = font; text.fontSize = size; text.color = color;
            text.fontStyle = bold ? FontStyle.Bold : FontStyle.Normal; text.alignment = align; text.raycastTarget = false;
            text.horizontalOverflow = HorizontalWrapMode.Wrap; text.verticalOverflow = VerticalWrapMode.Truncate; return text;
        }
        private Button Button(Transform parent, string title, float x, float y, float w, float h, Action action, Color color, Color foreground, bool round = true)
        {
            var rt = Panel(parent, title, x, y, w, h, color, round); var image = rt.GetComponent<Image>(); image.raycastTarget = true;
            var button = rt.gameObject.AddComponent<Button>(); button.targetGraphic = image;
            button.onClick.AddListener(() => action()); var colors = button.colors; colors.highlightedColor = new Color(.9f, .96f, .93f); colors.pressedColor = new Color(.76f, .88f, .82f); button.colors = colors;
            Text(rt, title, 8, 0, w - 16, h, 14, foreground, true, TextAnchor.MiddleCenter);
            return button;
        }
        private InputField Input(Transform parent, string value, string placeholder, float x, float y, float w, bool numeric = false)
        {
            var rt = Panel(parent, "Input", x, y, w, 50, Color.white); rt.GetComponent<Image>().raycastTarget = true;
            var field = rt.gameObject.AddComponent<InputField>(); field.targetGraphic = rt.GetComponent<Image>();
            field.textComponent = Text(rt, "", 12, 0, w - 24, 50, 16, Ink, false, TextAnchor.MiddleLeft);
            field.placeholder = Text(rt, placeholder, 12, 0, w - 24, 50, 14, Muted, false, TextAnchor.MiddleLeft);
            field.contentType = numeric ? InputField.ContentType.IntegerNumber : InputField.ContentType.Standard;
            field.text = value; return field;
        }
        private void Pill(Transform p, string title, float x, float y, float w)
        {
            var rt = Panel(p, title, x, y, w, 30, Pale); Text(rt, title, 10, 0, w - 20, 30, 11, Teal, true, TextAnchor.MiddleCenter);
        }
        private static Color Hex(string value) { Color c; ColorUtility.TryParseHtmlString("#" + value, out c); return c; }
        private static Sprite CreateRoundedSprite()
        {
            const int size = 32; var texture = new Texture2D(size, size, TextureFormat.RGBA32, false);
            for (int y = 0; y < size; y++) for (int x = 0; x < size; x++)
            {
                float dx = Mathf.Max(8 - x, x - 23), dy = Mathf.Max(8 - y, y - 23);
                float distance = Mathf.Sqrt(Mathf.Max(0, dx) * Mathf.Max(0, dx) + Mathf.Max(0, dy) * Mathf.Max(0, dy));
                texture.SetPixel(x, y, new Color(1, 1, 1, Mathf.Clamp01(8 - distance)));
            }
            texture.Apply(); return Sprite.Create(texture, new Rect(0, 0, size, size), new Vector2(.5f, .5f), 100, 0, SpriteMeshType.FullRect, new Vector4(10, 10, 10, 10));
        }
        private void OnDestroy()
        {
            PauseIntegration();
            ReleaseIntegration();
            ReleaseMobileBack();
#if UNITY_ANDROID && !UNITY_EDITOR
            uiDestroyed = true; imeQueryGeneration++;
#endif
            if (activeInstance == this) activeInstance = null;
            if (rounded != null) { Destroy(rounded.texture); Destroy(rounded); }
        }
    }
}
