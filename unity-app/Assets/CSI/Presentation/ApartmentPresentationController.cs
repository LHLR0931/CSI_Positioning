using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace CsiApp
{
    /// <summary>Virtual desktop/mobile presentation. No live CSI or site alignment.</summary>
    public sealed class ApartmentPresentationController : MonoBehaviour
    {
        public enum ViewMode { Overview, Plan, FirstPerson }
        [Tooltip("Use the independent portrait touch interface, never the desktop canvas.")]
        public bool mobileLayout;
        public GameObject architecture;
        public ApartmentWorldModel grid;
        public Camera viewCamera;
        public Camera interfaceCamera;
        public Material[] architectureMaterials;
        public Material markerMaterial;
        public Transform walker;
        [SerializeField] private GameObject interfaceRoot;
        private Text routeText, statusText, startText, targetText, viewText;
        private readonly List<int> path = new List<int>();
        private ViewMode mode;
        private int start = 0, target = 14;
        private float distanceTravelled, yaw, pitch = 23;
        private bool playing;
        private bool ended;
        private string destinationId = "grid";
        private string startPlaceId = "grid";
        private DemoNavigationRoute activeRoute;
        private DemoRouteVisual routeVisual;
        private Transform roomLabels;
        private readonly List<TextMesh> roomTexts = new List<TextMesh>();
        private Vector3 eyePosition;
        private Font font;
        private ApartmentMobilePresentationUi mobileUi;
        private AndroidPresentationBackHandler mobileBack;
        private bool presentationInitialized;
        private bool mobileSuspended, leaving;
        private string message = "";
        public ViewMode Mode { get { return mode; } }
        public bool IsPlaying { get { return playing; } }
        public IReadOnlyList<int> Route { get { return path; } }
        public int StartLabel { get { return start; } }
        public int TargetLabel { get { return target; } }
        public string StatusMessage { get { return message; } }
        public string DestinationId { get { return destinationId; } }
        public string StartPlaceId { get { return startPlaceId; } }
        public DemoNavigationRoute ActiveRoute { get { return activeRoute; } }
        public float DistanceTravelled { get { return distanceTravelled; } }
        public float RemainingDistance { get { return activeRoute == null ? 0 : Mathf.Max(0, activeRoute.Length - distanceTravelled); } }
        public string NextInstruction { get { return activeRoute == null ? "Choose a destination" : ended ? "Demo ended — choose a new route" : activeRoute.InstructionAtDistance(distanceTravelled); } }
        public bool HasEnded { get { return ended; } }
        public const float EyeHeight = 1.50f;

        private void Start()
        {
            if (!Application.isEditor && !mobileLayout) Screen.SetResolution(1600, 1000, FullScreenMode.Windowed);
            if (mobileLayout) Screen.orientation = ScreenOrientation.Portrait;
            Initialize();
        }

        public void Initialize()
        {
            if (presentationInitialized) return;
            if (grid == null || viewCamera == null || architecture == null)
                throw new InvalidOperationException("Generate ApartmentPresentation with its Editor builder first.");
            font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            grid.Initialize(false);
            CreateRoomLabels();
            routeVisual = GetComponent<DemoRouteVisual>();
            if (routeVisual == null) routeVisual = gameObject.AddComponent<DemoRouteVisual>();
            if (mobileLayout)
            {
                if (interfaceRoot != null) interfaceRoot.SetActive(false);
                mobileUi = GetComponent<ApartmentMobilePresentationUi>();
                if (mobileUi == null) mobileUi = gameObject.AddComponent<ApartmentMobilePresentationUi>();
                mobileUi.Build(this);
                Debug.Log("CSI_MOBILE_MODEL_ENTER: offline simulation; no CSI and no AR session.");
            }
            else BuildInterface();
            if (PlanningPreviewTransfer.ReturnToPlanning && PlanningPreviewTransfer.Labels != null)
            {
                var path = PlanningPreviewTransfer.Labels;
                ApplyRoute(DemoNavigationRoute.BuildFromGridPath(path), path[0], path[path.Count - 1]);
                SetView(PlanningPreviewTransfer.FirstPerson ? ViewMode.FirstPerson : ViewMode.Plan);
            }
            else { ConfigureRoute(0, 14); SetView(mobileLayout ? ViewMode.FirstPerson : ViewMode.Overview); }
            presentationInitialized = true;
            if (mobileLayout && mobileBack == null) mobileBack = new AndroidPresentationBackHandler();
        }

        public void ConfigureRoute(int from, int to)
        {
            ConfigureDestination(from, "grid", to);
        }

        public void ConfigureDestination(int from, string destination, int gridTarget = 14)
        {
            var next = DemoNavigationRoute.Build(from, destination, gridTarget);
            startPlaceId = "grid";
            ApplyRoute(next, from, gridTarget);
        }

        public void ConfigureFromPlace(string place, string destination, int gridTarget = 14)
        {
            var next = DemoNavigationRoute.BuildFromPlace(place, destination, gridTarget);
            startPlaceId = place.Trim().ToLowerInvariant();
            ApplyRoute(next, next.GridLabels[0], gridTarget);
        }

        private void ApplyRoute(DemoNavigationRoute next, int from, int gridTarget)
        {
            playing = ended = false; start = from; target = gridTarget; distanceTravelled = 0;
            destinationId = next.DestinationId; activeRoute = next;
            path.Clear(); path.AddRange(next.GridLabels);
            // Preserve the desktop grid-only inspection path, but avoid double lines on mobile.
            grid.SetRoute(mobileLayout ? null : path, start, next.ConnectorLabel);
            if (routeVisual != null) routeVisual.Build(next, false);
            if (walker != null) walker.position = next.Points[0] + Vector3.up * .14f;
            eyePosition = next.Points[0] + Vector3.up * EyeHeight;
            pitch = 23;
            if (next.Points.Count > 1) SetHeading(next.Points[1] - next.Points[0]);
            UpdateCopy(next.UsesIllustrativeConnector ? "Demo entrance / site verification pending" : "Grid route / simulated position");
            if (mobileLayout) Debug.Log("CSI_MOBILE_MODEL_ROUTE: " + from + " -> " + destinationId + " via " + next.ConnectorLabel + " / length=" + next.Length.ToString("F3") + " / illustrative=" + next.UsesIllustrativeConnector);
            if (mode == ViewMode.FirstPerson) PositionFirstPerson();
        }

        public void ResetRoute()
        {
            if (PlanningPreviewTransfer.ReturnToPlanning && PlanningPreviewTransfer.Labels != null)
            { var labels = PlanningPreviewTransfer.Labels; ApplyRoute(DemoNavigationRoute.BuildFromGridPath(labels), labels[0], labels[labels.Count - 1]); return; }
            if (startPlaceId == "grid") ConfigureDestination(start, destinationId, target);
            else ConfigureFromPlace(startPlaceId, destinationId, target);
        }
        public void EndRoute()
        {
            playing = false; ended = true;
            UpdateCopy("Demo ended by user. No real arrival was measured.");
            if (mobileLayout) Debug.Log("CSI_MOBILE_MODEL_END: manual demo end, not real arrival");
        }

        public void SetView(ViewMode value)
        {
            mode = value;
            foreach (var material in architectureMaterials)
                if (material != null) material.SetFloat("_CutHeight", value == ViewMode.FirstPerson ? 20f : .35f);
            SetArchitectureVisibility(value);
            viewCamera.nearClipPlane = .04f; viewCamera.farClipPlane = 80;
            viewCamera.fieldOfView = 67;
            if (value == ViewMode.FirstPerson)
            {
                viewCamera.orthographic = false;
                if (walker != null) eyePosition = walker.position + Vector3.up * (EyeHeight - walker.position.y);
                if (activeRoute != null) RecenterVirtualView(); else PositionFirstPerson();
            }
            else
            {
                var bounds = ApartmentMap.Current.PlanBounds;
                var center = ApartmentMap.ToWorld(bounds.center.x, bounds.center.y, .15f);
                var rotation = value == ViewMode.Plan ? Quaternion.Euler(90, 0, 0) : Quaternion.Euler(63, -13, 0);
                viewCamera.orthographic = true;
                if (mobileLayout)
                {
                    float halfWidth = 0, halfHeight = 0;
                    foreach (float x in new[] { bounds.xMin, bounds.xMax }) foreach (float y in new[] { bounds.yMin, bounds.yMax })
                        foreach (float h in new[] { 0f, .35f })
                        {
                            var offset = Quaternion.Inverse(rotation) * (ApartmentMap.ToWorld(x, y, h) - center);
                            halfWidth = Mathf.Max(halfWidth, Mathf.Abs(offset.x)); halfHeight = Mathf.Max(halfHeight, Mathf.Abs(offset.y));
                        }
                    viewCamera.orthographicSize = Mathf.Max(halfHeight + .25f, (halfWidth + .25f) / Mathf.Max(.1f, viewCamera.aspect));
                }
                else
                {
                    // Reserve the right side of the desktop image for its controls.
                    center += rotation * Vector3.right * 1.80f + rotation * Vector3.up * .60f;
                    viewCamera.orthographicSize = value == ViewMode.Plan ? 4.80f : 5.35f;
                }
                viewCamera.transform.SetPositionAndRotation(center + rotation * Vector3.back * 18, rotation);
            }
            grid.SetLabelsFacing(viewCamera);
            FaceRoomLabels();
            if (roomLabels != null) roomLabels.gameObject.SetActive(value != ViewMode.FirstPerson);
            if (walker != null) walker.gameObject.SetActive(value != ViewMode.FirstPerson);
            if (viewText != null) viewText.text = value == ViewMode.FirstPerson ? "FIRST PERSON  /  VIRTUAL CAMERA" : value == ViewMode.Plan ? "TOP VIEW  /  CONFIRMED PLAN" : "3D OVERVIEW  /  CUTAWAY";
            if (mobileUi != null) mobileUi.Refresh();
            if (mobileLayout) Debug.Log("CSI_MOBILE_MODEL_VIEW: " + value);
        }

        public void TogglePlayback()
        {
            if (leaving || mobileSuspended) return;
            if (activeRoute == null || activeRoute.Points.Count < 2) { UpdateCopy("Start and destination are the same. Choose another target."); return; }
            if (!playing && (ended || distanceTravelled >= activeRoute.Length - .0001f)) ResetRoute();
            playing = !playing;
            if (mode == ViewMode.FirstPerson) RecenterVirtualView();
            UpdateCopy(playing ? (activeRoute.UsesIllustrativeConnector ? "Simulated walk / amber segments are illustrative" : "Simulated walk / four-neighbor grid route") : "Paused / simulated position");
            if (mobileLayout) Debug.Log(playing ? "CSI_MOBILE_MODEL_PLAY" : "CSI_MOBILE_MODEL_PAUSE");
        }

        private void Update()
        {
            if (viewCamera == null || grid == null) return;
            if (leaving || mobileSuspended) return;
            if (mobileLayout && ((mobileBack != null && mobileBack.ConsumeBackRequest()) || Input.GetKeyDown(KeyCode.Escape)))
            { if (mobileUi != null && mobileUi.CloseDrawer()) return; ReturnHome(); return; }
            if (Input.GetKeyDown(KeyCode.Alpha1)) SetView(ViewMode.Overview);
            if (Input.GetKeyDown(KeyCode.Alpha2)) SetView(ViewMode.Plan);
            if (Input.GetKeyDown(KeyCode.Alpha3)) SetView(ViewMode.FirstPerson);
            if (Input.GetKeyDown(KeyCode.Space)) TogglePlayback();
            if (Input.GetKeyDown(KeyCode.R)) ResetRoute();
            if (!mobileLayout && Input.GetKeyDown(KeyCode.Escape)) { playing = false; Cursor.lockState = CursorLockMode.None; Cursor.visible = true; }
            if (playing) Advance(Time.deltaTime);
            else if (mode == ViewMode.FirstPerson && !mobileLayout) MoveVirtualCamera();
            grid.SetLabelsFacing(viewCamera);
            FaceRoomLabels();
        }

        private void Advance(float delta)
        {
            if (activeRoute == null) return;
            distanceTravelled = Mathf.Min(activeRoute.Length, distanceTravelled + Mathf.Max(0, delta) * .55f);
            Vector3 direction;
            Vector3 position = activeRoute.PointAtDistance(distanceTravelled, out direction);
            if (walker != null) walker.position = position + Vector3.up * .14f;
            if (routeVisual != null) routeVisual.SetDistance(distanceTravelled);
            if (mode == ViewMode.FirstPerson)
            {
                eyePosition = position + Vector3.up * EyeHeight;
                Vector3 facing = VirtualRouteFacing(position);
                yaw = Mathf.MoveTowardsAngle(yaw, Mathf.Atan2(facing.x, facing.z) * Mathf.Rad2Deg, delta * 130);
                PositionFirstPerson();
            }
            if (distanceTravelled >= activeRoute.Length - .0001f)
            {
                playing = false;
                UpdateCopy(activeRoute.DestinationId != "grid" ? "Demo entrance reached. Tap End; real arrival is not measured." : "Grid target reached. Tap End to finish the demo.");
            }
            else if (mobileUi != null) mobileUi.Refresh();
        }

        private void MoveVirtualCamera()
        {
            if (Input.GetMouseButton(1))
            {
                yaw += Input.GetAxisRaw("Mouse X") * 2.2f;
                pitch = Mathf.Clamp(pitch - Input.GetAxisRaw("Mouse Y") * 1.8f, -45, 65);
            }
            if (Input.GetKey(KeyCode.LeftArrow)) yaw -= Time.deltaTime * 65;
            if (Input.GetKey(KeyCode.RightArrow)) yaw += Time.deltaTime * 65;
            Vector3 local = new Vector3((Input.GetKey(KeyCode.D) ? 1 : 0) - (Input.GetKey(KeyCode.A) ? 1 : 0), 0,
                (Input.GetKey(KeyCode.W) ? 1 : 0) - (Input.GetKey(KeyCode.S) ? 1 : 0));
            eyePosition += Quaternion.Euler(0, yaw, 0) * local.normalized * (.75f * Time.deltaTime);
            // Free inspection stays inside the open, confirmed grid rectangle. No invented room walkability.
            eyePosition.x = Mathf.Clamp(eyePosition.x, 0, 3.2f); eyePosition.z = Mathf.Clamp(eyePosition.z, 0, 1.6f);
            eyePosition.y = EyeHeight;
            PositionFirstPerson();
        }
        private void PositionFirstPerson() { viewCamera.transform.SetPositionAndRotation(eyePosition, Quaternion.Euler(pitch, yaw, 0)); }
        public void SetMobileViewport(Rect viewport)
        {
            if (!mobileLayout || viewCamera == null) return;
            viewCamera.rect = viewport;
            viewCamera.aspect = Mathf.Max(1, Screen.width * viewport.width) / Mathf.Max(1, Screen.height * viewport.height);
            interfaceCamera.ResetAspect();
            if (mode != ViewMode.FirstPerson) SetView(mode);
        }
        public void RotateVirtualView(float yawDelta, float pitchDelta)
        {
            if (!mobileLayout || mobileSuspended || leaving || mode != ViewMode.FirstPerson) return;
            if (playing) PausePlayback("Paused for virtual view control. Tap Play to resume.");
            yaw += yawDelta; pitch = Mathf.Clamp(pitch + pitchDelta, -45, 65); PositionFirstPerson();
        }
        public void RecenterVirtualView()
        {
            if (activeRoute == null || mode != ViewMode.FirstPerson) return;
            Vector3 direction;
            Vector3 position = activeRoute.PointAtDistance(distanceTravelled, out direction);
            eyePosition = position + Vector3.up * EyeHeight;
            SetHeading(VirtualRouteFacing(position)); pitch = 23; PositionFirstPerson();
        }
        private Vector3 VirtualRouteFacing(Vector3 position)
        {
            // This controls only the model camera. It must never drive an AR camera.
            // A short look-ahead anticipates bends instead of freezing midway through
            // a delayed turn. At a symmetric U-turn, use the upcoming segment tangent.
            Vector3 aheadDirection;
            Vector3 ahead = activeRoute.PointAtDistance(Mathf.Min(activeRoute.Length, distanceTravelled + .35f), out aheadDirection);
            Vector3 facing = ahead - position;
            facing.y = 0;
            return facing.sqrMagnitude > .0025f ? facing.normalized : aheadDirection;
        }
        public void StepVirtualView(float metres)
        {
            if (!mobileLayout || mobileSuspended || leaving || mode != ViewMode.FirstPerson) return;
            if (playing) PausePlayback("Paused for virtual view control. Tap Play to resume.");
            eyePosition += Quaternion.Euler(0, yaw, 0) * Vector3.forward * Mathf.Clamp(metres, -.25f, .25f);
            eyePosition.x = Mathf.Clamp(eyePosition.x, 0, 3.2f); eyePosition.z = Mathf.Clamp(eyePosition.z, 0, 1.6f);
            eyePosition.y = EyeHeight; PositionFirstPerson();
            UpdateCopy("Virtual viewpoint moved. No CSI position was changed.");
        }
        public void PausePlayback(string reason)
        {
            playing = false;
            if (mode == ViewMode.FirstPerson) RecenterVirtualView();
            UpdateCopy(reason);
            if (mobileLayout) Debug.Log("CSI_MOBILE_MODEL_PAUSE: " + reason);
        }
        public void ReturnHome()
        {
            if (!mobileLayout || leaving) return;
            string nextScene = PlanningPreviewTransfer.ReturnToPlanning ? "LivePositioning" : "UiDemo";
            if (!Application.CanStreamedLevelBeLoaded(nextScene)) { UpdateCopy("Return scene is not included in this build."); return; }
            leaving = true; playing = false;
            ReleaseMobileBack();
            Debug.Log("CSI_MOBILE_MODEL_BACK: UiDemo");
            SceneManager.LoadScene(nextScene, LoadSceneMode.Single);
        }
        public void OpenArCamera()
        {
            if (!mobileLayout || leaving) return;
            if (!Application.CanStreamedLevelBeLoaded("ArCalibration")) { UpdateCopy("AR camera scene is not included in this build."); return; }
            leaving = true; playing = false; ReleaseMobileBack();
            Debug.Log("CSI_MOBILE_MODEL_OPEN_AR: real camera; no automatic virtual walk");
            SceneManager.LoadScene("ArCalibration", LoadSceneMode.Single);
        }
        private void OnApplicationPause(bool paused)
        {
            if (!mobileLayout) return;
            mobileSuspended = paused;
            if (paused)
            {
                PausePlayback("Paused in background. Tap Play to resume.");
                if (mobileUi != null) mobileUi.CancelGesture();
                if (mobileBack != null) mobileBack.DiscardPending();
                Debug.Log("CSI_MOBILE_MODEL_BACKGROUND");
            }
            else if (mobileUi != null) mobileUi.Refresh();
        }
        private void OnEnable()
        {
            if (mobileLayout && presentationInitialized && !leaving && mobileBack == null)
                mobileBack = new AndroidPresentationBackHandler();
        }
        private void OnDisable() { ReleaseMobileBack(); }
        private void OnDestroy() { ReleaseMobileBack(); }
        private void ReleaseMobileBack()
        {
            if (mobileBack == null) return;
            mobileBack.Dispose();
            mobileBack = null;
        }
        private void SetHeading(Vector3 direction) { yaw = Mathf.Atan2(direction.x, direction.z) * Mathf.Rad2Deg; }
        private static Vector3 Point(int label, float height = 0)
        { var p = ApartmentMap.Current.FindPoint(label); return ApartmentMap.ToWorld(p.x_m, p.y_m, height); }

        private void UpdateCopy(string message)
        {
            this.message = message;
            if (startText != null) startText.text = start + "  " + GridMap.Index(start);
            if (targetText != null) targetText.text = target + "  " + GridMap.Index(target);
            if (routeText != null)
            {
                var labels = new List<string>(); foreach (int label in path) labels.Add(label.ToString());
                routeText.text = string.Join("  >  ", labels.ToArray()) + "\n\n" + (activeRoute == null ? 0 : activeRoute.Length).ToString("F1") + " m   /   " + (path.Count - 1) + " grid segments";
            }
            if (statusText != null) statusText.text = message;
            if (mobileUi != null) mobileUi.Refresh();
        }

        private void CreateRoomLabels()
        {
            if (roomLabels != null) return;
            roomLabels = new GameObject("Overview room names / display labels").transform;
            roomLabels.SetParent(transform, false);
            foreach (var region in ApartmentMap.Current.regions)
            {
                if (region.model_role == "excluded_unknown") continue;
                string value = region.id == "living_room" ? "LIVING ROOM" : region.id == "corridor_extension" ? "CORRIDOR" : region.id.ToUpperInvariant();
                float x = (region.min_xy_m[0] + region.max_xy_m[0]) * .5f;
                float z = region.id == "living_room" ? region.max_xy_m[1] - .42f : (region.min_xy_m[1] + region.max_xy_m[1]) * .5f;
                var go = new GameObject("Room name / " + value, typeof(TextMesh)); go.transform.SetParent(roomLabels, false);
                go.transform.localPosition = ApartmentMap.ToWorld(x, z, .11f);
                var text = go.GetComponent<TextMesh>(); text.font = font; text.fontSize = 64;
                text.characterSize = (region.id == "corridor_extension" ? .21f : .28f) * 10 / text.fontSize;
                text.text = value; text.anchor = TextAnchor.MiddleCenter; text.alignment = TextAlignment.Center;
                text.fontStyle = FontStyle.Bold; text.color = new Color32(40, 72, 66, 255);
                var renderer = go.GetComponent<MeshRenderer>(); renderer.sharedMaterial = font.material;
                renderer.shadowCastingMode = ShadowCastingMode.Off; renderer.receiveShadows = false;
                roomTexts.Add(text);
            }
        }
        private void FaceRoomLabels()
        {
            if (viewCamera == null) return;
            foreach (var label in roomTexts) if (label != null) label.transform.rotation = viewCamera.transform.rotation;
        }
        private void SetArchitectureVisibility(ViewMode value)
        {
            if (architecture == null) return;
            foreach (var renderer in architecture.GetComponentsInChildren<MeshRenderer>(true))
            {
                // A display cutaway, not a changed survey or claim that exterior walls are absent.
                bool balconyExterior = renderer.name == "Wall_B01" || renderer.name == "Wall_B02" || renderer.name == "Wall_B03";
                renderer.enabled = !(value == ViewMode.FirstPerson && balconyExterior);
            }
        }

        private void BuildInterface()
        {
            if (interfaceRoot != null)
            {
                interfaceRoot.SetActive(false);
                if (Application.isPlaying) Destroy(interfaceRoot); else DestroyImmediate(interfaceRoot);
            }
            interfaceRoot = new GameObject("Presentation interface / simulation labels", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            interfaceRoot.layer = 5;
            interfaceRoot.transform.SetParent(transform, false);
            var canvas = interfaceRoot.GetComponent<Canvas>(); canvas.renderMode = RenderMode.ScreenSpaceCamera;
            canvas.worldCamera = interfaceCamera; canvas.planeDistance = .6f;
            var scaler = interfaceRoot.GetComponent<CanvasScaler>(); scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1600, 1000); scaler.matchWidthOrHeight = .5f;
            if (Application.isPlaying)
            {
                var events = GetComponentInChildren<EventSystem>();
                if (events == null)
                {
                    events = new GameObject("Presentation input events", typeof(EventSystem), typeof(StandaloneInputModule)).GetComponent<EventSystem>();
                    events.transform.SetParent(transform, false);
                }
                // Space/arrows belong to the virtual camera/playback, not to uGUI Submit/Move.
                events.sendNavigationEvents = false;
            }
            RectTransform root = interfaceRoot.GetComponent<RectTransform>();
            var navy = new Color32(20, 39, 48, 255); var paper = new Color32(245, 247, 246, 249);
            Panel(root, "View controls backing", 0, 112, 1230, 144, new Color32(237, 242, 239, 255));
            Panel(root, "Assumptions backing", 0, 828, 1230, 66, new Color32(237, 242, 239, 255));
            Panel(root, "Header", 0, 0, 1600, 112, navy);
            Label(root, "CSI APARTMENT", 36, 16, 900, 42, 33, Color.white, FontStyle.Bold);
            Label(root, "Measured plan  /  Blender architecture  /  Unity route simulation", 38, 65, 1050, 28, 19, new Color32(183, 209, 206, 255));
            Panel(root, "Simulation badge", 1220, 29, 340, 48, new Color32(247, 199, 104, 255));
            Label(root, "SIMULATION  /  NO LIVE CSI", 1235, 38, 310, 30, 18, navy, FontStyle.Bold);
            Panel(root, "Control panel", 1230, 132, 346, 742, paper);
            Label(root, "ROUTE STUDIO", 1252, 151, 300, 36, 25, navy, FontStyle.Bold);
            Label(root, "15 points  /  22 four-neighbor edges", 1252, 192, 300, 30, 17, new Color32(83, 107, 111, 255));
            Label(root, "START GRID", 1252, 245, 300, 26, 15, navy, FontStyle.Bold);
            startText = Label(root, "", 1303, 277, 200, 37, 24, navy, FontStyle.Bold);
            Button(root, "-", 1252, 276, 40, 39, () => ConfigureRoute((start + 14) % 15, target));
            Button(root, "+", 1510, 276, 40, 39, () => ConfigureRoute((start + 1) % 15, target));
            Label(root, "TARGET GRID", 1252, 341, 300, 26, 15, navy, FontStyle.Bold);
            targetText = Label(root, "", 1303, 373, 200, 37, 24, navy, FontStyle.Bold);
            Button(root, "-", 1252, 372, 40, 39, () => ConfigureRoute(start, (target + 14) % 15));
            Button(root, "+", 1510, 372, 40, 39, () => ConfigureRoute(start, (target + 1) % 15));
            routeText = Label(root, "", 1252, 444, 300, 100, 21, navy);
            Button(root, "PLAY / PAUSE", 1252, 556, 300, 54, TogglePlayback, true);
            Button(root, "RESET ROUTE", 1252, 623, 300, 44, () => ConfigureRoute(start, target));
            Label(root, "BALCONY TARGET UNAVAILABLE", 1252, 705, 300, 40, 17, new Color32(150, 92, 37, 255), FontStyle.Bold);
            Label(root, "T01 and the exit-grid connection\nare still unconfigured.", 1252, 751, 300, 64, 18, new Color32(83, 107, 111, 255));
            viewText = Label(root, "", 38, 140, 1130, 38, 23, navy, FontStyle.Bold);
            Button(root, "1  OVERVIEW", 38, 198, 188, 44, () => SetView(ViewMode.Overview));
            Button(root, "2  TOP VIEW", 238, 198, 188, 44, () => SetView(ViewMode.Plan));
            Button(root, "3  FIRST PERSON", 438, 198, 220, 44, () => SetView(ViewMode.FirstPerson));
            Panel(root, "Footer", 0, 894, 1600, 106, navy);
            statusText = Label(root, "", 36, 907, 1525, 32, 20, Color.white);
            Label(root, "WASD move in grid  |  Hold right mouse / arrow keys to look  |  Space play  |  R reset", 36, 944, 1520, 25, 17, new Color32(183, 209, 206, 255));
            Label(root, "Display heights only: walls 2.4 m / cabinets 1.1 m / eye 1.5 m. Plan dimensions are survey based.", 38, 845, 1160, 35, 17, new Color32(70, 95, 99, 255));
            Canvas.ForceUpdateCanvases();
        }

        private static RectTransform Box(Transform parent, string name, float x, float y, float width, float height)
        {
            var go = new GameObject(name, typeof(RectTransform)); go.layer = 5; var r = go.GetComponent<RectTransform>(); r.SetParent(parent, false);
            r.anchorMin = r.anchorMax = new Vector2(0, 1); r.pivot = new Vector2(0, 1); r.anchoredPosition = new Vector2(x, -y); r.sizeDelta = new Vector2(width, height); return r;
        }
        private static void Panel(Transform parent, string name, float x, float y, float w, float h, Color color)
        { var image = Box(parent, name, x, y, w, h).gameObject.AddComponent<Image>(); image.color = color; image.raycastTarget = false; }
        private Text Label(Transform parent, string text, float x, float y, float w, float h, int size, Color color, FontStyle style = FontStyle.Normal)
        {
            var t = Box(parent, "Text / " + text, x, y, w, h).gameObject.AddComponent<Text>(); t.font = font; t.text = text;
            t.fontSize = size; t.color = color; t.fontStyle = style; t.raycastTarget = false; t.alignment = TextAnchor.UpperLeft;
            t.horizontalOverflow = HorizontalWrapMode.Wrap; t.verticalOverflow = VerticalWrapMode.Truncate; return t;
        }
        private void Button(Transform parent, string text, float x, float y, float w, float h, UnityEngine.Events.UnityAction action, bool primary = false)
        {
            var r = Box(parent, "Button / " + text, x, y, w, h); var image = r.gameObject.AddComponent<Image>();
            image.color = primary ? new Color32(0, 121, 110, 255) : new Color32(223, 232, 230, 255);
            var button = r.gameObject.AddComponent<Button>(); button.targetGraphic = image; button.onClick.AddListener(action);
            var label = Label(r, text, 0, 0, w, h, 17, primary ? Color.white : new Color32(20, 61, 61, 255), FontStyle.Bold); label.alignment = TextAnchor.MiddleCenter;
        }
    }
}
