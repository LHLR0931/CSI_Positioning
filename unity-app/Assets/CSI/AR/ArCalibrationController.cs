using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.SpatialTracking;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using UnityEngine.XR.Management;
#if UNITY_ANDROID
using UnityEngine.Android;
#endif

namespace CsiApp
{
    /// <summary>Real ARCore meeting demo. Image-local placement is NOT measured site registration.</summary>
    public sealed class ArCalibrationController : MonoBehaviour
    {
        public string referenceImageName = "CSI Reference";
        public float defaultImageWidthMeters = .10f;
        private readonly ArCalibrationProfile profile = new ArCalibrationProfile();
        private XRManagerSettings xr;
        private GameObject runtimeRoot;
        private ARSession session;
        private ARTrackedImageManager images;
        private ARAnchorManager anchors;
        private ARAnchor anchor;
        private ARTrackedImage candidate;
        private Camera arCamera, fallbackCamera;
        private ArMeetingModel model;
        private AndroidPresentationBackHandler backHandler;
        private ArMeetingView meetingView = ArMeetingView.Miniature;
        private DemoNavigationRoute route;
        private int startLabel, gridTarget = 14;
        private string destinationId = "balcony";
        private string startPlace = "grid";
        private float simulatedDistance, demoMarkerHeight = 1.2f;
        private bool simPlaying;
        private MutableRuntimeReferenceImageLibrary library;
        private Texture2D referenceTexture;
        private AddReferenceImageJobState imageJob;
        private Guid imageGuid;
        private int generation, imageJobGeneration;
        private bool booting = true, ready, imageJobRunning, anchorPending, locked, needsRescan;
        private bool paused, exiting, cleaned, widthDirty;
        private string notice = "Starting AR camera...";
        private Font font;
        private RectTransform safe, page;
        private Rect lastSafe;
        private Vector2 lastScreen;
        private Text statusText, markerText, routeText, startText, targetText, playText, lockText, modeText, distanceText;
        private InputField widthInput, heightInput;
        private Button applyButton, lockButton, rescanButton, backButton, playButton;
        private GameObject drawer;
        private readonly Dictionary<string, Button> destinationButtons = new Dictionary<string, Button>();
        private static readonly Color Ink = new Color(.08f, .20f, .18f);
        private static readonly Color Teal = new Color(0, .46f, .38f);

        private IEnumerator Start()
        {
            Screen.orientation = ScreenOrientation.Portrait;
            BuildUi();
            ChangeRoute();
            backHandler = new AndroidPresentationBackHandler();
#if UNITY_ANDROID && !UNITY_EDITOR
            yield return GuardStartup(StartAndroid());
#else
            notice = "Camera AR requires an Android device. Use Home to return.";
#endif
            booting = false;
            yield break;
        }

#if UNITY_ANDROID && !UNITY_EDITOR
        // Drive nested startup iterators here so an exception in a provider's
        // coroutine cannot leave the screen permanently waiting with Back disabled.
        private IEnumerator GuardStartup(IEnumerator startup)
        {
            var pending = new Stack<IEnumerator>();
            pending.Push(startup);
            try
            {
                while (pending.Count > 0)
                {
                    var current = pending.Peek();
                    object instruction = null;
                    Exception failure = null;
                    bool hasNext = false;
                    try
                    {
                        hasNext = current.MoveNext();
                        if (hasNext) instruction = current.Current;
                    }
                    catch (Exception e) { failure = e; }
                    if (failure != null)
                    {
                        ready = false;
                        notice = "AR initialization failed: " + failure.Message + ". Use Back to return.";
                        Debug.LogException(failure);
                        break;
                    }
                    if (!hasNext)
                    {
                        pending.Pop();
                        (current as IDisposable)?.Dispose();
                    }
                    else if (instruction is IEnumerator nested) pending.Push(nested);
                    else yield return instruction;
                }
            }
            finally
            {
                booting = false;
                while (pending.Count > 0)
                {
                    try { (pending.Pop() as IDisposable)?.Dispose(); }
                    catch (Exception e) { Debug.LogException(e); }
                }
            }
        }

        private IEnumerator StartAndroid()
        {
            if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
            {
                bool answered = false;
                var callbacks = new PermissionCallbacks();
                callbacks.PermissionGranted += _ => answered = true;
                callbacks.PermissionDenied += _ => answered = true;
                callbacks.PermissionDeniedAndDontAskAgain += _ => answered = true;
                notice = "Allow camera access for the AR meeting demo.";
                Permission.RequestUserPermission(Permission.Camera, callbacks);
                while (!answered) yield return null;
            }
            if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
            { notice = "Camera permission denied. Enable it in Android settings, then reopen this lab."; yield break; }
            xr = XRGeneralSettings.Instance == null ? null : XRGeneralSettings.Instance.Manager;
            if (xr == null) { notice = "XR settings are missing. Back remains available."; yield break; }
            notice = "Initializing the Android AR provider...";
            yield return xr.InitializeLoader();
            if (xr.activeLoader == null) { notice = "ARCore provider could not initialize on this device."; yield break; }
            yield return ARSession.CheckAvailability();
            if (ARSession.state == ARSessionState.NeedsInstall)
            {
                notice = "Google Play Services for AR needs installation or update.";
                yield return ARSession.Install();
            }
            if (ARSession.state != ARSessionState.Ready && ARSession.state != ARSessionState.SessionInitializing
                && ARSession.state != ARSessionState.SessionTracking)
            { notice = "AR is unavailable: " + ARSession.state + ". Back remains available."; yield break; }
            try
            {
                xr.StartSubsystems();
                CreateRuntime();
                referenceTexture = Resources.Load<Texture2D>("AR/CSI_Reference");
                if (referenceTexture == null || !referenceTexture.isReadable)
                    throw new InvalidOperationException("Reference image missing or not readable. Rebuild its import settings.");
                ready = true;
                notice = "Open Controls, check the actual image width, then Apply & scan.";
            }
            catch (Exception e) { notice = "AR setup failed: " + e.Message; Debug.LogException(e); }
        }
#endif

        private void CreateRuntime()
        {
            runtimeRoot = new GameObject("AR lab runtime"); runtimeRoot.SetActive(false);
            runtimeRoot.transform.SetParent(transform, false);
            var sessionObject = new GameObject("AR Session", typeof(ARSession), typeof(ARInputManager));
            sessionObject.transform.SetParent(runtimeRoot.transform, false);
            session = sessionObject.GetComponent<ARSession>(); session.attemptUpdate = false;
            var originObject = new GameObject("XR Origin"); originObject.SetActive(false);
            originObject.transform.SetParent(runtimeRoot.transform, false);
            var offset = new GameObject("Camera Offset"); offset.transform.SetParent(originObject.transform, false);
            var cameraObject = new GameObject("AR Camera", typeof(Camera));
            cameraObject.transform.SetParent(offset.transform, false); cameraObject.tag = "MainCamera";
            arCamera = cameraObject.GetComponent<Camera>();
            arCamera.nearClipPlane = .01f; arCamera.farClipPlane = 20;
            arCamera.clearFlags = CameraClearFlags.SolidColor; arCamera.backgroundColor = Color.black;
            var pose = cameraObject.AddComponent<TrackedPoseDriver>();
            pose.SetPoseSource(TrackedPoseDriver.DeviceType.GenericXRDevice, TrackedPoseDriver.TrackedPose.ColorCamera);
            pose.trackingType = TrackedPoseDriver.TrackingType.RotationAndPosition;
            pose.updateType = TrackedPoseDriver.UpdateType.UpdateAndBeforeRender;
            var cameraManager = cameraObject.AddComponent<ARCameraManager>();
            cameraManager.requestedFacingDirection = CameraFacingDirection.World;
            cameraManager.autoFocusRequested = true;
            cameraObject.AddComponent<ARCameraBackground>();
            var origin = originObject.AddComponent<XROrigin>();
            origin.Camera = arCamera; origin.CameraFloorOffsetObject = offset;
            origin.CameraYOffset = 0; origin.RequestedTrackingOriginMode = XROrigin.TrackingOriginMode.Device;
            anchors = originObject.AddComponent<ARAnchorManager>();
            images = originObject.AddComponent<ARTrackedImageManager>(); images.enabled = false;
            images.requestedMaxNumberOfMovingImages = 1;
            originObject.SetActive(true); runtimeRoot.SetActive(true);
            fallbackCamera.enabled = false;
        }

        private bool SessionUsable()
        { return ARSession.state == ARSessionState.SessionInitializing || ARSession.state == ARSessionState.SessionTracking; }

        private void Update()
        {
            if (page == null || exiting) return;
            if ((backHandler != null && backHandler.ConsumeBackRequest()) || Input.GetKeyDown(KeyCode.Escape)) { HandleBack(); return; }
            if (lastSafe != Screen.safeArea || lastScreen != new Vector2(Screen.width, Screen.height)) FitSafeArea();
            if (imageJobRunning && imageJob.jobHandle.IsCompleted) FinishImageJob();
            candidate = null;
            if (ready && images != null && images.enabled && !paused && !needsRescan)
                foreach (var image in images.trackables)
                    if (image.referenceImage.guid == imageGuid && ValidImage(image)) { candidate = image; break; }
            if (locked && (paused || ARSession.state != ARSessionState.SessionTracking || anchor == null
                || anchor.trackingState != TrackingState.Tracking))
                InvalidateLock("Tracking interrupted. Model hidden; tap Rescan, then Lock model again.");
            bool idle = ready && !booting && !paused && !imageJobRunning && !anchorPending;
            applyButton.interactable = idle && SessionUsable();
            rescanButton.interactable = idle && SessionUsable();
            widthInput.interactable = !booting && !imageJobRunning && !anchorPending;
            heightInput.interactable = idle && meetingView == ArMeetingView.FullSizeOverlay;
            lockButton.interactable = idle && !locked && !needsRescan && !widthDirty && candidate != null
                && ARSession.state == ARSessionState.SessionTracking;
            backButton.interactable = !booting && !anchorPending;
            playButton.interactable = locked && !paused && !anchorPending && route != null && route.Length > .001f;
            if (locked && simPlaying && model != null)
            {
                simulatedDistance = Mathf.Min(route.Length, simulatedDistance + Time.deltaTime * .55f);
                model.SetDistance(simulatedDistance);
                if (simulatedDistance >= route.Length)
                {
                    simPlaying = false;
                    notice = "Simulated marker finished. This is not measured arrival.";
                    Debug.Log("AR_MEETING_SIM_COMPLETE: " + destinationId);
                }
            }
            playText.text = simPlaying ? "Pause marker" : "Play marker";
            lockText.text = locked ? "Locked" : "Lock overlay";
            markerText.text = locked ? (meetingView == ArMeetingView.Miniature ? "ANCHOR TRACKING / MODEL 1:20" : "ANCHOR TRACKING / FULL-SIZE DEMO") : candidate != null
                ? "IMAGE TRACKING / ready to lock" : needsRescan ? "RESCAN REQUIRED / model hidden" : "IMAGE NOT LOCKED";
            string imageStatus = candidate == null ? "Marker: not currently tracked" : "Marker: Tracking / "
                + candidate.size.x.ToString("F3") + " x " + candidate.size.y.ToString("F3") + " m";
            statusText.text = notice;
            distanceText.text = locked ? "SIM MARKER " + simulatedDistance.ToString("F1") + " / " + route.Length.ToString("F1")
                + " m  |  " + route.InstructionAtDistance(simulatedDistance) : "Session: " + ARSession.state + " / " + imageStatus;
        }

        private void LateUpdate()
        { if (locked && model != null && arCamera != null) model.FaceLabels(arCamera); }

        private bool ValidImage(ARTrackedImage image)
        {
            if (image.trackingState != TrackingState.Tracking || referenceTexture == null) return false;
            float height = profile.ImageWidthMeters * referenceTexture.height / referenceTexture.width;
            return Finite(image.size.x) && Finite(image.size.y) && image.size.x > 0 && image.size.y > 0
                && Mathf.Abs(image.size.x - profile.ImageWidthMeters) <= profile.ImageWidthMeters * .15f
                && Mathf.Abs(image.size.y - height) <= height * .15f;
        }
        private static bool Finite(float value) { return !float.IsNaN(value) && !float.IsInfinity(value); }

        private void ApplyWidthAndScan()
        {
            if (!ready || imageJobRunning || anchorPending || paused || !SessionUsable()) return;
            float width;
            if (!float.TryParse(widthInput.text.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out width)
                || !Finite(width) || width < .03f || width > 5)
            { notice = "Enter a width from 0.03 to 5 metres, using a decimal point."; return; }
            widthInput.DeactivateInputField();
            generation++; ClearAnchor(); images.enabled = false; candidate = null; needsRescan = false;
            profile.ImageWidthMeters = width; widthDirty = false;
            PlayerPrefs.SetFloat("CSI_AR_IMAGE_WIDTH_M", width); PlayerPrefs.Save();
            try
            {
                library = images.CreateRuntimeLibrary() as MutableRuntimeReferenceImageLibrary;
                if (library == null) throw new NotSupportedException("Provider does not support runtime image widths.");
                imageJob = library.ScheduleAddImageWithValidationJob(referenceTexture, referenceImageName, width);
                imageJobRunning = true; imageJobGeneration = generation;
                notice = "Validating reference image at " + width.ToString("F3") + " m...";
                drawer.SetActive(false);
            }
            catch (Exception e) { needsRescan = true; notice = "Image setup failed: " + e.Message; Debug.LogException(e); }
        }

        private void FinishImageJob()
        {
            imageJob.jobHandle.Complete(); imageJobRunning = false;
            if (imageJobGeneration != generation || paused || exiting) return;
            if (imageJob.status != AddReferenceImageJobStatus.Success || library == null || library.count != 1)
            { needsRescan = true; notice = "Reference image rejected: " + imageJob.status + ". Use a clearer image and rescan."; return; }
            imageGuid = library[0].guid;
            images.referenceLibrary = library; images.enabled = true;
            notice = "Aim at the full reference image. Wait for Tracking, then tap Lock model.";
        }

        private async void LockModel()
        {
            if (anchorPending || locked || exiting || !lockButton.interactable || candidate == null) return;
            if (meetingView == ArMeetingView.FullSizeOverlay &&
                (!float.TryParse(heightInput.text.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out demoMarkerHeight)
                || !Finite(demoMarkerHeight) || demoMarkerHeight < .1f || demoMarkerHeight > 3f))
            { notice = "Demo marker-to-floor height must be 0.10 to 3.00 m. This is not a surveyed site pose."; drawer.SetActive(true); return; }
            heightInput.DeactivateInputField();
            int requestGeneration = generation;
            Pose markerPose = new Pose(candidate.transform.position, candidate.transform.rotation);
            anchorPending = true; notice = "Creating an AR anchor...";
            ARAnchor addedAnchor = null;
            try
            {
                var result = await anchors.TryAddAnchorAsync(markerPose);
                if (!result.status.IsSuccess()) throw new InvalidOperationException("Anchor rejected: " + result.status);
                addedAnchor = result.value;
                float deadline = Time.realtimeSinceStartup + 3;
                while (!exiting && requestGeneration == generation && addedAnchor != null
                    && addedAnchor.trackingState != TrackingState.Tracking && Time.realtimeSinceStartup < deadline)
                    await Awaitable.NextFrameAsync();
                if (exiting || this == null || requestGeneration != generation || paused) return;
                if (addedAnchor == null || addedAnchor.trackingState != TrackingState.Tracking
                    || ARSession.state != ARSessionState.SessionTracking)
                    throw new InvalidOperationException("Anchor is not Tracking. Rescan and retry.");
                anchor = addedAnchor; addedAnchor = null;
                var modelObject = new GameObject("MEETING apartment / demo placement / no site alignment");
                model = modelObject.AddComponent<ArMeetingModel>();
                ArMeetingPlacement.Place(modelObject.transform, anchor.transform, arCamera.transform.forward, meetingView, demoMarkerHeight);
                model.Initialize(meetingView, route, route.GridLabels[0]); model.FaceLabels(arCamera);
                simulatedDistance = 0; simPlaying = false;
                locked = true;
                drawer.SetActive(false);
                notice = meetingView == ArMeetingView.Miniature ? "1:20 model locked. Move the real phone to inspect; Play moves only the blue marker."
                    : "1:1 translucent model DEMO. Floor height is assumed; +Z faces scanner. No site alignment.";
                Debug.Log("AR_MEETING_LOCKED: view=" + meetingView + "; scale=" + modelObject.transform.localScale.x
                    + "; demo_marker_height=" + demoMarkerHeight + "; site_alignment=false; destination=" + destinationId);
            }
            catch (Exception e)
            { if (!exiting && this != null) { InvalidateLock("Lock failed: " + e.Message); Debug.LogException(e); } }
            finally
            {
                if (addedAnchor != null && anchors != null && anchors.enabled && anchors.subsystem != null)
                    anchors.TryRemoveAnchor(addedAnchor);
                anchorPending = false;
            }
        }

        private void InvalidateLock(string message)
        {
            generation++; locked = false; needsRescan = true; candidate = null;
            simPlaying = false;
            if (model != null) model.gameObject.SetActive(false);
            if (images != null) images.enabled = false;
            notice = message;
        }
        private void ClearAnchor()
        {
            locked = false;
            simPlaying = false; simulatedDistance = 0;
            if (model != null) { model.gameObject.SetActive(false); Destroy(model.gameObject); model = null; }
            if (anchor != null)
            {
                if (anchors != null && anchors.enabled && !anchors.TryRemoveAnchor(anchor)) Destroy(anchor.gameObject);
                anchor = null;
            }
        }
        private void OnApplicationPause(bool value)
        {
            paused = value;
            if (value && backHandler != null) backHandler.DiscardPending();
            if (ready && value) InvalidateLock("App paused. Tap Rescan and lock again after returning.");
            // Keep ARSession and the camera active in the hierarchy: ARSession forwards
            // these callbacks to ARCore. Disabling the root also stops its subsystems
            // and can interfere with the provider's own pause/resume sequence.
        }
        private void GoBack()
        {
            if (booting || anchorPending || exiting) return;
            Debug.Log("AR_MEETING_BACK: returning to Home");
            exiting = true; Cleanup(); SceneManager.LoadScene("UiDemo");
        }
        private void HandleBack()
        {
            if ((widthInput != null && widthInput.isFocused) || (heightInput != null && heightInput.isFocused))
            {
                EndInputEditing();
                Debug.Log("AR_MEETING_BACK: input editing ended; values retained");
                return;
            }
            if (drawer != null && drawer.activeSelf)
            {
                EndInputEditing(); drawer.SetActive(false);
                Debug.Log("AR_MEETING_BACK: controls closed");
                return;
            }
            GoBack();
        }
        private void EndInputEditing()
        {
            if (widthInput != null) widthInput.DeactivateInputField();
            if (heightInput != null) heightInput.DeactivateInputField();
            if (EventSystem.current != null) EventSystem.current.SetSelectedGameObject(null);
        }
        private void Cleanup()
        {
            if (cleaned) return; cleaned = true; generation++;
            if (backHandler != null) { backHandler.Dispose(); backHandler = null; }
            if (imageJobRunning) { imageJob.jobHandle.Complete(); imageJobRunning = false; }
            ClearAnchor();
            if (runtimeRoot != null) { runtimeRoot.SetActive(false); Destroy(runtimeRoot); }
            if (xr != null && xr.isInitializationComplete) { xr.StopSubsystems(); xr.DeinitializeLoader(); }
            library = null;
        }
        private void OnDestroy() { exiting = true; Cleanup(); }

        private void BuildUi()
        {
            font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            var background = new GameObject("AR unavailable background", typeof(Camera));
            background.transform.SetParent(transform, false); fallbackCamera = background.GetComponent<Camera>();
            fallbackCamera.clearFlags = CameraClearFlags.SolidColor; fallbackCamera.backgroundColor = new Color(.06f, .11f, .13f);
            fallbackCamera.cullingMask = 0;
            var canvasObject = new GameObject("AR meeting canvas", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            canvasObject.transform.SetParent(transform, false);
            var canvas = canvasObject.GetComponent<Canvas>(); canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasObject.GetComponent<CanvasScaler>(); scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(390, 844); scaler.matchWidthOrHeight = 0;
            canvas.scaleFactor = Mathf.Max(1, Screen.width) / 390f;
            var events = new GameObject("AR meeting events", typeof(EventSystem), typeof(StandaloneInputModule));
            events.transform.SetParent(transform, false); events.GetComponent<EventSystem>().sendNavigationEvents = false;
            safe = new GameObject("Safe Area", typeof(RectTransform)).GetComponent<RectTransform>(); safe.SetParent(canvasObject.transform, false);
            page = new GameObject("AR meeting layout", typeof(RectTransform)).GetComponent<RectTransform>(); page.SetParent(safe, false);
            page.anchorMin = Vector2.zero; page.anchorMax = Vector2.one; page.offsetMin = page.offsetMax = Vector2.zero;
            Panel("Header", 0, 0, 390, 113);
            backButton = Button("< Home", 12, 8, 78, 40, GoBack);
            Label("AR WAYFINDING", 102, 13, 276, 28, 21, true);
            Label("CAMERA AR / DEMO PLACEMENT / NO LIVE CSI", 14, 56, 362, 20, 12, true);
            markerText = Label("IMAGE NOT LOCKED", 14, 83, 362, 22, 13, true);

            var footer = BottomGroup("AR action card", 150, 0);
            Panel("Footer backing", 0, 0, 390, 150, footer);
            statusText = Label("Starting AR...", 14, 12, 362, 43, 13, false, footer);
            lockButton = Button("Lock overlay", 14, 62, 112, 44, LockModel, footer); lockText = lockButton.GetComponentInChildren<Text>();
            playButton = Button("Play marker", 136, 62, 112, 44, ToggleMarkerPlayback, footer); playText = playButton.GetComponentInChildren<Text>();
            Button("Controls", 258, 62, 118, 44, ToggleDrawer, footer);
            distanceText = Label("", 14, 116, 362, 28, 11, false, footer);

            var controls = BottomGroup("Controls drawer", 396, 158); drawer = controls.gameObject;
            Panel("Drawer backing", 0, 0, 390, 396, controls);
            Label("SETUP & ROUTE", 14, 12, 230, 24, 16, true, controls);
            Button("Close", 296, 6, 80, 36, ToggleDrawer, controls);
            Button("1:20 model", 14, 47, 176, 40, () => ChangeView(ArMeetingView.Miniature), controls);
            Button("Full-size overlay", 200, 47, 176, 40, () => ChangeView(ArMeetingView.FullSizeOverlay), controls);
            modeText = Label("1:20 model / horizontal demo anchor", 14, 93, 362, 21, 11, true, controls);
            Label("Actual image width (m)", 14, 120, 176, 19, 12, true, controls);
            Label("DEMO marker height (m)", 200, 120, 176, 19, 12, true, controls);
            widthInput = NumberInput("Measured image width", 14, 143, 176, controls,
                PlayerPrefs.GetFloat("CSI_AR_IMAGE_WIDTH_M", .10f).ToString("F3", CultureInfo.InvariantCulture));
            heightInput = NumberInput("Illustrative marker height", 200, 143, 176, controls, "1.20");
            widthInput.onValueChanged.AddListener(_ => { widthDirty = true; ClearAnchor(); notice = "Width changed. Measure the full image, then Apply & scan."; });
            heightInput.onValueChanged.AddListener(_ => { if (meetingView == ArMeetingView.FullSizeOverlay) { ClearAnchor(); notice = "Demo height changed. Lock again; this is not site calibration."; } });
            applyButton = Button("Apply & scan", 14, 194, 176, 42, ApplyWidthAndScan, controls);
            rescanButton = Button("Rescan", 200, 194, 176, 42, ApplyWidthAndScan, controls);
            Button("From", 14, 242, 49, 36, CycleStartPlace, controls);
            Button("-", 69, 242, 35, 36, () => StepStart(-1), controls);
            startText = Label("0", 108, 249, 34, 24, 15, true, controls); startText.alignment = TextAnchor.UpperCenter;
            Button("+", 146, 242, 35, 36, () => StepStart(1), controls);
            Label("GRID", 200, 249, 44, 21, 11, true, controls);
            Button("-", 245, 242, 35, 36, () => StepGrid(-1), controls);
            targetText = Label("14", 283, 249, 42, 24, 15, true, controls); targetText.alignment = TextAnchor.UpperCenter;
            Button("+", 337, 242, 39, 36, () => StepGrid(1), controls);
            string[] ids = { "balcony", "office", "kitchen", "grid" };
            string[] names = { "Balcony", "Office", "Kitchen", "Grid" };
            for (int i = 0; i < ids.Length; i++)
            {
                string id = ids[i];
                destinationButtons[id] = Button(names[i], 14 + i * 93, 288, 83, 38, () => SelectDestination(id), controls);
            }
            routeText = Label("", 14, 335, 256, 46, 11, false, controls);
            Button("Reset marker", 276, 340, 100, 40, ResetMarker, controls);
            applyButton.interactable = lockButton.interactable = rescanButton.interactable = backButton.interactable = playButton.interactable = false;
            FitSafeArea();
        }

        private RectTransform BottomGroup(string name, float height, float bottom)
        {
            var rt = new GameObject(name, typeof(RectTransform)).GetComponent<RectTransform>(); rt.SetParent(page, false);
            rt.anchorMin = rt.anchorMax = new Vector2(.5f, 0); rt.pivot = new Vector2(.5f, 0);
            rt.sizeDelta = new Vector2(390, height); rt.anchoredPosition = new Vector2(0, bottom); return rt;
        }
        private InputField NumberInput(string name, float x, float y, float width, Transform parent, string value)
        {
            var rt = Panel(name, x, y, width, 42, parent); rt.GetComponent<Image>().color = Color.white;
            var input = rt.gameObject.AddComponent<InputField>(); input.targetGraphic = rt.GetComponent<Image>();
            input.textComponent = Label("", 10, 0, width - 20, 42, 19, false, rt); input.textComponent.alignment = TextAnchor.MiddleLeft;
            input.contentType = InputField.ContentType.DecimalNumber; input.text = value; return input;
        }
        private void ToggleDrawer()
        {
            EndInputEditing();
            drawer.SetActive(!drawer.activeSelf);
        }
        private void ChangeView(ArMeetingView value)
        {
            if (anchorPending || exiting || meetingView == value) return;
            ClearAnchor(); meetingView = value;
            modeText.text = value == ArMeetingView.Miniature ? "1:20 model / horizontal demo anchor"
                : "1:1 DEMO / floor below image / +Z toward scanner";
            notice = value == ArMeetingView.Miniature ? "Miniature selected. Scan if needed, then lock again."
                : "Full-size DEMO: enter marker-to-floor height, then lock. Not site registration.";
            Debug.Log("AR_MEETING_MODE: " + value + "; site_alignment=false");
        }
        private void StepStart(int delta)
        {
            if (anchorPending || exiting) return;
            startPlace = "grid";
            startLabel = (startLabel + 15 + delta) % 15; ChangeRoute();
        }
        private void CycleStartPlace()
        {
            if (anchorPending || exiting) return;
            string[] ids = { "grid", "balcony", "office", "kitchen" };
            startPlace = ids[(Array.IndexOf(ids, startPlace) + 1) % ids.Length]; ChangeRoute();
        }
        private void StepGrid(int delta)
        {
            if (anchorPending || exiting) return;
            gridTarget = (gridTarget + 15 + delta) % 15; destinationId = "grid"; ChangeRoute();
        }
        private void SelectDestination(string id)
        {
            if (anchorPending || exiting) return;
            destinationId = id; ChangeRoute();
        }
        private void ChangeRoute()
        {
            try
            {
                route = startPlace == "grid" ? DemoNavigationRoute.Build(startLabel, destinationId, gridTarget)
                    : DemoNavigationRoute.BuildFromPlace(startPlace, destinationId, gridTarget);
                simPlaying = false; simulatedDistance = 0;
                if (locked && model != null) model.SetRoute(route, route.GridLabels[0]);
                startText.text = startPlace == "grid" ? startLabel.ToString() : startPlace.Substring(0, 1).ToUpperInvariant();
                targetText.text = gridTarget.ToString();
                routeText.text = route.StartName + " > " + route.DestinationName + "\n"
                    + route.Length.ToString("F1") + " m / " + (route.UsesIllustrativeConnector ? "AMBER: DEMO approach" : "Four-neighbor simulation");
                foreach (var item in destinationButtons) item.Value.targetGraphic.color = item.Key == destinationId ? Teal : new Color(.32f, .46f, .42f);
                if (!booting) notice = "Route updated. Play moves the blue simulated marker only.";
                Debug.Log("AR_MEETING_ROUTE: " + route.StartName + " -> " + destinationId + "; connector=" + route.ConnectorLabel + "; illustrative=" + route.UsesIllustrativeConnector);
            }
            catch (Exception exception) { notice = "Route unavailable: " + exception.Message; Debug.LogException(exception); }
        }
        private void ToggleMarkerPlayback()
        {
            if (!locked || paused || model == null || route == null || route.Length < .001f) return;
            if (simulatedDistance >= route.Length) simulatedDistance = 0;
            simPlaying = !simPlaying;
            notice = simPlaying ? "Blue marker is simulated. Move the real phone yourself; no live CSI."
                : "Simulated marker paused. The camera remains live.";
            Debug.Log(simPlaying ? "AR_MEETING_SIM_PLAY" : "AR_MEETING_SIM_PAUSE");
        }
        private void ResetMarker()
        {
            simPlaying = false; simulatedDistance = 0;
            if (model != null) model.SetDistance(0);
            notice = "Simulated marker reset. Camera position unchanged.";
            Debug.Log("AR_MEETING_SIM_RESET");
        }
        private void FitSafeArea()
        {
            lastSafe = Screen.safeArea; lastScreen = new Vector2(Screen.width, Screen.height);
            safe.anchorMin = new Vector2(lastSafe.xMin / Screen.width, lastSafe.yMin / Screen.height);
            safe.anchorMax = new Vector2(lastSafe.xMax / Screen.width, lastSafe.yMax / Screen.height);
            safe.offsetMin = safe.offsetMax = Vector2.zero; Canvas.ForceUpdateCanvases();
        }
        private RectTransform Place(GameObject go, Transform parent, float x, float y, float w, float h)
        {
            var rt = go.GetComponent<RectTransform>(); rt.SetParent(parent, false);
            rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(0, 1);
            rt.anchoredPosition = new Vector2(x, -y); rt.sizeDelta = new Vector2(w, h); return rt;
        }
        private RectTransform Panel(string name, float x, float y, float w, float h, Transform parent = null)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Image));
            var rt = Place(go, parent == null ? page : parent, x, y, w, h); go.GetComponent<Image>().color = new Color(.96f, .98f, .96f, .95f); return rt;
        }
        private Text Label(string value, float x, float y, float w, float h, int size, bool bold = false, Transform parent = null)
        {
            var go = new GameObject("Label", typeof(RectTransform), typeof(Text)); Place(go, parent == null ? page : parent, x, y, w, h);
            var text = go.GetComponent<Text>(); text.font = font; text.text = value; text.fontSize = size; text.color = Ink;
            text.fontStyle = bold ? FontStyle.Bold : FontStyle.Normal; text.raycastTarget = false; return text;
        }
        private Button Button(string title, float x, float y, float w, float h, Action action, Transform parent = null)
        {
            var rt = Panel(title, x, y, w, h, parent); rt.GetComponent<Image>().color = Teal;
            var button = rt.gameObject.AddComponent<Button>(); button.targetGraphic = rt.GetComponent<Image>();
            button.onClick.AddListener(() => action());
            var label = Label(title, 6, 0, w - 12, h, 14, true, rt); label.color = Color.white; label.alignment = TextAnchor.MiddleCenter;
            return button;
        }
    }
}
