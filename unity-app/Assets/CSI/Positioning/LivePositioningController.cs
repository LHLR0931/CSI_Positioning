using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace CsiApp
{
    // Real received positions and measured site geometry. No playback or synthetic movement.
    public sealed partial class LivePositioningController : MonoBehaviour
    {
        private static readonly Color Ink = new Color32(25, 58, 53, 255);
        private static readonly Color Teal = new Color32(0, 119, 102, 255);
        private static readonly Color Pale = new Color32(225, 239, 232, 255);
        private static readonly Color Amber = new Color32(179, 104, 25, 255);
        private PositioningSession connection;
        private SiteNavigationConfig config;
        private LiveNavigationState navigation;
        private SiteArTracker ar;
        private AndroidPresentationBackHandler backHandler;
        private Font font;
        private RectTransform safe, pageHeader, mainContent, modal, modalContent, modalViewport, modalFeedbackPanel, mapRect, infoRect, destinationsRect, noticeRect, navigationFooter;
        private ScrollRect mainScroll, modalScroll;
        private Image background;
        private Text sourceText, positionText, instructionText, noticeText, goalText, arStateText, modalFeedback;
        private Button startButton, pauseButton, endButton, arrivalButton, handoffButton;
        private SiteModelOverlay siteModelOverlay;
        private LivePositioningMapGraphic map;
        private readonly List<Text> gridNumbers = new List<Text>();
        private string destinationId = "balcony", lastIdentity = "", lastModel = "", notice = "";
        private int gridTarget = 14, connectionRevision = -1;
        private bool gridDestination = true, arMode, wasCalibrated, wasVerified, quitting, arAlignmentKnown, previousArAlignment;
        private bool modelCheckMode, showSiteModel = true;
        private float nextRefresh;
        private Rect lastSafe;
        private Vector2 lastScreen;
        private Transform visualMap, verificationRoot, arRouteRoot;
        private string visualRouteKey = "";
        private Material gridMaterial, routeMaterial, targetMaterial, walkedMaterial;
        private Mesh arrowMesh;

        private void Start()
        {
            Screen.orientation = ScreenOrientation.Portrait;
            connection = PositioningSession.Instance;
            connection.SetPredictionPolling(false);
            config = SiteNavigationConfig.Load();
            navigation = new LiveNavigationState(config);
            if (PlanningPreviewTransfer.ReturnToPlanning && PlanningPreviewTransfer.Labels != null && PlanningPreviewTransfer.Labels.Count > 1)
            {
                plannedStart = PlanningPreviewTransfer.Labels[0]; gridTarget = PlanningPreviewTransfer.Labels[PlanningPreviewTransfer.Labels.Count - 1];
                selectedCandidate = PlanningPreviewTransfer.Candidate; PlanningPreviewTransfer.ReturnToPlanning = false;
            }
            navigation.Plan(plannedStart, gridTarget, selectedCandidate);
            ar = gameObject.AddComponent<SiteArTracker>();
            backHandler = new AndroidPresentationBackHandler();
            font = CsiLocalization.UiFont;
            BuildUi();
            notice = config.LastError ?? "";
            if (connection.Backend != null && !connection.Backend.IsLiveTransport)
                notice = "Recorded data cannot drive positioning. Open Connection to reconnect to the live server.";
            Refresh();
        }

        private void Update()
        {
            if (connection == null || navigation == null || quitting) return;
            if ((backHandler != null && backHandler.ConsumeBackRequest()) || Input.GetKeyDown(KeyCode.Escape)) HandleBack();
            if (lastSafe != Screen.safeArea || lastScreen != new Vector2(Screen.width, Screen.height)) FitSafe();
            UpdateKeyboard();
            bool observationsReset = false;
            if (connectionRevision != connection.ConnectionRevision)
            {
                connectionRevision = connection.ConnectionRevision;
                navigation.ResetObservations(); lastIdentity = "";
                observationsReset = true;
            }
            if (connection.SelectedModelId != lastModel)
            {
                bool initialModelResolved = !observationsReset && navigation.Preparing && navigation.RawLabel < 0;
                int checkpoint = navigation.PreparationCheckpointLabel;
                // A deliberate model switch may revisit the same sequence/window as before.
                // Reset observations, keep an active route paused, then accept that model's row.
                if (!observationsReset) navigation.ResetObservations();
                if (lastModel.Length > 0)
                    notice = navigation.Active ? "Model changed. The route position is frozen; review the selected estimate, then Resume."
                        : "Model changed. Showing only this model's received estimate.";
                lastModel = connection.SelectedModelId; lastIdentity = "";
                if (initialModelResolved) navigation.BeginPreparation(checkpoint);
            }
            if (pendingPreparation != -2 && arMode && (!ar.IsVerified || !ar.IsTracking)) SuspendForScan();
            if (pendingPreparation != -2 && connection.Backend.IsLiveTransport && connection.Backend.Meta != null
                && connection.Backend.Connection != CsiConnectionState.Disconnected)
                PrepareAtPoint(pendingPreparation);
            var result = connection.SelectedResult;
            bool fresh = connection.HasFreshLiveSelected;
            navigation.SetCsiAvailable(fresh, fresh ? "" : "Selected model is unavailable, stale, or server is not ready.");
            if (result != null && connection.Backend.PredictionPollingEnabled)
            {
                var p = result.Prediction;
                string identity = (p.session_id ?? "") + "|" + result.ModelId + "|" + p.seq + "|" + p.timestamp + "|" + result.EffectiveSource;
                if (identity != lastIdentity)
                {
                    // Even an unavailable/replay-declared row must reach the state machine:
                    // an explicit source change pauses an AR approach as well as grid navigation.
                    navigation.AcceptCsi(p, result.EffectiveSource, fresh);
                    lastIdentity = identity;
                }
            }
            bool aligned = ar.IsVerified && ar.IsTracking;
            if (!arAlignmentKnown || aligned != previousArAlignment)
            {
                // No AR message should replace a normal 2D pause/source explanation every frame.
                if (aligned || arAlignmentKnown)
                    navigation.SetArAlignment(aligned, aligned ? "Measured AR alignment verified by the operator." : "AR alignment is not verified or tracking was interrupted.");
                arAlignmentKnown = true; previousArAlignment = aligned;
            }
            if (arMode && !aligned && connection.Backend.PredictionPollingEnabled && (navigation.Preparing || navigation.Active && !navigation.Paused)) SuspendForScan();
            if (aligned) navigation.AcceptAr(ar.MapCameraPosition, true);
            if ((wasCalibrated && !ar.IsCalibrated) || (wasVerified && !ar.IsVerified)) ClearVerificationChecks();
            wasCalibrated = ar.IsCalibrated; wasVerified = ar.IsVerified;
            if (connection.Recording && navigation.CurrentLabel >= 0 &&
                (navigation.InApproach ? aligned : fresh))
                connection.RecordNavigation(navigation.CurrentPosition, navigation.PositionSource,
                    navigation.Active ? navigation.Paused ? "paused" : navigation.InApproach ? "ar_approach" : "grid_navigation" : "positioning");
            UpdateArVisuals();
            if (Time.unscaledTime >= nextRefresh) { nextRefresh = Time.unscaledTime + .2f; Refresh(); }
        }

        private void ConfirmArrival()
        {
            if (!connection.StopLog()) { SetNotice(connection.LogMessage); return; }
            navigation.ConfirmArrival(); if (navigation.Arrived) connection.SetPredictionPolling(false); SetNotice(navigation.Notice);
        }
        private void SetNotice(string value)
        {
            notice = value ?? "";
            arActionFeedback = notice; arActionFeedbackUntil = Time.unscaledTime + 6f;
            if (modalFeedback != null)
            {
                modalFeedback.text = notice;
                modalFeedbackPanel.gameObject.SetActive(!string.IsNullOrEmpty(notice));
                Stretch(modalScroll.GetComponent<RectTransform>(), 0, 60, 0, string.IsNullOrEmpty(notice) ? 12 : 112);
            }
            Refresh();
        }
        private static string DestinationName(string id) { return id == "balcony" ? "Balcony entrance" : id == "office" ? "Office entrance" : id == "kitchen" ? "Kitchen entrance" : id; }

        private void OpenConnection()
        {
            OpenModal("Prediction connection");
            ModalText("Use the backend computer's HTTP address. Source remains MOCK or unspecified when reported that way.", 70);
            var address = ModalInput("Computer IP and port", connection.Address, "http://computer-ip:8765", 48);
            Text error = ModalText("", 60);
            ModalButton("Save & connect", () => { navigation.Pause(); if (connection.Connect(address.text)) { lastIdentity = ""; CloseModal(); } else error.text = connection.LogMessage; SetNotice(connection.LogMessage); });
            ModalButton("Disconnect", () => { SuspendForScan(); connection.Disconnect(); SetNotice("Disconnected. Last position is historical."); CloseModal(); });
            ModalButton("Select received model", OpenModels);
            ModalText("Connection: " + connection.Backend.Connection + "\n" +
                (connection.Backend.Status == null ? connection.Backend.Notice : connection.Backend.Status.state + "\n" + connection.Backend.Notice), 100);
            FinishModal();
        }
        private void OpenModels()
        {
            OpenModal("Position model");
            ModalText("Changing model pauses navigation and saves any active recording. Only a fresh selected result can continue a grid route.", 77);
            foreach (var value in connection.AvailableModelIds)
            {
                string id = value;
                ModalButton(id == CsiModels.Unspecified ? "Server model (unspecified)" : CsiModels.Display(id),
                    () => { navigation.Pause(); connection.SelectModel(id); lastIdentity = "";
                        if (connection.SelectedModelId != id) { SetNotice(connection.LogMessage); return; }
                        CloseModal(); SetNotice("Selected " + CsiModels.Display(id) + ". Check the current result before resuming."); });
            }
            FinishModal();
        }
        private void OpenModelCheck()
        {
            if (!connection.StopLog()) { SetNotice(connection.LogMessage); return; }
            SuspendForScan(); modelCheckMode = true; showSiteModel = true; showSiteGrid = true;
            SetCameraMode(true, false);
            OpenArControls();
        }
        private void OpenModelCheckControls()
        {
            OpenModal("Model check help");
            ModalText("No server or destination is needed. Keep the measured image fixed upright on the wall.", 62);
            ModalButton("Image measurements", OpenImageSetup);
            ModalButton("Back to camera", OpenArControls);
            ModalText("Scan image, wait for image tracking, then Lock alignment. The room appears before you approve it. Tap Check floor points and compare 0, 4, then 10 with the real tape before each confirmation.", 116);
            ModalText("Hide walls / Show walls compares the room outline. Hide grid removes point labels. Hide controls leaves the camera clear; Controls restores them. Tracking warnings remain visible.", 100);
            ModalText("Compare wall bases and doorway sides from several positions. Heights are illustrative. If anything is shifted, use More > Scan reference again; do not confirm a mismatch.", 96);
            FinishModal();
        }
        private void OpenImageSetup() { OpenSiteSetup(true); }
        private void OpenSiteSetup() { OpenSiteSetup(true); }
        private void OpenSiteSetup(bool imageOnly)
        {
            imageOnly = true;
            OpenModal(imageOnly ? "Measured reference image" : "Measured site setup");
            ModalText("Fixed upright wall image. Origin = grid 0; +x = 0 to 4; +y = 0 to 10. Measure the image CENTRE. Saving ends the route and requires a new scan.", 100);
            var values = new Dictionary<string, InputField>();
            values["width"] = ModalInput("Full image width (m)", Number(config.imageWidthM), "Measured width", 45);
            values["x"] = ModalInput("Image centre map x (m)", Number(config.imageX), "x metres", 45);
            values["y"] = ModalInput("Image centre map y (m)", Number(config.imageY), "y metres", 45);
            values["height"] = ModalInput("Image centre above floor (m)", Number(config.imageHeightM), "height metres", 45);
            values["facing"] = ModalInput("Image front: 0 = +y; 90 = +x (degrees)", Number(config.imageFacingDeg), "Facing degrees", 45);
            Toggle measured = ModalToggle("I measured the fixed image pose and size", config.markerMeasured);
            var paths = new Dictionary<string, InputField>();
            var checkedPaths = new Dictionary<string, Toggle>();
            foreach (string id in imageOnly ? new string[0] : new[] { "balcony", "office", "kitchen" })
            {
                var destination = config.FindDestination(id);
                ModalText(DestinationName(id) + " / connector " + SiteNavigationConfig.ConnectorFor(id) +
                    "\nOrdered waypoints AFTER the connector, ending at the measured standing target. One x,y pair per line (metres). Include turns.", 100);
                paths[id] = ModalInput(id + " approach x,y", Waypoints(destination), "x,y\nx,y", 130, true);
                checkedPaths[id] = ModalToggle("I checked this route, doorway and target onsite", destination != null && destination.surveyed);
            }
            Text errorText = ModalText("", 80);
            ModalButton("Save site setup / end current route", () => {
                var candidate = JsonUtility.FromJson<SiteNavigationConfig>(JsonUtility.ToJson(config));
                float width, x, y, height, facing;
                if (!TryNumber(values["width"].text, out width) || !TryNumber(values["x"].text, out x) ||
                    !TryNumber(values["y"].text, out y) || !TryNumber(values["height"].text, out height) || !TryNumber(values["facing"].text, out facing))
                { errorText.text = "Use finite decimal numbers and a decimal point. No units in number fields."; return; }
                candidate.imageWidthM = width; candidate.imageX = x; candidate.imageY = y; candidate.imageHeightM = height;
                candidate.imageFacingDeg = facing; candidate.markerMeasured = measured.isOn;
                foreach (string id in imageOnly ? new string[0] : new[] { "balcony", "office", "kitchen" })
                {
                    SiteWaypoint[] points; string error;
                    if (!ParseWaypoints(paths[id].text, out points, out error)) { errorText.text = DestinationName(id) + ": " + error; return; }
                    var d = candidate.FindDestination(id);
                    d.approach = points; d.surveyed = checkedPaths[id].isOn;
                }
                if (!connection.StopLog()) { errorText.text = connection.LogMessage; return; }
                if (!candidate.Save()) { errorText.text = candidate.LastError; return; }
                pendingPreparation = -2; navigation.End(); connection.SetPredictionPolling(false); config = candidate; navigation = new LiveNavigationState(config);
                routeConfirmed = displayChosen = false; navigation.Plan(plannedStart, gridTarget, selectedCandidate);
                lastIdentity = ""; arAlignmentKnown = false; ClearVerificationChecks(); ar.Shutdown();
                if (arMode) ar.Begin(config);
                CloseModal(); SetNotice("Site setup saved. Scan, lock and verify several grid points before using AR.");
            });
            FinishModal();
        }
        private void OpenArControls()
        {
            CloseModal();
            arControlsHidden = false;
            if (!arMode) SetCameraMode(true, false);
            Refresh();
        }
        private void ClearVerificationChecks()
        {
            Array.Clear(checkedFloorPoints, 0, checkedFloorPoints.Length);
            checkingFloor = false; floorCheckStep = 0;
        }
        private static string Number(float value) { return value.ToString("0.###", CultureInfo.InvariantCulture); }
        private static bool TryNumber(string value, out float result)
        { return float.TryParse((value ?? "").Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out result) && !float.IsNaN(result) && !float.IsInfinity(result); }
        private static string Waypoints(SiteDestinationConfig d)
        {
            var rows = new List<string>();
            if (d != null && d.approach != null) foreach (var p in d.approach) if (p != null) rows.Add(Number(p.x) + "," + Number(p.y));
            return string.Join("\n", rows.ToArray());
        }
        private static bool ParseWaypoints(string text, out SiteWaypoint[] points, out string error)
        {
            var result = new List<SiteWaypoint>(); error = null; points = null;
            foreach (var row in (text ?? "").Replace("\r", "").Split('\n'))
            {
                if (string.IsNullOrWhiteSpace(row)) continue;
                string[] pair = row.Split(','); float x, y;
                if (pair.Length != 2 || !TryNumber(pair[0], out x) || !TryNumber(pair[1], out y))
                { error = "Use one finite x,y pair per line, with decimal points."; return false; }
                result.Add(new SiteWaypoint { x = x, y = y });
                if (result.Count > 64) { error = "Use at most 64 waypoints."; return false; }
            }
            points = result.ToArray(); return true;
        }

        private void UpdateArVisuals()
        {
            if (ar.MapRoot == null || !ar.IsCalibrated)
            {
                if (siteModelOverlay != null) siteModelOverlay.gameObject.SetActive(false);
                return;
            }
            if (visualMap != ar.MapRoot)
            {
                visualMap = ar.MapRoot; visualRouteKey = "";
                siteModelOverlay = null;
                verificationRoot = new GameObject("Check real grid 0 4 10").transform; verificationRoot.SetParent(visualMap, false);
                EnsureMaterials();
                for (int i = 0; i < 15; i++)
                {
                    var point = SiteRenderGeometry.CreateSphere("Check grid " + i);
                    point.transform.SetParent(verificationRoot, false); point.transform.localPosition = new Vector3(i % 5 * .8f, .025f, i / 5 * .8f);
                    point.transform.localScale = Vector3.one * (i == 0 || i == 4 || i == 10 ? .12f : .07f);
                    point.GetComponent<Renderer>().sharedMaterial = gridMaterial;
                    var labelObject = new GameObject("Grid label " + i, typeof(TextMesh)); labelObject.transform.SetParent(verificationRoot, false);
                    labelObject.transform.localPosition = point.transform.localPosition + Vector3.up * .13f;
                    var text = labelObject.GetComponent<TextMesh>(); text.text = i.ToString(); text.font = font; text.fontSize = 40; text.characterSize = .035f; text.anchor = TextAnchor.MiddleCenter; text.color = Color.white;
                    labelObject.GetComponent<MeshRenderer>().sharedMaterial = font.material;
                }
                arRouteRoot = new GameObject("Verified live navigation route").transform; arRouteRoot.SetParent(visualMap, false);
            }
            if (arMode && showSiteModel && siteModelOverlay == null)
            {
                var overlayObject = new GameObject("Surveyed apartment overlay / 1:1");
                overlayObject.transform.SetParent(visualMap, false);
                siteModelOverlay = overlayObject.AddComponent<SiteModelOverlay>();
                siteModelOverlay.Initialize();
            }
            if (siteModelOverlay != null) siteModelOverlay.gameObject.SetActive(arMode && showSiteModel && ar.IsTracking);
            verificationRoot.gameObject.SetActive(arMode && ar.IsTracking && (showSiteGrid || checkingFloor));
            foreach (var label in verificationRoot.GetComponentsInChildren<TextMesh>())
                if (ar.Camera != null) label.transform.rotation = Quaternion.LookRotation(label.transform.position - ar.Camera.transform.position, Vector3.up);
            bool visible = arMode && !modelCheckMode && ar.IsVerified && ar.IsTracking && routeConfirmed;
            arRouteRoot.gameObject.SetActive(visible);
            if (!visible) return;
            string key = navigation.DestinationId + "|" + navigation.ProgressIndex + "|" + navigation.Paused + "|" + navigation.Arrived + "|";
            foreach (var p in navigation.Route) key += Number(p.x) + "," + Number(p.z) + ";";
            if (key == visualRouteKey) return;
            visualRouteKey = key;
            for (int i = arRouteRoot.childCount - 1; i >= 0; i--) Destroy(arRouteRoot.GetChild(i).gameObject);
            if (navigation.Route.Count < 2) return;
            for (int i = 1; i < navigation.Route.Count; i++)
            {
                var a = navigation.Route[i - 1]; var b = navigation.Route[i]; var delta = b - a;
                Material segmentMaterial = i <= navigation.ProgressIndex || navigation.Paused ? walkedMaterial : routeMaterial;
                var lineObject = new GameObject("Fixed route segment " + i, typeof(LineRenderer)); lineObject.transform.SetParent(arRouteRoot, false);
                var line = lineObject.GetComponent<LineRenderer>(); line.useWorldSpace = false; line.positionCount = 2;
                line.widthMultiplier = i <= navigation.ProgressIndex ? .022f : .045f; line.sharedMaterial = segmentMaterial;
                line.SetPosition(0, a + Vector3.up * .035f); line.SetPosition(1, b + Vector3.up * .035f);
                if (i <= navigation.ProgressIndex) continue;
                if (delta.sqrMagnitude < .001f) continue;
                var arrow = new GameObject("Direction arrow", typeof(MeshFilter), typeof(MeshRenderer)); arrow.transform.SetParent(arRouteRoot, false);
                arrow.transform.localPosition = (a + b) * .5f + Vector3.up * .04f; arrow.transform.localRotation = Quaternion.LookRotation(delta.normalized, Vector3.up);
                arrow.GetComponent<MeshFilter>().sharedMesh = arrowMesh; arrow.GetComponent<MeshRenderer>().sharedMaterial = segmentMaterial;
            }
            var target = SiteRenderGeometry.CreateSphere("Measured route target"); target.transform.SetParent(arRouteRoot, false);
            target.transform.localPosition = navigation.Route[navigation.Route.Count - 1] + Vector3.up * .07f; target.transform.localScale = Vector3.one * .14f;
            target.GetComponent<Renderer>().sharedMaterial = targetMaterial;
        }
        private void EnsureMaterials()
        {
            if (gridMaterial != null) return;
            Shader shader = Resources.Load<Shader>("Maps/ApartmentMapUnlit") ?? Shader.Find("Unlit/Color");
            gridMaterial = new Material(shader) { color = new Color(.1f, .55f, .9f) };
            routeMaterial = new Material(shader) { color = new Color(0, .8f, .62f) };
            walkedMaterial = new Material(shader) { color = new Color(.38f, .45f, .46f) };
            targetMaterial = new Material(shader) { color = new Color(1, .65f, .08f) };
            arrowMesh = new Mesh { name = "Positioning direction arrow" };
            arrowMesh.vertices = new[] { new Vector3(0, 0, .16f), new Vector3(-.10f, 0, -.08f), new Vector3(.10f, 0, -.08f) };
            arrowMesh.triangles = new[] { 0, 2, 1, 0, 1, 2 }; arrowMesh.RecalculateNormals();
        }

        private float modalY;
        private void OpenModal(string title)
        {
            CloseModal();
            modal = Panel(safe, 12, 70, 366, 600, new Color32(246, 249, 244, 255));
            modal.anchorMin = Vector2.zero; modal.anchorMax = Vector2.one; modal.pivot = new Vector2(.5f, .5f);
            modal.offsetMin = new Vector2(12, 12); modal.offsetMax = new Vector2(-12, -70);
            MakeText(modal, title, 14, 14, 272, 34, 19, true);
            MakeButton(modal, "X", 308, 10, 44, 40, CloseModal);
            modalScroll = MakeScroll(modal, "Scrollable controls", out modalContent, out modalViewport);
            Stretch(modalScroll.GetComponent<RectTransform>(), 0, 60, 0, 12);
            // Fixed feedback stays visible while a failed action leaves the dialog open.
            // In particular, save failures must never disappear behind the modal surface.
            var feedback = Panel(modal, 12, 0, 342, 96, Color.white);
            feedback.anchorMin = feedback.anchorMax = feedback.pivot = Vector2.zero;
            feedback.anchoredPosition = new Vector2(12, 8);
            modalFeedback = MakeText(feedback, "", 8, 8, 326, 80, 12);
            modalFeedbackPanel = feedback;
            feedback.gameObject.SetActive(false);
            modalY = 0;
        }
        private void CloseModal()
        {
            if (modal != null) { modal.gameObject.SetActive(false); Destroy(modal.gameObject); }
            modal = modalContent = modalViewport = modalFeedbackPanel = null; modalScroll = null; arStateText = null; modalFeedback = null;
        }
        private Text ModalText(string value, float height)
        { var t = MakeText(modalContent, value, 14, modalY, 338, height, 13); modalY += height + 12; return t; }
        private void ModalButton(string title, Action action)
        { MakeButton(modalContent, title, 14, modalY, 338, 46, action); modalY += 58; }
        private InputField ModalInput(string title, string value, string placeholder, float height, bool multiline = false)
        {
            MakeText(modalContent, title, 14, modalY, 338, 30, 12, true); modalY += 34;
            var box = Panel(modalContent, 14, modalY, 338, height, Color.white);
            var input = box.gameObject.AddComponent<InputField>(); input.targetGraphic = box.GetComponent<Image>();
            var text = MakeText(box, "", 10, 7, 318, height - 14, 17); text.supportRichText = false;
            var hint = MakeText(box, placeholder, 10, 7, 318, height - 14, 15); hint.color = new Color(.45f, .52f, .49f);
            input.textComponent = text; input.placeholder = hint; input.lineType = multiline ? InputField.LineType.MultiLineNewline : InputField.LineType.SingleLine;
            input.text = value ?? ""; modalY += height + 18; return input;
        }
        private Toggle ModalToggle(string title, bool value)
        {
            var row = Panel(modalContent, 14, modalY, 338, 62, Pale);
            var toggle = row.gameObject.AddComponent<Toggle>();
            var box = Panel(row, 10, 18, 26, 26, Color.white);
            var mark = Panel(box, 5, 5, 16, 16, Teal);
            toggle.targetGraphic = box.GetComponent<Image>(); toggle.graphic = mark.GetComponent<Image>(); toggle.isOn = value;
            MakeText(row, title, 48, 9, 280, 49, 13); modalY += 74; return toggle;
        }
        private void FinishModal() { modalContent.sizeDelta = new Vector2(366, Mathf.Max(100, modalY + 16)); modalScroll.verticalNormalizedPosition = 1; }
        private void UpdateKeyboard()
        {
            if (modal == null) return;
            float keyboard = TouchScreenKeyboard.visible ? TouchScreenKeyboard.area.height : 0;
#if UNITY_ANDROID && !UNITY_EDITOR
            var selected = EventSystem.current == null ? null : EventSystem.current.currentSelectedGameObject;
            var input = selected == null ? null : selected.GetComponent<InputField>();
            if (input != null && input.isFocused && keyboard < 1) keyboard = Screen.height * .4f;
#endif
            modal.offsetMin = new Vector2(12, keyboard > 0 ? keyboard / Mathf.Max(.1f, Screen.width / 390f) + 8 : 12);
            var current = EventSystem.current == null ? null : EventSystem.current.currentSelectedGameObject;
            var field = current == null ? null : current.GetComponent<InputField>();
            if (field == null || !field.isFocused || modalViewport == null) return;
            var fieldCorners = new Vector3[4]; var viewCorners = new Vector3[4];
            field.GetComponent<RectTransform>().GetWorldCorners(fieldCorners); modalViewport.GetWorldCorners(viewCorners);
            if (fieldCorners[0].y < viewCorners[0].y + 12)
                modalContent.anchoredPosition += Vector2.up * ((viewCorners[0].y + 12 - fieldCorners[0].y) / Mathf.Max(.1f, Screen.width / 390f));
        }
        private void FitSafe()
        {
            lastSafe = Screen.safeArea; lastScreen = new Vector2(Screen.width, Screen.height);
            safe.anchorMin = new Vector2(lastSafe.xMin / Screen.width, lastSafe.yMin / Screen.height);
            safe.anchorMax = new Vector2(lastSafe.xMax / Screen.width, lastSafe.yMax / Screen.height);
            safe.offsetMin = safe.offsetMax = Vector2.zero;
        }
        private static void Stretch(RectTransform rt, float left, float top, float right, float bottom)
        { rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one; rt.offsetMin = new Vector2(left, bottom); rt.offsetMax = new Vector2(-right, -top); }
        private RectTransform Panel(Transform parent, float x, float y, float width, float height, Color color)
        {
            var go = new GameObject("Panel", typeof(RectTransform), typeof(Image)); var rt = go.GetComponent<RectTransform>(); rt.SetParent(parent, false);
            rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(0, 1); rt.anchoredPosition = new Vector2(x, -y); rt.sizeDelta = new Vector2(width, height);
            go.GetComponent<Image>().color = color; return rt;
        }
        private Text MakeText(Transform parent, string value, float x, float y, float width, float height, int size, bool bold = false)
        {
            var go = new GameObject("Text", typeof(RectTransform), typeof(Text)); var rt = go.GetComponent<RectTransform>(); rt.SetParent(parent, false);
            rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(0, 1); rt.anchoredPosition = new Vector2(x, -y); rt.sizeDelta = new Vector2(width, height);
            var t = go.GetComponent<Text>(); t.font = font; t.text = value; t.fontSize = size; t.color = Ink; t.raycastTarget = false;
            t.fontStyle = bold ? FontStyle.Bold : FontStyle.Normal; return t;
        }
        private Button MakeButton(Transform parent, string title, float x, float y, float width, float height, Action action)
        {
            var box = Panel(parent, x, y, width, height, Pale); var b = box.gameObject.AddComponent<Button>();
            b.targetGraphic = box.GetComponent<Image>(); b.onClick.AddListener(() => action());
            var text = MakeText(box, title, 5, 4, width - 10, height - 8, 13, true); text.alignment = TextAnchor.MiddleCenter; return b;
        }
        private ScrollRect MakeScroll(Transform parent, string name, out RectTransform content, out RectTransform viewport)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(ScrollRect)); go.transform.SetParent(parent, false);
            var sr = go.GetComponent<ScrollRect>(); sr.horizontal = false; sr.vertical = true; sr.movementType = ScrollRect.MovementType.Clamped;
            var viewObject = new GameObject("Viewport", typeof(RectTransform), typeof(Image), typeof(Mask)); viewport = viewObject.GetComponent<RectTransform>(); viewport.SetParent(go.transform, false); Stretch(viewport, 0, 0, 0, 0);
            viewObject.GetComponent<Image>().color = Color.white; viewObject.GetComponent<Mask>().showMaskGraphic = false;
            content = new GameObject("Content", typeof(RectTransform)).GetComponent<RectTransform>(); content.SetParent(viewport, false);
            content.anchorMin = content.anchorMax = content.pivot = new Vector2(.5f, 1); content.anchoredPosition = Vector2.zero;
            sr.viewport = viewport; sr.content = content; sr.scrollSensitivity = 28; return sr;
        }
        private void OnApplicationPause(bool paused)
        { if (paused && navigation != null) { SuspendForScan(); ClearVerificationChecks(); } }
        private void OnDestroy()
        {
            if (backHandler != null) backHandler.Dispose();
            if (connection != null) connection.SetPredictionPolling(false);
            if (ar != null) ar.Shutdown();
            if (gridMaterial != null) Destroy(gridMaterial);
            if (routeMaterial != null) Destroy(routeMaterial);
            if (targetMaterial != null) Destroy(targetMaterial);
            if (walkedMaterial != null) Destroy(walkedMaterial);
            if (arrowMesh != null) Destroy(arrowMesh);
        }
    }

    [RequireComponent(typeof(CanvasRenderer))]
    public sealed class LivePositioningMapGraphic : MaskableGraphic
    {
        public IList<Vector3> Route, Alternative;
        public int ProgressIndex, SelectedCandidate;
        public bool Frozen;
        public Vector3 PlannedStart;
        public Vector3 CurrentPosition, Target;
        public bool HasPosition, HasTarget, Current;
        private Rect Bounds()
        {
            var apartment = ApartmentMap.Current.PlanBounds;
            float x0 = Mathf.Min(0, apartment.xMin), y0 = Mathf.Min(0, apartment.yMin);
            float x1 = Mathf.Max(3.2f, apartment.xMax), y1 = Mathf.Max(1.6f, apartment.yMax);
            if (Route != null) foreach (var p in Route) { x0 = Mathf.Min(x0, p.x); y0 = Mathf.Min(y0, p.z); x1 = Mathf.Max(x1, p.x); y1 = Mathf.Max(y1, p.z); }
            if (HasTarget) { x0 = Mathf.Min(x0, Target.x); y0 = Mathf.Min(y0, Target.z); x1 = Mathf.Max(x1, Target.x); y1 = Mathf.Max(y1, Target.z); }
            if (HasPosition) { x0 = Mathf.Min(x0, CurrentPosition.x); y0 = Mathf.Min(y0, CurrentPosition.z); x1 = Mathf.Max(x1, CurrentPosition.x); y1 = Mathf.Max(y1, CurrentPosition.z); }
            return Rect.MinMaxRect(x0 - .3f, y0 - .3f, x1 + .3f, y1 + .3f);
        }
        public Vector2 MapPoint(Vector3 point)
        {
            var bounds = Bounds(); var rect = rectTransform.rect;
            float scale = Mathf.Min((rect.width - 20) / bounds.width, (rect.height - 20) / bounds.height);
            return rect.center + new Vector2(point.x - bounds.center.x, point.z - bounds.center.y) * scale;
        }
        protected override void OnPopulateMesh(VertexHelper vh)
        {
            vh.Clear();
            var apartment = ApartmentMap.Current;
            foreach (var region in apartment.regions)
            {
                if (region.model_role == "excluded_unknown") continue;
                float x0 = region.min_xy_m[0], y0 = region.min_xy_m[1], x1 = region.max_xy_m[0], y1 = region.max_xy_m[1];
                Color floor = region.id == "balcony" ? new Color32(220, 235, 220, 255)
                    : region.id == "office" ? new Color32(235, 229, 222, 255)
                    : region.id == "kitchen" ? new Color32(243, 233, 207, 255) : new Color32(248, 250, 247, 255);
                Quad(vh, MapPoint(new Vector3(x0, 0, y0)), MapPoint(new Vector3(x1, 0, y0)),
                    MapPoint(new Vector3(x1, 0, y1)), MapPoint(new Vector3(x0, 0, y1)), floor);
            }
            foreach (var boundary in apartment.boundaries)
            {
                if (boundary.kind == "open_connection") continue;
                Line(vh, MapPoint(ApartmentMap.ToWorld(boundary.start_xy_m)), MapPoint(ApartmentMap.ToWorld(boundary.end_xy_m)),
                    boundary.kind == "region_boundary_unspecified" ? 1 : 3,
                    boundary.kind == "region_boundary_unspecified" ? new Color32(180, 193, 188, 255) : new Color32(88, 117, 107, 255));
            }
            var grid = new Color32(166, 194, 184, 255);
            for (int i = 0; i < 15; i++)
                foreach (int j in GridMap.Neighbours(i)) if (j > i) Line(vh, P(i), P(j), 1.5f, grid);
            if (Alternative != null) for (int i = 1; i < Alternative.Count; i++) Line(vh, MapPoint(Alternative[i - 1]), MapPoint(Alternative[i]), 3, SelectedCandidate == 0 ? new Color32(200, 142, 55, 210) : new Color32(80, 164, 146, 210));
            if (Route != null) for (int i = 1; i < Route.Count; i++)
            {
                Color color = i <= ProgressIndex || Frozen ? new Color32(155, 168, 162, 255) : SelectedCandidate == 1 ? new Color32(179, 104, 25, 255) : new Color32(0, 119, 102, 255);
                var a = MapPoint(Route[i - 1]); var b = MapPoint(Route[i]); Line(vh, a, b, i <= ProgressIndex ? 3 : 5, color);
                if (i > ProgressIndex) Arrow(vh, (a+b)*.5f, (b-a).normalized, color);
            }
            Disc(vh, MapPoint(PlannedStart), 16, new Color32(80, 133, 180, 255));
            for (int i = 0; i < 15; i++) Disc(vh, P(i), 12, new Color32(247, 250, 246, 255));
            if (HasTarget) { Disc(vh, MapPoint(Target), 17, new Color32(218, 145, 35, 255)); Disc(vh, MapPoint(Target), 12, new Color32(255, 244, 223, 255)); }
            if (HasPosition) { Disc(vh, MapPoint(CurrentPosition), 15, Current ? new Color32(45, 113, 214, 255) : new Color32(134, 151, 159, 255)); Disc(vh, MapPoint(CurrentPosition), 9, Color.white); }
        }
        private Vector2 P(int label) { return MapPoint(new Vector3(label % 5 * .8f, 0, label / 5 * .8f)); }
        private static void Quad(VertexHelper vh, Vector2 a, Vector2 b, Vector2 c, Vector2 d, Color color)
        {
            int n = vh.currentVertCount;
            vh.AddVert(a, color, Vector2.zero); vh.AddVert(b, color, Vector2.zero); vh.AddVert(c, color, Vector2.zero); vh.AddVert(d, color, Vector2.zero);
            vh.AddTriangle(n, n + 1, n + 2); vh.AddTriangle(n, n + 2, n + 3);
        }
        private static void Line(VertexHelper vh, Vector2 a, Vector2 b, float width, Color color)
        {
            var normal = new Vector2(-(b - a).normalized.y, (b - a).normalized.x) * width * .5f; int n = vh.currentVertCount;
            vh.AddVert(a + normal, color, Vector2.zero); vh.AddVert(b + normal, color, Vector2.zero); vh.AddVert(b - normal, color, Vector2.zero); vh.AddVert(a - normal, color, Vector2.zero);
            vh.AddTriangle(n, n + 1, n + 2); vh.AddTriangle(n, n + 2, n + 3);
        }
        private static void Arrow(VertexHelper vh, Vector2 center, Vector2 direction, Color color)
        {
            Vector2 normal = new Vector2(-direction.y, direction.x); int n = vh.currentVertCount;
            vh.AddVert(center + direction * 9, color, Vector2.zero);
            vh.AddVert(center - direction * 5 + normal * 6, color, Vector2.zero);
            vh.AddVert(center - direction * 5 - normal * 6, color, Vector2.zero); vh.AddTriangle(n, n+1, n+2);
        }
        private static void Disc(VertexHelper vh, Vector2 center, float radius, Color color)
        {
            int first = vh.currentVertCount; vh.AddVert(center, color, Vector2.zero);
            for (int i = 0; i <= 24; i++)
            {
                float a = i * Mathf.PI * 2 / 24; vh.AddVert(center + new Vector2(Mathf.Cos(a), Mathf.Sin(a)) * radius, color, Vector2.zero);
                if (i > 0) vh.AddTriangle(first, first + i, first + i + 1);
            }
        }
    }
}
