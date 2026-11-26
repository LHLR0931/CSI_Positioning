using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace CsiApp
{
    public sealed partial class LivePositioningController
    {
        private int plannedStart, selectedCandidate;
        private bool routeConfirmed, displayChosen;
        private Button originChoice, targetChoice, routeOne, routeTwo, viewChoice, cancelPreparationButton, topPreview, firstPersonPreview;
        private Text phaseText, routeSummary, directionArrow, nextPointText, distanceText;
        private readonly List<Text> roomNames = new List<Text>();
        private readonly List<Vector3> roomPositions = new List<Vector3>();
        private float preparationOpened;
        private int pendingPreparation = -2;

        private void BuildUi()
        {
            var canvasObject = new GameObject("Live positioning UI", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            canvasObject.transform.SetParent(transform, false);
            canvasObject.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasObject.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(390, 844); scaler.matchWidthOrHeight = 0;
            if (UnityEngine.EventSystems.EventSystem.current == null)
                new GameObject("Positioning input", typeof(UnityEngine.EventSystems.EventSystem), typeof(UnityEngine.EventSystems.StandaloneInputModule));
            var full = new GameObject("Screen", typeof(RectTransform), typeof(Image)); full.transform.SetParent(canvasObject.transform, false);
            Stretch(full.GetComponent<RectTransform>(), 0, 0, 0, 0); background = full.GetComponent<Image>(); background.color = new Color32(246, 249, 244, 255);
            safe = new GameObject("Safe area", typeof(RectTransform)).GetComponent<RectTransform>(); safe.SetParent(canvasObject.transform, false);
            pageHeader = Panel(safe, 0, 0, 390, 64, Color.clear); pageHeader.GetComponent<Image>().raycastTarget = false;
            MakeButton(pageHeader, "Home", 12, 12, 62, 42, HandleBack);
            MakeText(pageHeader, "Positioning", 84, 16, 170, 34, 23, true);
            MakeButton(pageHeader, "Settings", 272, 12, 106, 42, OpenActions);
            mainScroll = MakeScroll(safe, "Positioning content", out mainContent, out var viewport);
            Stretch(mainScroll.GetComponent<RectTransform>(), 0, 70, 0, 76); mainContent.sizeDelta = new Vector2(390, 558);
            phaseText = MakeText(mainContent, "Choose your route", 12, 0, 366, 28, 20, true);
            sourceText = MakeText(mainContent, "", 12, 28, 366, 40, 11);
            sourceText.gameObject.SetActive(false); // Full source diagnostics remain in Settings.
            topPreview = MakeButton(mainContent, "Model top view", 12, 38, 176, 36, () => OpenVirtualPreview(false));
            firstPersonPreview = MakeButton(mainContent, "First person demo", 202, 38, 176, 36, () => OpenVirtualPreview(true));
            infoRect = Panel(mainContent, 12, 38, 366, 102, Color.white);
            directionArrow = MakeText(infoRect, "", 8, 9, 66, 76, 54, true); directionArrow.alignment = TextAnchor.MiddleCenter;
            instructionText = MakeText(infoRect, "", 82, 9, 272, 29, 21, true);
            nextPointText = MakeText(infoRect, "", 82, 40, 272, 24, 14);
            distanceText = MakeText(infoRect, "", 82, 68, 272, 25, 18, true);
            mapRect = Panel(mainContent, 12, 84, 366, 300, Pale);
            var mapObject = new GameObject("Measured map drawing", typeof(RectTransform), typeof(LivePositioningMapGraphic)); mapObject.transform.SetParent(mapRect, false);
            Stretch(mapObject.GetComponent<RectTransform>(), 0, 0, 0, 0); map = mapObject.GetComponent<LivePositioningMapGraphic>(); map.raycastTarget = false;
            foreach (var region in ApartmentMap.Current.regions)
            {
                if (region.model_role == "excluded_unknown") continue;
                string title = region.id == "balcony" ? "BALCONY" : region.id == "office" ? "OFFICE" : region.id == "kitchen" ? "KITCHEN" : region.id == "corridor_extension" ? "CORRIDOR" : "LIVING ROOM";
                var label = MakeText(mapRect, title, 0, 0, 95, 20, 9, true); label.alignment = TextAnchor.MiddleCenter; label.color = new Color32(85, 104, 96, 255);
                roomNames.Add(label); roomPositions.Add(new Vector3((region.min_xy_m[0] + region.max_xy_m[0]) / 2, 0,
                    region.id == "living_room" ? 2.35f : (region.min_xy_m[1] + region.max_xy_m[1]) / 2));
            }
            for (int n = 0; n < 15; n++) { var label = MakeText(mapRect, n.ToString(), 0, 0, 24, 22, 11, true); label.alignment = TextAnchor.MiddleCenter; gridNumbers.Add(label); }
            positionText = MakeText(mainContent, "", 12, 363, 366, 32, 12); positionText.gameObject.SetActive(false);
            destinationsRect = Panel(mainContent, 12, 396, 366, 148, Color.white);
            originChoice = MakeButton(destinationsRect, "Start: 0", 10, 10, 167, 40, () => OpenPointPicker(true));
            targetChoice = MakeButton(destinationsRect, "Target: 14", 189, 10, 167, 40, () => OpenPointPicker(false));
            routeSummary = MakeText(destinationsRect, "", 10, 58, 346, 30, 21, true);
            routeOne = MakeButton(destinationsRect, "Route 1", 10, 100, 167, 38, () => ChooseCandidate(0));
            routeTwo = MakeButton(destinationsRect, "Route 2", 189, 100, 167, 38, () => ChooseCandidate(1));
            noticeRect = Panel(mainContent, 12, 554, 366, 54, new Color32(255, 242, 218, 255));
            noticeText = MakeText(noticeRect, "", 10, 7, 346, 40, 13);
            navigationFooter = Panel(safe, 0, 0, 390, 70, new Color32(246, 249, 244, 255));
            navigationFooter.anchorMin = navigationFooter.anchorMax = navigationFooter.pivot = new Vector2(.5f, 0); navigationFooter.anchoredPosition = Vector2.zero;
            startButton = MakeButton(navigationFooter, "Confirm route", 12, 10, 366, 46, PrimaryRouteAction);
            pauseButton = MakeButton(navigationFooter, "Retry check", 12, 64, 114, 38, RetryPreparation);
            cancelPreparationButton = MakeButton(navigationFooter, "Cancel check", 138, 64, 114, 38, CancelPositionCheck);
            endButton = MakeButton(navigationFooter, "Replan", 12, 64, 176, 38, RequestNewPlan);
            viewChoice = MakeButton(navigationFooter, "AR camera", 202, 64, 176, 38, () => ChooseDisplay(true));
            BuildArHud(); FitSafe();
        }

        private void Refresh()
        {
            if (sourceText == null || connection == null) return;
            bool polling = connection.Backend.PredictionPollingEnabled;
            var r = connection.SelectedResult;
            string server = connection.Backend.Status == null ? connection.Backend.Connection.ToString() : connection.Backend.Status.state;
            sourceText.text = "Server: " + server + "  /  Model: " + CsiModels.Display(connection.SelectedModelId) + "\n" +
                (polling ? "Source: " + (r == null ? "waiting" : r.EffectiveSource) + (connection.HasFreshLiveSelected ? " / current" : " / waiting for fresh live data") : "Position requests off / planning only");
            phaseText.text = routeConfirmed ? "Grid " + plannedStart + " \u2192 " + gridTarget : "Choose your route";
            string raw = navigation.RawLabel < 0 ? "--" : navigation.RawLabel.ToString();
            positionText.text = "CSI estimate: " + raw + " / accepted route point: " + (navigation.CurrentLabel < 0 ? "--" : navigation.CurrentLabel.ToString());
            originChoice.GetComponentInChildren<Text>().text = "Start: " + plannedStart;
            targetChoice.GetComponentInChildren<Text>().text = "Target: " + gridTarget;
            originChoice.interactable = targetChoice.interactable = !routeConfirmed;
            routeSummary.text = "Total distance: " + navigation.TotalDistance.ToString("F1") + " m";
            routeOne.interactable = !routeConfirmed && navigation.CandidateRoutes.Count > 0;
            routeTwo.gameObject.SetActive(navigation.CandidateRoutes.Count > 1); routeTwo.interactable = !routeConfirmed;
            routeOne.GetComponent<Image>().color = selectedCandidate == 0 ? new Color32(0, 119, 102, 255) : Pale;
            routeOne.GetComponentInChildren<Text>().color = selectedCandidate == 0 ? Color.white : Ink;
            routeTwo.GetComponent<Image>().color = selectedCandidate == 1 ? Amber : new Color32(248, 237, 218, 255);
            routeTwo.GetComponentInChildren<Text>().color = selectedCandidate == 1 ? Color.white : Ink;
            string arrow, title, next, distance;
            Guidance(false, out arrow, out title, out next, out distance);
            directionArrow.text = arrow; instructionText.text = title; nextPointText.text = next; distanceText.text = distance;
            startButton.GetComponentInChildren<Text>().text = PrimaryRouteLabel();
            startButton.interactable = pendingPreparation == -2 && (!navigation.Preparing || navigation.ReadyToStart);
            if (navigation.Preparing && arMode && (!ar.IsVerified || !ar.IsTracking)) startButton.interactable = false;
            endButton.GetComponentInChildren<Text>().text = navigation.Active ? "End route" : "Replan";
            bool checking = navigation.Preparing || pendingPreparation != -2;
            topPreview.gameObject.SetActive(!routeConfirmed); firstPersonPreview.gameObject.SetActive(!routeConfirmed);
            infoRect.gameObject.SetActive(routeConfirmed); destinationsRect.gameObject.SetActive(!routeConfirmed);
            pauseButton.gameObject.SetActive(navigation.Preparing); cancelPreparationButton.gameObject.SetActive(checking);
            endButton.gameObject.SetActive(routeConfirmed && !checking && !navigation.Arrived);
            viewChoice.gameObject.SetActive(routeConfirmed && !navigation.Arrived);
            PositionButton(viewChoice, checking ? 264 : 202, 64, checking ? 114 : 176, 38);
            PositionButton(cancelPreparationButton, navigation.Preparing ? 138 : 12, 64, navigation.Preparing ? 114 : 240, 38);
            navigationFooter.sizeDelta = new Vector2(390, routeConfirmed && !navigation.Arrived ? 116 : 70);
            Stretch(mainScroll.GetComponent<RectTransform>(), 0, 70, 0, routeConfirmed && !navigation.Arrived ? 122 : 76);
            string warning = RelevantWarning(); noticeText.text = warning;
            noticeRect.gameObject.SetActive(!string.IsNullOrEmpty(warning));
            noticeRect.anchoredPosition = new Vector2(12, routeConfirmed ? -148 : -554);
            mapRect.anchoredPosition = new Vector2(12, routeConfirmed ? string.IsNullOrEmpty(warning) ? -152 : -214 : -84);
            mainContent.sizeDelta = new Vector2(390, routeConfirmed ? string.IsNullOrEmpty(warning) ? 462 : 526 : string.IsNullOrEmpty(warning) ? 554 : 616);
            map.Route = navigation.Route; map.ProgressIndex = navigation.Active || navigation.Arrived ? navigation.ProgressIndex : 0;
            map.Frozen = navigation.Paused; map.HasPosition = navigation.HasPosition && polling; map.CurrentPosition = navigation.CurrentPosition;
            map.Current = connection.HasFreshLiveSelected && !navigation.Paused; map.HasTarget = true; map.Target = GridPosition(gridTarget);
            map.PlannedStart = GridPosition(plannedStart); map.SelectedCandidate = selectedCandidate;
            map.Alternative = null;
            if (!routeConfirmed && navigation.CandidateRoutes.Count > 1)
            { var alt = new List<Vector3>(); foreach (int n in navigation.CandidateRoutes[1 - selectedCandidate]) alt.Add(GridPosition(n)); map.Alternative = alt; }
            map.SetVerticesDirty();
            for (int i = 0; i < gridNumbers.Count; i++) PlaceMapLabel(gridNumbers[i], GridPosition(i));
            for (int i = 0; i < roomNames.Count; i++) PlaceMapLabel(roomNames[i], roomPositions[i]);
            RefreshArHud();
        }

        private void PlaceMapLabel(Text label, Vector3 point)
        { var rt = label.rectTransform; rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(.5f, .5f); rt.anchoredPosition = map.MapPoint(point); }
        private static Vector3 GridPosition(int label) { return new Vector3(label % 5 * .8f, 0, label / 5 * .8f); }
        private bool TryTarget(out Vector3 target) { target = GridPosition(gridTarget); return true; }
        private static void PositionButton(Button button, float x, float y, float width, float height)
        {
            var rect = button.GetComponent<RectTransform>(); rect.anchoredPosition = new Vector2(x, -y); rect.sizeDelta = new Vector2(width, height);
            var label = button.GetComponentInChildren<Text>().rectTransform; label.sizeDelta = new Vector2(width - 10, height - 8);
        }

        // Presentation only: position acceptance, progress and arrival stay in LiveNavigationState.
        // Never parse rendered/localized Text to decide a turn or an action.
        private void Guidance(bool cameraHeading, out string arrow, out string title, out string next, out string distance, Vector3? mapForward = null)
        {
            arrow = "\u2191";
            distance = (navigation.Active || navigation.Arrived ? "Remaining: " + navigation.RemainingDistance.ToString("F1")
                : "Total distance: " + navigation.TotalDistance.ToString("F1")) + " m";
            if (pendingPreparation != -2) { arrow = "..."; title = "Connecting"; next = "Waiting for the server"; return; }
            if (navigation.Preparing)
            {
                arrow = navigation.ReadyToStart ? "\u2713" : "\u25cf";
                title = navigation.ReadyToStart ? "Ready to depart" : "Stand at grid " + navigation.PreparationCheckpointLabel;
                next = navigation.ReadyToStart ? "Start when ready" : "Position check: " + navigation.PreparationStableCount + " / 3";
                return;
            }
            if (navigation.Arrived) { arrow = "\u2713"; title = "Arrival confirmed"; next = "Grid " + gridTarget; return; }
            if (!navigation.Active)
            {
                title = displayChosen ? "Start at grid " + plannedStart : "Route confirmed";
                next = displayChosen ? "Face the 14-9-4 side" : "Choose 2D or AR"; return;
            }
            if (navigation.Paused) { arrow = "\u2016"; title = "Route paused"; next = "Check position to continue"; return; }
            if (!navigation.CsiAvailable) { arrow = "!"; title = "Position unavailable"; next = "Wait for fresh live data"; return; }
            if (navigation.AtTarget) { arrow = "\u2713"; title = "At target estimate"; next = "Confirm only if you are there"; return; }
            next = "Next grid " + navigation.GridLabels[Mathf.Min(navigation.ProgressIndex + 1, navigation.GridLabels.Count - 1)];
            Vector3 forward = cameraHeading ? mapForward ?? ar.MapCameraForward : navigation.ProgressIndex == 0 ? Vector3.right
                : navigation.Route[navigation.ProgressIndex] - navigation.Route[navigation.ProgressIndex - 1];
            forward.y = 0;
            if (!SiteNavigationConfig.Finite(forward.x) || !SiteNavigationConfig.Finite(forward.z) || forward.sqrMagnitude < .01f)
            { arrow = "!"; title = cameraHeading ? "Raise phone" : "Check route direction"; return; }
            Vector3 direction = navigation.NextTarget - navigation.CurrentPosition; direction.y = 0;
            float angle = Vector3.SignedAngle(forward.normalized, direction, Vector3.up);
            if (Mathf.Abs(angle) >= 135) { arrow = "\u2193"; title = "Turn back"; }
            else if (angle >= 25) { arrow = "\u2192"; title = "Turn right"; }
            else if (angle <= -25) { arrow = "\u2190"; title = "Turn left"; }
            else { arrow = "\u2191"; title = "Straight"; }
        }

        private string RelevantWarning()
        {
            bool positionStage = navigation.Active || navigation.Preparing || pendingPreparation != -2;
            if (positionStage && (connection.Backend.Connection == CsiConnectionState.Disconnected
                || connection.Backend.Connection == CsiConnectionState.Reconnecting || connection.Backend.Connection == CsiConnectionState.Error))
                return "Connection lost. Check Settings.";
            if (positionStage && connection.Backend.PredictionPollingEnabled && !connection.HasFreshLiveSelected)
                return "Waiting for fresh live position.";
            if (navigation.Preparing && !navigation.ReadyToStart)
            {
                if (navigation.RawLabel >= 0 && navigation.RawLabel != navigation.PreparationCheckpointLabel)
                    return "CSI estimate: " + navigation.RawLabel + ". Stand at grid " + navigation.PreparationCheckpointLabel + ".";
                if (Time.unscaledTime - preparationOpened > 12) return "Still checking. Retry or cancel.";
            }
            if (navigation.Active && !navigation.Paused && navigation.RawLabel >= 0 && !navigation.GridLabels.Contains(navigation.RawLabel))
                return "Position estimate is off the route.";
            // Routine plan/connection text belongs in Settings, not over the map.
            if (Time.unscaledTime < arActionFeedbackUntil && !string.IsNullOrEmpty(notice)
                && (notice.IndexOf("fail", StringComparison.OrdinalIgnoreCase) >= 0 || notice.IndexOf("error", StringComparison.OrdinalIgnoreCase) >= 0
                    || notice.StartsWith("Choose different", StringComparison.Ordinal) || notice.StartsWith("Position check cancelled", StringComparison.Ordinal)))
                return CompactMessage(notice);
            if (!string.IsNullOrEmpty(config.LastError)) return CompactMessage(config.LastError);
            return "";
        }
        private static string CompactMessage(string value)
        {
            if (string.IsNullOrEmpty(value)) return "";
            int line = value.IndexOf('\n'); if (line >= 0) value = value.Substring(0, line);
            int sentence = value.IndexOf(". ", StringComparison.Ordinal); if (sentence >= 0) value = value.Substring(0, sentence + 1);
            return value.Length > 112 ? value.Substring(0, 109) + "..." : value;
        }
        private void RetryPreparation()
        {
            int label = navigation.Preparing ? navigation.PreparationCheckpointLabel : pendingPreparation;
            if (label == -2) return;
            CloseModal(); SuspendForScan(); PrepareAtPoint(label); Refresh();
        }
        private void CancelPositionCheck()
        {
            CloseModal(); SuspendForScan(); SetNotice("Position check cancelled. Route kept.");
        }
        private string PrimaryRouteLabel()
        {
            if (pendingPreparation != -2) return "Connecting...";
            if (navigation.Arrived) return "Plan another route";
            if (navigation.Preparing) return navigation.ReadyToStart ? "Start route" : "Checking position...";
            if (navigation.Active) return navigation.Paused ? "Check & resume" : navigation.AtTarget ? "Confirm arrival" : "Pause route";
            if (!routeConfirmed) return "Confirm route";
            if (!displayChosen) return "Choose 2D or AR";
            return "I am at start / ready";
        }
        private void PrimaryRouteAction()
        {
            if (navigation.Arrived) { ResetPlanning(); return; }
            if (navigation.Preparing) { if (navigation.ReadyToStart) StartRoute(); return; }
            if (navigation.Active) { if (!navigation.Paused && navigation.AtTarget) ConfirmArrival(); else PauseOrResume(); return; }
            if (!routeConfirmed) { routeConfirmed = true; OpenDisplayChoice(); Refresh(); return; }
            if (!displayChosen) { OpenDisplayChoice(); return; }
            PrepareAtPoint(-1);
        }
        private void OpenDisplayChoice()
        {
            OpenModal("Choose navigation view"); ModalText("Your route is fixed. 2D uses the map. AR first checks the fixed reference image; return to the start afterwards.", 88);
            ModalButton("Use 2D map", () => ChooseDisplay(false)); ModalButton("Use AR camera", () => ChooseDisplay(true)); FinishModal();
        }
        private void ChooseDisplay(bool camera)
        {
            if (!routeConfirmed) { SetNotice("Confirm the planned route first."); return; }
            CloseModal(); displayChosen = true; modelCheckMode = false; SetCameraMode(camera, false);
            if (camera && (!ar.IsVerified || !ar.IsTracking)) SuspendForScan();
            SetNotice(camera ? "Scan and verify the reference image, then return to the route start. Preparation does not advance the route." : "Return to grid " + plannedStart + ", facing the 14-9-4 side. Confirm when ready.");
        }
        private void PrepareAtPoint(int label)
        {
            if (!routeConfirmed || modelCheckMode) { SetNotice("Confirm a route first."); return; }
            if (arMode && (!ar.IsVerified || !ar.IsTracking)) { SetNotice("Complete the AR floor check first, or return to the map."); return; }
            if (!connection.Backend.IsLiveTransport || connection.Backend.Meta == null || connection.Backend.Connection == CsiConnectionState.Disconnected)
            {
                pendingPreparation = label; connection.SetPredictionPolling(false);
                if (CsiBackendClient.ValidBaseUrl(connection.Address)) connection.Connect(connection.Address); else OpenConnection();
                SetNotice("Connecting before the position check. Position requests remain off until server metadata is available. Cancel check stops this preparation."); return;
            }
            pendingPreparation = -2;
            connection.SetPredictionPolling(true); navigation.ResetObservations(); connectionRevision = connection.ConnectionRevision; lastModel = connection.SelectedModelId; lastIdentity = "";
            if (!navigation.BeginPreparation(label)) { SetNotice(navigation.Notice); return; }
            preparationOpened = Time.unscaledTime; notice = "";
            SetNotice("Stand on grid " + navigation.PreparationCheckpointLabel + ". Waiting for fresh, distinct live windows; the route will not advance yet.");
        }
        private void StartRoute()
        {
            if (!connection.HasFreshLiveSelected) { SetNotice("Wait for a fresh live model result and a ready server."); return; }
            if (arMode && (!ar.IsVerified || !ar.IsTracking)) { SetNotice("Restore AR alignment or return to the map."); return; }
            navigation.StartPrepared(); notice = ""; arActionFeedback = ""; Refresh();
        }
        private void PauseOrResume()
        {
            if (!navigation.Active) return;
            if (navigation.Paused) { OpenRecovery(); return; }
            navigation.Pause(); SetNotice(navigation.Notice);
        }
        private void OpenRecovery()
        {
            OpenModal("Resume the fixed route");
            ModalText("Return to a point on your chosen route. Select where you are standing; fresh CSI must confirm it before Resume. The original route stays fixed.", 90);
            foreach (int point in navigation.GridLabels)
            { int label = point; ModalButton("I am at grid " + label, () => { CloseModal(); PrepareAtPoint(label); }); }
            ModalButton("End route", OpenEnd); FinishModal();
        }
        private void OpenRouteOptions()
        {
            OpenModal("Route options");
            ModalText(navigation.Notice, 85);
            if (navigation.Preparing || pendingPreparation != -2)
            { ModalButton("Retry check", RetryPreparation); ModalButton("Cancel check", CancelPositionCheck); }
            if (navigation.Active) ModalButton(navigation.Paused ? "Choose recovery point" : "Pause route", () => { CloseModal(); PauseOrResume(); });
            if (routeConfirmed) ModalButton(navigation.Active ? "End route" : "Replan", RequestNewPlan);
            ModalButton("Connection details", OpenConnection); FinishModal();
        }
        private void OpenGridTarget() { OpenPointPicker(false); }
        private void OpenPointPicker(bool start)
        {
            if (routeConfirmed) { SetNotice("Use Replan to change a confirmed route."); return; }
            OpenModal(start ? "Choose planning start" : "Choose destination");
            ModalText("Select a floor label from 0 to 14. This plans the route; it does not claim a measured current position.", 76);
            for (int i = 0; i < 15; i++)
            { int label = i; ModalButton("Grid " + label, () => { if ((start ? gridTarget : plannedStart) == label) { SetNotice("Choose different start and target points."); return; }
                if (start) plannedStart = label; else gridTarget = label; selectedCandidate = 0; navigation.Plan(plannedStart, gridTarget); CloseModal(); notice = ""; Refresh(); }); }
            FinishModal();
        }
        private void ChooseCandidate(int index)
        { if (routeConfirmed) return; if (navigation.SelectCandidate(index)) { selectedCandidate = index; notice = ""; } Refresh(); }
        private void RequestNewPlan() { if (navigation.Active) OpenEnd(); else ResetPlanning(); }
        private void ResetPlanning()
        {
            if (!connection.StopLog()) { SetNotice(connection.LogMessage); return; }
            CloseModal(); pendingPreparation = -2; connection.SetPredictionPolling(false); navigation.End();
            routeConfirmed = displayChosen = false; modelCheckMode = false; selectedCandidate = 0;
            navigation.Plan(plannedStart, gridTarget); lastIdentity = ""; notice = ""; SetCameraMode(false, false); Refresh();
        }
        private void SuspendForScan()
        { pendingPreparation = -2; navigation.Pause(); connection.SetPredictionPolling(false); navigation.ResetObservations(); lastIdentity = ""; }
        private void SetCameraMode(bool enabled, bool openControls = true)
        {
            bool returningToMap = arMode && !enabled;
            if (!enabled) modelCheckMode = false;
            arMode = enabled; background.color = enabled ? Color.clear : new Color32(246, 249, 244, 255);
            if (enabled && !ar.Ready && !ar.IsBusy) ar.Begin(config);
            if (enabled) arControlsHidden = false;
            if (returningToMap)
            {
                arActionFeedback = "";
                notice = navigation.Active || navigation.Preparing || navigation.Arrived ? navigation.Notice
                    : routeConfirmed ? "2D map selected. Return to grid " + plannedStart + ", facing the 14-9-4 side. Confirm when ready."
                    : "Select start and target points, then confirm your route. Position requests are off.";
            }
            Refresh();
        }
        private void OpenActions()
        {
            OpenModal("Positioning settings"); ModalButton("Server connection", OpenConnection);
            if (navigation.Preparing || pendingPreparation != -2)
            { ModalButton("Retry check", RetryPreparation); ModalButton("Cancel check", CancelPositionCheck); }
            else if (navigation.Active) ModalButton(navigation.Paused ? "Choose recovery point" : "Pause route", () => { CloseModal(); PauseOrResume(); });
            if (routeConfirmed && !navigation.Arrived) ModalButton(navigation.Active ? "End route" : "Replan", RequestNewPlan);
            if (connection.AvailableModelIds.Length > 1) ModalButton("Available server models", OpenModels);
            ModalButton("Reference image measurements", OpenImageSetup); ModalButton("Model overlay check", OpenModelCheck);
            ModalButton(connection.Recording ? "Stop & save logs" : "Record received data", () => { if (connection.Recording) connection.StopLog(); else connection.StartLog(); SetNotice(connection.LogMessage); });
            ModalText(sourceText.text + "\n" + positionText.text, 110);
            ModalText("CSI drives position and progress. AR supplies camera heading and aligned display. Logs contain estimates, not independent truth.\n" + connection.LogMessage, 120); FinishModal();
        }
        private void OpenEnd()
        {
            OpenModal("End this route?"); ModalText("The current route will end without confirming arrival. You can then choose a new plan.", 66);
            ModalButton("End and plan again", ResetPlanning); ModalButton("Continue route", CloseModal); FinishModal();
        }
        private void HandleBack()
        {
            var current = UnityEngine.EventSystems.EventSystem.current == null ? null : UnityEngine.EventSystems.EventSystem.current.currentSelectedGameObject;
            var input = current == null ? null : current.GetComponent<InputField>();
            if (input != null && input.isFocused) { input.DeactivateInputField(); return; }
            if (modal != null) { CloseModal(); return; }
            if (arMode) { SetCameraMode(false, false); return; }
            OpenModal("Return Home?"); ModalText("Leaving ends this navigation session. Your saved endpoint and reference image measurements are retained.", 77);
            ModalButton("Return Home", () => { if (!connection.StopLog()) { SetNotice(connection.LogMessage); return; }
                navigation.End(); connection.SetPredictionPolling(false); quitting = true; ar.Shutdown(); SceneManager.LoadScene("UiDemo"); });
            ModalButton("Stay here", CloseModal); FinishModal();
        }
        private void OpenVirtualPreview(bool firstPerson)
        {
            if (navigation.Active || routeConfirmed) { SetNotice("Model simulation is available while planning. It never drives the live route."); return; }
            if (!connection.StopLog()) { SetNotice(connection.LogMessage); return; }
            connection.SetPredictionPolling(false);
            PlanningPreviewTransfer.Labels = new List<int>(navigation.GridLabels); PlanningPreviewTransfer.Candidate = selectedCandidate;
            PlanningPreviewTransfer.FirstPerson = firstPerson; PlanningPreviewTransfer.ReturnToPlanning = true;
            quitting = true; ar.Shutdown(); SceneManager.LoadScene("ApartmentMobilePresentation");
        }
    }
    public static class PlanningPreviewTransfer
    {
        public static List<int> Labels;
        public static int Candidate;
        public static bool FirstPerson, ReturnToPlanning;
    }
}
