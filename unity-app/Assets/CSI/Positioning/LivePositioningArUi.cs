using UnityEngine;
using UnityEngine.UI;

namespace CsiApp
{
    public sealed partial class LivePositioningController
    {
        private RectTransform arHud, arHudStatus, arHudFooter;
        private Text arHudTitle, arHudDetail, arHudArrow, arHudDistance, arHudWarning;
        private Button arPrimary, arRestore;
        private bool arControlsHidden, checkingFloor, showSiteGrid = true;
        private int floorCheckStep;
        private string arActionFeedback = "";
        private float arActionFeedbackUntil;
        private readonly bool[] checkedFloorPoints = new bool[3];
        private static readonly int[] FloorLabels = { 0, 4, 10 };

        private void BuildArHud()
        {
            arHud = new GameObject("AR camera controls", typeof(RectTransform)).GetComponent<RectTransform>();
            arHud.SetParent(safe, false); Stretch(arHud, 0, 0, 0, 0);
            arHudStatus = Panel(arHud, 12, 12, 366, 104, new Color32(20, 34, 40, 232));
            arHudArrow = MakeText(arHudStatus, "", 8, 8, 62, 80, 54, true); arHudArrow.alignment = TextAnchor.MiddleCenter;
            arHudTitle = MakeText(arHudStatus, "", 80, 9, 274, 29, 21, true);
            arHudDetail = MakeText(arHudStatus, "", 80, 40, 274, 24, 14);
            arHudDistance = MakeText(arHudStatus, "", 80, 68, 274, 25, 18, true);
            arHudWarning = MakeText(arHudStatus, "", 10, 101, 346, 26, 12, true);
            arHudWarning.resizeTextForBestFit = true; arHudWarning.resizeTextMinSize = 11; arHudWarning.resizeTextMaxSize = 12;
            arHudTitle.color = arHudDetail.color = arHudArrow.color = arHudDistance.color = Color.white;
            arHudWarning.color = new Color32(255, 216, 151, 255);
            arHudFooter = Panel(arHud, 0, 0, 390, 66, new Color32(20, 34, 40, 220));
            arHudFooter.anchorMin = arHudFooter.anchorMax = arHudFooter.pivot = new Vector2(.5f, 0);
            arHudFooter.anchoredPosition = Vector2.zero;
            MakeButton(arHudFooter, "Map", 12, 10, 72, 46, () => { CloseModal(); SetCameraMode(false, false); });
            arPrimary = MakeButton(arHudFooter, "Scan image", 94, 10, 190, 46, ArPrimaryAction);
            MakeButton(arHudFooter, "More", 294, 10, 84, 46, OpenArMenu);
            arRestore = MakeButton(arHud, "Controls", 0, 0, 94, 38, () => { arControlsHidden = false; Refresh(); });
            var rect = arRestore.GetComponent<RectTransform>(); rect.anchorMin = rect.anchorMax = rect.pivot = new Vector2(1, 0);
            rect.anchoredPosition = new Vector2(-12, 12); arHud.gameObject.SetActive(false);
        }

        private string DefaultPositioningNotice() { return navigation.Notice; }
        private void RefreshArHud()
        {
            pageHeader.gameObject.SetActive(!arMode); mainScroll.gameObject.SetActive(!arMode);
            navigationFooter.gameObject.SetActive(!arMode); arHud.gameObject.SetActive(arMode);
            if (!arMode) return;
            arHudStatus.gameObject.SetActive(true); arHudFooter.gameObject.SetActive(!arControlsHidden);
            arRestore.gameObject.SetActive(arControlsHidden);
            arHudArrow.text = "\u25ce"; arHudDistance.text = "";
            arHudTitle.text = ar.IsBusy ? "Starting camera" : !ar.IsCalibrated ? ar.HasCandidate ? "Ready to lock" : "Scan reference image"
                : ar.IsVerified ? "Alignment checked" : "Check floor points";
            string detail = ar.IsBusy ? "Keep the camera facing the room" : !ar.IsCalibrated ? ar.HasCandidate ? "Keep the image still; lock alignment" : "Aim at the complete wall image"
                : "Compare the real floor marks";
            string warning = "";
            if (checkingFloor && ar.IsCalibrated)
            {
                arHudTitle.text = floorCheckStep < 3 ? "Check grid " + FloorLabels[floorCheckStep] : "Three points checked";
                detail = floorCheckStep == 0 ? "Match position and floor height"
                    : floorCheckStep == 1 ? "Match 0 to 4 direction and distance"
                    : floorCheckStep == 2 ? "Match 0 to 10 direction and distance"
                    : "Confirm only if all three match";
                arHudDistance.text = "Confirm only after comparing";
            }
            else if (ar.IsVerified && ar.IsTracking)
            {
                if (modelCheckMode) { arHudTitle.text = "Compare the model"; detail = "Check wall bases and doorway sides"; arHudDistance.text = "Walls and grid: More"; }
                else if (routeConfirmed)
                { RefreshArNavigationGuidance(ar.MapCameraForward); detail = arHudDetail.text; warning = RelevantWarning(); }
                else { arHudTitle.text = "Alignment checked"; detail = "Use Map to choose a route"; }
            }
            if (!ar.IsBusy && ar.Ready && !ar.IsTracking)
            { arHudArrow.text = "!"; arHudTitle.text = "Tracking unavailable"; detail = "Scan again or return to Map"; arHudDistance.text = navigation.Active ? "Route progress is paused" : "Keep the camera facing the room"; }
            if (!ar.IsBusy && !ar.IsCalibrated && !ar.HasCandidate && (!ar.Ready || !config.markerMeasured)) warning = CompactMessage(ar.Message);
            if (siteModelOverlay != null && showSiteModel && siteModelOverlay.Status.StartsWith("Model overlay unavailable")) warning = "Model overlay unavailable. See More.";
            arHudDetail.text = detail;
            SetArWarning(warning);
            string primary = ar.IsBusy ? "Please wait" : !ar.IsCalibrated ? ar.HasCandidate ? "Lock alignment" : "Scan image"
                : checkingFloor ? floorCheckStep < 3 ? FloorLabels[floorCheckStep] + " matches" : "Confirm alignment"
                : !ar.IsVerified ? "Check floor points" : !ar.IsTracking ? "Scan again" : modelCheckMode ? "Scan again" : PrimaryRouteLabel();
            arPrimary.GetComponentInChildren<Text>().text = primary;
            arPrimary.interactable = !ar.IsBusy && !(ar.IsVerified && ar.IsTracking && (pendingPreparation != -2 || navigation.Preparing && !navigation.ReadyToStart));
        }
        // The same assignment method supports Editor layout checks on an explicitly
        // labelled test background, without fabricating an AR tracker/alignment state.
        private void RefreshArNavigationGuidance(Vector3 mapForward)
        {
            string arrow, title, next, distance;
            Guidance(true, out arrow, out title, out next, out distance, mapForward);
            arHudArrow.text = arrow; arHudTitle.text = title; arHudDetail.text = next; arHudDistance.text = distance;
        }
        private void SetArWarning(string warning)
        {
            arHudWarning.text = warning; arHudWarning.gameObject.SetActive(!string.IsNullOrEmpty(warning));
            arHudStatus.sizeDelta = new Vector2(366, string.IsNullOrEmpty(warning) ? 104 : 132);
        }
        private void ScanArAgain()
        {
            CloseModal(); SuspendForScan(); ClearVerificationChecks(); arControlsHidden = false; notice = ""; arActionFeedback = "";
            if (!arMode) SetCameraMode(true, false); else ar.Scan(config); Refresh();
        }
        private void ArPrimaryAction()
        {
            CloseModal(); if (ar.IsBusy) return;
            if (!ar.IsCalibrated)
            { if (ar.HasCandidate) { ClearVerificationChecks(); ar.LockAlignment(); } else ScanArAgain(); }
            else if (checkingFloor)
            {
                if (!ar.IsTracking) { ClearVerificationChecks(); Refresh(); return; }
                if (floorCheckStep < checkedFloorPoints.Length) { checkedFloorPoints[floorCheckStep] = true; floorCheckStep++; }
                else if (checkedFloorPoints[0] && checkedFloorPoints[1] && checkedFloorPoints[2]) { ar.VerifyAlignment(); checkingFloor = !ar.IsVerified; }
            }
            else if (!ar.IsVerified) { checkingFloor = true; floorCheckStep = 0; showSiteGrid = true; }
            else if (!ar.IsTracking || modelCheckMode) ScanArAgain();
            else PrimaryRouteAction();
            Refresh();
        }
        private void OpenArMenu()
        {
            OpenModal("AR controls");
            if (navigation.Preparing || pendingPreparation != -2)
            { ModalButton("Retry check", RetryPreparation); ModalButton("Cancel check", CancelPositionCheck); }
            else if (navigation.Active && navigation.Paused) ModalButton("Choose recovery point", OpenRecovery);
            else if (navigation.Active && navigation.AtTarget) ModalButton("Pause route", () => { CloseModal(); PauseOrResume(); });
            ModalButton(showSiteModel ? "Hide walls" : "Show walls", () => { showSiteModel = !showSiteModel; CloseModal(); Refresh(); });
            if (!checkingFloor) ModalButton(showSiteGrid ? "Hide grid" : "Show grid", () => { showSiteGrid = !showSiteGrid; CloseModal(); Refresh(); });
            ModalButton("Hide controls", () => { arControlsHidden = true; CloseModal(); Refresh(); });
            ModalButton("Scan reference again", ScanArAgain); ModalButton("Image measurements", OpenImageSetup);
            ModalButton(modelCheckMode ? "Back to navigation view" : "Inspect apartment model", () =>
            { if (modelCheckMode) { modelCheckMode = false; CloseModal(); Refresh(); } else OpenModelCheck(); });
            ModalButton("Connection & recording", OpenArDetails);
            if (routeConfirmed && !navigation.Arrived) ModalButton(navigation.Active ? "End route" : "Replan", RequestNewPlan);
            FinishModal();
        }
        private void OpenArDestination() { CloseModal(); SetCameraMode(false, false); RequestNewPlan(); }
        private void OpenArDetails()
        {
            OpenModal("AR details"); ModalText(ar.Message, 88);
            if (siteModelOverlay != null) ModalText(siteModelOverlay.Status, 58);
            ModalText(sourceText.text, 85); ModalButton("Connection", OpenConnection);
            if (connection.AvailableModelIds.Length > 1) ModalButton("Available server models", OpenModels);
            ModalButton(connection.Recording ? "Stop & save logs" : "Start received-data recording", () =>
            { if (connection.Recording) connection.StopLog(); else connection.StartLog(); SetNotice(connection.LogMessage); }); FinishModal();
        }
    }
}
