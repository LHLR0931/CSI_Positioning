using System;
using System.Collections.Generic;
using UnityEngine;

namespace CsiApp
{
    public enum LiveNavigationPhase { None, Planning, Preparing, Ready, Active, Paused, Arrived }
    // Plan, raw observations and accepted cursor are separate. Repeated-window guards are provisional.
    public sealed class LiveNavigationState
    {
        public readonly List<Vector3> Route = new List<Vector3>();
        public readonly List<int> GridLabels = new List<int>();
        public readonly List<List<int>> CandidateRoutes = new List<List<int>>();
        public const int PreparationWindowsRequired = 3, AdjacentWindowsRequired = 2, JumpWindowsRequired = 3;
        // Compatibility only: external approaches are disabled.
        public const float ArrivalPromptRadiusM = .30f, HandoffAgreementM = .60f, ApproachDeviationM = .60f;
        public Vector3 CurrentPosition { get; private set; }
        public Vector3 RawPosition { get; private set; }
        public int RawLabel { get { return LatestCsiLabel; } }
        public int LatestCsiLabel { get; private set; } = -1;
        public int CurrentLabel { get; private set; } = -1;
        public int TargetLabel { get; private set; } = -1;
        public int ConnectorLabel { get { return TargetLabel; } }
        public int PlannedStartLabel { get; private set; } = -1;
        public int PlannedTargetLabel { get { return TargetLabel; } }
        public int SelectedCandidateIndex { get; private set; }
        public int ProgressIndex { get; private set; }
        public int PreparationCheckpointLabel { get; private set; } = -1;
        public int PreparationStableCount { get; private set; }
        public bool PreparationIsResume { get; private set; }
        public bool PreparationHasWindowBoundary { get; private set; }
        public bool Planned { get { return GridLabels.Count > 1; } }
        public bool Preparing { get; private set; }
        public bool ReadyToStart { get; private set; }
        public bool HasPosition { get; private set; }
        public bool Active { get; private set; }
        public bool Paused { get; private set; }
        public bool AtTarget { get; private set; }
        public bool Arrived { get; private set; }
        public bool InApproach { get { return false; } }
        public bool ArVerified { get; private set; }
        public bool ArTracking { get; private set; }
        public bool CsiAvailable { get; private set; }
        public string DestinationId { get { return Planned ? "grid" : null; } }
        public string PositionSource { get; private set; } = "no accepted position";
        public string Notice { get; private set; } = "Choose a planning start and target.";
        public string ArNotice { get; private set; }
        public float ApproachProgress { get { return ProgressIndex * .8f; } }
        public LiveNavigationPhase Phase { get { return Arrived ? LiveNavigationPhase.Arrived
            : Preparing ? ReadyToStart ? LiveNavigationPhase.Ready : LiveNavigationPhase.Preparing
            : Active ? Paused ? LiveNavigationPhase.Paused : LiveNavigationPhase.Active
            : Planned ? LiveNavigationPhase.Planning : LiveNavigationPhase.None; } }
        private readonly SiteNavigationConfig config;
        private readonly Dictionary<string, long> sequences = new Dictionary<string, long>();
        private readonly Dictionary<string, DateTime> times = new Dictionary<string, DateTime>();
        private readonly HashSet<string> windows = new HashSet<string>();
        private string identity, csiSource = "unknown", serverState;
        private bool needsConnectionObservation = true, baselineNeeded;
        private int preparationIndex, pendingIndex = -1, pendingCount;
        private DateTime baselineTime, baselineEnd;
        public LiveNavigationState(SiteNavigationConfig configuration) { config = configuration ?? new SiteNavigationConfig(); }

        public bool Plan(int start, int target, int candidateIndex = 0)
        {
            if (Active) { Notice = "End the current route before planning another one."; return false; }
            var candidates = GridRoutePlanner.Candidates(start, target);
            if (candidates.Count == 0 || candidateIndex < 0 || candidateIndex >= candidates.Count)
            { Notice = start == target ? "Choose different start and target points." : "No valid grid route is available."; return false; }
            CandidateRoutes.Clear(); foreach (var path in candidates) CandidateRoutes.Add(new List<int>(path));
            PlannedStartLabel = start; TargetLabel = target;
            Preparing = ReadyToStart = Paused = Arrived = AtTarget = HasPosition = false;
            CurrentLabel = -1; ProgressIndex = 0; ResetObservationStorage(); ApplyCandidate(candidateIndex);
            Notice = "Planning start is manual. Review the route before preparing live positioning."; return true;
        }
        public bool SelectCandidate(int index)
        {
            if (Active || Preparing || index < 0 || index >= CandidateRoutes.Count)
            { Notice = "Select a route while planning, before preparing to depart."; return false; }
            ApplyCandidate(index); Notice = "Route selected. The manual planning point is not a CSI observation."; return true;
        }
        private void ApplyCandidate(int index)
        {
            SelectedCandidateIndex = index; GridLabels.Clear(); GridLabels.AddRange(CandidateRoutes[index]); Route.Clear();
            foreach (int label in GridLabels) { var p = ApartmentMap.Current.FindPoint(label); Route.Add(ApartmentMap.ToWorld(p.x_m, p.y_m)); }
        }
        public bool BeginPreparation(int checkpointLabel = -1)
        {
            if (!Planned || Arrived) { Notice = "Choose and confirm a new route first."; return false; }
            bool resume = Active;
            if (checkpointLabel < 0) checkpointLabel = resume ? GridLabels[ProgressIndex] : PlannedStartLabel;
            int index = GridLabels.IndexOf(checkpointLabel);
            if (index < 0 || !resume && checkpointLabel != PlannedStartLabel)
            { Notice = "Initial preparation uses the planned start; a resume checkpoint must be on the route."; return false; }
            if (Active) Paused = true;
            Preparing = true; ReadyToStart = AtTarget = false; PreparationIsResume = resume;
            preparationIndex = index; PreparationCheckpointLabel = checkpointLabel;
            PreparationStableCount = 0; baselineNeeded = true; PreparationHasWindowBoundary = false; ResetPending();
            Notice = "Checking point " + checkpointLabel + ". First response is a baseline; progress remains frozen."; return true;
        }
        public bool StartPrepared()
        {
            if (!Preparing || !ReadyToStart || !CsiAvailable || needsConnectionObservation
                || LatestCsiLabel != PreparationCheckpointLabel || PreparationStableCount < PreparationWindowsRequired)
            { Notice = "Wait for fresh ready live windows matching the checkpoint before Start."; return false; }
            ProgressIndex = preparationIndex; AdoptCursor(); Active = true; Paused = false;
            Preparing = ReadyToStart = false; Arrived = false; ResetPending(); RefreshAtTarget();
            Notice = "Started explicitly. Only later confirmed live windows update the fixed route."; return true;
        }
        public void ResetObservations()
        {
            ResetObservationStorage(); CancelPreparation(); ResetPending();
            if (HasPosition) PositionSource = "historical / CSI / " + csiSource;
            PauseWith("Connection changed. Keep the plan and prepare again with new live windows.");
        }
        private void ResetObservationStorage()
        {
            sequences.Clear(); times.Clear(); windows.Clear(); identity = null;
            LatestCsiLabel = -1; CsiAvailable = false; serverState = null; needsConnectionObservation = true;
        }
        public Vector3 NextTarget { get { return Route.Count == 0 ? CurrentPosition : Route[Mathf.Min(ProgressIndex + 1, Route.Count - 1)]; } }
        public float RemainingDistance { get { return Route.Count == 0 ? 0 : Mathf.Max(0, Route.Count - 1 - ProgressIndex) * .8f; } }
        public float TotalDistance { get { return Mathf.Max(0, Route.Count - 1) * .8f; } }

        public bool AcceptCsi(CsiPrediction prediction, string source, bool fresh = true)
        {
            string nextSource = string.IsNullOrWhiteSpace(source) ? "source unspecified" : source.Trim().ToLowerInvariant();
            bool sourceChanged = identity != null && csiSource != nextSource;
            if (sourceChanged) ObservationChanged();
            if (nextSource == "replay")
            { CsiAvailable = false; needsConnectionObservation = true; CancelPreparation(); PauseWith("Replay cannot drive live navigation. Prepare with live data."); return false; }
            string error;
            if (!CsiValidation.Prediction(prediction, out error)) { SetCsiAvailable(false, error); return false; }
            string nextIdentity = (prediction.session_id ?? "unidentified") + "|" + CsiValidation.ModelId(prediction);
            bool changed = identity != null && identity != nextIdentity;
            if (changed) ObservationChanged();
            DateTime stamp, oldTime; long oldSeq; CsiValidation.TryTime(prediction.timestamp, out stamp);
            if (sequences.TryGetValue(nextIdentity, out oldSeq) && prediction.seq <= oldSeq
                || times.TryGetValue(nextIdentity, out oldTime) && stamp < oldTime) return false;
            identity = nextIdentity; sequences[identity] = prediction.seq; times[identity] = stamp; csiSource = nextSource;
            LatestCsiLabel = prediction.label; RawPosition = ApartmentMap.ToWorld(prediction.x_m, prediction.y_m);
            serverState = string.IsNullOrEmpty(prediction.state) ? "ready" : prediction.state;
            CsiAvailable = fresh && serverState == "ready" && csiSource == "live";
            if (CsiAvailable) needsConnectionObservation = false;
            bool newWindow = windows.Add(identity + "|" + WindowKey(prediction));
            if (!CsiAvailable)
            {
                ResetPending(); ResetPreparationEvidence();
                if (Active && !Paused) PauseWith("CSI is stale, degraded or not declared live. Progress frozen.");
                else if (Preparing) Notice = "Waiting for fresh ready live data; mock/unspecified cannot prepare navigation.";
                return true;
            }
            if (sourceChanged || changed || !newWindow) return true;
            if (Preparing) ObservePreparation(prediction, stamp);
            else if (Active && !Paused) ObserveProgress(prediction.label);
            return true;
        }
        private static string WindowKey(CsiPrediction p)
        { return !string.IsNullOrEmpty(p.window_id) ? p.window_id : p.window != null ? p.window.start + "/" + p.window.end : p.timestamp; }
        private void ObservePreparation(CsiPrediction prediction, DateTime stamp)
        {
            if (baselineNeeded)
            {
                baselineNeeded = false; baselineTime = stamp; PreparationStableCount = 0; ReadyToStart = false;
                PreparationHasWindowBoundary = prediction.window != null && CsiValidation.TryTime(prediction.window.end, out baselineEnd);
                Notice = "Baseline received. Wait for three different new windows at point " + PreparationCheckpointLabel + "."; return;
            }
            DateTime start;
            if (stamp <= baselineTime || PreparationHasWindowBoundary && (prediction.window == null
                || !CsiValidation.TryTime(prediction.window.start, out start) || start <= baselineEnd))
            { Notice = "Waiting for a window entirely after the preparation baseline."; return; }
            if (prediction.label != PreparationCheckpointLabel)
            { PreparationStableCount = 0; ReadyToStart = false; Notice = "Planning point " + PreparationCheckpointLabel + "; CSI predicts " + prediction.label + ". Progress frozen."; return; }
            PreparationStableCount = Math.Min(PreparationWindowsRequired, PreparationStableCount + 1);
            ReadyToStart = PreparationStableCount >= PreparationWindowsRequired;
            Notice = ReadyToStart ? "Checkpoint observations stable. Press Start explicitly; not a ground-truth measurement."
                : "Checking point " + PreparationCheckpointLabel + ": " + PreparationStableCount + "/3 new windows.";
        }
        private void ObserveProgress(int label)
        {
            // Keep the accepted cursor, but do not offer arrival while current raw evidence disagrees.
            if (label != TargetLabel) AtTarget = false;
            int index = GridLabels.IndexOf(label);
            if (index < 0)
            { ResetPending(); Notice = "CSI predicts off-route point " + label + ". Plan and progress retained; return or replan explicitly."; return; }
            if (index == ProgressIndex) { ResetPending(); RefreshAtTarget(); Notice = "Current route point confirmed by live CSI."; return; }
            if (pendingIndex != index) { pendingIndex = index; pendingCount = 1; } else pendingCount++;
            int required = index == ProgressIndex + 1 ? AdjacentWindowsRequired : JumpWindowsRequired;
            if (pendingCount < required)
            { Notice = "CSI point " + label + ": waiting for " + pendingCount + "/" + required + " distinct new windows before changing progress."; return; }
            bool backwards = index < ProgressIndex; ProgressIndex = index; AdoptCursor(); ResetPending(); RefreshAtTarget();
            Notice = AtTarget ? "Repeated target estimate. Confirm arrival only if physically there."
                : backwards ? "Repeated CSI confirms backtracking. Remaining route restored." : "Accepted progress on the fixed route.";
        }
        private void AdoptCursor()
        { CurrentLabel = GridLabels[ProgressIndex]; CurrentPosition = Route[ProgressIndex]; HasPosition = true; PositionSource = "CSI / " + csiSource; }
        private void ResetPending() { pendingIndex = -1; pendingCount = 0; }
        private void ResetPreparationEvidence()
        { if (Preparing) { PreparationStableCount = 0; ReadyToStart = false; baselineNeeded = true; } }
        private void CancelPreparation() { Preparing = ReadyToStart = false; PreparationStableCount = 0; baselineNeeded = true; }
        private void ObservationChanged()
        {
            needsConnectionObservation = true; CsiAvailable = false; CancelPreparation(); ResetPending();
            PauseWith("Model, session or source changed. Keep the plan and prepare again explicitly.");
        }
        public void SetCsiAvailable(bool available, string message = null)
        {
            bool before = CsiAvailable;
            CsiAvailable = available && !needsConnectionObservation && LatestCsiLabel >= 0 && serverState == "ready" && csiSource == "live";
            if (!CsiAvailable)
            {
                ResetPending(); ResetPreparationEvidence();
                if (Active && !Paused) PauseWith(message ?? "Live CSI unavailable. Progress frozen; prepare before resuming.");
                else if (before && Preparing) Notice = message ?? "Live CSI unavailable. Checking checkpoint again.";
            }
        }
        public void Pause() { PauseWith("Navigation paused. Prepare at a route checkpoint before resuming."); }
        private void PauseWith(string message) { if (Active) Paused = true; AtTarget = false; Notice = message; }
        public bool Resume() { return BeginPreparation(); }
        public void End()
        {
            Active = Paused = Arrived = AtTarget = false; CancelPreparation(); ResetPending();
            Notice = "Route ended. Plan remains visible; explicitly plan or prepare a new run.";
        }
        public bool ConfirmArrival()
        {
            if (!Active || Paused || !AtTarget || !CsiAvailable || LatestCsiLabel != TargetLabel)
            { Notice = "A current confirmed target and physical arrival confirmation are required."; return false; }
            Arrived = true; Active = false; Paused = false; CancelPreparation();
            Notice = "Arrival confirmed manually. Not independent positioning accuracy evidence."; return true;
        }
        private void RefreshAtTarget() { AtTarget = Active && !Paused && ProgressIndex == GridLabels.Count - 1; }
        // AR supplies heading/alignment only. It never changes CSI position or grid validity.
        public void SetArAlignment(bool verified, string message = null)
        { string error; ArVerified = verified && config.ValidateMarker(out error); if (!ArVerified) ArTracking = false; ArNotice = message; }
        public bool AcceptAr(Vector3 mapPosition, bool tracking)
        { ArTracking = tracking && ArVerified && SiteNavigationConfig.Finite(mapPosition.x) && SiteNavigationConfig.Finite(mapPosition.y) && SiteNavigationConfig.Finite(mapPosition.z); return ArTracking; }
        public bool StartGrid(int label) { Notice = "Plan and prepare before Start. Direct start disabled."; return false; }
        public bool Start(string id) { Notice = "Only planned point-to-point navigation is supported."; return false; }
        public bool ConfirmApproach() { Notice = "External destination approaches have been removed."; return false; }
        public string Instruction()
        {
            if (Preparing) return Notice;
            if (!Active) return Arrived ? "Arrival confirmed manually." : "Review the plan and prepare at its start.";
            if (Paused) return Notice;
            if (!CsiAvailable) return "Stop: fresh ready live CSI unavailable.";
            if (AtTarget) return "At target estimate. Confirm arrival only if physically there.";
            return TurnInstruction(ProgressIndex == 0 ? Vector3.right : Route[ProgressIndex] - Route[ProgressIndex - 1], false);
        }
        public string Instruction(Vector3 mapForward)
        {
            if (!Active || Paused || AtTarget || !CsiAvailable) return Instruction();
            if (!ArVerified || !ArTracking) return "Restore verified AR tracking or use 2D map.";
            return TurnInstruction(mapForward, true);
        }
        private string TurnInstruction(Vector3 forward, bool cameraHeading)
        {
            forward.y = 0;
            if (!SiteNavigationConfig.Finite(forward.x) || !SiteNavigationConfig.Finite(forward.z) || forward.sqrMagnitude < .01f)
                return cameraHeading ? "Raise the phone for a reliable horizontal heading." : "Check the planned route direction.";
            Vector3 direction = NextTarget - CurrentPosition; direction.y = 0;
            float angle = Vector3.SignedAngle(forward.normalized, direction, Vector3.up);
            string turn = Mathf.Abs(angle) >= 135 ? "Turn back \u2193" : angle >= 25 ? "Turn right \u2192" : angle <= -25 ? "Turn left \u2190" : "Straight \u2191";
            return turn + " toward grid " + GridLabels[Mathf.Min(ProgressIndex + 1, GridLabels.Count - 1)] + "; " + RemainingDistance.ToString("F1") + " m remaining.";
        }
    }
}
