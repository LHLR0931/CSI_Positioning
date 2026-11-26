using System;
using System.Collections.Generic;
using UnityEngine;

namespace CsiApp.Editor
{
    // Synthetic observations test state transitions; no test fixture is saved as site evidence.
    public static class LiveNavigationChecks
    {
        private static int assertions;
        private static void Check(bool value, string message)
        { assertions++; if (!value) throw new InvalidOperationException("Live navigation check failed: " + message); }
        private static readonly DateTime Epoch = new DateTime(2026, 10, 25, 0, 0, 0, DateTimeKind.Utc);
        private static CsiPrediction Result(int label, long seq, string window = null, string session = "test-A", string model = "KNN", string state = "ready")
        {
            var probabilities = new float[15]; probabilities[label] = 1;
            return new CsiPrediction { label = label, seq = seq, session_id = session, model_id = model, source = "live", state = state,
                window_id = window ?? "window-" + seq, timestamp = Epoch.AddSeconds(seq * 2).ToString("o"),
                x = label % 5, y = label / 5, x_m = label % 5 * .8f, y_m = label / 5 * .8f, confidence = 1, probs = probabilities };
        }
        private static CsiPrediction WindowResult(int label, long seq, double start, double end)
        {
            var r = Result(label, seq); r.timestamp = Epoch.AddSeconds(end).ToString("o");
            r.window = new CsiWindow { start = Epoch.AddSeconds(start).ToString("o"), end = r.timestamp, frames = 16 }; return r;
        }
        private static SiteNavigationConfig Marker()
        { return new SiteNavigationConfig { markerMeasured = true, imageWidthM = .14f, imageX = .2f, imageY = -.4f, imageHeightM = 1.2f, imageFacingDeg = 0 }; }
        private static LiveNavigationState Running(int start, int target, ref long sequence)
        {
            var n = new LiveNavigationState(Marker()); Check(n.Plan(start, target), "plan before preparation");
            Check(n.BeginPreparation(), "begin initial preparation");
            n.AcceptCsi(Result(start, ++sequence), "live");
            Check(n.PreparationStableCount == 0 && !n.ReadyToStart, "first response is baseline only");
            for (int i = 0; i < 3; i++) n.AcceptCsi(Result(start, ++sequence), "live");
            Check(n.ReadyToStart && !n.Active && !n.HasPosition, "three new windows do not auto start");
            Check(n.StartPrepared(), "explicit prepared start"); return n;
        }
        private static string Labels(LiveNavigationState n) { return string.Join(",", n.GridLabels); }
        public static int Run()
        {
            assertions = 0; string error;
            Check(!new SiteNavigationConfig().ValidateMarker(out error), "unmeasured marker remains invalid");
            var bad = Marker(); bad.imageWidthM = float.NaN;
            Check(!bad.ValidateMarker(out error) && !bad.Save(), "nonfinite marker cannot be saved");
            var partial = Marker(); partial.destinations = new[] { partial.FindDestination("balcony"), partial.FindDestination("office") };
            Check(!partial.Save(), "partial legacy configuration cannot be persisted");
            foreach (float yaw in new[] { 0f, 90f, 180f, 270f })
            {
                var config = Marker(); config.imageFacingDeg = yaw; Pose p = config.MarkerMapPose;
                var normal = new Vector3(Mathf.Sin(yaw * Mathf.Deg2Rad), 0, Mathf.Cos(yaw * Mathf.Deg2Rad));
                Check(Vector3.Distance(p.rotation * Vector3.up, normal) < .0001f, "reference front normal " + yaw);
                Check(Vector3.Distance(p.rotation * Vector3.forward, Vector3.up) < .0001f, "reference top upright " + yaw);
                var observed = new Pose(new Vector3(12, 1.7f, -4), Quaternion.Euler(11, 33, 17));
                var frame = SiteArTracker.ComposeMapPose(observed, p);
                Check(Vector3.Distance(frame.position + frame.rotation * p.position, observed.position) < .0001f, "map transform reconstructs image " + yaw);
            }
            for (int from = 0; from < 15; from++) for (int to = 0; to < 15; to++)
            {
                var routes = GridRoutePlanner.Candidates(from, to);
                if (from == to) { Check(routes.Count == 0, "same point has no fabricated trip"); continue; }
                bool valid = routes.Count >= 1 && routes.Count <= 2;
                foreach (var path in routes)
                {
                    valid &= path[0] == from && path[path.Count - 1] == to;
                    valid &= path.Count - 1 == Math.Abs(from % 5 - to % 5) + Math.Abs(from / 5 - to / 5);
                    for (int i = 1; i < path.Count; i++) valid &= ApartmentMap.Current.HasEdge(path[i - 1], path[i]);
                }
                Check(valid, "recommended shortest grid routes " + from + " to " + to);
            }
            var alternatives = GridRoutePlanner.Candidates(0, 14);
            Check(alternatives.Count == 2 && GridRoutePlanner.Turns(alternatives[0]) == 1 && GridRoutePlanner.Turns(alternatives[1]) == 1,
                "opposite corners recommend two distinct one-turn paths");
            Check(string.Join(",", alternatives[0]) != string.Join(",", alternatives[1]), "candidates are distinct");
            Check(GridRoutePlanner.Candidates(-1, 14).Count == 0, "invalid planning point rejected");
            var n = new LiveNavigationState(Marker());
            Check(!n.StartGrid(14) && !n.Start("balcony") && !n.ConfirmApproach(), "old direct/approach entry points safely disabled");
            Check(!n.Plan(0, 0) && n.Plan(0, 14) && n.SelectCandidate(1), "manual plan and candidate selection");
            Check(!n.HasPosition && n.RawLabel == -1 && Math.Abs(n.TotalDistance - 4.8f) < .001f, "planning is not positioning");
            Check(!n.BeginPreparation(5) && n.BeginPreparation(), "initial checkpoint must be the chosen start");
            Check(!n.SelectCandidate(0), "cannot replace route while preparing");
            n.AcceptCsi(Result(0, 1), "mock"); Check(!n.CsiAvailable && !n.ReadyToStart && n.RawLabel == 0, "mock visible but cannot prepare");
            n.AcceptCsi(Result(0, 2), "source unspecified"); Check(!n.CsiAvailable, "unspecified source cannot prepare");
            Check(!n.AcceptCsi(Result(0, 3), "replay"), "replay cannot prepare live navigation");
            n.End(); n.Plan(0, 14); n.BeginPreparation();
            n.AcceptCsi(Result(14, 1), "live"); Check(n.PreparationStableCount == 0 && !n.AtTarget, "wrong baseline does not move to target");
            n.AcceptCsi(Result(0, 2, "same-window"), "live"); n.AcceptCsi(Result(0, 3, "same-window"), "live");
            Check(n.PreparationStableCount == 1, "increasing seq with same window counts only once");
            Check(!n.AcceptCsi(Result(0, 3), "live") && n.PreparationStableCount == 1, "duplicate sequence ignored");
            n.AcceptCsi(Result(1, 4), "live"); Check(n.PreparationStableCount == 0, "nonmatching point breaks consecutive evidence");
            n.AcceptCsi(Result(0, 5), "live"); n.AcceptCsi(Result(0, 6), "live"); n.AcceptCsi(Result(0, 7), "live");
            Check(n.ReadyToStart && !n.Active, "three matching independent windows ready but not started");
            n.SetCsiAvailable(false); Check(!n.ReadyToStart && !n.StartPrepared(), "staleness revokes ready qualification");

            var bounded = new LiveNavigationState(Marker()); bounded.Plan(0, 14); bounded.BeginPreparation();
            bounded.AcceptCsi(WindowResult(0, 1, 0, 1.6), "live");
            bounded.AcceptCsi(WindowResult(0, 2, .4, 2), "live");
            bounded.AcceptCsi(WindowResult(0, 3, 1.6, 3.2), "live");
            Check(bounded.PreparationHasWindowBoundary && bounded.PreparationStableCount == 0, "overlap or touching baseline end does not count");
            bounded.AcceptCsi(WindowResult(0, 4, 1.7, 3.3), "live");
            bounded.AcceptCsi(WindowResult(0, 5, 3.4, 5), "live");
            bounded.AcceptCsi(WindowResult(0, 6, 5.1, 6.7), "live");
            Check(bounded.ReadyToStart && bounded.StartPrepared(), "whole new windows permit explicit start");
            var fallback = new LiveNavigationState(Marker()); fallback.Plan(0, 14); fallback.BeginPreparation();
            for (int i = 1; i <= 4; i++) { var r = Result(0, i); r.window_id = null; fallback.AcceptCsi(r, "live"); }
            Check(!fallback.PreparationHasWindowBoundary && fallback.ReadyToStart, "missing boundaries use distinct source timestamps explicitly");

            long seq = 0; n = Running(0, 14, ref seq); string fixedRoute = Labels(n); Vector3 frozen = n.CurrentPosition;
            Check(n.Instruction().Contains("Straight"), "initial 2D direction uses user-agreed +x reference");
            n.AcceptCsi(Result(14, ++seq), "live"); Check(n.RawLabel == 14 && n.ProgressIndex == 0 && !n.AtTarget, "single target jump never consumes route");
            n.AcceptCsi(Result(0, ++seq), "live"); n.AcceptCsi(Result(1, ++seq), "live");
            Check(n.ProgressIndex == 0 && n.CurrentPosition == frozen, "one adjacent window does not move cursor");
            n.AcceptCsi(Result(1, ++seq), "live"); Check(n.ProgressIndex == 1 && n.CurrentLabel == 1 && Labels(n) == fixedRoute, "confirmed adjacent progress preserves entire plan");
            float distance = n.RemainingDistance;
            n.AcceptCsi(Result(5, ++seq), "live"); n.AcceptCsi(Result(5, ++seq), "live"); n.AcceptCsi(Result(5, ++seq), "live");
            Check(n.RawLabel == 5 && n.ProgressIndex == 1 && n.RemainingDistance == distance && Labels(n) == fixedRoute, "off-route evidence cannot snap or replan");
            n.AcceptCsi(Result(14, ++seq), "live"); n.AcceptCsi(Result(14, ++seq), "live");
            Check(!n.AtTarget && n.ProgressIndex == 1, "two distant jumps still held");
            n.AcceptCsi(Result(14, ++seq), "live"); Check(n.AtTarget && !n.Arrived && n.Active && Labels(n) == fixedRoute, "three repeated target windows only prompt arrival");
            n.AcceptCsi(Result(0, ++seq), "live"); n.AcceptCsi(Result(0, ++seq), "live");
            Check(!n.AtTarget && n.ProgressIndex == n.GridLabels.Count - 1 && !n.ConfirmArrival(), "raw disagreement hides arrival without moving accepted cursor");
            n.AcceptCsi(Result(0, ++seq), "live"); Check(n.ProgressIndex == 0 && !n.AtTarget && n.RemainingDistance > distance, "confirmed backtracking restores remaining distance");
            n.Pause(); frozen = n.CurrentPosition; n.AcceptCsi(Result(1, ++seq), "live");
            Check(n.Paused && n.CurrentPosition == frozen && !n.Instruction().Contains("Straight"), "paused raw data cannot resume or issue movement");
            Check(n.BeginPreparation(9), "explicit resume checkpoint on selected route");
            for (int i = 0; i < 4; i++) n.AcceptCsi(Result(9, ++seq), "live");
            Check(n.ReadyToStart && n.Paused && n.StartPrepared() && n.CurrentLabel == 9 && Labels(n) == fixedRoute, "checkpoint preparation resumes same fixed plan explicitly");
            n.SetArAlignment(true); n.AcceptAr(new Vector3(100, 1.5f, -100), true);
            Check(!n.Paused && n.CurrentLabel == 9, "AR camera bounds do not gate 2D or move CSI position");
            n.SetArAlignment(false); n.AcceptAr(Vector3.zero, false); Check(!n.Paused, "AR invalidity does not pause pure 2D logic");
            n.AcceptCsi(Result(14, ++seq), "live"); n.AcceptCsi(Result(14, ++seq), "live");
            Check(n.AtTarget && n.ConfirmArrival() && n.Arrived && !n.Active, "manual arrival after confirmed adjacent progress");
            Check(!n.BeginPreparation(), "arrived route does not restart itself");
            n.End(); Check(!n.Active && n.Planned && Labels(n) == fixedRoute && n.Phase == LiveNavigationPhase.Planning, "End preserves reviewable plan but stops navigation");

            seq = 0; n = Running(0, 14, ref seq); frozen = n.CurrentPosition;
            Check(!n.Plan(1, 14), "active plan cannot be silently replaced");
            n.AcceptCsi(Result(1, ++seq, state: "degraded"), "live");
            Check(n.Paused && !n.CsiAvailable && n.CurrentPosition == frozen, "degraded positioning freezes active progress");
            n.AcceptCsi(Result(0, ++seq), "live"); Check(n.Paused && !n.StartPrepared(), "ready recovery cannot automatically resume");
            n.Resume(); for (int i = 0; i < 4; i++) n.AcceptCsi(Result(0, ++seq), "live");
            Check(n.ReadyToStart && n.StartPrepared(), "Resume initiates preparation rather than adopting raw point");
            Check(!n.AcceptCsi(Result(1, seq), "mock") && n.Paused && !n.CsiAvailable, "source change pauses even duplicate row");
            n.SetCsiAvailable(true); Check(!n.CsiAvailable && !n.StartPrepared(), "availability flag cannot bypass rejected source change");
            n.ResetObservations(); Check(n.RawLabel == -1 && n.Paused && Labels(n) == fixedRoute, "connection reset preserves plan and clears evidence");
            n.AcceptCsi(Result(0, 0, session: "test-B"), "live"); Check(n.Paused, "new session sequence zero never auto resumes");
            n.BeginPreparation(); n.AcceptCsi(Result(0, 1, session: "test-B"), "live");
            n.AcceptCsi(Result(0, 2, session: "test-B"), "live");
            n.AcceptCsi(Result(0, 3, session: "test-B", model: "GCN"), "live");
            Check(!n.Preparing && !n.ReadyToStart && n.Paused, "model change cancels partially accumulated preparation");
            return assertions;
        }
        public static void RunUnity()
        { Debug.Log("CSI_LIVE_NAVIGATION_PASS " + Run() + " assertions / synthetic state checks; no site accuracy claim"); }
    }
}
