using System;
using UnityEngine;

namespace CsiApp.Editor
{
    public static class RealReplayChecks
    {
        private static int assertions;
        private static void Check(bool value, string message)
        { assertions++; if (!value) throw new InvalidOperationException("Real replay check failed: " + message); }
        public static int Run()
        {
            assertions = 0;
            var asset = Resources.Load<TextAsset>(RealReplayData.BundledResource);
            Check(asset != null, "real recorded dataset is bundled");
            var data = RealReplayData.Parse(asset.text);
            Check(data.WindowCount == 751 && data.Replay.Predictions.Count == 3004, "751 windows and four genuine model rows per window");
            Check(data.Title.Contains("seed 42") && data.Declaration.Contains("previously inspected"), "single-seed diagnostic evaluation is declared");
            int[] correct = { 379, 382, 457, 556 };
            double[] means = { .80221947844845687, .81084674073561391, .69553969212408773, .44601676274676488 };
            for (int i = 0; i < 4; i++)
            {
                var metric = data.MetricsFor(CsiModels.Ids[i]);
                Check(metric.Count == 751 && metric.Correct == correct[i], "paired accuracy count " + CsiModels.Ids[i]);
                Check(Math.Abs(metric.MeanError - means[i]) < .00000001, "independent grid-error mean " + CsiModels.Ids[i]);
                Check(data.Replay.Predictions[i].window_id == "d1-r9-w000" && data.Replay.Predictions[i].model_id == CsiModels.Ids[i], "models grouped in canonical order");
            }
            var boundary = data.Replay.Predictions[375 * 4];
            Check(boundary.window_id == "d1-r10-w000" && boundary.seq == 375, "round boundary does not reset model sequence");
            Check(data.Replay.Predictions[0].timestamp == "2026-10-08T13:27:20.887+11:00", "original observation time and timezone retained");
            Check(data.Replay.Predictions[0].truth_source == "capture_station_label" && data.Replay.Source == "recorded_csi", "real capture provenance differs from synthetic fixture");
            var state = new DeveloperSessionState();
            foreach (var p in data.Replay.Predictions) Check(state.Accept(p, "replay", DateTime.UtcNow), "ordered real row accepted " + p.seq + "/" + p.model_id);
            Check(state.WindowId == "d1-r10-w375", "all rounds are playable");
            string window = state.WindowId;
            foreach (string id in CsiModels.Ids) { state.SelectModel(id); Check(state.PredictionForSelected != null && state.WindowId == window, "model selection does not advance window"); }
            Check(!state.Accept(data.Replay.Predictions[data.Replay.Predictions.Count - 1], "replay", DateTime.UtcNow), "duplicate delivery rejected");
            Check(state.MetricsFor("KNN").Count == 751, "selection and duplicate delivery do not inflate measured count");
            state.SetGroundTruth(14);
            Check(data.MetricsFor("KNN").Correct == 379 && state.MetricsFor("KNN").Correct == 379, "manual truth cannot rewrite recorded replay metrics");
            state.ResetData(); state.Accept(data.Replay.Predictions[0], "replay", DateTime.UtcNow);
            Check(state.MetricsFor("KNN").Count == 1 && data.MetricsFor("KNN").Count == 751, "restart clears played metrics, not full-batch statistics");
            state.ResetData(); state.Accept(data.Replay.Predictions[0], "live", DateTime.UtcNow);
            Check(state.GroundTruth == -1 && state.MetricsFor("KNN").Count == 0, "live transport never consumes automatic capture truth");
            var r9 = RealReplayData.Parse(asset.text, "r9"); var r10 = RealReplayData.Parse(asset.text, "r10");
            Check(r9.WindowCount == 375 && r10.WindowCount == 376, "round filters retain exact window counts");
            Check(r9.MetricsFor("KNN").Count == 375 && r10.MetricsFor("KNN").Count == 376, "round statistics are scoped to selected data");
            var again = CsiJson.ParseReplay(data.CanonicalJson);
            Check(again.Predictions.Count == 3004 && again.Predictions[1500].seq == 375, "canonical JSON can be cached and reloaded without retraining");
            bool rejected = false;
            try { RealReplayData.Parse(asset.text.Replace("d1-final-cls_ampphase_pp2_seed42", "undeclared-model-version")); }
            catch (FormatException) { rejected = true; }
            Check(rejected, "unknown model version cannot inherit the D1 batch declaration");
            rejected = false;
            try { RealReplayData.Parse(asset.text.Replace("\"model\": \"GCN\"", "\"model\": \"KNN\"")); }
            catch (FormatException) { rejected = true; }
            Check(rejected, "duplicate model rows cannot masquerade as four-algorithm comparisons");
            return assertions;
        }
        public static void RunUnity() { Debug.Log("CSI_REAL_REPLAY_PASS " + Run() + " assertions / archived D1 data; not new live experiments"); }
    }
}
