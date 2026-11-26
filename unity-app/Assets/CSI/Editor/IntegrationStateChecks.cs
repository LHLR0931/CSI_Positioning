using System;

namespace CsiApp.Editor
{
    // Also compiled as a small standalone check by the implementation workflow.
    public static class IntegrationStateChecks
    {
        private static int assertions;
        private static void Check(bool condition, string message)
        { assertions++; if (!condition) throw new Exception("Integration check failed: " + message); }
        private static CsiPrediction P(string model = "KNN", long seq = 1, string window = "w1", string session = "s1", int label = 7)
        {
            var probs = new float[15]; probs[label] = 1;
            string timestamp = "2026-10-21T00:00:" + seq.ToString("D2") + "Z";
            return new CsiPrediction { model_id = model, seq = seq, window_id = window, session_id = session,
                timestamp = timestamp, label = label, x = label % 5, y = label / 5,
                window = new CsiWindow { start = DateTimeOffset.Parse(timestamp).AddSeconds(-1.6).ToString("o"), end = timestamp, frames = 16 },
                expected_m = new CsiExpected { x = label % 5 * .8f, y = label / 5 * .8f },
                x_m = label % 5 * .8f, y_m = label / 5 * .8f, confidence = 1, probs = probs, source = "mock", state = "ready" };
        }
        public static int Run()
        {
            assertions = 0; var state = new DeveloperSessionState(); string error;
            Check(state.GroundTruth == -1 && double.IsNaN(state.ErrorMeters), "no invented initial truth");
            Check(state.Accept(P(), "live", DateTime.UtcNow), "first actual row accepted");
            Check(state.PredictionForSelected.EffectiveSource == "mock", "connection mode cannot relabel mock live");
            Check(state.Accept(P("GCN"), "live", DateTime.UtcNow), "same-window different model accepted");
            Check(!state.Accept(P(), "live", DateTime.UtcNow), "duplicate rejected");
            state.SetGroundTruth(6); Check(Math.Abs(state.ErrorMeters - .8) < .0001, "manual independent truth uses meters");
            state.StartRecording(); Check(state.RecordCount == 2, "record only two actually received models");
            state.SelectModel("ConcatMLP"); Check(state.PredictionForSelected == null && state.RecordCount == 2, "missing model waits, no fabricated rows");
            Check(state.Accept(P("KNN", 2, "w2"), "live", DateTime.UtcNow), "next window accepted");
            Check(state.ResultFor("GCN") == null && state.RecordCount == 3, "previous-window models cleared");
            Check(!state.Accept(P("GCN", 1), "live", DateTime.UtcNow), "late previous window rejected");
            int rows = state.RecordCount; state.ResetData();
            Check(state.RecordCount == rows && !state.Recording && state.GroundTruth == -1, "mode reset preserves record but clears data/truth");
            Check(state.Accept(P(null), "live", DateTime.UtcNow), "v0.1 model unspecified accepted");
            Check(state.UnknownResult != null && state.ResultFor("KNN") == null, "unlabelled result never assigned to KNN");
            state.SelectModel(CsiModels.Unspecified); Check(state.PredictionForSelected != null, "explicit unlabelled display available");
            state.ResetData(); var replay = P(); replay.source = "live"; replay.ground_truth_label = 6; replay.truth_source = "fixture_label";
            state.Accept(replay, "replay", DateTime.UtcNow); state.SelectModel("KNN");
            Check(state.GroundTruth == 6 && state.TruthSource == "replay:fixture_label", "recorded replay truth tagged");
            Check(state.PredictionForSelected.EffectiveSource == "replay", "replay of live data remains replay transport");
            state.SetGroundTruth(-1); Check(state.GroundTruth == -1 && double.IsNaN(state.ErrorMeters), "clear overrides replay truth");
            state.UseRecordedTruth(); Check(state.GroundTruth == 6, "restore recorded truth");
            state.ResetData(); state.SelectModel("KNN"); state.Accept(replay, "live", DateTime.UtcNow);
            Check(state.GroundTruth == -1, "live payload cannot automatically set truth");
            state.SetGroundTruth(7); state.StartRecording();
            Check(state.Accept(P("KNN", 0, "restart", "s2"), "live", DateTime.UtcNow) == false, "older foreign-session timestamp rejected");
            var restarted = P("KNN", 0, "restart", "s2"); restarted.timestamp = "2026-10-21T00:01:00Z";
            restarted.window.start = "2026-10-21T00:00:58.4Z"; restarted.window.end = restarted.timestamp;
            Check(state.Accept(restarted, "live", DateTime.UtcNow), "new identified server session can reset sequence");
            Check(state.GroundTruth == -1 && !state.Recording, "new server session clears old manual truth/stops recording");
            Check(!state.Accept(P("KNN", 2, "w2", "s1"), "live", DateTime.UtcNow), "retired session packets rejected");
            var bad = P(); bad.label = 15; Check(!CsiValidation.Prediction(bad, out error), "label out of range");
            bad = P(); bad.x_m = 2; Check(!CsiValidation.Prediction(bad, out error), "indices not mistaken for meter coords");
            bad = P(); bad.confidence = float.NaN; Check(!CsiValidation.Prediction(bad, out error), "nonfinite confidence");
            bad = P(); bad.probs[0] = .3f; Check(!CsiValidation.Prediction(bad, out error), "probability sum");
            bad = P(); bad.probs[7] = .2f; bad.probs[0] = .8f; bad.confidence = .2f;
            Check(!CsiValidation.Prediction(bad, out error), "label must be argmax");
            state.ClearRecords(); Check(state.RecordCount == 0 && state.Csv().StartsWith("run_id,"), "clear and CSV header");
            state.ResetData(); state.Accept(P(), "live", DateTime.UtcNow); state.SetGroundTruth(6);
            state.StartRecording(false); Check(state.RecordCount == 0, "live recording does not relabel a preexisting prediction");
            state.Accept(P("KNN", 2, "w2"), "live", DateTime.UtcNow);
            Check(state.RecordCount == 1, "future-only recording captures the next actual window");
#if UNITY_EDITOR
            string single = UnityEngine.JsonUtility.ToJson(P(null));
            Check(CsiJson.ParsePredictions(single)[0].model_id == null || CsiJson.ParsePredictions(single)[0].model_id == "", "single v0.1 JSON stays unlabelled");
            string batch = "{\"source\":\"mock\",\"session_id\":\"s1\",\"predictions\":[" + UnityEngine.JsonUtility.ToJson(P()) + "," + UnityEngine.JsonUtility.ToJson(P("GCN")) + "]}";
            Check(CsiJson.ParsePredictions(batch).Count == 2, "atomic JSON batch parsed");
            Check(!CsiJson.ParseStatus("{\"state\":\"training\",\"progress\":null}").Progress.HasValue, "null progress preserved");
            Check(Math.Abs(CsiJson.ParseStatus("{\"state\":\"calibrating\",\"progress\":0.4}").Progress.Value - .4f) < .001f, "numeric progress parsed");
            bool rejected = false; try { CsiJson.ParsePredictions(single.Replace("\"label\":7", "\"label\":15")); } catch (FormatException) { rejected = true; }
            Check(rejected, "JSON invalid label rejected");
            rejected = false; try { CsiJson.ParsePredictions(single.Replace("\"x_m\":1.6", "\"x_m\":null")); } catch (FormatException) { rejected = true; }
            Check(rejected, "JSON null coordinate rejected");
            string nullOptional = single.Replace("\"expected_m\":" + CsiJson.Property(single, "expected_m"), "\"expected_m\":null")
                .Replace("\"window\":" + CsiJson.Property(single, "window"), "\"window\":null");
            var optional = CsiJson.ParsePredictions(nullOptional)[0];
            Check(optional.expected_m == null && optional.window == null, "explicit null optional objects remain null");
            string fixturePath = System.IO.Path.GetFullPath(System.IO.Path.Combine(UnityEngine.Application.dataPath, "../../tests/fixtures/integration/replay.json"));
            var realistic = CsiJson.ParseReplay(System.IO.File.ReadAllText(fixturePath));
            Check(realistic.Predictions.Count == 24 && realistic.Meta.grid.cols == 5, "committed realistic replay fixture parsed");
            Check(realistic.Predictions[0].source == "mock" && realistic.Predictions[0].truth_source == "fixture_label", "realistic fixture provenance and truth preserved");
            string liveFixture = UnityEngine.JsonUtility.ToJson(realistic.Predictions[0]);
            liveFixture = System.Text.RegularExpressions.Regex.Replace(liveFixture, ",?\"ground_truth_label\":-?[0-9]+", "");
            liveFixture = liveFixture.Replace(",\"truth_source\":\"fixture_label\"", "");
            var noTruth = CsiJson.ParsePredictions(liveFixture)[0];
            Check(noTruth.ground_truth_label == -1 && string.IsNullOrEmpty(noTruth.truth_source), "omitted live truth stays unset through native JSON parsing");
            var liveState = new DeveloperSessionState(); liveState.SelectModel(noTruth.model_id); liveState.Accept(noTruth, "live", DateTime.UtcNow);
            Check(liveState.GroundTruth == -1 && double.IsNaN(liveState.ErrorMeters), "realistic no-truth live result never invents error");
#endif
            return assertions;
        }
#if UNITY_EDITOR
        public static void RunUnity()
        { UnityEngine.Debug.Log("CSI_INTEGRATION_STATE_PASS " + Run() + " assertions including JSON parsing"); }
#endif
#if CSI_INTEGRATION_CHECKS_STANDALONE
        public static int Main()
        { try { Console.WriteLine("CSI_INTEGRATION_STATE_PASS " + Run() + " assertions"); return 0; } catch (Exception e) { Console.Error.WriteLine(e); return 1; } }
#endif
    }
}
