using System;
using UnityEngine;

namespace CsiApp.Editor
{
    public static class PositioningSessionChecks
    {
        public static void RunUnity()
        {
            int checks = 0;
            Action<bool, string> check = (ok, message) => { checks++; if (!ok) throw new Exception("Positioning freshness: " + message); };
            DateTime now = new DateTime(2026, 10, 22, 0, 0, 0, DateTimeKind.Utc);
            var state = new DeveloperSessionState();
            Func<string, long, DateTime, CsiPrediction> prediction = (model, seq, time) => {
                var probabilities = new float[15]; probabilities[7] = 1;
                return new CsiPrediction { model_id = model, seq = seq, timestamp = time.ToString("o"), source = "mock",
                    state = "ready", label = 7, x = 2, y = 1, x_m = 1.6f, y_m = .8f, probs = probabilities, confidence = 1 };
            };
            state.Accept(prediction("KNN", 1, now), "live", now);
            var first = state.PredictionForSelected;
            check(PositioningSession.IsFreshResult(first, now.AddSeconds(2)), "fresh selected model usable");
            check(!PositioningSession.IsFreshResult(first, now.AddSeconds(4)), "old selected model expires");
            state.Accept(prediction("GCN", 2, now.AddSeconds(4)), "live", now.AddSeconds(4));
            check(state.PredictionForSelected == null, "missing selected result in new window cannot be replaced by other model");
            check(!PositioningSession.IsFreshResult(first, now.AddSeconds(4)), "peer update never refreshes older selected result");
            state.ResetData(); state.Accept(prediction("KNN", 3, now), "replay", now);
            check(!PositioningSession.IsFreshResult(state.PredictionForSelected, now), "replay never becomes live navigation");
            state.ResetData();
            var declaredReplay = prediction("KNN", 3, now); declaredReplay.source = " RePlAy ";
            state.Accept(declaredReplay, "live", now);
            check(!PositioningSession.IsFreshResult(state.PredictionForSelected, now), "HTTP transport cannot turn declared replay into live navigation");
            state.ResetData(); state.Accept(prediction("KNN", 4, now.AddSeconds(8)), "live", now);
            check(!PositioningSession.IsFreshResult(state.PredictionForSelected, now), "future source clock is not fresh");
            state.ResetData(); state.Accept(prediction("KNN", 5, now.AddSeconds(-8)), "live", now);
            check(!PositioningSession.IsFreshResult(state.PredictionForSelected, now), "recent receipt does not refresh old observation");
            check(!PositioningSession.IsDeclaredLive(first), "HTTP mock cannot qualify as declared real live positioning");
            state.ResetData();
            var logicalLiveFixture = prediction("F-MLP", 6, now); logicalLiveFixture.source = "live";
            state.Accept(logicalLiveFixture, "live", now); state.SelectModel("F-MLP");
            check(PositioningSession.IsDeclaredLive(state.PredictionForSelected), "source filter accepts an explicit live declaration (logic fixture, not a field result)");
            check(PositioningSession.PreferredModel(new CsiMeta { default_model_id = "F-MLP", supported_models = new[] { "KNN", "F-MLP" } }) == "F-MLP", "server default replaces hard-coded KNN");
            check(PositioningSession.PreferredModel(new CsiMeta { supported_models = new[] { "F-MLP" } }) == "F-MLP", "single advertised model can be selected before results exist");
            check(PositioningSession.PreferredModel(null) == CsiModels.Unspecified, "no metadata does not invent a model");
            check(PositioningSession.PreferredModel(new CsiMeta { supported_models = new[] { "KNN" } }, "F-MLP") == "F-MLP", "first observed model wins when the server declares no default");
            var gate = new PredictionPollingGate(); int pending = gate.Epoch;
            check(gate.Allows(pending), "enabled prediction request owns the current token");
            gate.SetEnabled(false);
            check(!gate.Allows(pending), "closing the gate invalidates an in-flight prediction");
            gate.SetEnabled(true);
            check(!gate.Allows(pending), "closing then reopening cannot revive the old response");
            int current = gate.Epoch; gate.SetEnabled(true);
            check(gate.Allows(current), "idempotent enable preserves the current request");
            CheckReplayImport(check, prediction, now);
            Debug.Log("CSI_POSITIONING_SESSION_PASS " + checks + " assertions");
        }
        private static void CheckReplayImport(Action<bool, string> check, Func<string, long, DateTime, CsiPrediction> make, DateTime now)
        {
            var go = new GameObject("Prediction gate and replay checks");
            try
            {
                var client = go.AddComponent<CsiBackendClient>();
                client.SetPredictionPolling(false);
                var meta = new CsiMeta { grid = new CsiGrid { cols = 5, rows = 3, spacing_m = .8f }, cells = new CsiCell[15] };
                for (int label = 0; label < 15; label++)
                    meta.cells[label] = new CsiCell { label = label, x = label % 5, y = label / 5, x_m = label % 5 * .8f, y_m = label / 5 * .8f };
                var rows = new[] { make("KNN", 1, now), make("KNN", 2, now.AddSeconds(1)), make("GCN", 1, now), make("GCN", 2, now.AddSeconds(1)) };
                foreach (var row in rows)
                {
                    row.session_id = "unit-replay"; row.window_id = "w" + row.seq;
                    // JsonUtility serializes null inline DTOs as default objects on some Unity versions.
                    // Give this wire-format fixture valid optional objects rather than emitting an
                    // invalid zero-frame window or an expected coordinate of (0,0).
                    DateTime end; CsiValidation.TryTime(row.timestamp, out end);
                    row.window = new CsiWindow { start = end.AddSeconds(-1.6).ToString("o"), end = row.timestamp, frames = 16 };
                    row.expected_m = new CsiExpected { x = row.x_m, y = row.y_m };
                }
                string json = "{\"meta\":" + JsonUtility.ToJson(meta) + ",\"predictions\":[" +
                    string.Join(",", Array.ConvertAll(rows, row => JsonUtility.ToJson(row))) + "]}";
                bool loaded = client.LoadReplayJson(json);
                check(loaded, "canonical replay import succeeds while live polling is disabled: " + client.Notice);
                check(client.ReplayCount == 2, "nonadjacent model rows are grouped into two shared windows");
                check(!client.IsLiveTransport && !client.PredictionPollingEnabled, "replay does not reopen live position reading");
                int emitted = 0; string window = null;
                client.PredictionReceived += (row, mode, received) => { emitted++; window = row.window_id; check(mode == "replay", "replay delivery retains replay transport"); };
                client.StepReplay();
                check(emitted == 2 && window == "w1", "first replay step emits both models from the earliest window");
                client.StepReplay();
                check(emitted == 4 && window == "w2", "second replay step emits the second shared window");
                check(!client.LoadReplayJson("{}") && client.ReplayCount == 2, "invalid import preserves the last usable in-memory replay");
                client.Disconnect();
                check(client.IsLiveTransport && !client.PredictionPollingEnabled, "leaving replay does not bypass the prediction gate");
            }
            finally { UnityEngine.Object.DestroyImmediate(go); }
        }
    }
}
