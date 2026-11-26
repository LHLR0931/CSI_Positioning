using System;
using System.Collections.Generic;
using UnityEngine;

namespace CsiApp
{
    // Offline inference results, not a live service and not generated demonstration values.
    public sealed class RealReplayData
    {
        public const string BundledResource = "Replay/replay_d1_r9r10_seed42";
        public CsiReplay Replay { get; private set; }
        public string CanonicalJson { get; private set; }
        public string Title { get; private set; }
        public string Declaration { get; private set; }
        public string RoundFilter { get; private set; }
        public int WindowCount { get; private set; }
        private readonly Dictionary<string, DeveloperMetrics> metrics = new Dictionary<string, DeveloperMetrics>();
        public DeveloperMetrics MetricsFor(string model) { DeveloperMetrics value; return metrics.TryGetValue(model, out value) ? value : new DeveloperMetrics(); }

        [Serializable] private sealed class RawMeta
        {
            public string session, capture_date, calibration_round, preprocessing, test_rounds_used_for;
            public string[] train_rounds, test_rounds;
            public int seed;
            public CsiGrid grid;
        }
        [Serializable] private sealed class RawRecord
        {
            public string session, round, window_id, model, model_version, t_start, t_end;
            public int window_index, true_label, pred_label;
            public float true_x_m, true_y_m, pred_x_m, pred_y_m, confidence;
            public float[] probs;
        }
        [Serializable] private sealed class RawFile { public RawMeta meta; public RawRecord[] records; }
        [Serializable] private sealed class WireFile
        { public string source, session_id; public CsiMeta meta; public CsiPrediction[] predictions; }
        private sealed class Group
        { public string Key; public DateTime Time; public readonly Dictionary<string, RawRecord> Rows = new Dictionary<string, RawRecord>(); }

        public static RealReplayData Parse(string json, string roundFilter = "all")
        {
            if (roundFilter != "all" && roundFilter != "r9" && roundFilter != "r10") throw new FormatException("Choose all, r9 or r10.");
            if (CsiJson.Property(json, "records") == null) return FromCanonical(json);
            var raw = new RawFile(); JsonUtility.FromJsonOverwrite(json, raw);
            if (raw.meta == null || raw.records == null || raw.records.Length == 0 || raw.meta.grid == null
                || raw.meta.grid.cols != 5 || raw.meta.grid.rows != 3 || Math.Abs(raw.meta.grid.spacing_m - .8f) > .001f)
                throw new FormatException("Recorded data requires records and the agreed 15-point 0.8 m grid.");
            if (raw.meta.session != "d1" || raw.meta.seed != 42 || raw.meta.calibration_round != "r8")
                throw new FormatException("This recorded-data adapter expects the declared D1 final seed-42 export.");
            if (raw.meta.train_rounds == null || raw.meta.train_rounds.Length != 8 || raw.meta.test_rounds == null
                || raw.meta.test_rounds.Length != 2 || raw.meta.test_rounds[0] != "r9" || raw.meta.test_rounds[1] != "r10")
                throw new FormatException("Recorded data must declare training r1-r8 and held-out rounds r9/r10.");
            for (int i = 0; i < 8; i++) if (raw.meta.train_rounds[i] != "r" + (i + 1))
                throw new FormatException("Recorded training-round declaration differs from D1 final.");
            var groups = new Dictionary<string, Group>();
            foreach (var row in raw.records)
            {
                if (row == null || Array.IndexOf(CsiModels.Ids, row.model) < 0 || row.session != raw.meta.session
                    || string.IsNullOrEmpty(row.window_id) || (row.round != "r9" && row.round != "r10")
                    || row.model_version != "d1-final-cls_ampphase_pp2_seed42"
                    || row.true_label < 0 || row.true_label > 14 || row.pred_label < 0 || row.pred_label > 14)
                    throw new FormatException("Recorded row has an invalid identity, model or truth label.");
                if (roundFilter != "all" && row.round != roundFilter) continue;
                DateTime start, end;
                if (!CsiValidation.TryTime(row.t_start, out start) || !CsiValidation.TryTime(row.t_end, out end) || start > end)
                    throw new FormatException("Recorded frame times must include their original timezone.");
                if (!CsiValidation.Finite(row.true_x_m) || !CsiValidation.Finite(row.true_y_m)
                    || !CsiValidation.Finite(row.pred_x_m) || !CsiValidation.Finite(row.pred_y_m)
                    || Math.Abs(row.true_x_m - row.true_label % 5 * .8f) > .005f
                    || Math.Abs(row.true_y_m - row.true_label / 5 * .8f) > .005f
                    || Math.Abs(row.pred_x_m - row.pred_label % 5 * .8f) > .005f
                    || Math.Abs(row.pred_y_m - row.pred_label / 5 * .8f) > .005f)
                    throw new FormatException("Recorded grid coordinates disagree with labels.");
                string key = row.session + "|" + row.window_id;
                Group group;
                if (!groups.TryGetValue(key, out group)) { group = new Group { Key = key, Time = end }; groups.Add(key, group); }
                if (group.Rows.ContainsKey(row.model)) throw new FormatException("Duplicate model/window in recorded data.");
                if (group.Time != end) throw new FormatException("Models in one window have different observation times.");
                foreach (var previous in group.Rows.Values)
                    if (previous.true_label != row.true_label || previous.t_start != row.t_start || previous.round != row.round)
                        throw new FormatException("Models in one window disagree on capture truth or frame range.");
                group.Rows.Add(row.model, row);
            }
            var ordered = new List<Group>(groups.Values);
            ordered.Sort((a, b) => { int value = a.Time.CompareTo(b.Time); return value != 0 ? value : string.CompareOrdinal(a.Key, b.Key); });
            if (ordered.Count == 0) throw new FormatException("No windows in the selected round.");
            var cells = new CsiCell[15];
            for (int i = 0; i < 15; i++) cells[i] = new CsiCell { label = i, x = i % 5, y = i / 5, x_m = i % 5 * .8f, y_m = i / 5 * .8f };
            var meta = new CsiMeta { contract_version = "0.1", source = "recorded_csi", session_id = raw.meta.session,
                grid = raw.meta.grid, cells = cells, supported_models = CsiModels.Ids, window_frames = 16, frame_rate_hz = 10 };
            var predictions = new List<CsiPrediction>();
            for (int index = 0; index < ordered.Count; index++)
            {
                var group = ordered[index];
                if (group.Rows.Count != 4) throw new FormatException("Each recorded window must contain all four actual models.");
                foreach (string id in CsiModels.Ids)
                {
                    var row = group.Rows[id];
                    var p = new CsiPrediction { type = "prediction", contract_version = "0.1", source = "recorded_csi", session_id = row.session,
                        model_id = row.model, window_id = row.window_id, seq = index, timestamp = row.t_end,
                        window = new CsiWindow { start = row.t_start, end = row.t_end, frames = 16 },
                        label = row.pred_label, x = row.pred_label % 5, y = row.pred_label / 5,
                        x_m = row.pred_x_m, y_m = row.pred_y_m, confidence = row.confidence, probs = row.probs,
                        ground_truth_label = row.true_label, truth_source = "capture_station_label", state = "ready", expected_m = new CsiExpected() };
                    if (p.probs != null) for (int i = 0; i < p.probs.Length && i < 15; i++)
                    { p.expected_m.x += p.probs[i] * (i % 5) * .8f; p.expected_m.y += p.probs[i] * (i / 5) * .8f; }
                    string error; if (!CsiValidation.Prediction(p, out error)) throw new FormatException(error);
                    predictions.Add(p);
                }
            }
            var wire = new WireFile { source = "recorded_csi", session_id = raw.meta.session, meta = meta, predictions = predictions.ToArray() };
            var result = FromCanonical(JsonUtility.ToJson(wire));
            result.RoundFilter = roundFilter;
            result.Title = "D1 / " + (roundFilter == "all" ? "r9 + r10" : roundFilter) + " / seed " + raw.meta.seed;
            result.Declaration = "Captured " + raw.meta.capture_date + ". Train r1-r8; calibration r8.\nHeld-out diagnostic replay; scores previously inspected.\nNot a fresh blind test, live inference or a walking trial.";
            return result;
        }
        public static RealReplayData FromCanonical(string json)
        {
            var replay = CsiJson.ParseReplay(json);
            var result = new RealReplayData { Replay = replay, CanonicalJson = json, RoundFilter = "all",
                Title = "External recorded replay", Declaration = "Source: " + (replay.Source ?? "unspecified") + ".\nRecorded truth only; playback is not live positioning." };
            var windows = new HashSet<string>(); var observations = new HashSet<string>();
            foreach (var p in replay.Predictions)
            {
                string window = (p.session_id ?? "") + "|" + (p.window_id ?? p.timestamp);
                windows.Add(window);
                if (!observations.Add(window + "|" + CsiValidation.ModelId(p))) throw new FormatException("Replay contains duplicate model/window rows.");
                if (p.ground_truth_label >= 0 && !string.IsNullOrEmpty(p.truth_source))
                {
                    DeveloperMetrics value; string model = CsiValidation.ModelId(p);
                    if (!result.metrics.TryGetValue(model, out value)) { value = new DeveloperMetrics(); result.metrics.Add(model, value); }
                    value.Add(p.label, p.ground_truth_label);
                }
            }
            result.WindowCount = windows.Count; return result;
        }
    }
}
