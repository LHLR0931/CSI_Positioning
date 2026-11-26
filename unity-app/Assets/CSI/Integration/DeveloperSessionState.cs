using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace CsiApp
{
    public sealed class DeveloperMetrics
    {
        public int Count { get; private set; }
        public int Correct { get; private set; }
        private double totalError;
        public double Accuracy { get { return Count == 0 ? double.NaN : (double)Correct / Count; } }
        public double MeanError { get { return Count == 0 ? double.NaN : totalError / Count; } }
        internal void Add(int prediction, int truth)
        { Count++; if (prediction == truth) Correct++; totalError += GridMap.Error(prediction, truth); }
    }
    // Real received rows and replay rows live here; fixed UI examples stay in DeveloperDemoData.
    public sealed class DeveloperSessionState
    {
        public readonly string[] Models = CsiModels.Ids;
        private readonly Dictionary<string, CsiResult> current = new Dictionary<string, CsiResult>();
        private readonly Dictionary<string, long> lastSequences = new Dictionary<string, long>();
        private readonly Dictionary<string, DateTime> lastTimes = new Dictionary<string, DateTime>();
        private readonly HashSet<string> retiredSessions = new HashSet<string>();
        private readonly HashSet<string> recorded = new HashSet<string>();
        private readonly List<string> rows = new List<string>();
        private readonly Dictionary<string, DeveloperMetrics> metrics = new Dictionary<string, DeveloperMetrics>();
        private readonly HashSet<string> measured = new HashSet<string>();
        public DeveloperMetrics MetricsFor(string model)
        { DeveloperMetrics result; return metrics.TryGetValue(model, out result) ? result : new DeveloperMetrics(); }
        private string activeSession, activeWindow;
        private DateTime windowTime = DateTime.MinValue;
        private int manualTruth = -1;
        private bool manualOverride;
        public string SelectedModelId { get; private set; }
        public string Notice { get; private set; }
        public string RunId { get; private set; }
        public bool Recording { get; private set; }
        public int RecordCount { get { return rows.Count; } }
        public int GroundTruth { get { return EffectiveTruth; } }
        public DeveloperSessionState() { SelectedModelId = "KNN"; Notice = "Waiting for received results."; }
        public string WindowId { get { return activeWindow ?? ""; } }
        public CsiResult PredictionForSelected { get { return ResultFor(SelectedModelId); } }
        public CsiResult UnknownResult { get { return ResultFor(CsiModels.Unspecified); } }
        public int EffectiveTruth
        {
            get
            {
                if (manualOverride) return manualTruth;
                var r = PredictionForSelected;
                return ReplayTruth(r) ? r.Prediction.ground_truth_label : -1;
            }
        }
        public string TruthSource
        {
            get
            {
                if (manualOverride) return manualTruth < 0 ? "unset" : "manual";
                var r = PredictionForSelected;
                return ReplayTruth(r) ? "replay:" + r.Prediction.truth_source : "unset";
            }
        }
        public double ErrorMeters
        {
            get { var r = PredictionForSelected; return r == null || EffectiveTruth < 0 ? double.NaN : GridMap.Error(r.Prediction.label, EffectiveTruth); }
        }
        private static bool ReplayTruth(CsiResult r)
        { return r != null && r.TransportMode == "replay" && r.Prediction.ground_truth_label >= 0 && r.Prediction.ground_truth_label < 15 && !string.IsNullOrEmpty(r.Prediction.truth_source); }
        public CsiResult ResultFor(string model)
        { CsiResult result; return current.TryGetValue(model ?? CsiModels.Unspecified, out result) ? result : null; }
        public void SelectModel(string id)
        { SelectedModelId = string.IsNullOrWhiteSpace(id) ? CsiModels.Unspecified : id; Notice = PredictionForSelected == null ? "Waiting for this model in the current window." : "Showing received " + CsiModels.Display(SelectedModelId) + " result."; }
        public void SetGroundTruth(int label)
        {
            if (label < -1 || label > 14) throw new ArgumentOutOfRangeException("label");
            Recording = false; manualOverride = true; manualTruth = label;
            Notice = label < 0 ? "Truth cleared. Recording stopped." : "Manual truth selected. Recording stopped.";
        }
        public void UseReplayTruth()
        { Recording = false; manualOverride = false; manualTruth = -1; Notice = "Using recorded replay truth where available. Recording stopped."; }
        public void UseRecordedTruth() { UseReplayTruth(); }
        public void ResetData()
        {
            Recording = false; current.Clear(); lastSequences.Clear(); lastTimes.Clear(); retiredSessions.Clear();
            activeSession = activeWindow = null; windowTime = DateTime.MinValue;
            manualOverride = false; manualTruth = -1; Notice = "Data cleared. Existing exported/run rows are retained.";
            metrics.Clear(); measured.Clear();
        }
        public bool Accept(CsiPrediction p, string transportMode, DateTime receivedUtc)
        {
            string error;
            if (!CsiValidation.Prediction(p, out error)) { Notice = error; return false; }
            DateTime timestamp; CsiValidation.TryTime(p.timestamp, out timestamp);
            string session = string.IsNullOrEmpty(p.session_id) ? "unidentified-server-session" : p.session_id;
            string model = CsiValidation.ModelId(p);
            if (retiredSessions.Contains(session)) { Notice = "Ignored a retired server session."; return false; }
            if (activeSession != null && activeSession != session)
            {
                if (timestamp < windowTime) { Notice = "Ignored an older server session."; return false; }
                retiredSessions.Add(activeSession); current.Clear(); lastSequences.Clear(); lastTimes.Clear();
                activeWindow = null; windowTime = DateTime.MinValue;
                Recording = false; manualTruth = -1; manualOverride = false;
                metrics.Clear(); measured.Clear();
            }
            activeSession = session;
            long oldSeq; DateTime oldTime;
            if (lastSequences.TryGetValue(model, out oldSeq) && p.seq <= oldSeq)
            {
                // A v0.1 server provides no restart identity. Never guess a restart from old packets.
                Notice = "Ignored duplicate/out-of-order seq. Reconnect after a v0.1 server restart."; return false;
            }
            if (lastTimes.TryGetValue(model, out oldTime) && timestamp < oldTime)
            { Notice = "Ignored older prediction timestamp."; return false; }
            string window = !string.IsNullOrEmpty(p.window_id) ? p.window_id
                : (p.window != null ? p.window.start + "/" + p.window.end : p.timestamp);
            if (activeWindow != null && window != activeWindow && timestamp < windowTime)
            { Notice = "Ignored a result from an older window."; return false; }
            if (window != activeWindow)
            { current.Clear(); activeWindow = window; windowTime = timestamp; }
            lastSequences[model] = p.seq; lastTimes[model] = timestamp;
            var result = new CsiResult { Prediction = p, TransportMode = transportMode,
                EffectiveSource = CsiValidation.EffectiveSource(p.source, transportMode), ReceivedUtc = receivedUtc.ToUniversalTime() };
            current[model] = result;
            int metricTruth = ReplayTruth(result) ? p.ground_truth_label : manualOverride ? manualTruth : -1;
            string metricKey = transportMode + "|" + session + "|" + model + "|" + window;
            if (metricTruth >= 0 && measured.Add(metricKey))
            {
                DeveloperMetrics value;
                if (!metrics.TryGetValue(model, out value)) { value = new DeveloperMetrics(); metrics.Add(model, value); }
                value.Add(p.label, metricTruth);
            }
            if (Recording) Capture(result);
            Notice = model == CsiModels.Unspecified ? "Received an unlabelled server model; it is not assigned to a model button."
                : "Received " + CsiModels.Display(model) + " / " + result.EffectiveSource + ".";
            return true;
        }
        public void StartRecording(bool includeCurrent = true)
        {
            rows.Clear(); recorded.Clear(); RunId = "received_" + DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
            Recording = true; if (includeCurrent) foreach (var r in current.Values) Capture(r);
            Notice = "Recording actual received rows; missing model results are not invented.";
        }
        public void StopRecording() { Recording = false; Notice = "Recording stopped."; }
        public void ClearRecords() { Recording = false; rows.Clear(); recorded.Clear(); RunId = null; Notice = "Record cleared."; }
        private void Capture(CsiResult r)
        {
            var p = r.Prediction;
            string identity = r.TransportMode + "|" + (p.session_id ?? activeSession) + "|" + r.ModelId + "|" + p.seq + "|" + (p.window_id ?? p.timestamp);
            if (!recorded.Add(identity)) return;
            int truth = manualOverride ? manualTruth : ReplayTruth(r) ? p.ground_truth_label : -1;
            string truthSource = manualOverride ? truth < 0 ? "unset" : "manual" : ReplayTruth(r) ? "replay:" + p.truth_source : "unset";
            string error = truth < 0 ? "" : GridMap.Error(p.label, truth).ToString("F4", CultureInfo.InvariantCulture);
            rows.Add(Row(RunId, r.EffectiveSource, r.TransportMode, p.source, p.session_id, p.window_id,
                p.seq.ToString(CultureInfo.InvariantCulture), p.timestamp, r.ModelId,
                p.label.ToString(CultureInfo.InvariantCulture), truth < 0 ? "" : truth.ToString(CultureInfo.InvariantCulture),
                truthSource, error, r.ReceivedUtc.ToString("o"), DateTime.UtcNow.ToString("o")));
        }
        private static string Row(params string[] values)
        {
            var b = new StringBuilder();
            for (int i = 0; i < values.Length; i++) { if (i != 0) b.Append(','); b.Append('"').Append((values[i] ?? "").Replace("\"", "\"\"")).Append('"'); }
            return b.ToString();
        }
        public string Csv()
        { return "run_id,source,transport_mode,declared_source,session_id,window_id,seq,prediction_timestamp,model_id,predicted_label,ground_truth_label,truth_source,error_m,received_utc,recorded_utc\n" + string.Join("\n", rows.ToArray()) + "\n"; }
    }
}
