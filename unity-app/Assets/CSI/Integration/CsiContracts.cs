using System;
using System.Collections.Generic;
using System.Globalization;

namespace CsiApp
{
    public enum CsiConnectionState { Disconnected, Connecting, Connected, Reconnecting, Error }

    public static class CsiModels
    {
        public const string Unspecified = "Unspecified";
        public static readonly string[] Ids = { "KNN", "GCN", "ConcatMLP", "ConcatMLP-mask" };
        public static string Display(string id)
        { return id == "ConcatMLP" ? "MLP" : id == "ConcatMLP-mask" ? "MLP-mask" : id; }
    }

    [Serializable] public sealed class CsiGrid { public int cols, rows; public float spacing_m; }
    [Serializable] public sealed class CsiCell { public int label, x, y; public float x_m, y_m; }
    [Serializable] public sealed class CsiDevice { public string id, kind; public float x_m, y_m, h_m; }
    [Serializable] public sealed class CsiLink { public string id, pi, ap; }
    [Serializable] public sealed class CsiMeta
    {
        public string contract_version, source, session_id, default_model_id;
        public CsiGrid grid;
        public CsiCell[] cells;
        public CsiDevice[] devices;
        public CsiLink[] links;
        public string[] supported_models;
        public int window_frames;
        public float frame_rate_hz;
    }
    [Serializable] public sealed class CsiLinkHealth { public bool online; public float fps, last_frame_age_s; }
    [Serializable] public sealed class CsiStatusLinks
    { public CsiLinkHealth piA_apX, piA_apY, piB_apX, piB_apY; }
    [Serializable] public sealed class CsiStatus
    {
        public string type, contract_version, timestamp, state, calibrated_at, message, source, session_id;
        // progress is nullable in the wire protocol. The parser fills this explicitly.
        [NonSerialized] public float? Progress;
        public CsiStatusLinks links;
    }
    [Serializable] public sealed class CsiWindow { public string start, end; public int frames; }
    [Serializable] public sealed class CsiExpected { public float x, y; }
    [Serializable] public sealed class CsiPrediction
    {
        public string type, contract_version, model_id, window_id, session_id, source;
        public long seq = -1;
        public string timestamp;
        public CsiWindow window;
        public int label = -1, x = -1, y = -1;
        public float x_m = float.NaN, y_m = float.NaN, confidence = float.NaN;
        public float[] probs;
        public CsiExpected expected_m;
        public string[] links_used;
        public string state;
        // Optional replay extension; never accepted as automatic truth in live transport.
        public int ground_truth_label = -1;
        public string truth_source;
    }
    public sealed class CsiResult
    {
        public CsiPrediction Prediction { get; internal set; }
        public string EffectiveSource { get; internal set; }
        public string TransportMode { get; internal set; }
        public DateTime ReceivedUtc { get; internal set; }
        public string ModelId { get { return CsiValidation.ModelId(Prediction); } }
    }
    public sealed class CsiReplay
    {
        public CsiMeta Meta;
        public string Source, SessionId;
        public readonly List<CsiPrediction> Predictions = new List<CsiPrediction>();
    }
    public static class CsiValidation
    {
        public static bool Finite(float n) { return !float.IsNaN(n) && !float.IsInfinity(n); }
        public static string ModelId(CsiPrediction p)
        { return p == null || string.IsNullOrWhiteSpace(p.model_id) ? CsiModels.Unspecified : p.model_id; }
        public static bool TryTime(string text, out DateTime utc)
        {
            utc = DateTime.MinValue; DateTimeOffset value;
            if (string.IsNullOrWhiteSpace(text) || text.IndexOf('T') < 0) return false;
            bool zone = text.EndsWith("Z", StringComparison.OrdinalIgnoreCase)
                || text.Length >= 6 && (text[text.Length - 6] == '+' || text[text.Length - 6] == '-') && text[text.Length - 3] == ':';
            if (!zone || !DateTimeOffset.TryParse(text, CultureInfo.InvariantCulture, DateTimeStyles.None, out value)) return false;
            utc = value.UtcDateTime; return true;
        }
        public static bool Meta(CsiMeta meta, out string error)
        {
            error = null;
            if (meta == null || meta.grid == null || meta.grid.cols != 5 || meta.grid.rows != 3
                || !Finite(meta.grid.spacing_m) || Math.Abs(meta.grid.spacing_m - .8f) > .001f)
            { error = "Map must be the agreed 5 x 3 grid at 0.8 m spacing."; return false; }
            if (meta.cells == null || meta.cells.Length != 15)
            { error = "Metadata must contain all 15 cells."; return false; }
            var labels = new HashSet<int>();
            foreach (var c in meta.cells)
                if (c == null || c.label < 0 || c.label > 14 || !labels.Add(c.label)
                    || c.x != c.label % 5 || c.y != c.label / 5 || !Finite(c.x_m) || !Finite(c.y_m)
                    || Math.Abs(c.x_m - c.x * .8f) > .005f || Math.Abs(c.y_m - c.y * .8f) > .005f)
                { error = "Metadata cell labels, indices and meter coordinates disagree."; return false; }
            return true;
        }
        public static bool Prediction(CsiPrediction p, out string error)
        {
            error = null; DateTime timestamp;
            if (p == null || p.label < 0 || p.label > 14 || p.seq < 0 || !TryTime(p.timestamp, out timestamp))
            { error = "Prediction requires label 0-14, nonnegative seq and an ISO timestamp."; return false; }
            if (p.x != p.label % 5 || p.y != p.label / 5 || !Finite(p.x_m) || !Finite(p.y_m)
                || Math.Abs(p.x_m - p.x * .8f) > .005f || Math.Abs(p.y_m - p.y * .8f) > .005f)
            { error = "Prediction indices and meter coordinates disagree with its label."; return false; }
            if (p.probs == null || p.probs.Length != 15 || !Finite(p.confidence) || p.confidence < 0 || p.confidence > 1)
            { error = "Prediction requires 15 probabilities and finite confidence 0-1."; return false; }
            double sum = 0; float max = -1;
            foreach (float probability in p.probs)
            {
                if (!Finite(probability) || probability < 0 || probability > 1)
                { error = "Probability is outside 0-1 or not finite."; return false; }
                sum += probability; max = Math.Max(max, probability);
            }
            if (Math.Abs(sum - 1) > .025 || Math.Abs(p.confidence - p.probs[p.label]) > .005f
                || p.probs[p.label] < max - .00001f)
            { error = "Probability sum, confidence or winning label is inconsistent."; return false; }
            if (p.expected_m != null && (!Finite(p.expected_m.x) || !Finite(p.expected_m.y)
                || p.expected_m.x < -.005f || p.expected_m.x > 3.205f || p.expected_m.y < -.005f || p.expected_m.y > 1.605f))
            { error = "Expected position is outside the grid or not finite."; return false; }
            if (p.expected_m != null)
            {
                double expectedX = 0, expectedY = 0;
                for (int i = 0; i < 15; i++) { expectedX += p.probs[i] * (i % 5) * .8; expectedY += p.probs[i] * (i / 5) * .8; }
                if (Math.Abs(p.expected_m.x - expectedX) > .03 || Math.Abs(p.expected_m.y - expectedY) > .03)
                { error = "Expected position disagrees with probability-weighted meter coordinates."; return false; }
            }
            if (p.ground_truth_label < -1 || p.ground_truth_label > 14)
            { error = "Optional recorded truth must be label 0-14 or unset."; return false; }
            if (!string.IsNullOrEmpty(p.state) && p.state != "ready" && p.state != "degraded")
            { error = "Predictions are valid only in ready or degraded server state."; return false; }
            if (p.window != null)
            {
                DateTime start, end;
                if (!TryTime(p.window.start, out start) || !TryTime(p.window.end, out end)
                    || start > end || end != timestamp || p.window.frames < 1)
                { error = "Prediction window times or frame count are invalid."; return false; }
            }
            return true;
        }
        public static string EffectiveSource(string declared, string transport)
        {
            declared = (declared ?? "").Trim();
            if (transport == "replay") return "replay";
            if (string.Equals(declared, "mock", StringComparison.OrdinalIgnoreCase)) return "mock";
            if (string.Equals(declared, "replay", StringComparison.OrdinalIgnoreCase)) return "replay";
            if (string.Equals(declared, "live", StringComparison.OrdinalIgnoreCase)) return "live";
            return "server (source unspecified)";
        }
    }
}
