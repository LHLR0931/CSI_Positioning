using System;
using System.Collections.Generic;
using System.Globalization;

namespace CsiApp
{
    // Deliberately independent of Unity: the route/state rules can be checked without an Editor.
    public static class GridMap
    {
        public const int Columns = 5, Rows = 3;
        public const double Spacing = 0.8;
        public static bool Valid(int label) { return label >= 0 && label < Columns * Rows; }
        public static int Label(int column, int row)
        {
            if (column < 0 || column >= Columns || row < 0 || row >= Rows)
                throw new ArgumentOutOfRangeException("Out of range");
            return row * Columns + column;
        }
        public static string Index(int label) { return "(" + label % Columns + ", " + label / Columns + ")"; }
        public static List<int> Neighbours(int label)
        {
            if (!Valid(label)) throw new ArgumentOutOfRangeException("label");
            var result = new List<int>();
            int x = label % Columns, y = label / Columns;
            if (x > 0) result.Add(label - 1);
            if (y < Rows - 1) result.Add(label + Columns);
            if (x < Columns - 1) result.Add(label + 1);
            if (y > 0) result.Add(label - Columns);
            return result;
        }
        public static List<int> Route(int start, int target)
        {
            if (!Valid(start) || !Valid(target)) throw new ArgumentOutOfRangeException("label");
            var previous = new Dictionary<int, int>();
            var queue = new Queue<int>();
            previous[start] = -1; queue.Enqueue(start);
            while (queue.Count > 0)
            {
                int point = queue.Dequeue();
                if (point == target) break;
                foreach (int next in Neighbours(point))
                    if (!previous.ContainsKey(next)) { previous[next] = point; queue.Enqueue(next); }
            }
            var path = new List<int>();
            for (int node = target; node != -1; node = previous[node]) path.Add(node);
            path.Reverse(); return path;
        }
        public static double Error(int prediction, int truth)
        {
            int dx = prediction % Columns - truth % Columns;
            int dy = prediction / Columns - truth / Columns;
            return Math.Sqrt(dx * dx + dy * dy) * Spacing;
        }
    }

    public sealed class PositionSample
    {
        public long Sequence;
        public int Label;
        public string Source = "Demo";
        public DateTime ReceivedUtc;
    }

    public sealed class NavigationState
    {
        public string Page = "Home";
        // Both views are virtual map previews. Changing view never claims AR calibration.
        public bool TopView = true;
        public bool TrackingLostDemo, Stale, Outside, Navigating, Paused, Recording;
        public int Prediction = 12, GroundTruth = -1, Target = 14;
        public bool Balcony;
        public long Sequence;
        public string Notice = "", RunId = "";
        public PositionSample LastRaw;
        public List<int> Path = new List<int>();
        public double PlannedGridLength;
        public readonly List<double> Errors = new List<double>();
        private readonly List<string> records = new List<string>();
        // The confirmed map has a door opening, but no configured T01 or exit label yet.
        public bool DestinationAvailable { get { return !Balcony && GridMap.Valid(Target); } }
        public string Destination { get { return Balcony ? "Balcony / endpoint pending" : "Grid " + GridMap.Index(Target); } }
        public bool CanAdvance { get { return !Stale && !Outside && (!Navigating || !Paused); } }
        public bool CanResume { get { return DestinationAvailable && !Stale && !Outside && !TrackingLostDemo; } }
        public double MeanError
        {
            get { double sum = 0; foreach (double e in Errors) sum += e; return Errors.Count == 0 ? double.NaN : sum / Errors.Count; }
        }
        public NavigationState() { Preview(); }
        public void Preview()
        {
            Path = DestinationAvailable ? GridMap.Route(Prediction, Target) : new List<int>();
            PlannedGridLength = DestinationAvailable ? (Path.Count - 1) * GridMap.Spacing : double.NaN;
        }
        public void SetTarget(bool balcony, int label)
        {
            if (!balcony && !GridMap.Valid(label)) throw new ArgumentOutOfRangeException("label");
            Balcony = balcony; Target = balcony ? -1 : label;
            Navigating = false; Paused = false; Preview();
            Notice = DestinationAvailable ? "Grid route preview ready" : "Balcony unavailable: T01 and exit label are pending.";
        }
        public bool Start()
        {
            if (!DestinationAvailable) { Notice = "Unavailable: destination endpoint and exit are pending."; return false; }
            if (Stale || Outside) { Notice = "A valid start is required"; return false; }
            if (TrackingLostDemo) { Notice = "Restore simulated tracking before starting."; return false; }
            Navigating = true; Paused = false; Preview(); Notice = "Demo navigation started"; return true;
        }
        public void End(bool arrived)
        {
            Navigating = false; Paused = false;
            Notice = arrived ? "Arrival confirmed by user" : "Navigation ended early";
        }
        public void PauseOrResume()
        {
            if (!Navigating) return;
            if (Paused && !CanResume) { Notice = "Resolve tracking / position status first"; return; }
            Paused = !Paused;
            if (!Paused) Path = GridMap.Route(Prediction, Target);
            Notice = Paused ? "Navigation paused" : "Demo navigation resumed";
        }
        public void InvalidateAr()
        {
            TrackingLostDemo = true; if (Navigating) Paused = true;
            Notice = "Simulated tracking loss. Demo paused; no AR session is running.";
        }
        public void RestoreTrackingDemo()
        {
            TrackingLostDemo = false;
            Notice = "Simulated tracking restored. Resume explicitly; this is not AR calibration.";
        }
        public void SetStale(bool value)
        {
            Stale = value; if (value && Navigating) Paused = true;
            Notice = value ? "CSI stale: last valid estimate frozen" : "Demo data restored. Resume explicitly.";
        }
        public void SetOutside(bool value)
        {
            Outside = value; if (Navigating) Paused = true;
            Notice = value ? "Outside CSI area (simulated). Last estimate frozen." : "Re-entry selected. Send a fresh demo sample.";
            if (!value) Stale = true;
        }
        public bool Accept(PositionSample sample)
        {
            if (sample == null || !GridMap.Valid(sample.Label) || sample.Sequence <= Sequence) return false;
            if (sample.Source != "Demo") return false; // Live adapter is intentionally not wired in this prototype.
            LastRaw = sample; Sequence = sample.Sequence;
            if (Outside) return false;
            Stale = false; Prediction = sample.Label;
            if (Navigating && !Paused) Path = GridMap.Route(Prediction, Target);
            else if (!Navigating) Preview();
            if (Recording)
            {
                double error = GroundTruth < 0 ? double.NaN : GridMap.Error(Prediction, GroundTruth);
                if (!double.IsNaN(error)) Errors.Add(error);
                records.Add(string.Join(",", new[] { RunId, "demo", Sequence.ToString(CultureInfo.InvariantCulture),
                    sample.ReceivedUtc.ToString("o"), Prediction.ToString(CultureInfo.InvariantCulture),
                    GroundTruth < 0 ? "" : GroundTruth.ToString(CultureInfo.InvariantCulture),
                    double.IsNaN(error) ? "" : error.ToString("F4", CultureInfo.InvariantCulture) }));
            }
            return true;
        }
        public void DemoSample()
        {
            if (Outside) { Notice = "Outside: CSI marker remains at the last valid estimate"; return; }
            int label;
            if (Navigating) label = !Paused && Path.Count > 1 ? Path[1] : Prediction;
            else label = (Prediction + 1) % 15;
            Accept(new PositionSample { Label = label, Sequence = Sequence + 1, ReceivedUtc = DateTime.UtcNow });
            if (Navigating && Path.Count == 1)
                Notice = Balcony ? "Exit reached. Balcony arrival still needs confirmation." : "Target estimate reached. Confirm arrival manually.";
            else Notice = "New demo sample received";
        }
        public void StartRecording()
        {
            RunId = "demo_" + DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff");
            records.Clear(); Errors.Clear(); Recording = true; Notice = "Recording demo samples locally";
        }
        public void StopRecording() { Recording = false; GroundTruth = -1; Notice = "Recording stopped; ground truth cleared"; }
        public string Csv()
        {
            return "run_id,source,sequence,received_utc,predicted_label,ground_truth_label,error_m\n" + string.Join("\n", records.ToArray()) + "\n";
        }
        public int RecordCount { get { return records.Count; } }
    }
}
