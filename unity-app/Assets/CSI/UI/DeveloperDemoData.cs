using System;
using System.Collections.Generic;
using System.Globalization;

namespace CsiApp
{
    // Hand-authored UI fixtures, never inferred model outputs or experimental measurements.
    // All four model columns belong to the same sample; changing the selected model
    // cannot advance the sample, change its truth, or add a duplicate CSV row.
    public sealed class DeveloperDemoData
    {
        public const string BatchId = "ui-samples-fourlink-v1";
        public const int ModelCount = 4;
        private static readonly string[] Models = { "KNN", "GCN", "ConcatMLP", "ConcatMLP-mask" };
        private static readonly int[] Truth = { 0, 4, 7, 10, 12, 14 };
        private static readonly int[,] Predictions = {
            { 0, 1, 5, 0 }, { 3, 4, 4, 9 }, { 6, 7, 8, 7 },
            { 11, 10, 10, 5 }, { 12, 13, 7, 12 }, { 9, 14, 13, 14 }
        };
        private readonly List<string> records = new List<string>();
        private readonly HashSet<int> recordedSamples = new HashSet<int>();
        private int? truthOverride;

        public int ModelIndex { get; private set; }
        public int SampleIndex { get; private set; }
        public int SampleCount { get { return Truth.Length; } }
        public string Model { get { return ModelName(ModelIndex); } }
        public string SampleId { get { return IdFor(SampleIndex); } }
        public int Prediction { get { return PredictionFor(SampleIndex, ModelIndex); } }
        public int GroundTruth { get { return truthOverride.HasValue ? truthOverride.Value : Truth[SampleIndex]; } }
        public string TruthSource { get { return !truthOverride.HasValue ? "sample_label" : GroundTruth < 0 ? "unset" : "manual"; } }
        public double Error { get { return GroundTruth < 0 ? double.NaN : GridMap.Error(Prediction, GroundTruth); } }
        public bool Recording { get; private set; }
        public string RunId { get; private set; }
        public string Notice { get; private set; }
        public int RecordCount { get { return records.Count; } }
        public int RecordedSampleCount { get { return recordedSamples.Count; } }

        public DeveloperDemoData() { SampleIndex = 2; Notice = "Fixed examples for interface review. No live backend."; }
        public static string ModelName(int index)
        {
            if (index < 0 || index >= ModelCount) throw new ArgumentOutOfRangeException("index");
            return Models[index];
        }
        public string IdFor(int index)
        {
            ValidateSample(index);
            return "S" + (index + 1).ToString("D2", CultureInfo.InvariantCulture);
        }
        public int TruthFor(int index) { ValidateSample(index); return Truth[index]; }
        public int PredictionFor(int sample, int model)
        {
            ValidateSample(sample); ModelName(model);
            return Predictions[sample, model];
        }
        private void ValidateSample(int index)
        {
            if (index < 0 || index >= SampleCount) throw new ArgumentOutOfRangeException("index");
        }
        public void SelectModel(int index)
        {
            ModelName(index); ModelIndex = index;
            Notice = Model + " / " + SampleId + " — same sample, illustrative output.";
        }
        public void SelectSample(int index)
        {
            ValidateSample(index);
            if (index == SampleIndex) return;
            SampleIndex = index; truthOverride = null;
            if (Recording) CaptureCurrent();
            Notice = SampleId + " selected. Truth restored from its example label.";
        }
        public void SetGroundTruth(int label)
        {
            if (label != -1 && !GridMap.Valid(label)) throw new ArgumentOutOfRangeException("label");
            Recording = false; truthOverride = label;
            Notice = label < 0 ? "Truth cleared. Recording stopped." : "Manual truth selected. Recording stopped.";
        }
        public void UseSampleTruth()
        {
            Recording = false; truthOverride = null;
            Notice = "Example truth restored. Recording stopped.";
        }
        public void StartRecording()
        {
            RunId = "sample_" + DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
            records.Clear(); recordedSamples.Clear(); Recording = true;
            CaptureCurrent();
            Notice = "Recording examples. Choose another sample to add its four model rows.";
        }
        public void StopRecording()
        {
            Recording = false;
            Notice = "Recording stopped. Export before starting a new run.";
        }
        private void CaptureCurrent()
        {
            if (!recordedSamples.Add(SampleIndex)) return;
            string recordedUtc = DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture);
            for (int model = 0; model < ModelCount; model++)
            {
                int prediction = PredictionFor(SampleIndex, model);
                double error = GroundTruth < 0 ? double.NaN : GridMap.Error(prediction, GroundTruth);
                records.Add(string.Join(",", new[] { RunId, "sample", BatchId, SampleId, ModelName(model),
                    prediction.ToString(CultureInfo.InvariantCulture),
                    GroundTruth < 0 ? "" : GroundTruth.ToString(CultureInfo.InvariantCulture), TruthSource,
                    double.IsNaN(error) ? "" : error.ToString("F4", CultureInfo.InvariantCulture), recordedUtc }));
            }
        }
        public string Csv()
        {
            return "run_id,source,batch_id,sample_id,model_id,predicted_label,ground_truth_label,truth_source,error_m,recorded_utc\n"
                + string.Join("\n", records.ToArray()) + "\n";
        }
    }
}
