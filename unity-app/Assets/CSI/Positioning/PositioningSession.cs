using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace CsiApp
{
    // The HTTP connection survives UI/AR scene changes; replay never drives live navigation.
    public sealed class PositioningSession : MonoBehaviour
    {
        private static PositioningSession instance;
        public static PositioningSession Instance
        {
            get
            {
                if (instance == null) new GameObject("Positioning connection").AddComponent<PositioningSession>();
                return instance;
            }
        }
        public CsiBackendClient Backend { get; private set; }
        public DeveloperSessionState Results { get; private set; } = new DeveloperSessionState();
        public string SelectedModelId { get { return Results.SelectedModelId; } }
        public CsiResult SelectedResult { get { return Results.PredictionForSelected; } }
        public string Address { get { return PlayerPrefs.GetString("csi.endpoint", ""); } }
        public string SavedLog { get; private set; } = "";
        public string LogMessage { get; private set; } = "";
        public bool Recording { get { return Results.Recording; } }
        public int ConnectionRevision { get; private set; }
        public event Action Changed;
        private bool suspended, previousFresh;
        private readonly List<string> navigationRows = new List<string>();
        private readonly List<string> availableModels = new List<string>();
        private CsiMeta lastMeta;
        private bool modelSelectionExplicit, selectedObservedModel;
        public bool PredictionPollingEnabled { get { return Backend != null && Backend.PredictionPollingEnabled; } }
        public string[] AvailableModelIds
        {
            get
            {
                var all = new List<string>(availableModels);
                if (all.Count == 0) all.Add(CsiModels.Unspecified);
                if (SelectedModelId != CsiModels.Unspecified && !all.Contains(SelectedModelId)) all.Add(SelectedModelId);
                return all.ToArray();
            }
        }
        private float nextNavigationLog;
        private CsiConnectionState previousConnection;

        public bool HasFreshSelected
        {
            get
            {
                var result = SelectedResult;
                if (suspended || Backend == null || !Backend.PredictionPollingEnabled || !Backend.IsLiveTransport || Backend.Connection != CsiConnectionState.Connected
                    || Backend.Status == null || Backend.Status.state != "ready" || result == null || result.TransportMode != "live"
                    || (!string.IsNullOrEmpty(result.Prediction.state) && result.Prediction.state != "ready")) return false;
                return IsFreshResult(result, DateTime.UtcNow);
            }
        }
        // HTTP transport alone is not evidence of real CSI. Mock remains usable in Developer diagnostics.
        public bool HasFreshLiveSelected
        { get { return HasFreshSelected && IsDeclaredLive(SelectedResult); } }
        public static bool IsDeclaredLive(CsiResult result)
        { return result != null && string.Equals(result.EffectiveSource, "live", StringComparison.OrdinalIgnoreCase); }
        // Existing provisional freshness limits, evaluated for the selected model, not its peers.
        public static bool IsFreshResult(CsiResult result, DateTime now)
        {
            DateTime observed;
            if (result == null || result.TransportMode != "live" || result.EffectiveSource == "replay"
                || !CsiValidation.TryTime(result.Prediction.timestamp, out observed)) return false;
            double receivedAge = (now - result.ReceivedUtc).TotalSeconds, observedAge = (now - observed).TotalSeconds;
            return receivedAge >= 0 && receivedAge <= 3 && observedAge >= -5 && observedAge <= 3;
        }
        private void Awake()
        {
            if (instance != null && instance != this) { Destroy(gameObject); return; }
            instance = this; DontDestroyOnLoad(gameObject);
            Backend = gameObject.AddComponent<CsiBackendClient>();
            Results.SelectModel(CsiModels.Unspecified);
            Backend.PredictionReceived += Receive;
            Backend.Changed += ConnectionChanged;
        }
        private void Receive(CsiPrediction prediction, string transport, DateTime received)
        {
            if (transport != "live" || !Backend.PredictionPollingEnabled) return;
            bool wasRecording = Results.Recording;
            if (!Results.Accept(prediction, transport, received)) return;
            string model = CsiValidation.ModelId(prediction);
            if (!availableModels.Contains(model)) availableModels.Add(model);
            if (!modelSelectionExplicit && !selectedObservedModel)
            {
                // Honour an explicit server default. Otherwise prefer the first actually received model
                // over a remembered button or the first item of a supported-model declaration.
                Results.SelectModel(PreferredModel(Backend.Meta, model));
                selectedObservedModel = true;
            }
            if (wasRecording && !Results.Recording) SaveLog();
            Notify();
        }
        private void ConnectionChanged()
        {
            if (Backend.Connection != previousConnection && (Backend.Connection == CsiConnectionState.Connecting
                || Backend.Connection == CsiConnectionState.Disconnected))
            {
                ConnectionRevision++;
                StopLog(); Results.ResetData();
                selectedObservedModel = false;
            }
            if (!ReferenceEquals(lastMeta, Backend.Meta))
            {
                lastMeta = Backend.Meta; availableModels.Clear();
                if (lastMeta != null)
                {
                    if (lastMeta.supported_models != null)
                        foreach (var id in lastMeta.supported_models) AddModel(id);
                    AddModel(lastMeta.default_model_id);
                    if (!modelSelectionExplicit || !availableModels.Contains(SelectedModelId))
                        Results.SelectModel(PreferredModel(lastMeta));
                }
            }
            previousConnection = Backend.Connection; Notify();
        }
        private void AddModel(string id)
        { if (!string.IsNullOrWhiteSpace(id) && !availableModels.Contains(id.Trim())) availableModels.Add(id.Trim()); }
        public static string PreferredModel(CsiMeta meta, string observedModel = null)
        {
            if (meta != null && !string.IsNullOrWhiteSpace(meta.default_model_id)) return meta.default_model_id.Trim();
            if (!string.IsNullOrWhiteSpace(observedModel)) return observedModel.Trim();
            if (meta != null && meta.supported_models != null)
                foreach (var id in meta.supported_models) if (!string.IsNullOrWhiteSpace(id)) return id.Trim();
            return CsiModels.Unspecified;
        }
        private void Update()
        {
            bool fresh = HasFreshSelected;
            if (fresh != previousFresh) { previousFresh = fresh; Notify(); }
        }
        private void Notify() { Changed?.Invoke(); }
        public bool Connect(string address)
        {
            address = (address ?? "").Trim();
            if (!CsiBackendClient.ValidBaseUrl(address)) { LogMessage = "Enter the computer HTTP(S) base address."; Notify(); return false; }
            if (!StopLog()) return false;
            PlayerPrefs.SetString("csi.endpoint", address); PlayerPrefs.Save();
            Results.ResetData(); Results.SelectModel(CsiModels.Unspecified);
            modelSelectionExplicit = selectedObservedModel = false;
            availableModels.Clear(); lastMeta = null;
            Backend.Connect(address); return true;
        }
        public void Disconnect() { StopLog(); Backend.Disconnect(); }
        // Positioning owns this gate while planning/preparing. Developer uses the client's gate directly.
        // Leaving a cached replay stops its coroutine; it never silently becomes a live position.
        public void SetPredictionPolling(bool enabled)
        {
            bool wasReplay = !Backend.IsLiveTransport;
            if (!wasReplay && Backend.PredictionPollingEnabled == enabled
                && (enabled || string.IsNullOrEmpty(Results.WindowId))) return;
            StopLog();
            if (wasReplay) Backend.Disconnect();
            Backend.SetPredictionPolling(enabled);
            Results.ResetData(); selectedObservedModel = false;
            ConnectionRevision++; Notify();
        }
        public void SelectModel(string id)
        {
            if (id == SelectedModelId) return;
            if (!StopLog()) return;
            modelSelectionExplicit = true;
            Results.SelectModel(id); PlayerPrefs.SetString("csi.navigation.model", SelectedModelId); PlayerPrefs.Save(); Notify();
        }
        public bool StartLog()
        {
            if (!HasFreshSelected) { LogMessage = "Wait for a fresh selected model and ready server."; Notify(); return false; }
            if (!StopLog()) return false;
            Results.SetGroundTruth(-1); Results.StartRecording(false); navigationRows.Clear(); nextNavigationLog = 0;
            SavedLog = ""; LogMessage = "Recording received predictions and navigation observations; no independent truth."; Notify(); return true;
        }
        public bool StopLog()
        {
            Results.StopRecording(); return SaveLog();
        }
        public void RecordNavigation(Vector3 position, string source, string state)
        {
            if (!Results.Recording || Time.unscaledTime < nextNavigationLog) return;
            nextNavigationLog = Time.unscaledTime + .25f;
            var r = SelectedResult;
            navigationRows.Add(string.Join(",", new[] { Q(DateTime.UtcNow.ToString("o")), Q(source),
                position.x.ToString("F4", CultureInfo.InvariantCulture), position.z.ToString("F4", CultureInfo.InvariantCulture),
                Q(state), Q(SelectedModelId), Q(r == null ? "" : r.Prediction.session_id),
                Q(r == null ? "" : r.Prediction.seq.ToString(CultureInfo.InvariantCulture)), Q(r == null ? "" : r.EffectiveSource) }));
        }
        private static string Q(string value) { return "\"" + (value ?? "").Replace("\"", "\"\"") + "\""; }
        private bool SaveLog()
        {
            if (Results.RecordCount == 0 && navigationRows.Count == 0) return true;
            try
            {
                string stem = "navigation_" + Results.RunId;
                string received = Path.Combine(Application.persistentDataPath, stem + "_predictions.csv");
                string motion = Path.Combine(Application.persistentDataPath, stem + "_motion.csv");
                WriteAtomic(received, Results.Csv());
                WriteAtomic(motion, "observed_utc,position_source,map_x_m,map_y_m,navigation_state,model_id,session_id,last_csi_seq,declared_data_source\n"
                    + string.Join("\n", navigationRows.ToArray()) + "\n");
                SavedLog = received + "\n" + motion; LogMessage = "Saved locally: " + stem;
                Debug.Log("CSI_NAV_CSV: " + SavedLog); return true;
            }
            catch (Exception e) { LogMessage = "CSV save failed: " + e.Message; Debug.LogError(LogMessage); return false; }
        }
        private static void WriteAtomic(string path, string content)
        {
            string temporary = path + ".new"; File.WriteAllText(temporary, content, new UTF8Encoding(false));
            if (File.Exists(path)) File.Replace(temporary, path, null); else File.Move(temporary, path);
        }
        private void OnApplicationPause(bool pause)
        {
            suspended = pause;
            if (pause) StopLog();
            else if (Backend != null && Backend.IsLiveTransport && Backend.Connection != CsiConnectionState.Disconnected
                && CsiBackendClient.ValidBaseUrl(Address)) Connect(Address);
            Notify();
        }
        private void OnApplicationQuit() { StopLog(); }
        private void OnDestroy()
        {
            if (instance != this) return;
            StopLog();
            if (Backend != null) { Backend.PredictionReceived -= Receive; Backend.Changed -= ConnectionChanged; }
            instance = null;
        }
    }
}
