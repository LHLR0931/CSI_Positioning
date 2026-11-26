using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace CsiApp
{
    // HTTP transport only. It never trains or runs positioning models.
    public sealed class CsiBackendClient : MonoBehaviour
    {
        public CsiMeta Meta { get; private set; }
        public CsiStatus Status { get; private set; }
        public CsiConnectionState Connection { get; private set; } = CsiConnectionState.Disconnected;
        public string Notice { get; private set; } = "Not connected.";
        public bool ReplayAvailable { get { return replay != null && replay.Predictions.Count > 0; } }
        public bool ReplayPlaying { get; private set; }
        public int ReplayIndex { get; private set; }
        public int ReplayCount { get { return replayGroups.Count; } }
        public bool PredictionPollingEnabled { get { return predictionPolling.Enabled; } }
        public string CachePath { get { return Path.Combine(Application.persistentDataPath, "csi_cached_replay.json"); } }
        // Provisional UI warning, not a measured navigation-validity threshold.
        // Source age assumes server and phone clocks are synchronized.
        public bool IsStale { get { return !replayMode && (!PredictionPollingEnabled || Connection != CsiConnectionState.Connected || lastFreshUtc == DateTime.MinValue
            || (DateTime.UtcNow - lastFreshUtc).TotalSeconds > 3
            || SourceAgeSeconds > 3 || SourceAgeSeconds < -5); } }
        public double SourceAgeSeconds { get { return lastSourceUtc == DateTime.MinValue ? double.NaN : (DateTime.UtcNow - lastSourceUtc).TotalSeconds; } }
        public DateTime LastPredictionReceivedUtc { get { return lastFreshUtc; } }
        public event Action Changed;
        public event Action<CsiPrediction, string, DateTime> PredictionReceived;
        private CsiReplay replay;
        private readonly List<List<CsiPrediction>> replayGroups = new List<List<CsiPrediction>>();
        private readonly Dictionary<string, long> sequences = new Dictionary<string, long>();
        private readonly Dictionary<string, DateTime> timestamps = new Dictionary<string, DateTime>();
        private readonly HashSet<string> retiredSessions = new HashSet<string>();
        private string activeSession;
        private UnityWebRequest activeRequest;
        private UnityWebRequest activePredictionRequest;
        private readonly PredictionPollingGate predictionPolling = new PredictionPollingGate();
        private Coroutine replayRoutine;
        private string baseUrl;
        private bool replayMode, lastStale = true;
        public bool IsLiveTransport { get { return !replayMode; } }
        private DateTime lastFreshUtc = DateTime.MinValue;
        private DateTime lastSourceUtc = DateTime.MinValue;
        private int generation;
        private const int MaxPayloadBytes = 16 * 1024 * 1024;

        private void Notify() { var callback = Changed; if (callback != null) callback(); }
        private void Update() { bool stale = IsStale; if (stale != lastStale) { lastStale = stale; Notify(); } }
        public static bool ValidBaseUrl(string value)
        { Uri uri; return Uri.TryCreate(value, UriKind.Absolute, out uri) && (uri.Scheme == "http" || uri.Scheme == "https") && string.IsNullOrEmpty(uri.UserInfo) && string.IsNullOrEmpty(uri.Query) && string.IsNullOrEmpty(uri.Fragment); }
        public void Connect(string address)
        {
            CancelOperations(); replayMode = false; sequences.Clear(); timestamps.Clear(); retiredSessions.Clear(); activeSession = null;
            lastFreshUtc = lastSourceUtc = DateTime.MinValue;
            Meta = null; Status = null;
            if (!ValidBaseUrl(address)) { Connection = CsiConnectionState.Error; Notice = "Enter an HTTP(S) server base URL without credentials, query or fragment."; Notify(); return; }
            baseUrl = address.TrimEnd('/'); Connection = CsiConnectionState.Connecting; Notice = "Connecting to server..."; Notify();
            StartCoroutine(Poll(generation));
        }
        public void Disconnect()
        { CancelOperations(); replayMode = false; Connection = CsiConnectionState.Disconnected; Notice = "Disconnected. Last received results are historical."; Notify(); }
        // Stops only prediction GETs. Metadata/status and the remote CSI capture keep running.
        // The epoch also rejects a late response if polling is switched off and on again.
        public void SetPredictionPolling(bool enabled)
        {
            if (!predictionPolling.SetEnabled(enabled)) return;
            lastFreshUtc = DateTime.MinValue;
            if (!enabled && activePredictionRequest != null) activePredictionRequest.Abort();
            if (!replayMode) Notice = enabled ? "Position reading enabled; waiting for a new result."
                : "Position reading off. Server status remains available.";
            Notify();
        }
        private void CancelOperations()
        {
            generation++; ReplayPlaying = false; StopAllCoroutines(); replayRoutine = null;
            if (activeRequest != null) { activeRequest.Abort(); activeRequest.Dispose(); activeRequest = null; }
            activePredictionRequest = null;
        }
        private IEnumerator Request(string path, Action<long, string, string> completed, int predictionEpoch = -1)
        {
            using (var request = UnityWebRequest.Get(baseUrl + path))
            {
                if (predictionEpoch >= 0 && !predictionPolling.Allows(predictionEpoch)) yield break;
                activeRequest = request;
                if (predictionEpoch >= 0) activePredictionRequest = request;
                request.timeout = 5;
                try
                {
                    yield return request.SendWebRequest();
                    // Aborting a prediction must not reconnect or interrupt status polling.
                    if (predictionEpoch >= 0 && !predictionPolling.Allows(predictionEpoch)) yield break;
                    long code = request.responseCode;
                    string error = request.result == UnityWebRequest.Result.Success ? null : request.error;
                    string body = request.downloadHandler == null ? "" : request.downloadHandler.text;
                    if (request.downloadedBytes > MaxPayloadBytes) error = "Response exceeds the 16 MB limit.";
                    // Event handlers can change mode or reconnect while processing a result.
                    // The completed request no longer belongs to those cancellation paths.
                    if (activeRequest == request) activeRequest = null;
                    if (activePredictionRequest == request) activePredictionRequest = null;
                    completed(code, body, error);
                }
                finally
                {
                    if (activeRequest == request) activeRequest = null;
                    if (activePredictionRequest == request) activePredictionRequest = null;
                }
            }
        }
        private IEnumerator Poll(int expectedGeneration)
        {
            while (expectedGeneration == generation)
            {
                string fault = null; CsiMeta candidateMeta = null;
                yield return Request("/api/meta", (code, body, error) => {
                    if (error != null || code != 200) { fault = error ?? "Metadata HTTP " + code; return; }
                    try { candidateMeta = CsiJson.ParseMeta(body); } catch (Exception e) { fault = e.Message; }
                });
                if (expectedGeneration != generation) yield break;
                if (fault != null) { Connection = CsiConnectionState.Reconnecting; Notice = fault + " Retrying in 2 s."; Notify(); yield return new WaitForSecondsRealtime(2); continue; }
                Meta = candidateMeta; Connection = CsiConnectionState.Connected;
                Notice = PredictionPollingEnabled ? "Server connected; waiting for predictions." : "Server connected; position reading off."; Notify();
                while (expectedGeneration == generation)
                {
                    fault = null;
                    yield return Request("/api/status", (code, body, error) => {
                        if (error != null || code != 200) { fault = error ?? "Status HTTP " + code; return; }
                        try
                        {
                            Status = CsiJson.ParseStatus(body);
                            if (string.IsNullOrEmpty(Status.source)) Status.source = Meta.source;
                            if (string.IsNullOrEmpty(Status.session_id)) Status.session_id = Meta.session_id;
                            Notice = string.IsNullOrEmpty(Status.message) ? "Server: " + Status.state : Status.message;
                        }
                        catch (Exception e) { fault = e.Message; }
                    });
                    if (expectedGeneration != generation) yield break;
                    if (fault == null && PredictionPollingEnabled)
                    {
                        int predictionEpoch = predictionPolling.Epoch;
                        yield return Request("/api/predict/latest", (code, body, error) => {
                            if (!predictionPolling.Allows(predictionEpoch)) return;
                            if (code == 204 && error == null) { Notice = "No prediction yet; server: " + (Status == null ? "unknown" : Status.state); return; }
                            if (error != null || code != 200) { fault = error ?? "Prediction HTTP " + code; return; }
                            try
                            {
                                string source = Status == null ? Meta.source : Status.source ?? Meta.source;
                                string session = Status == null ? Meta.session_id : Status.session_id ?? Meta.session_id;
                                var predictions = CsiJson.ParsePredictions(body, source, session);
                                bool fresh = false;
                                foreach (var prediction in predictions)
                                {
                                    if (!predictionPolling.Allows(predictionEpoch)) break;
                                    fresh |= Deliver(prediction, "live");
                                }
                                if (fresh) Notice = "Received " + predictions.Count + " server result(s).";
                            }
                            catch (Exception e) { fault = e.Message; }
                        }, predictionEpoch);
                    }
                    if (expectedGeneration != generation) yield break;
                    if (fault != null)
                    { Connection = CsiConnectionState.Reconnecting; Notice = fault + " Retrying in 2 s."; Notify(); yield return new WaitForSecondsRealtime(2); break; }
                    Connection = CsiConnectionState.Connected; Notify();
                    yield return new WaitForSecondsRealtime(.5f);
                }
            }
        }
        private bool Deliver(CsiPrediction prediction, string transport)
        {
            if (transport != "replay" && !PredictionPollingEnabled) return false;
            DateTime timestamp; CsiValidation.TryTime(prediction.timestamp, out timestamp);
            string session = string.IsNullOrEmpty(prediction.session_id) ? "unidentified" : prediction.session_id;
            if (transport != "replay")
            {
                if (retiredSessions.Contains(session)) return false;
                if (activeSession != null && activeSession != session)
                {
                    if (timestamp < lastSourceUtc) return false;
                    retiredSessions.Add(activeSession);
                }
                activeSession = session;
            }
            string key = session + "|" + CsiValidation.ModelId(prediction);
            long previous; DateTime previousTime;
            if (transport != "replay" && (sequences.TryGetValue(key, out previous) && prediction.seq <= previous
                || timestamps.TryGetValue(key, out previousTime) && timestamp < previousTime)) return false;
            sequences[key] = prediction.seq; timestamps[key] = timestamp;
            DateTime received = DateTime.UtcNow;
            if (transport != "replay") { lastFreshUtc = received; lastSourceUtc = timestamp; }
            var callback = PredictionReceived; if (callback != null) callback(prediction, transport, received);
            return true;
        }
        public void DownloadReplay(string address)
        {
            CancelOperations(); replayMode = true;
            replay = null; replayGroups.Clear(); ReplayIndex = 0; Meta = null; Status = null;
            if (!ValidBaseUrl(address)) { Connection = CsiConnectionState.Error; Notice = "Enter a valid HTTP(S) base URL first."; Notify(); return; }
            baseUrl = address.TrimEnd('/'); Connection = CsiConnectionState.Connecting; Notice = "Downloading replay..."; Notify();
            StartCoroutine(Download(generation));
        }
        private IEnumerator Download(int expectedGeneration)
        {
            string fault = null, json = null; CsiReplay parsed = null;
            yield return Request("/api/replay", (code, body, error) => {
                if (error != null || code != 200) { fault = error ?? "Replay HTTP " + code; return; }
                try { parsed = CsiJson.ParseReplay(body); json = body; } catch (Exception e) { fault = e.Message; }
            });
            if (expectedGeneration != generation) yield break;
            if (fault == null)
            {
                try
                {
                    // Validate before replacing the previous valid cache.
                    string temporary = CachePath + ".new"; File.WriteAllText(temporary, json);
                    if (File.Exists(CachePath)) File.Replace(temporary, CachePath, null);
                    else File.Move(temporary, CachePath);
                    SetReplay(parsed); Notice = "Replay cached on this device. Step or Play to view it.";
                }
                catch (Exception e) { fault = "Could not cache replay: " + e.Message; }
            }
            Connection = fault == null ? CsiConnectionState.Disconnected : CsiConnectionState.Error;
            if (fault != null) Notice = fault; Notify();
        }
        public bool LoadCachedReplay()
        {
            CancelOperations(); replayMode = true; Connection = CsiConnectionState.Disconnected;
            replay = null; replayGroups.Clear(); ReplayIndex = 0; Meta = null; Status = null;
            try
            {
                if (!File.Exists(CachePath)) { Notice = "No cached replay yet. Download one from a server."; Notify(); return false; }
                if (new FileInfo(CachePath).Length > MaxPayloadBytes) throw new FormatException("Replay cache exceeds 16 MB.");
                SetReplay(CsiJson.ParseReplay(File.ReadAllText(CachePath))); Notice = "Cached replay loaded; no server connection required."; Notify(); return true;
            }
            catch (Exception e) { Notice = "Replay cache invalid: " + e.Message; Notify(); return false; }
        }
        // Importers can adapt their schema before calling this method. No legacy cache is written.
        public bool LoadReplayJson(string json)
        {
            CsiReplay parsed;
            try
            {
                if (string.IsNullOrWhiteSpace(json)) throw new FormatException("Replay file is empty.");
                if (System.Text.Encoding.UTF8.GetByteCount(json) > MaxPayloadBytes) throw new FormatException("Replay exceeds 16 MB.");
                parsed = CsiJson.ParseReplay(json);
            }
            catch (Exception e) { Notice = "Replay import invalid: " + e.Message; Notify(); return false; }
            CancelOperations(); replayMode = true; Connection = CsiConnectionState.Disconnected;
            SetReplay(parsed); Notice = "Recorded replay loaded. Step or Play to inspect its windows."; Notify(); return true;
        }
        private void SetReplay(CsiReplay value)
        {
            replay = value; Meta = value.Meta; Status = null; replayGroups.Clear(); ReplayIndex = 0; ReplayPlaying = false;
            sequences.Clear(); timestamps.Clear(); retiredSessions.Clear(); activeSession = null;
            lastFreshUtc = lastSourceUtc = DateTime.MinValue;
            var groups = new Dictionary<string, List<CsiPrediction>>();
            var sessionStarts = new Dictionary<string, DateTime>();
            foreach (var p in value.Predictions)
            {
                string window = !string.IsNullOrEmpty(p.window_id) ? p.window_id
                    : p.window != null ? p.window.start + "/" + p.window.end : p.timestamp;
                string key = (p.session_id ?? "") + "|" + window;
                List<CsiPrediction> group;
                if (!groups.TryGetValue(key, out group))
                { group = new List<CsiPrediction>(); groups.Add(key, group); replayGroups.Add(group); }
                group.Add(p);
                DateTime time, first; CsiValidation.TryTime(p.timestamp, out time);
                string session = p.session_id ?? "";
                if (!sessionStarts.TryGetValue(session, out first) || time < first) sessionStarts[session] = time;
            }
            replayGroups.Sort((a, b) => {
                string aSession = a[0].session_id ?? "", bSession = b[0].session_id ?? "";
                int order = sessionStarts[aSession].CompareTo(sessionStarts[bSession]);
                if (order != 0) return order;
                order = string.Compare(aSession, bSession, StringComparison.Ordinal);
                if (order != 0) return order;
                DateTime aTime, bTime; CsiValidation.TryTime(a[0].timestamp, out aTime); CsiValidation.TryTime(b[0].timestamp, out bTime);
                order = aTime.CompareTo(bTime);
                if (order != 0) return order;
                return a[0].seq.CompareTo(b[0].seq);
            });
        }
        public void PlayReplay()
        {
            if (!ReplayAvailable) { Notice = "Load a cached replay first."; Notify(); return; }
            if (ReplayIndex >= ReplayCount) { Notice = "Replay finished. Reset before playing again."; Notify(); return; }
            if (ReplayPlaying) return;
            replayMode = true; ReplayPlaying = true; replayRoutine = StartCoroutine(Play()); Notify();
        }
        private IEnumerator Play()
        {
            while (ReplayPlaying && ReplayIndex < ReplayCount)
            {
                EmitReplayGroup(); yield return new WaitForSecondsRealtime(.3f);
            }
            ReplayPlaying = false; replayRoutine = null; Notice = ReplayIndex >= ReplayCount ? "Replay finished; these are recorded rows." : "Replay paused."; Notify();
        }
        public void PauseReplay()
        { ReplayPlaying = false; if (replayRoutine != null) StopCoroutine(replayRoutine); replayRoutine = null; Notice = "Replay paused."; Notify(); }
        public void StepReplay()
        { PauseReplay(); if (!ReplayAvailable || ReplayIndex >= ReplayCount) { Notice = "No next replay window. Load or reset replay."; Notify(); return; } replayMode = true; EmitReplayGroup(); }
        private void EmitReplayGroup()
        {
            foreach (var p in replayGroups[ReplayIndex]) Deliver(p, "replay");
            ReplayIndex++; Notice = "Replay window " + ReplayIndex + " / " + ReplayCount + "."; Notify();
        }
        public void ResetReplay()
        { PauseReplay(); ReplayIndex = 0; Notice = "Replay reset. Clear the displayed session before stepping from the beginning."; Notify(); }
        private void OnDestroy() { CancelOperations(); }
    }

    // Separate from transport so close/reopen races can be checked without a network or phone.
    public sealed class PredictionPollingGate
    {
        public bool Enabled { get; private set; } = true;
        public int Epoch { get; private set; }
        public bool SetEnabled(bool enabled)
        { if (Enabled == enabled) return false; Enabled = enabled; Epoch++; return true; }
        public bool Allows(int epoch) { return Enabled && epoch == Epoch; }
    }
}
