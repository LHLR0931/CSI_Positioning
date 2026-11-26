using System;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

namespace CsiApp
{
    public sealed partial class CsiAppUi
    {
        private enum DeveloperMode { Replay, Live }
        private DeveloperMode developerMode = DeveloperMode.Replay;
        private readonly DeveloperSessionState session = new DeveloperSessionState();
        private CsiBackendClient backend;
        private PositioningSession sharedConnection;
        private RealReplayData replayData;
        private string replayRound = "all", serverModelId = CsiModels.Unspecified;
        private bool developerEntered, integrationDirty, replayDownloadPending;
        private int developerConnectionRevision;
        private float nextIntegrationRender;
        private Text connectionText;
        private Action integrationChanged;
        private Action<CsiPrediction, string, DateTime> integrationPrediction;
        private bool IsRecording { get { return session.Recording; } }
        private CsiResult DisplayResult { get { return session.PredictionForSelected; } }
        private string RealReplayCache { get { return Path.Combine(Application.persistentDataPath, "real_replay_d1_seed42.json"); } }

        private void InitializeIntegration()
        {
            sharedConnection = PositioningSession.Instance;
            backend = sharedConnection.Backend;
            developerConnectionRevision = sharedConnection.ConnectionRevision;
            // Home initialization only subscribes; it must not stop a retained Live connection.
            integrationChanged = () => { SyncDeveloperConnection(); integrationDirty = true; };
            backend.Changed += integrationChanged;
            integrationPrediction = (prediction, transport, received) => {
                if (state.Page != "Developer") return;
                if ((developerMode == DeveloperMode.Live && transport == "live") ||
                    (developerMode == DeveloperMode.Replay && transport == "replay"))
                {
                    if (!SyncDeveloperConnection()) return;
                    bool wasRecording = session.Recording;
                    if (session.Accept(prediction, transport, received))
                    {
                        string id = CsiValidation.ModelId(prediction);
                        if (Array.IndexOf(CsiModels.Ids, id) < 0) { serverModelId = id; session.SelectModel(id); }
                    }
                    if (wasRecording && !session.Recording) SaveCurrent();
                    integrationDirty = true;
                }
            };
            backend.PredictionReceived += integrationPrediction;
        }
        private bool SyncDeveloperConnection()
        {
            if (sharedConnection == null || developerMode != DeveloperMode.Live || state.Page != "Developer") return true;
            if (developerConnectionRevision == sharedConnection.ConnectionRevision) return true;
            // Preserve exported rows, clear old dedup/truth/current estimates after explicit reconnect.
            if (!StopAndSave()) return false;
            developerConnectionRevision = sharedConnection.ConnectionRevision;
            session.ResetData(); serverModelId = CsiModels.Unspecified;
            return true;
        }
        private void EnterDeveloper()
        {
            state.Page = "Developer"; drawer = ""; connectionText = null;
            if (developerMode == DeveloperMode.Replay && (!developerEntered || backend.IsLiveTransport))
                LoadRealReplay(replayRound);
            else if (developerMode == DeveloperMode.Live)
            {
                backend.SetPredictionPolling(true);
                SyncDeveloperConnection();
            }
            developerEntered = true; Render();
        }
        private bool LeaveDeveloper()
        {
            if (!StopAndSave()) return false;
            if (backend != null && backend.ReplayPlaying) backend.PauseReplay();
            state.Page = "Home"; drawer = ""; connectionText = null; return true;
        }
        private void ReleaseIntegration()
        {
            if (backend == null) return;
            backend.Changed -= integrationChanged; backend.PredictionReceived -= integrationPrediction;
        }
        private void UpdateIntegration()
        {
            if (backend == null) return;
            SyncDeveloperConnection();
            if (replayDownloadPending && backend.Connection != CsiConnectionState.Connecting)
            {
                replayDownloadPending = false;
                if (backend.ReplayAvailable)
                {
                    try { replayData = RealReplayData.FromCanonical(File.ReadAllText(backend.CachePath)); }
                    catch (Exception e) { replayData = null; state.Notice = e.Message; }
                }
            }
            if (connectionText != null) connectionText.text = ConnectionSummary();
            if (!integrationDirty || state.Page != "Developer" || drawer.Length > 0 || Time.unscaledTime < nextIntegrationRender) return;
#if ENABLE_LEGACY_INPUT_MANAGER
            if (UnityEngine.Input.touchCount > 0 || UnityEngine.Input.GetMouseButton(0)) return;
#endif
            nextIntegrationRender = Time.unscaledTime + .2f; integrationDirty = false; Render();
        }
        private string SourceSummary()
        {
            if (developerMode == DeveloperMode.Replay)
            {
                string declared = replayData == null ? "not loaded" : replayData.Replay.Source ?? "source unspecified";
                return "REPLAY / " + (declared == "mock" ? "MOCK / synthetic" : declared == "recorded_csi" ? "recorded CSI / offline" : declared);
            }
            var result = DisplayResult;
            string source = result == null ? "waiting" : result.EffectiveSource;
            return "LIVE / " + source + (backend.IsStale ? " / STALE" : "") + (IsRecording ? " / REC" : "");
        }
        private string ConnectionSummary()
        {
            return backend.Connection + (backend.Status == null ? "" : " / " + backend.Status.state +
                (backend.Status.Progress.HasValue ? " / " + (backend.Status.Progress.Value * 100).ToString("F0") + "%" : "")) + "\n" + backend.Notice;
        }
        private string DeveloperFooterNotice()
        {
            if (developerMode != DeveloperMode.Replay || backend == null || backend.IsLiveTransport || !backend.ReplayAvailable)
                return state.Notice;
            string playback = backend.ReplayPlaying ? "Replay playing" : backend.ReplayIndex >= backend.ReplayCount
                ? "Replay finished" : "Replay paused";
            return playback + " / window " + backend.ReplayIndex + " of " + backend.ReplayCount
                + (string.IsNullOrEmpty(state.Notice) ? "" : "\n" + state.Notice);
        }
        private bool LoadRealReplay(string round)
        {
            if (!StopAndSave()) return false;
            try
            {
                string raw;
                if (File.Exists(RealReplayCache)) raw = File.ReadAllText(RealReplayCache);
                else
                {
                    var asset = Resources.Load<TextAsset>(RealReplayData.BundledResource);
                    if (asset == null) throw new IOException("Bundled real D1 replay is missing.");
                    raw = asset.text;
                }
                RealReplayData parsed;
                try { parsed = RealReplayData.Parse(raw, round); }
                catch
                {
                    var asset = Resources.Load<TextAsset>(RealReplayData.BundledResource);
                    if (asset == null) throw; raw = asset.text; parsed = RealReplayData.Parse(raw, round);
                }
                if (CsiJson.Property(raw, "records") == null) throw new FormatException("Default D1 replay must retain the real export metadata and records.");
                developerMode = DeveloperMode.Replay; backend.SetPredictionPolling(false);
                session.ResetData(); session.SelectModel("KNN"); serverModelId = CsiModels.Unspecified;
                if (!backend.LoadReplayJson(parsed.CanonicalJson)) throw new FormatException(backend.Notice);
                replayData = parsed; replayRound = round; replayDownloadPending = false;
                try
                {
                    string pending = RealReplayCache + ".new"; File.WriteAllText(pending, raw);
                    if (File.Exists(RealReplayCache)) File.Replace(pending, RealReplayCache, null); else File.Move(pending, RealReplayCache);
                }
                catch (Exception e) { Debug.LogWarning("Real replay loaded; local cache write failed: " + e.Message); }
                backend.StepReplay(); state.Notice = "Loaded " + parsed.WindowCount + " real recorded windows.";
                integrationDirty = true; return true;
            }
            catch (Exception e) { state.Notice = "Real replay could not load: " + e.Message; return false; }
        }
        private void ChangeMode(DeveloperMode mode)
        {
            if (mode == developerMode) return;
            if (!StopAndSave()) return;
            drawer = ""; connectionText = null;
            if (mode == DeveloperMode.Replay) LoadRealReplay(replayRound);
            else
            {
                developerMode = mode; replayDownloadPending = false;
                session.ResetData(); session.SelectModel("KNN"); serverModelId = CsiModels.Unspecified;
                state.Notice = "";
                backend.SetPredictionPolling(true);
                string address = PlayerPrefs.GetString("csi.endpoint", "");
                if (CsiBackendClient.ValidBaseUrl(address)) backend.Connect(address);
                else { backend.Disconnect(); state.Notice = "Open connection settings and enter the real server address."; }
                developerConnectionRevision = sharedConnection.ConnectionRevision;
            }
            Render();
        }
        private void DrawDeveloperUnified()
        {
            Button(page, "Replay", 24, 108, 165, 36, () => ChangeMode(DeveloperMode.Replay),
                developerMode == DeveloperMode.Replay ? Teal : Pale, developerMode == DeveloperMode.Replay ? Color.white : Ink);
            Button(page, "Live", 201, 108, 165, 36, () => ChangeMode(DeveloperMode.Live),
                developerMode == DeveloperMode.Live ? Teal : Pale, developerMode == DeveloperMode.Live ? Color.white : Ink);
            Panel(page, "Data source and evaluation scope", 24, 151, 342, 86, Color.white);
            bool replay = developerMode == DeveloperMode.Replay;
            Text(page, replay ? replayData == null ? "Real recorded replay / not loaded" : replayData.Title : ConnectionSummary(),
                34, 158, 322, replay ? 18 : 39, 12, Ink, true);
            Text(page, replay ? replayData == null ? "Open Replay controls to reload the bundled D1 results." : replayData.Declaration
                : "Current server output only. No automatic ground truth.\nManual static truth labels must be updated when you move.",
                34, replay ? 179 : 199, 322, replay ? 53 : 32, 10, Muted);
            for (int i = 0; i < 4; i++)
            {
                string id = CsiModels.Ids[i];
                var button = Button(page, CsiModels.Display(id), 24 + i * 87, 246, 81, 38,
                    () => Run(() => session.SelectModel(id)), session.SelectedModelId == id ? Teal : Color.white,
                    session.SelectedModelId == id ? Color.white : Ink);
                button.GetComponentInChildren<Text>().fontSize = 12;
            }
            var result = DisplayResult;
            string modelLine = result == null ? "Waiting for selected model in this window"
                : CsiModels.Display(result.ModelId) + " / " + (replay ? "window " + backend.ReplayIndex + "/" + backend.ReplayCount : "seq " + result.Prediction.seq);
            if (!replay && session.ResultFor(serverModelId) != null)
                Button(page, "Server model: " + serverModelId, 24, 289, 342, 23,
                    () => Run(() => session.SelectModel(serverModelId)), Pale, Teal).GetComponentInChildren<Text>().fontSize = 11;
            else Text(page, modelLine, 24, 289, 342, 20, 11, Muted);
            Map(24, 314, 342, 177, false);
            Panel(page, "Current window metrics", 24, 499, 342, 77, Color.white);
            Text(page, "CURRENT PREDICTION", 34, 506, 112, 13, 9, Muted, true);
            Text(page, replay ? "RECORDED TRUTH" : "MANUAL TRUTH", 152, 506, 110, 13, 9, Muted, true);
            Text(page, "ERROR (m)", 277, 506, 80, 13, 9, Muted, true);
            Text(page, result == null ? "--" : result.Prediction.label.ToString(), 34, 524, 100, 26, 22, Ink, true);
            Text(page, session.GroundTruth < 0 ? "--" : session.GroundTruth.ToString(), 152, 524, 100, 26, 22, Ink, true);
            Text(page, double.IsNaN(session.ErrorMeters) ? "N/A" : session.ErrorMeters.ToString("F2"), 277, 524, 78, 26, 22, Ink, true);
            Text(page, "Truth: " + session.TruthSource, 34, 555, 322, 15, 10, Muted);
            Panel(page, "Batch metrics independent of playback", 24, 583, 342, 134, Color.white);
            Text(page, replay ? "FULL BATCH / " + (replayData == null ? 0 : replayData.WindowCount) + " unique windows"
                : "LIVE / independently labelled selected model", 34, 590, 322, 17, 11, Ink, true);
            Text(page, "MODEL", 34, 611, 97, 14, 10, Muted, true);
            Text(page, "ACCURACY", 139, 611, 73, 14, 10, Muted, true, TextAnchor.UpperRight);
            Text(page, "MEAN (m)", 224, 611, 65, 14, 10, Muted, true, TextAnchor.UpperRight);
            Text(page, "N", 304, 611, 49, 14, 10, Muted, true, TextAnchor.UpperRight);
            for (int i = 0; i < (replay ? 4 : 1); i++)
            {
                string id = replay ? CsiModels.Ids[i] : session.SelectedModelId;
                var value = replay ? replayData == null ? new DeveloperMetrics() : replayData.MetricsFor(id) : session.MetricsFor(id);
                Text(page, CsiModels.Display(id), 34, 630 + i * 19, 97, 17, 11, Ink);
                Text(page, value.Count == 0 ? "N/A" : (value.Accuracy * 100).ToString("F1") + "%", 139, 630 + i * 19, 73, 17, 11, Ink, false, TextAnchor.UpperRight);
                Text(page, value.Count == 0 ? "N/A" : value.MeanError.ToString("F3"), 224, 630 + i * 19, 65, 17, 11, Ink, false, TextAnchor.UpperRight);
                Text(page, value.Count.ToString(), 304, 630 + i * 19, 49, 17, 11, Ink, false, TextAnchor.UpperRight);
            }
            if (!replay) Text(page, "Buttons select received results, not server inference.\nOther models require their own actual server output.",
                34, 661, 322, 43, 11, Muted);
            if (replay) Button(page, "Replay controls / " + backend.ReplayIndex + " of " + backend.ReplayCount,
                24, 725, 342, 30, () => Open("Replay controls"), Pale, Teal).GetComponentInChildren<Text>().fontSize = 12;
        }
        private void ConnectionSettings(RectTransform p)
        {
            Text(p, "LIVE PREDICTION SERVER", 24, 96, 342, 22, 12, Teal, true);
            connectionText = Text(p, ConnectionSummary(), 24, 126, 342, 62, 13, Muted);
            Text(p, "BASE ADDRESS / computer IP and port", 24, 202, 342, 22, 11, Muted, true);
            var input = Input(p, PlayerPrefs.GetString("csi.endpoint", ""), "http://<computer-ip>:8765", 24, 232, 342);
            var error = Text(p, formError, 24, 464, 342, 54, 13, Orange); formBottom = error.rectTransform;
            input.onValueChanged.AddListener(value => { formError = ""; error.text = ""; });
            Action<bool> connect = download => {
                string address = input.text.Trim();
                if (!CsiBackendClient.ValidBaseUrl(address)) { error.text = "Use a valid HTTP(S) base URL."; return; }
                if (!StopAndSave()) { error.text = state.Notice; return; }
                session.ResetData(); session.SelectModel("KNN"); serverModelId = CsiModels.Unspecified;
                PlayerPrefs.SetString("csi.endpoint", address); PlayerPrefs.Save();
                developerMode = download ? DeveloperMode.Replay : DeveloperMode.Live;
                state.Page = "Developer"; drawer = ""; connectionText = null;
                backend.SetPredictionPolling(!download);
                if (download) { replayData = null; replayDownloadPending = true; backend.DownloadReplay(address); }
                else backend.Connect(address);
                developerConnectionRevision = sharedConnection.ConnectionRevision;
                state.Notice = backend.Notice; Render();
            };
            Button(p, "Save & connect Live", 24, 300, 342, 48, () => connect(false), Teal, Color.white);
            Button(p, "Disconnect", 24, 361, 165, 44, () => Run(() => {
                if (!StopAndSave()) { formError = state.Notice; return; }
                formError = ""; backend.Disconnect();
            }), Pale, Teal);
            Button(p, "Link status", 201, 361, 165, 44, () => Open("Link status"), Pale, Teal);
            Button(p, "Download external replay", 24, 416, 342, 44, () => connect(true), Color.white, Ink);
            Text(p, "Optional: external replay requires /api/replay.\nBundled D1 replay works without any server.", 24, 521, 342, 38, 11, Muted);
        }
        private void IntegrationControls(RectTransform p)
        {
            Text(p, SourceSummary() + "\n" + (IsRecording ? "Recording received rows" : "Not recording"), 24, 97, 342, 45, 13, Muted);
            Button(p, developerMode == DeveloperMode.Replay ? "Replay controls" : "Connection settings", 24, 156, 342, 46,
                () => Open(developerMode == DeveloperMode.Replay ? "Replay controls" : "Settings"), Pale, Teal);
            Button(p, developerMode == DeveloperMode.Replay ? "Recorded truth information" : "Manual static ground truth", 24, 217, 342, 46, () => Open("Ground truth"), Color.white, Ink);
            Button(p, IsRecording ? "Stop & save recording" : "Start recording", 24, 278, 342, 46, () => Run(() => {
                if (IsRecording) { if (!StopAndSave()) return; }
                else
                {
                    if (developerMode == DeveloperMode.Live && (backend.Connection != CsiConnectionState.Connected || backend.IsStale
                        || backend.Status == null || (backend.Status.state != "ready" && backend.Status.state != "degraded")))
                    { state.Notice = "Wait for fresh ready/degraded server data before recording."; return; }
                    if (!SaveCurrent()) return;
                    session.StartRecording(developerMode == DeveloperMode.Replay);
                    state.Notice = "Recording received windows. Full replay batch metrics are independent of recording.";
                }
                if (developerMode == DeveloperMode.Replay) drawer = "";
            }), Teal, Color.white);
            Button(p, "Results & CSV export", 24, 339, 342, 46, () => Open("Results"), Pale, Teal);
            if (developerMode == DeveloperMode.Replay)
                Button(p, "Connection settings", 24, 400, 342, 46, () => Open("Settings"), Color.white, Ink);
            Text(p, developerMode == DeveloperMode.Replay
                ? "Pause replay only pauses playback. Stop recording\nseparately to save. Source changes and app pauses\nstop and save runs."
                : state.Notice, 24, developerMode == DeveloperMode.Replay ? 477 : 400, 342, 68, 13, Muted);
        }
        private void IntegrationTruth(RectTransform p)
        {
            if (developerMode == DeveloperMode.Replay)
            {
                Text(p, "Replay truth comes from the capture station label.\nIt is not the predicted position. Manual truth cannot\nchange full-batch replay accuracy or mean error.", 24, 97, 342, 90, 15, Ink);
                Text(p, replayData == null ? "No replay loaded." : replayData.Declaration, 24, 212, 342, 100, 14, Muted);
                Button(p, "Use recorded truth", 24, 355, 342, 48, () => Run(() => { if (!StopAndSave()) return; session.UseRecordedTruth(); }), Teal, Color.white);
                return;
            }
            int truth = session.GroundTruth;
            Text(p, "Manual static truth: independently confirm where\nthe person is standing. Clear it before walking.\nChanging truth stops and saves the current run.", 24, 94, 342, 68, 14, Muted);
            for (int n = 0; n < 15; n++)
            {
                int label = n;
                Button(p, n + "\n" + GridMap.Index(n), 24 + n % 5 * 69, 180 + (2 - n / 5) * 54, 65, 46,
                    () => Run(() => { if (!StopAndSave()) return; session.SetGroundTruth(label); }),
                    truth == n ? Teal : Color.white, truth == n ? Color.white : Ink);
            }
            Button(p, "Clear ground truth", 24, 354, 342, 44, () => Run(() => { if (!StopAndSave()) return; session.SetGroundTruth(-1); }), Pale, Teal);
            Text(p, "No independent truth means N/A. Live statistics use\nonly future received windows with a manual label.\nNever copy the prediction into the truth field.", 24, 435, 342, 80, 13, Muted);
        }
        private void IntegrationResults(RectTransform p)
        {
            Text(p, session.RecordCount + " recorded rows", 24, 104, 342, 45, 25, Ink, true);
            Text(p, developerMode == DeveloperMode.Replay
                ? "Full-batch metrics on the main page use the loaded\nrecorded dataset, not the recorded/playback row count.\nCSV contains only rows you actually recorded."
                : "Live CSV records actual received rows and provenance.\nManual static truth is explicitly labelled.\nNo truth gives a blank error.", 24, 180, 342, 100, 14, Muted);
            Button(p, "Export CSV", 24, 350, 342, 52, () => Run(() => {
                if (session.RecordCount == 0) state.Notice = "No recorded rows to export. Start recording first.";
                else SaveCurrent();
            }), Teal, Color.white);
            Text(p, "Saved in this app's Android files directory.\nA new run saves the previous run first.", 24, 440, 342, 60, 13, Muted);
            Text(p, state.Notice, 24, 505, 342, 60, 12, Muted);
        }
        private void ReplayControls(RectTransform p)
        {
            Text(p, "Window " + backend.ReplayIndex + " / " + backend.ReplayCount + "\nPlayback speed is for review, not measured latency.", 24, 96, 342, 55, 13, Muted);
            Button(p, backend.ReplayPlaying ? "Pause replay" : "Play replay", 24, 157, 342, 42,
                () => Run(() => { if (backend.ReplayPlaying) backend.PauseReplay(); else backend.PlayReplay(); drawer = ""; }), Teal, Color.white);
            Button(p, "Next window", 24, 207, 165, 42, () => Run(() => { backend.StepReplay(); drawer = ""; }), Pale, Teal);
            Button(p, "Restart replay", 201, 207, 165, 42, () => Run(() => {
                if (!StopAndSave()) return; session.ResetData(); backend.ResetReplay(); backend.StepReplay(); drawer = "";
            }), Pale, Teal);
            Text(p, "D1 round / full-batch statistics scope", 24, 265, 342, 20, 12, Ink, true);
            string[] labels = { "All 751", "r9 / 375", "r10 / 376" }, rounds = { "all", "r9", "r10" };
            for (int i = 0; i < 3; i++)
            {
                string round = rounds[i];
                Button(p, labels[i], 24 + i * 116, 294, 110, 42, () => Run(() => { if (LoadRealReplay(round)) drawer = ""; }),
                    replayRound == round ? Teal : Pale, replayRound == round ? Color.white : Ink).GetComponentInChildren<Text>().fontSize = 12;
            }
            Button(p, "Reload bundled / local real D1", 24, 354, 342, 42, () => Run(() => { if (LoadRealReplay(replayRound)) drawer = ""; }), Color.white, Ink);
            Button(p, "Load external cached replay", 24, 406, 342, 42, () => Run(() => {
                if (!StopAndSave()) return;
                try
                {
                    var data = RealReplayData.Parse(File.ReadAllText(backend.CachePath));
                    session.ResetData(); backend.SetPredictionPolling(false);
                    if (!backend.LoadReplayJson(data.CanonicalJson)) throw new FormatException(backend.Notice);
                    replayData = data; backend.StepReplay(); drawer = "";
                }
                catch (Exception e) { state.Notice = "External replay could not load: " + e.Message; }
            }), Pale, Teal);
            Text(p, "Default is real D1, not the former synthetic fixture.\nExternal recordings retain their declared provenance.", 24, 480, 342, 60, 12, Muted);
        }
        private void LinkStatus(RectTransform p)
        {
            connectionText = Text(p, ConnectionSummary(), 24, 98, 342, 66, 13, Muted);
            var links = backend.Status == null ? null : backend.Status.links;
            string[] names = { "piA_apX", "piA_apY", "piB_apX", "piB_apY" };
            CsiLinkHealth[] health = links == null ? new CsiLinkHealth[4] : new[] { links.piA_apX, links.piA_apY, links.piB_apX, links.piB_apY };
            for (int i = 0; i < 4; i++)
            {
                var h = health[i];
                Text(p, names[i] + "  " + (h == null ? "unreported" : (h.online ? "online" : "offline") + " / " + h.fps.ToString("F1") + " fps / age " + h.last_frame_age_s.ToString("F1") + " s"),
                    24, 185 + i * 51, 342, 44, 14, Ink);
            }
            Text(p, "Server-reported link status. Input frame rate\nis not prediction update rate. Reopen to refresh.", 24, 413, 342, 70, 13, Muted);
            Button(p, "Refresh details", 24, 506, 342, 44, () => Open("Link status"), Pale, Teal);
        }
        private bool SaveCurrent()
        {
            if (session.RecordCount == 0) return true;
            try
            {
                string name = session.RunId + ".csv", path = Path.Combine(Application.persistentDataPath, name);
                string temporary = path + ".new"; File.WriteAllText(temporary, session.Csv());
                if (File.Exists(path)) File.Replace(temporary, path, null); else File.Move(temporary, path);
                Debug.Log("CSI_CSV: " + path + " / rows=" + session.RecordCount); state.Notice = "Saved locally: " + name; return true;
            }
            catch (Exception e) { state.Notice = "CSV save failed: " + e.Message; Debug.LogError(state.Notice); return false; }
        }
        private bool StopAndSave()
        {
            if (session.Recording) session.StopRecording();
            return SaveCurrent();
        }
        private void PauseIntegration()
        {
            if (backend == null) return;
            StopAndSave(); if (backend.ReplayPlaying) backend.PauseReplay();
        }
    }
}
