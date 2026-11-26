using System;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace CsiApp.Editor
{
    // Isolated UI/state fixtures. The synthetic navigation renders are not field evidence.
    public static class FrontendV090Checks
    {
        private const BindingFlags Flags = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
        private const BindingFlags Statics = BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public;
        private static int checks;
        private static object Get(object o, string n) { return o.GetType().GetField(n, Flags).GetValue(o); }
        private static void Set(object o, string n, object v) { o.GetType().GetField(n, Flags).SetValue(o, v); }
        private static object Call(object o, string n, params object[] args)
        {
            foreach (var m in o.GetType().GetMethods(Flags))
                if (m.Name == n && m.GetParameters().Length == args.Length)
                    try { return m.Invoke(o, args); } catch (TargetInvocationException e) { throw e.InnerException ?? e; }
            throw new MissingMethodException(n);
        }
        private static void Check(bool value, string reason)
        { checks++; if (!value) throw new InvalidOperationException("Frontend v0.9.0: " + reason); }
        private static bool Visible(object owner, string field)
        { return ((Component)Get(owner, field)).gameObject.activeInHierarchy; }
        private static void VisibleGlyphs(object owner, params string[] fields)
        {
            foreach (string field in fields)
            {
                var text = (Text)Get(owner, field);
                Check(text.gameObject.activeInHierarchy && text.cachedTextGenerator.vertexCount >= 4,
                    field + " has rendered glyphs, not just a nonempty text property");
            }
        }
        private static void DeletePage(CsiAppUi ui)
        {
            var page = (RectTransform)Get(ui, "page");
            if (page != null) UnityEngine.Object.DestroyImmediate(page.gameObject);
            Set(ui, "page", null);
        }
        private static CsiPrediction Prediction(int label, int seq)
        {
            var probabilities = new float[15]; probabilities[label] = 1;
            return new CsiPrediction { label = label, seq = seq, session_id = "editor-ui-fixture", model_id = "KNN",
                source = "live", state = "ready", window_id = "fixture-" + seq,
                timestamp = new DateTime(2026, 10, 26, 0, 0, 0, DateTimeKind.Utc).AddSeconds(seq * 2).ToString("o"),
                x = label % 5, y = label / 5, x_m = label % 5 * .8f, y_m = label / 5 * .8f,
                confidence = 1, probs = probabilities };
        }
        [MenuItem("CSI/Checks/Frontend v0.9.0 bilingual layout")]
        public static void RunUnity()
        {
            checks = 0;
            string output = Path.GetFullPath(Path.Combine(Application.dataPath, "../QA/2026-10-26/frontend-v090"));
            Directory.CreateDirectory(output);
            bool previousLanguage = CsiLocalization.IsChinese;
            if (Application.isBatchMode && string.IsNullOrEmpty(SceneManager.GetActiveScene().path))
                EditorSceneManager.OpenScene("Assets/CSI/Scenes/UiDemo.unity", OpenSceneMode.Single);
            var priorScene = SceneManager.GetActiveScene();
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Additive);
            SceneManager.SetActiveScene(scene);
            var sf = typeof(PositioningSession).GetField("instance", Statics);
            var uf = typeof(CsiAppUi).GetField("activeInstance", Statics);
            var oldSession = sf.GetValue(null); var oldUi = uf.GetValue(null);
            var oldOrientation = Screen.orientation;
            CsiAppUi ui = null;
            try
            {
                CsiLocalization.SetLanguage(false, false); sf.SetValue(null, null); uf.SetValue(null, null);
                var owner = new GameObject("UI QA isolated session");
                var session = owner.AddComponent<PositioningSession>();
                if (session.Backend == null) Set(session, "<Backend>k__BackingField", owner.AddComponent<CsiBackendClient>());
                sf.SetValue(null, session); session.Backend.SetPredictionPolling(false);
                var uiOwner = new GameObject("UI QA Home"); ui = uiOwner.AddComponent<CsiAppUi>();
                if (Get(ui, "canvasRect") == null) Call(ui, "Awake");
                CsiLocalization.ApplyNow();
                Check(CsiLocalization.UiFont.name != "LegacyRuntime", "bundled CJK font loaded");
                CsiLocalization.UiFont.RequestCharactersInTexture("中文导航剩余↑←→↓✓", 24);
                foreach (char glyph in "中文导航剩余↑←→↓✓") Check(CsiLocalization.UiFont.HasCharacter(glyph), "font contains " + glyph);
                Render(uiOwner, ui, output, "01-home-en.png", false);
                CsiLocalization.SetLanguage(true, false);
                Check(CsiLocalization.Translate("Settings") != "Settings", "Chinese settings label exists");
                foreach (string id in new[] { "CSI", "AR", "KNN", "GCN", "MLP", "MLP-mask", "F-MLP", "d1_r9r10_seed42" })
                    Check(CsiLocalization.Translate(id) == id, "stable model/data identifier " + id);
                Render(uiOwner, ui, output, "02-home-zh.png", false);
                DeletePage(ui); Set(ui, "drawer", "Home settings"); Call(ui, "Render");
                Render(uiOwner, ui, output, "03-settings-zh.png", false);
                DeletePage(ui); Set(ui, "drawer", "Language settings"); Call(ui, "Render");
                Render(uiOwner, ui, output, "04-language-zh.png", false);
                DeletePage(ui); Set(ui, "drawer", "Settings"); Call(ui, "Render");
                var inputs = uiOwner.GetComponentsInChildren<InputField>(true);
                Check(inputs.Length > 0, "connection input exists");
                inputs[0].text = "http://192.168.1.10:8765"; CsiLocalization.ApplyNow();
                Check(inputs[0].textComponent.text == "http://192.168.1.10:8765", "translation never rewrites address input");
                Render(uiOwner, ui, output, "05-connection-zh.png", false);
                DeletePage(ui); Set(ui, "drawer", "");
                ((NavigationState)Get(ui, "state")).Page = "Developer";
                var replay = RealReplayData.Parse(Resources.Load<TextAsset>(RealReplayData.BundledResource).text);
                Set(ui, "replayData", replay); Set(ui, "developerEntered", true);
                session.Backend.LoadReplayJson(replay.CanonicalJson); session.Backend.StepReplay(); Call(ui, "Render");
                Render(uiOwner, ui, output, "06-developer-zh.png", true);
                uiOwner.SetActive(false);

                var navOwner = new GameObject("UI QA positioning");
                var controller = navOwner.AddComponent<LivePositioningController>();
                if (Get(controller, "navigation") == null) Call(controller, "Start");
                var nav = (LiveNavigationState)Get(controller, "navigation");
                string fixedRoute = string.Join(",", nav.GridLabels);
                Call(controller, "Refresh");
                Check(!Visible(controller, "endButton") && !Visible(controller, "viewChoice"), "planning has no redundant replan or disabled AR button");
                Check(!session.PredictionPollingEnabled, "planning keeps polling off");
                Render(navOwner, controller, output, "07-planning-zh.png", true);
                VisibleGlyphs(controller, "phaseText", "routeSummary");
                CsiLocalization.SetLanguage(false, false); Call(controller, "Refresh");
                Render(navOwner, controller, output, "08-planning-en.png", true);
                Check(string.Join(",", nav.GridLabels) == fixedRoute && !session.PredictionPollingEnabled, "language change preserves route and gate");
                Set(controller, "routeConfirmed", true); Set(controller, "displayChosen", true);
                Check(nav.BeginPreparation(), "fixture begins at planned start"); int seq = 0;
                for (int i = 0; i < 4; i++) nav.AcceptCsi(Prediction(0, ++seq), "live");
                Check(nav.StartPrepared(), "fixture uses baseline + three distinct windows before explicit start");
                for (int label = 1; label <= 2; label++) for (int n = 0; n < 2; n++) nav.AcceptCsi(Prediction(label, ++seq), "live");
                Call(controller, "Refresh");
                Check(((Text)Get(controller, "directionArrow")).text == "↑" && Math.Abs(nav.RemainingDistance - 3.2f) < .001f,
                    "accepted route point 2 gives straight and 3.2m");
                Render(navOwner, controller, output, "09-navigation-en-fixture.png", true);
                VisibleGlyphs(controller, "directionArrow", "instructionText", "nextPointText", "distanceText");
                CsiLocalization.SetLanguage(true, false);
                Render(navOwner, controller, output, "10-navigation-zh-fixture.png", true);
                for (int label = 3; label <= 4; label++) for (int n = 0; n < 2; n++) nav.AcceptCsi(Prediction(label, ++seq), "live");
                Call(controller, "Refresh");
                Check(((Text)Get(controller, "directionArrow")).text == "←" && Math.Abs(nav.RemainingDistance - 1.6f) < .001f,
                    "turn appears at accepted grid 4, not before it");
                Render(navOwner, controller, output, "11-turn-zh-fixture.png", true);
                VisibleGlyphs(controller, "directionArrow", "instructionText", "nextPointText", "distanceText");
                Set(controller, "arMode", true); ((Image)Get(controller, "background")).color = Color.clear;
                FrontendV080Checks.RenderPage(navOwner, controller, output, "12-ar-turn-zh-fixture.png", "EDITOR / synthetic state / NO CAMERA", false, () =>
                {
                    Call(controller, "RefreshArNavigationGuidance", Vector3.right); Call(controller, "SetArWarning", "");
                    ((Button)Get(controller, "arPrimary")).GetComponentInChildren<Text>().text = (string)Call(controller, "PrimaryRouteLabel");
                });
                VisibleGlyphs(controller, "arHudArrow", "arHudTitle", "arHudDetail", "arHudDistance");
                Check(((RectTransform)Get(controller, "arHudStatus")).sizeDelta.y + ((RectTransform)Get(controller, "arHudFooter")).sizeDelta.y <= 844 * .24f,
                    "AR standard chrome occupies at most 24 percent of reference screen height");
                Check(!((SiteArTracker)Get(controller, "ar")).Ready && !session.PredictionPollingEnabled, "render never starts hardware or network polling");
                CsiLocalization.SetLanguage(false, false);
                Check(CsiLocalization.Translate("Settings") == "Settings", "English roundtrip restores source");
                Check(string.Join(",", nav.GridLabels) == fixedRoute && nav.ProgressIndex == 4, "language and render changes preserve fixed route/progress");
                File.WriteAllText(Path.Combine(output, "editor-checks.txt"), "Editor fixtures, not phone/field evidence.\nNo network or camera started; no preferences changed.\nAssertions: " + checks + "\n");
                Debug.Log("CSI_FRONTEND_V090_PASS " + checks + " assertions / " + output);
            }
            finally
            {
                if (ui != null)
                {
                    var sprite = (Sprite)Get(ui, "rounded");
                    if (sprite != null) { UnityEngine.Object.DestroyImmediate(sprite.texture); UnityEngine.Object.DestroyImmediate(sprite); Set(ui, "rounded", null); }
                }
                if (scene.IsValid()) EditorSceneManager.CloseScene(scene, true);
                sf.SetValue(null, oldSession); uf.SetValue(null, oldUi); Screen.orientation = oldOrientation;
                CsiLocalization.SetLanguage(previousLanguage, false);
                if (priorScene.IsValid() && priorScene.isLoaded) SceneManager.SetActiveScene(priorScene);
            }
        }
        private static void Render(GameObject owner, object controller, string output, string file, bool map)
        { FrontendV080Checks.RenderPage(owner, controller, output, file, "EDITOR QA / NO CAMERA / not field evidence", map); }
    }
}
