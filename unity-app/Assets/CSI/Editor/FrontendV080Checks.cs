using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace CsiApp.Editor
{
    // Editor UI integration and render checks. No device, network request or AR provider is started.
    public static class FrontendV080Checks
    {
        private const BindingFlags InstanceFlags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        private const BindingFlags StaticFlags = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
        private static int checks;
        private static void Check(bool ok, string message)
        { checks++; if (!ok) throw new InvalidOperationException("Frontend v0.8.0 check: " + message); }
        private static object Get(object owner, string name) { return owner.GetType().GetField(name, InstanceFlags).GetValue(owner); }
        private static void Set(object owner, string name, object value) { owner.GetType().GetField(name, InstanceFlags).SetValue(owner, value); }
        private static object Call(object owner, string name, params object[] args)
        {
            foreach (var method in owner.GetType().GetMethods(InstanceFlags))
                if (method.Name == name && method.GetParameters().Length == args.Length)
                    try { return method.Invoke(owner, args); }
                    catch (TargetInvocationException e) { throw e.InnerException ?? e; }
            throw new MissingMethodException(owner.GetType().Name, name);
        }
        private static string PathText(IList<int> labels) { return string.Join(",", labels); }
        private static void RemoveModal(LivePositioningController controller)
        {
            var modal = (RectTransform)Get(controller, "modal");
            if (modal != null) UnityEngine.Object.DestroyImmediate(modal.gameObject);
            // The destroyed Unity object compares null, so the real CloseModal will only clear its references.
            Call(controller, "CloseModal");
        }
        [MenuItem("CSI/Checks/Frontend v0.8.0 and Editor renders")]
        public static void RunUnity()
        {
            if (Application.isPlaying) throw new InvalidOperationException("Run these isolated checks outside Play mode.");
            checks = 0;
            string output = Path.GetFullPath(Path.Combine(Application.dataPath, "../QA/2026-10-26/frontend-v090/baseline"));
            Directory.CreateDirectory(output);
            var originalScene = SceneManager.GetActiveScene();
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Additive);
            SceneManager.SetActiveScene(scene);
            var sessionField = typeof(PositioningSession).GetField("instance", StaticFlags);
            var uiField = typeof(CsiAppUi).GetField("activeInstance", StaticFlags);
            var previousSession = sessionField.GetValue(null); var previousUi = uiField.GetValue(null);
            var previousLabels = PlanningPreviewTransfer.Labels; int previousCandidate = PlanningPreviewTransfer.Candidate;
            bool previousFirst = PlanningPreviewTransfer.FirstPerson, previousReturn = PlanningPreviewTransfer.ReturnToPlanning;
            var previousOrientation = Screen.orientation;
            CsiAppUi developerUi = null;
            try
            {
                sessionField.SetValue(null, null); uiField.SetValue(null, null);
                PlanningPreviewTransfer.ReturnToPlanning = false; PlanningPreviewTransfer.Labels = null;
                var sessionObject = new GameObject("Editor QA isolated positioning session");
                var session = sessionObject.AddComponent<PositioningSession>();
                // Awake is normally not called in edit mode. Inject only the dependencies needed here;
                // do not call its runtime DontDestroyOnLoad or read/write remembered model settings.
                if (session.Backend == null)
                    Set(session, "<Backend>k__BackingField", sessionObject.AddComponent<CsiBackendClient>());
                sessionField.SetValue(null, session);
                session.Backend.SetPredictionPolling(false);
                var owner = new GameObject("Editor QA positioning page");
                var controller = owner.AddComponent<LivePositioningController>();
                if (Get(controller, "navigation") == null) Call(controller, "Start");
                var nav = (LiveNavigationState)Get(controller, "navigation");
                Check(nav.Planned && nav.PlannedStartLabel == 0 && nav.TargetLabel == 14, "manual default plan initializes");
                Check(!session.PredictionPollingEnabled && !nav.Preparing && !nav.Active && !nav.HasPosition,
                    "planning does not read or invent live position");
                Check(nav.CandidateRoutes.Count == 2, "two route choices appear for opposite corners");
                if (CanRender()) RenderPage(owner, controller, output, "01-positioning-plan-editor.png", "EDITOR QA / planning / no device");

                Call(controller, "ChooseCandidate", 1); string selected = PathText(nav.GridLabels);
                Check(nav.SelectedCandidateIndex == 1 && selected == PathText(nav.CandidateRoutes[1]), "route button selects exact second candidate");
                Call(controller, "PrimaryRouteAction");
                Check((bool)Get(controller, "routeConfirmed") && Get(controller, "modal") != null,
                    "Confirm route opens the explicit display choice");
                RemoveModal(controller); Call(controller, "ChooseDisplay", false);
                Check((bool)Get(controller, "displayChosen") && !(bool)Get(controller, "arMode")
                    && !session.PredictionPollingEnabled && !nav.Preparing && !nav.Active,
                    "choosing 2D retains gate off until the separate readiness action");
                Check(PathText(nav.GridLabels) == selected, "view selection preserves confirmed route");
                if (CanRender()) RenderPage(owner, controller, output, "02-positioning-confirmed-editor.png", "EDITOR QA / confirmed plan / no live position");

                var copied = new List<int>(nav.GridLabels);
                PlanningPreviewTransfer.Labels = new List<int>(copied); PlanningPreviewTransfer.Candidate = 1;
                PlanningPreviewTransfer.FirstPerson = true; PlanningPreviewTransfer.ReturnToPlanning = true;
                var preview = DemoNavigationRoute.BuildFromGridPath(PlanningPreviewTransfer.Labels);
                Check(PathText(preview.GridLabels) == selected && preview.DestinationId == "grid" && !preview.UsesIllustrativeConnector,
                    "simulation transfer preserves the exact selected grid route");
                Check(Math.Abs(preview.Length - nav.TotalDistance) < .001f && preview.Points.Count == copied.Count,
                    "simulation preview has the same geometry and distance");
                Check(!session.PredictionPollingEnabled && !nav.Active, "preview construction does not start real navigation");
                var returnedOwner = new GameObject("Editor QA return from model preview");
                var returned = returnedOwner.AddComponent<LivePositioningController>();
                if (Get(returned, "navigation") == null) Call(returned, "Start");
                var returnedPlan = (LiveNavigationState)Get(returned, "navigation");
                Check(PathText(returnedPlan.GridLabels) == selected && returnedPlan.SelectedCandidateIndex == 1
                    && !returnedPlan.Active && !session.PredictionPollingEnabled,
                    "return from preview restores exact candidate without starting positioning");
                UnityEngine.Object.DestroyImmediate(returnedOwner);
                PlanningPreviewTransfer.ReturnToPlanning = false;

                // Render only the HUD over a declared solid Editor background. Never start a camera.
                Set(controller, "arMode", true); ((Image)Get(controller, "background")).color = Color.clear;
                Call(controller, "Refresh");
                Check(!((ScrollRect)Get(controller, "mainScroll")).gameObject.activeSelf
                    && ((RectTransform)Get(controller, "arHud")).gameObject.activeSelf, "AR HUD replaces the large planning page");
                Check(!((SiteArTracker)Get(controller, "ar")).Ready && !session.PredictionPollingEnabled,
                    "HUD render does not initialize AR or live prediction");
                if (CanRender()) RenderPage(owner, controller, output, "03-ar-hud-editor-no-camera.png", "EDITOR QA / NO CAMERA / solid test background");
                Call(controller, "ResetPlanning");
                Check(!(bool)Get(controller, "routeConfirmed") && !(bool)Get(controller, "displayChosen")
                    && !(bool)Get(controller, "arMode") && !session.PredictionPollingEnabled && !nav.Active,
                    "return to planning clears confirmation and leaves prediction gate off");
                Check(((ScrollRect)Get(controller, "mainScroll")).gameObject.activeSelf, "2D planning page returns");
                var font = (Font)Get(controller, "font"); font.RequestCharactersInTexture("\u2191\u2192\u2190\u2193", 13);
                foreach (char arrow in "\u2191\u2192\u2190\u2193") Check(font.HasCharacter(arrow), "direction font supports arrow " + arrow);
                UnityEngine.Object.DestroyImmediate(owner);

                var replayAsset = Resources.Load<TextAsset>(RealReplayData.BundledResource);
                Check(replayAsset != null, "real D1 replay is included as a resource");
                var real = RealReplayData.Parse(replayAsset.text);
                Check(real.WindowCount == 751, "bundled replay has 751 recorded windows");
                var uiOwner = new GameObject("Editor QA Developer replay page"); developerUi = uiOwner.AddComponent<CsiAppUi>();
                if (Get(developerUi, "canvasRect") == null) Call(developerUi, "Awake");
                var oldPage = (RectTransform)Get(developerUi, "page");
                if (oldPage != null) UnityEngine.Object.DestroyImmediate(oldPage.gameObject);
                Set(developerUi, "page", null);
                var uiState = (NavigationState)Get(developerUi, "state"); uiState.Page = "Developer";
                Set(developerUi, "replayData", real); Set(developerUi, "developerEntered", true);
                Check(session.Backend.LoadReplayJson(real.CanonicalJson), "real replay loads without a network request or cache write");
                session.Backend.StepReplay(); Call(developerUi, "Render");
                var results = (DeveloperSessionState)Get(developerUi, "session");
                Check(results.PredictionForSelected != null && results.PredictionForSelected.TransportMode == "replay",
                    "Developer renders a recorded window, not a fabricated live row");
                Check(!session.PredictionPollingEnabled && !session.HasFreshLiveSelected, "Replay cannot qualify formal positioning");
                if (CanRender()) RenderPage(uiOwner, developerUi, output, "04-developer-real-replay-editor.png", "EDITOR QA / real D1 recorded results / not live");
                File.WriteAllText(Path.Combine(output, "editor-checks.txt"), "Editor QA only; not device screenshots.\nNo network or AR provider started. No PlayerPrefs changed.\nAssertions: " + checks
                    + "\nGraphics available: " + CanRender() + "\nData: D1 seed42 r9+r10, 751 recorded windows.\n");
                Debug.Log("CSI_FRONTEND_V080_PASS " + checks + " assertions / Editor QA only / " + output);
            }
            finally
            {
                if (developerUi != null)
                {
                    // Runtime OnDestroy uses delayed Destroy for its sprite; clean it safely in edit mode first.
                    var rounded = (Sprite)Get(developerUi, "rounded");
                    if (rounded != null) { UnityEngine.Object.DestroyImmediate(rounded.texture); UnityEngine.Object.DestroyImmediate(rounded); Set(developerUi, "rounded", null); }
                }
                if (scene.IsValid()) EditorSceneManager.CloseScene(scene, true);
                sessionField.SetValue(null, previousSession); uiField.SetValue(null, previousUi);
                PlanningPreviewTransfer.Labels = previousLabels; PlanningPreviewTransfer.Candidate = previousCandidate;
                PlanningPreviewTransfer.FirstPerson = previousFirst; PlanningPreviewTransfer.ReturnToPlanning = previousReturn;
                Screen.orientation = previousOrientation;
                if (originalScene.IsValid() && originalScene.isLoaded) SceneManager.SetActiveScene(originalScene);
            }
        }
        private static bool CanRender() { return SystemInfo.graphicsDeviceType != GraphicsDeviceType.Null; }
        internal static void RenderPage(GameObject owner, object controller, string output, string filename, string note,
            bool expectMap = true, Action afterRefresh = null)
        {
            var cameraObject = new GameObject("Editor QA render camera", typeof(Camera));
            var camera = cameraObject.GetComponent<Camera>(); camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color32(59, 72, 80, 255); camera.cullingMask = 1 << 5;
            camera.transform.position = new Vector3(0, 0, -100); camera.nearClipPlane = .1f; camera.farClipPlane = 200;
            var target = new RenderTexture(1080, 2340, 24); var pixels = new Texture2D(1080, 2340, TextureFormat.RGB24, false);
            var previous = RenderTexture.active;
            var mapMeshes = new Dictionary<Graphic, Mesh>();
            try
            {
                target.Create(); camera.targetTexture = target;
                foreach (var canvas in owner.GetComponentsInChildren<Canvas>(true))
                {
                    canvas.renderMode = RenderMode.ScreenSpaceCamera; canvas.worldCamera = camera; canvas.planeDistance = 1;
                    var scaler = canvas.GetComponent<CanvasScaler>(); if (scaler != null) scaler.uiScaleMode = CanvasScaler.ScaleMode.ConstantPixelSize;
                    canvas.scaleFactor = 1080f / 390f; SetLayer(canvas.gameObject, 5);
                }
                var safe = (RectTransform)Get(controller, "safe"); safe.anchorMin = Vector2.zero; safe.anchorMax = Vector2.one; safe.offsetMin = safe.offsetMax = Vector2.zero;
                Canvas.ForceUpdateCanvases();
                if (controller is LivePositioningController) Call(controller, "Refresh");
                else if (controller is CsiAppUi) Call(controller, "FitPage");
                if (afterRefresh != null) afterRefresh();
                CsiLocalization.ApplyNow();
                var labelObject = new GameObject("Editor QA declaration", typeof(RectTransform), typeof(Text)); labelObject.transform.SetParent(safe, false); labelObject.layer = 5;
                var label = labelObject.GetComponent<Text>(); label.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf"); label.text = note;
                label.fontSize = 9; label.color = Color.magenta; label.alignment = TextAnchor.MiddleCenter; label.raycastTarget = false;
                var rect = label.rectTransform; rect.anchorMin = rect.anchorMax = rect.pivot = new Vector2(.5f, 0); rect.sizeDelta = new Vector2(386, 15); rect.anchoredPosition = new Vector2(0, 2);
                Canvas.ForceUpdateCanvases(); PrepareMapMeshes(owner, mapMeshes); camera.Render();
                Canvas.ForceUpdateCanvases(); PrepareMapMeshes(owner, mapMeshes); camera.Render();
                if (expectMap && (!(controller is LivePositioningController) || !(bool)Get(controller, "arMode")))
                    Check(mapMeshes.Count > 0, filename + " contains a populated custom map graphic");
                RenderTexture.active = target; pixels.ReadPixels(new Rect(0, 0, 1080, 2340), 0, 0); pixels.Apply();
                File.WriteAllBytes(Path.Combine(output, filename), pixels.EncodeToPNG()); UnityEngine.Object.DestroyImmediate(labelObject);
                var metrics = new System.Text.StringBuilder();
                foreach (var text in owner.GetComponentsInChildren<Text>(true))
                    if (text.gameObject.activeInHierarchy && !string.IsNullOrEmpty(text.text))
                        metrics.AppendLine(text.gameObject.name + " | " + text.text.Replace("\n", " / ") + " | font=" + text.fontSize
                            + " rect=" + text.rectTransform.rect.width + "x" + text.rectTransform.rect.height
                            + " preferred=" + text.preferredWidth + "x" + text.preferredHeight
                            + " vertices=" + text.cachedTextGenerator.vertexCount);
                File.WriteAllText(Path.Combine(output, Path.GetFileNameWithoutExtension(filename) + "-text.txt"), metrics.ToString());
            }
            finally
            {
                foreach (var pair in mapMeshes)
                {
                    if (pair.Key != null) pair.Key.canvasRenderer.Clear();
                    UnityEngine.Object.DestroyImmediate(pair.Value);
                }
                camera.targetTexture = null; RenderTexture.active = previous; target.Release();
                UnityEngine.Object.DestroyImmediate(target); UnityEngine.Object.DestroyImmediate(pixels); UnityEngine.Object.DestroyImmediate(cameraObject);
            }
        }
        private static void PrepareMapMeshes(GameObject owner, Dictionary<Graphic, Mesh> meshes)
        {
            foreach (var graphic in owner.GetComponentsInChildren<Graphic>(true))
            {
                if (!(graphic is LivePositioningMapGraphic) && !(graphic is GridGraphic)) continue;
                if (!graphic.isActiveAndEnabled || !graphic.gameObject.activeInHierarchy) continue;
                Check(graphic.rectTransform.rect.width > 0 && graphic.rectTransform.rect.height > 0,
                    graphic.GetType().Name + " has a laid-out rectangle");
                // These runtime MonoBehaviours are not ExecuteAlways. A batch Editor render can
                // lay out their Text children without scheduling their own mesh rebuild. Mirror
                // uGUI Graphic.DoMeshGeneration here, after layout, without changing runtime code.
                // Use the exact overload: Graphic also has the obsolete OnPopulateMesh(Mesh).
                var populate = graphic.GetType().GetMethod("OnPopulateMesh", InstanceFlags, null,
                    new[] { typeof(VertexHelper) }, null);
                Check(populate != null, graphic.GetType().Name + " exposes its vertex generator");
                if (!meshes.TryGetValue(graphic, out var mesh))
                {
                    mesh = new Mesh { name = "Editor QA " + graphic.GetType().Name };
                    meshes.Add(graphic, mesh);
                }
                graphic.canvasRenderer.cull = false;
                graphic.SetAllDirty(); graphic.Rebuild(CanvasUpdate.PreRender);
                using (var vertices = new VertexHelper())
                {
                    try { populate.Invoke(graphic, new object[] { vertices }); }
                    catch (TargetInvocationException e) { throw e.InnerException ?? e; }
                    foreach (var component in graphic.GetComponents<Component>())
                        if (component is IMeshModifier modifier) modifier.ModifyMesh(vertices);
                    vertices.FillMesh(mesh);
                }
                Check(mesh.vertexCount > 0, graphic.GetType().Name + " generated a nonempty map mesh");
                var renderer = graphic.canvasRenderer;
                renderer.materialCount = 1;
                renderer.SetMaterial(graphic.materialForRendering ?? graphic.defaultMaterial, 0);
                renderer.SetTexture(graphic.mainTexture != null ? graphic.mainTexture : Texture2D.whiteTexture);
                renderer.SetColor(Color.white); renderer.SetMesh(mesh);
                // Assert the submitted renderer mesh too: geometry in an unused Mesh is not a render.
                var submitted = renderer.GetMesh();
                Check(submitted != null && submitted.vertexCount == mesh.vertexCount && submitted.vertexCount > 0,
                    graphic.GetType().Name + " submitted map vertices to its CanvasRenderer");
            }
        }
        private static void SetLayer(GameObject owner, int layer)
        { owner.layer = layer; foreach (Transform child in owner.transform) SetLayer(child.gameObject, layer); }
    }
}
