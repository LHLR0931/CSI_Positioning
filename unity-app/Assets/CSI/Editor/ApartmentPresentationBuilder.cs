using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

namespace CsiApp.Editor
{
    /// <summary>Separate model presentation; never edits the mobile scene list, XR or map source.</summary>
    public static class ApartmentPresentationBuilder
    {
        public const string ScenePath = "Assets/CSI/Scenes/ApartmentPresentation.unity";
        public const string FbxPath = "Assets/CSI/Models/ApartmentPresentation/apartment_presentation.fbx";
        private const string OutputFolder = "QA/2026-10-15/presentation";

        [MenuItem("CSI/Presentation/Generate separate scene")]
        public static void Prepare()
        {
            var open = SceneManager.GetSceneByPath(ScenePath);
            if (open.IsValid() && open.isLoaded && open.isDirty)
                throw new InvalidOperationException("Save or close the existing ApartmentPresentation scene first.");
            if (open.IsValid() && open.isLoaded) EditorSceneManager.CloseScene(open, true);
            var survey = ApartmentInspectionBuilder.ValidateGeometry(ApartmentMap.Current);
            var importer = AssetImporter.GetAtPath(FbxPath) as ModelImporter;
            if (importer == null) throw new InvalidOperationException("The presentation FBX is missing: " + FbxPath);
            importer.globalScale = 1f; importer.useFileScale = true;
            importer.importCameras = false; importer.importLights = false; importer.isReadable = true;
            importer.SaveAndReimport();
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(FbxPath);
            if (prefab == null) throw new InvalidOperationException("FBX model import failed.");

            string generated = "Assets/CSI/Generated/ApartmentPresentation/run_" + DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff");
            Directory.CreateDirectory(generated); AssetDatabase.Refresh();
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Additive);
            SceneManager.SetActiveScene(scene);
            var root = new GameObject("OFFLINE PRESENTATION / not AR, not live CSI");
            var controller = root.AddComponent<ApartmentPresentationController>();
            controller.architecture = (GameObject)PrefabUtility.InstantiatePrefab(prefab, scene);
            controller.architecture.name = "Blender architecture / planar dimensions surveyed / heights illustrative";
            controller.architecture.transform.SetParent(root.transform, false);
            var importEvidence = AlignAndValidate(controller.architecture);
            controller.architectureMaterials = MakeMaterials(controller.architecture, generated);
            VerifyOpenings(controller.architecture);

            var overlay = new GameObject("Survey grid and route overlay / source unchanged");
            overlay.transform.SetParent(root.transform, false);
            controller.grid = overlay.AddComponent<ApartmentWorldModel>();
            controller.grid.wallDisplayHeight = 0; controller.grid.furnitureDisplayHeight = 0;
            controller.grid.Initialize(false);
            var remove = new List<GameObject>();
            foreach (Transform child in controller.grid.GeometryRoot)
            {
                bool keep = child.name.StartsWith("Grid node ") || child.name.StartsWith("Grid edge ");
                if (child.name.StartsWith("Label / "))
                {
                    int label; keep = int.TryParse(child.name.Substring("Label / ".Length), out label) && GridMap.Valid(label);
                }
                if (!keep) remove.Add(child.gameObject);
            }
            foreach (var item in remove) UnityEngine.Object.DestroyImmediate(item);
            foreach (var label in controller.grid.GetComponentsInChildren<TextMesh>()) label.characterSize *= .55f;
            PersistGrid(controller.grid, generated);

            var cameraObject = new GameObject("Presentation camera / virtual only", typeof(Camera), typeof(AudioListener));
            cameraObject.transform.SetParent(root.transform, false);
            controller.viewCamera = cameraObject.GetComponent<Camera>();
            controller.viewCamera.clearFlags = CameraClearFlags.SolidColor;
            controller.viewCamera.backgroundColor = new Color32(237, 242, 239, 255);
            controller.viewCamera.allowHDR = false; controller.viewCamera.allowMSAA = true;
            controller.viewCamera.aspect = 1.6f;
            controller.viewCamera.cullingMask = ~(1 << 5);
            var uiCameraObject = new GameObject("Presentation UI camera / isolated overlay", typeof(Camera));
            uiCameraObject.transform.SetParent(root.transform, false);
            controller.interfaceCamera = uiCameraObject.GetComponent<Camera>();
            controller.interfaceCamera.clearFlags = CameraClearFlags.Depth;
            controller.interfaceCamera.cullingMask = 1 << 5;
            controller.interfaceCamera.depth = 20; controller.interfaceCamera.nearClipPlane = .1f; controller.interfaceCamera.farClipPlane = 2;
            controller.interfaceCamera.allowHDR = false; controller.interfaceCamera.aspect = 1.6f;
            var sun = new GameObject("Presentation lighting", typeof(Light)); sun.transform.SetParent(root.transform, false);
            var light = sun.GetComponent<Light>(); light.type = LightType.Directional; light.intensity = 1.05f;
            light.color = new Color(1f, .97f, .91f); light.shadows = LightShadows.Soft;
            sun.transform.rotation = Quaternion.Euler(52, -34, 0);
            RenderSettings.ambientMode = AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(.68f, .74f, .77f);
            RenderSettings.ambientEquatorColor = new Color(.53f, .59f, .61f);
            RenderSettings.ambientGroundColor = new Color(.38f, .42f, .40f);
            RenderSettings.fog = false;
            var walker = GameObject.CreatePrimitive(PrimitiveType.Sphere); walker.name = "Simulated walker / no sensor input";
            walker.transform.SetParent(root.transform, false); walker.transform.localScale = Vector3.one * .18f;
            UnityEngine.Object.DestroyImmediate(walker.GetComponent<Collider>());
            controller.markerMaterial = new Material(Shader.Find("Standard")) { name = "Simulated walker amber", color = new Color32(245, 174, 52, 255) };
            AssetDatabase.CreateAsset(controller.markerMaterial, generated + "/walker.mat");
            walker.GetComponent<Renderer>().sharedMaterial = controller.markerMaterial;
            controller.walker = walker.transform; walker.transform.position = new Vector3(0, .14f, 0);
            controller.SetView(ApartmentPresentationController.ViewMode.Overview);
            AssetDatabase.SaveAssets();
            if (!EditorSceneManager.SaveScene(scene, ScenePath)) throw new InvalidOperationException("Could not save separate presentation scene.");

            var evidence = new PresentationEvidence
            {
                generated_utc = DateTime.UtcNow.ToString("o"), scene_path = ScenePath, fbx_path = FbxPath,
                fbx_sha256 = Hash(FbxPath), canonical_map_sha256 = survey.source_sha256,
                import_validation = importEvidence, opening_validation = "PASS: all three 0.8 m openings clear at 1 m height (triangle ray tests)",
                route_validation = ValidateRoutes(),
                limitations = "Simulation only. No live CSI or XR. T01/exit not configured. Walls 2.4 m, furniture 1.1 m, eye 1.5 m are display assumptions. Unknown space excluded."
            };
            string output = AbsoluteOutput(); Directory.CreateDirectory(output);
            File.WriteAllText(Path.Combine(output, "presentation-evidence.json"), JsonUtility.ToJson(evidence, true));
            Debug.Log("CSI_PRESENTATION_PREPARE_PASS: " + ScenePath + " | " + importEvidence);
        }

        private static string AlignAndValidate(GameObject model)
        {
            Transform origin = Marker(model, "Survey_0"), right = Marker(model, "Survey_4"), up = Marker(model, "Survey_10");
            Vector3 x = right.position - origin.position, y = up.position - origin.position;
            if (Mathf.Abs(x.magnitude - 3.2f) > .001f || Mathf.Abs(y.magnitude - 1.6f) > .001f || Mathf.Abs(Vector3.Dot(x.normalized, y.normalized)) > .001f)
                throw new InvalidOperationException("FBX survey basis does not match 3.2 m x 1.6 m: " + x + " / " + y);
            Quaternion correction = Quaternion.Inverse(Quaternion.LookRotation(y.normalized, Vector3.Cross(y, x).normalized));
            model.transform.rotation = correction * model.transform.rotation;
            model.transform.position -= origin.position;
            RequireNear(origin.position, Vector3.zero, "Survey_0");
            RequireNear(right.position, new Vector3(3.2f, 0, 0), "Survey_4");
            RequireNear(up.position, new Vector3(0, 0, 1.6f), "Survey_10");
            float low = float.PositiveInfinity, high = float.NegativeInfinity;
            foreach (var filter in model.GetComponentsInChildren<MeshFilter>())
                foreach (var vertex in filter.sharedMesh.vertices)
                { float height = filter.transform.TransformPoint(vertex).y; low = Mathf.Min(low, height); high = Mathf.Max(high, height); }
            if (low < -.025f || high < 2.39f || high > 2.41f)
                throw new InvalidOperationException("FBX vertical axis/height is inconsistent: min=" + low + " max=" + high + ". No reflection correction is allowed.");
            return "PASS: rigid rotation/translation only; Survey_0=(0,0,0), Survey_4=(3.2,0,0), Survey_10=(0,0,1.6); height range=" + low.ToString("F4") + ".." + high.ToString("F4") + " m";
        }
        private static Transform Marker(GameObject root, string name)
        { foreach (var t in root.GetComponentsInChildren<Transform>()) if (t.name == name) return t; throw new InvalidOperationException("Missing FBX marker " + name); }
        private static void RequireNear(Vector3 a, Vector3 b, string label)
        { if (Vector3.Distance(a, b) > .001f) throw new InvalidOperationException(label + " imported at " + a + " expected " + b); }

        private static Material[] MakeMaterials(GameObject architecture, string generated)
        {
            var shader = Resources.Load<Shader>("Presentation/ApartmentPresentationSurface");
            if (shader == null) throw new InvalidOperationException("Presentation surface shader is missing.");
            var result = new List<Material>();
            foreach (var renderer in architecture.GetComponentsInChildren<MeshRenderer>())
            {
                var colors = renderer.sharedMaterials; var replacement = new Material[colors.Length];
                for (int i = 0; i < colors.Length; i++)
                {
                    var color = colors[i] != null && colors[i].HasProperty("_Color") ? colors[i].color : new Color(.74f, .80f, .77f);
                    // This FBX exports Blender linear base colors; this project renders in Gamma.
                    if (QualitySettings.activeColorSpace == ColorSpace.Gamma) color = color.gamma;
                    var material = new Material(shader) { name = renderer.name + " presentation", color = color };
                    material.SetFloat("_CutHeight", .35f); material.SetFloat("_Glossiness", .16f);
                    AssetDatabase.CreateAsset(material, generated + "/architecture_" + result.Count.ToString("D3") + ".mat");
                    result.Add(material); replacement[i] = material;
                }
                renderer.sharedMaterials = replacement;
            }
            return result.ToArray();
        }

        private static void PersistGrid(ApartmentWorldModel model, string generated)
        {
            int number = 0;
            foreach (var filter in model.GetComponentsInChildren<MeshFilter>(true))
            {
                if (filter.GetComponent<TextMesh>() != null || filter.sharedMesh == null || AssetDatabase.Contains(filter.sharedMesh)) continue;
                AssetDatabase.CreateAsset(filter.sharedMesh, generated + "/grid_" + (number++).ToString("D3") + ".asset");
            }
            number = 0;
            foreach (var material in model.GetGeneratedMaterialsForPersistence())
                if (material != null && !AssetDatabase.Contains(material)) AssetDatabase.CreateAsset(material, generated + "/grid_material_" + (number++).ToString("D2") + ".mat");
            model.SealAsInspectionAssets();
        }

        private static void VerifyOpenings(GameObject architecture)
        {
            // Intersect actual imported triangles, not just transforms or material visibility.
            foreach (var opening in ApartmentMap.Current.openings)
            {
                Vector3 a = ApartmentMap.ToWorld(opening.start_xy_m, 1), b = ApartmentMap.ToWorld(opening.end_xy_m, 1);
                Vector3 tangent = (b - a).normalized, normal = new Vector3(-tangent.z, 0, tangent.x);
                foreach (float fraction in new[] { .1f, .3f, .5f, .7f, .9f })
                {
                    Vector3 origin = Vector3.Lerp(a, b, fraction) - normal * .12f;
                    foreach (var mesh in architecture.GetComponentsInChildren<MeshFilter>())
                    {
                        var vertices = mesh.sharedMesh.vertices; var triangles = mesh.sharedMesh.triangles;
                        for (int i = 0; i < triangles.Length; i += 3)
                            if (HitsTriangle(origin, normal, .24f, mesh.transform.TransformPoint(vertices[triangles[i]]), mesh.transform.TransformPoint(vertices[triangles[i + 1]]), mesh.transform.TransformPoint(vertices[triangles[i + 2]])))
                                throw new InvalidOperationException("FBX mesh obstructs " + opening.drawing_id + " at height 1 m: " + mesh.name);
                    }
                }
            }
        }
        private static bool HitsTriangle(Vector3 origin, Vector3 direction, float maxDistance, Vector3 a, Vector3 b, Vector3 c)
        {
            Vector3 e1 = b - a, e2 = c - a, p = Vector3.Cross(direction, e2); float determinant = Vector3.Dot(e1, p);
            if (Mathf.Abs(determinant) < 1e-7f) return false;
            float inverse = 1 / determinant; Vector3 t = origin - a; float u = Vector3.Dot(t, p) * inverse;
            if (u < 0 || u > 1) return false;
            Vector3 q = Vector3.Cross(t, e1); float v = Vector3.Dot(direction, q) * inverse;
            if (v < 0 || u + v > 1) return false;
            float distance = Vector3.Dot(e2, q) * inverse; return distance >= 0 && distance <= maxDistance;
        }
        private static string ValidateRoutes()
        {
            for (int from = 0; from < 15; from++) for (int to = 0; to < 15; to++)
            {
                var route = GridMap.Route(from, to);
                if (route[0] != from || route[route.Count - 1] != to || route.Count - 1 != Math.Abs(from % 5 - to % 5) + Math.Abs(from / 5 - to / 5))
                    throw new InvalidOperationException("Invalid shortest grid route.");
                for (int i = 1; i < route.Count; i++) if (!ApartmentMap.Current.HasEdge(route[i - 1], route[i])) throw new InvalidOperationException("Route leaves configured edges.");
            }
            return "PASS: all 225 start/target pairs use shortest four-neighbor paths on the confirmed graph; no external destination enabled";
        }

        [MenuItem("CSI/Presentation/Generate and render evidence")]
        public static void PrepareAndRender()
        {
            PrepareBatchContext(); Prepare();
            var controller = UnityEngine.Object.FindFirstObjectByType<ApartmentPresentationController>();
            controller.Initialize();
            string output = AbsoluteOutput();
            Render(controller, ApartmentPresentationController.ViewMode.Overview, Path.Combine(output, "unity-overview.png"));
            Render(controller, ApartmentPresentationController.ViewMode.Plan, Path.Combine(output, "unity-route-plan.png"));
            Render(controller, ApartmentPresentationController.ViewMode.FirstPerson, Path.Combine(output, "unity-first-person.png"));
            controller.SetView(ApartmentPresentationController.ViewMode.Overview);
            Debug.Log("CSI_PRESENTATION_RENDER_PASS: " + output);
        }
        private static void PrepareBatchContext()
        {
            if (Application.isBatchMode) EditorSceneManager.OpenScene("Assets/CSI/Scenes/UiDemo.unity", OpenSceneMode.Single);
        }
        private static void Render(ApartmentPresentationController controller, ApartmentPresentationController.ViewMode mode, string file)
        {
            if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.Null) throw new InvalidOperationException("Render needs graphics; omit -nographics.");
            var camera = controller.viewCamera; var uiCamera = controller.interfaceCamera; var previous = RenderTexture.active; var oldTarget = camera.targetTexture;
            var target = new RenderTexture(1600, 1000, 24, RenderTextureFormat.ARGB32) { antiAliasing = 4 };
            var pixels = new Texture2D(1600, 1000, TextureFormat.RGB24, false);
            try
            {
                target.Create(); camera.targetTexture = target; uiCamera.targetTexture = target; camera.aspect = uiCamera.aspect = 1.6f;
                controller.SetView(mode); Canvas.ForceUpdateCanvases(); camera.Render(); uiCamera.Render();
                Canvas.ForceUpdateCanvases(); camera.Render(); uiCamera.Render(); RenderTexture.active = target;
                pixels.ReadPixels(new Rect(0, 0, 1600, 1000), 0, 0); pixels.Apply(); File.WriteAllBytes(file, pixels.EncodeToPNG());
            }
            finally { camera.targetTexture = oldTarget; uiCamera.targetTexture = null; RenderTexture.active = previous; target.Release(); UnityEngine.Object.DestroyImmediate(target); UnityEngine.Object.DestroyImmediate(pixels); }
        }

        [MenuItem("CSI/Presentation/Build Windows standalone")]
        public static void BuildWindows()
        {
            if (!File.Exists(ScenePath)) { PrepareBatchContext(); Prepare(); }
            string output = Path.GetFullPath(Path.Combine(Application.dataPath, "../Builds/Presentation/CSI-Apartment-Presentation.exe"));
            Directory.CreateDirectory(Path.GetDirectoryName(output));
            // Explicit scenes avoid modifying EditorBuildSettings.scenes or the mobile product settings.
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions { scenes = new[] { ScenePath }, target = BuildTarget.StandaloneWindows64,
                locationPathName = output, options = BuildOptions.Development });
            Debug.Log("CSI_PRESENTATION_BUILD_RESULT: " + report.summary.result + " / " + output + " / bytes=" + report.summary.totalSize + " / errors=" + report.summary.totalErrors);
            if (report.summary.result != BuildResult.Succeeded) throw new InvalidOperationException("Presentation Windows build failed.");
        }
        private static string AbsoluteOutput() { return Path.GetFullPath(Path.Combine(Application.dataPath, "..", OutputFolder)); }
        private static string Hash(string path)
        { using (var sha = SHA256.Create()) using (var stream = File.OpenRead(path)) return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", ""); }
        [Serializable] private sealed class PresentationEvidence
        {
            public string generated_utc, scene_path, fbx_path, fbx_sha256, canonical_map_sha256;
            public string import_validation, opening_validation, route_validation, limitations;
        }
    }
}
