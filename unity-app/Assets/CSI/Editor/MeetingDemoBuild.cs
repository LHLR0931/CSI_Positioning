using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace CsiApp.Editor
{
    /// <summary>Meeting-only resources and checks; never changes the measured map.</summary>
    public static class MeetingDemoBuild
    {
        public static void PrepareArchitectureResource()
        {
            const string folder = "Assets/Resources/Presentation/MeetingArchitectureMaterials";
            const string target = "Assets/Resources/Presentation/ApartmentArchitecture.prefab";
            Directory.CreateDirectory(folder);
            AssetDatabase.Refresh();
            var scene = SceneManager.GetSceneByPath(ApartmentPresentationBuilder.ScenePath);
            bool close = !scene.IsValid() || !scene.isLoaded;
            if (close) scene = EditorSceneManager.OpenScene(ApartmentPresentationBuilder.ScenePath, OpenSceneMode.Additive);
            ApartmentPresentationController source = null;
            foreach (var root in scene.GetRootGameObjects())
            {
                source = root.GetComponentInChildren<ApartmentPresentationController>();
                if (source != null) break;
            }
            if (source == null || source.architecture == null)
                throw new InvalidOperationException("Validated Blender architecture is missing.");
            var wrapper = new GameObject("Apartment architecture / surveyed plane, illustrative heights");
            try
            {
                var copy = UnityEngine.Object.Instantiate(source.architecture);
                copy.name = "Blender architecture / corrected survey transform";
                copy.transform.SetParent(wrapper.transform, false);
                copy.transform.localPosition = source.architecture.transform.localPosition;
                copy.transform.localRotation = source.architecture.transform.localRotation;
                copy.transform.localScale = source.architecture.transform.localScale;
                var materials = new Dictionary<Material, Material>();
                foreach (var renderer in copy.GetComponentsInChildren<Renderer>(true))
                {
                    var originals = renderer.sharedMaterials;
                    for (int i = 0; i < originals.Length; i++)
                    {
                        if (originals[i] == null) continue;
                        Material material;
                        if (!materials.TryGetValue(originals[i], out material))
                        {
                            string path = folder + "/surface_" + materials.Count.ToString("D3") + ".mat";
                            material = AssetDatabase.LoadAssetAtPath<Material>(path);
                            if (material == null)
                            {
                                material = new Material(originals[i]); AssetDatabase.CreateAsset(material, path);
                            }
                            else EditorUtility.CopySerialized(originals[i], material);
                            material.name = "Meeting architecture " + materials.Count;
                            if (material.HasProperty("_CutHeight")) material.SetFloat("_CutHeight", 20f);
                            EditorUtility.SetDirty(material); materials.Add(originals[i], material);
                        }
                        originals[i] = material;
                    }
                    renderer.sharedMaterials = originals;
                }
                ValidateMarker(wrapper, "Survey_0", Vector3.zero);
                ValidateMarker(wrapper, "Survey_4", new Vector3(3.2f, 0, 0));
                ValidateMarker(wrapper, "Survey_10", new Vector3(0, 0, 1.6f));
                if (PrefabUtility.SaveAsPrefabAsset(wrapper, target) == null)
                    throw new InvalidOperationException("AR architecture prefab was not saved.");
                AssetDatabase.SaveAssets();
                Debug.Log("CSI_MEETING_ARCHITECTURE_PASS: " + target + " / 0,4,10 aligned / independent materials");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(wrapper);
                if (close) EditorSceneManager.CloseScene(scene, true);
            }
        }

        private static void ValidateMarker(GameObject root, string name, Vector3 expected)
        {
            foreach (var t in root.GetComponentsInChildren<Transform>(true))
                if (t.name == name)
                {
                    if (Vector3.Distance(root.transform.InverseTransformPoint(t.position), expected) > .002f)
                        throw new InvalidOperationException("Resource transform moved " + name);
                    return;
                }
            throw new InvalidOperationException("Resource lacks " + name);
        }

        public static void ValidateDemoRoutes()
        {
            int cases = 0;
            foreach (string destination in new[] { "balcony", "office", "kitchen", "grid" })
                for (int start = 0; start < 15; start++)
                    for (int target = 0; target < (destination == "grid" ? 15 : 1); target++)
                    {
                        var route = DemoNavigationRoute.Build(start, destination, target);
                        if (route.Points.Count == 0 || route.GridLabels.Count == 0 || route.GridLabels[0] != start)
                            throw new InvalidOperationException("Demo route lost its start.");
                        int connector = destination == "balcony" ? 5 : destination == "office" ? 12 : destination == "kitchen" ? 14 : target;
                        if (route.ConnectorLabel != connector || route.GridLabels[route.GridLabels.Count - 1] != connector)
                            throw new InvalidOperationException("Demo route connector mismatch.");
                        for (int i = 1; i < route.GridLabels.Count; i++)
                            if (!ApartmentMap.Current.HasEdge(route.GridLabels[i - 1], route.GridLabels[i]))
                                throw new InvalidOperationException("Non-neighbor grid segment.");
                        float length = 0;
                        for (int i = 1; i < route.Points.Count; i++)
                        {
                            float edge = Vector3.Distance(route.Points[i - 1], route.Points[i]);
                            if (edge < .0001f || float.IsNaN(edge)) throw new InvalidOperationException("Degenerate route edge.");
                            CheckNoBarrierCrossing(route.Points[i - 1], route.Points[i]);
                            length += edge;
                        }
                        if (Mathf.Abs(length - route.Length) > .001f)
                            throw new InvalidOperationException("Demo route length is inconsistent.");
                        if (route.UsesIllustrativeConnector != (destination != "grid"))
                            throw new InvalidOperationException("Demo/source provenance is inconsistent.");
                        cases++;
                    }
            foreach (string startPlace in new[] { "balcony", "office", "kitchen" })
                foreach (string destination in new[] { "balcony", "office", "kitchen", "grid" })
                    for (int target = 0; target < (destination == "grid" ? 15 : 1); target++)
                    {
                        var route = DemoNavigationRoute.BuildFromPlace(startPlace, destination, target);
                        Vector3 expected = startPlace == "balcony" ? new Vector3(-.522f, 0, 1.2f) :
                            startPlace == "office" ? new Vector3(2f, 0, 2.58f) : new Vector3(4.4f, 0, 2.58f);
                        if (route.Points.Count == 0 || Vector3.Distance(route.Points[0], expected) > .002f)
                            throw new InvalidOperationException("Place route did not start at its explicit DEMO stand-off.");
                        if (startPlace == destination && (route.Points.Count != 1 || route.Length > .001f))
                            throw new InvalidOperationException("Same-place demo must not create a round trip.");
                        for (int i = 1; i < route.Points.Count; i++) CheckNoBarrierCrossing(route.Points[i - 1], route.Points[i]);
                        for (int i = 1; i < route.GridLabels.Count; i++)
                            if (!ApartmentMap.Current.HasEdge(route.GridLabels[i - 1], route.GridLabels[i]))
                                throw new InvalidOperationException("Place route crossed a missing grid edge.");
                        cases++;
                    }
            var turnRegression = DemoNavigationRoute.BuildFromPlace("balcony", "kitchen");
            if (turnRegression.InstructionAtDistance(.89f) != "Turn back now")
                throw new InvalidOperationException("Balcony departure U-turn must not be called a right turn or rounded to 0.0 m.");
            if (turnRegression.InstructionAtDistance(.49f) != "Turn right now")
                throw new InvalidOperationException("Near-turn instruction must be immediately actionable.");
            if (Shader.Find("CSI/ARMeetingSurface") == null)
                throw new InvalidOperationException("Full-size AR transparent surface shader is missing.");
            Debug.Log("CSI_MEETING_ROUTE_CHECKS_PASS: " + cases + " routes / 15 grid starts / confirmed connectors / simulation only");
            Debug.Log("CSI_MEETING_TURN_REGRESSION_PASS: near turn and 180-degree turn / AR overlay shader present");
        }

        private static void CheckNoBarrierCrossing(Vector3 from, Vector3 to)
        {
            var a = new Vector2(from.x, from.z);
            var direction = new Vector2(to.x - from.x, to.z - from.z);
            foreach (var wall in ApartmentMap.Current.boundaries)
            {
                if (wall.kind == "open_connection" || wall.kind == "region_boundary_unspecified") continue;
                var c = new Vector2(wall.start_xy_m[0], wall.start_xy_m[1]);
                var edge = new Vector2(wall.end_xy_m[0], wall.end_xy_m[1]) - c;
                float denominator = Cross(direction, edge);
                if (Mathf.Abs(denominator) < .00001f) continue;
                float t = Cross(c - a, edge) / denominator;
                float u = Cross(c - a, direction) / denominator;
                if (t >= -.00001f && t <= 1.00001f && u >= -.00001f && u <= 1.00001f)
                    throw new InvalidOperationException("Demo centreline intersects surveyed barrier " + wall.id + " / " + from + " -> " + to);
            }
        }
        private static float Cross(Vector2 a, Vector2 b) { return a.x * b.y - a.y * b.x; }
    }
}
