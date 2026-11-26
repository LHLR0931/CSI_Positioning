using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace CsiApp
{
    /// <summary>Local-metre lines, floor arrows and optional simulated marker. No camera or floor ownership.</summary>
    public sealed class DemoRouteVisual : MonoBehaviour
    {
        public float floorOffset = .075f;
        public float lineWidth = .045f;
        private Transform generated, marker;
        private DemoNavigationRoute route;
        private Material material;
        private readonly List<Mesh> meshes = new List<Mesh>();
        private int renderLayer;
        public float Distance { get; private set; }
        private static readonly Color RouteColor = new Color32(0, 132, 113, 255);
        private static readonly Color ApproachColor = new Color32(215, 139, 48, 255);

        public void Build(DemoNavigationRoute value, bool showMarker = false)
        {
            if (value == null) throw new ArgumentNullException("value");
            Clear(); route = value;
            var shader = Resources.Load<Shader>("Maps/ApartmentMapUnlit");
            if (shader == null) throw new InvalidOperationException("ApartmentMapUnlit resource missing.");
            material = new Material(shader) { name = "Demo route / not measured navigation" };
            material.color = Color.white;
            generated = Child("Demo route arrows / local metres", transform);
            for (int i = 1; i < route.Points.Count; i++)
            {
                Vector3 a = route.Points[i - 1] + Vector3.up * floorOffset, b = route.Points[i] + Vector3.up * floorOffset;
                Color color = route.IsIllustrativeSegment(i) ? ApproachColor : RouteColor;
                Strip("Route segment " + i, a, b, lineWidth, color);
                Vector3 direction = (b - a).normalized;
                float length = Vector3.Distance(a, b);
                int count = Mathf.Max(1, Mathf.FloorToInt(length / .55f));
                for (int n = 0; n < count; n++)
                {
                    Vector3 mid = Vector3.Lerp(a, b, (n + .5f) / count) + Vector3.up * .003f;
                    Vector3 side = Vector3.Cross(Vector3.up, direction);
                    MakeMesh("Floor arrow " + i + "/" + n,
                        new[] { mid + direction * .13f, mid - direction * .085f + side * .105f, mid - direction * .085f - side * .105f },
                        new[] { 0, 1, 2 }, color, generated);
                }
            }
            Vector3 end = route.Points[route.Points.Count - 1] + Vector3.up * floorOffset;
            Color targetColor = route.UsesIllustrativeConnector ? ApproachColor : RouteColor;
            Vector3[] corners = { end + new Vector3(-.16f, 0, -.16f), end + new Vector3(.16f, 0, -.16f),
                end + new Vector3(.16f, 0, .16f), end + new Vector3(-.16f, 0, .16f) };
            for (int i = 0; i < 4; i++) Strip("Target outline " + i, corners[i], corners[(i + 1) % 4], .025f, targetColor);
            if (showMarker)
            {
                marker = Child("Simulated progress / not camera pose", generated);
                MakeMesh("Simulated position", new[] { new Vector3(0, 0, .13f), new Vector3(-.10f, 0, -.10f), new Vector3(.10f, 0, -.10f) },
                    new[] { 0, 1, 2 }, new Color32(52, 112, 203, 255), marker);
            }
            SetDistance(0);
        }
        public void SetDistance(float metres)
        {
            if (route == null) return;
            Distance = Mathf.Clamp(metres, 0, route.Length);
            if (marker == null) return;
            Vector3 direction;
            marker.localPosition = route.PointAtDistance(Distance, out direction) + Vector3.up * (floorOffset + .025f);
            marker.localRotation = Quaternion.LookRotation(direction, Vector3.up);
        }
        public void SetRenderLayer(int layer)
        {
            if (layer < 0 || layer > 31) throw new ArgumentOutOfRangeException("layer");
            renderLayer = layer;
            if (generated != null) foreach (var item in generated.GetComponentsInChildren<Transform>(true)) item.gameObject.layer = layer;
        }
        private Transform Child(string name, Transform parent)
        {
            var go = new GameObject(name); go.layer = renderLayer; go.transform.SetParent(parent, false); return go.transform;
        }
        private void Strip(string name, Vector3 a, Vector3 b, float width, Color color)
        {
            Vector3 side = Vector3.Cross(Vector3.up, (b - a).normalized) * width / 2;
            MakeMesh(name, new[] { a - side, b - side, b + side, a + side }, new[] { 0, 1, 2, 0, 2, 3 }, color, generated);
        }
        private void MakeMesh(string name, Vector3[] vertices, int[] indices, Color color, Transform parent)
        {
            var colors = new Color[vertices.Length]; for (int i = 0; i < colors.Length; i++) colors[i] = color;
            var mesh = new Mesh { name = name, vertices = vertices, triangles = indices, colors = colors }; mesh.RecalculateBounds(); meshes.Add(mesh);
            var go = Child(name, parent).gameObject; go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var renderer = go.AddComponent<MeshRenderer>(); renderer.sharedMaterial = material;
            renderer.shadowCastingMode = ShadowCastingMode.Off; renderer.receiveShadows = false;
        }
        public void Clear()
        {
            if (generated != null) { generated.gameObject.SetActive(false); DestroyOwned(generated.gameObject); }
            foreach (var mesh in meshes) DestroyOwned(mesh); meshes.Clear(); DestroyOwned(material);
            generated = marker = null; material = null; route = null; Distance = 0;
        }
        private static void DestroyOwned(UnityEngine.Object value)
        { if (value == null) return; if (Application.isPlaying) Destroy(value); else DestroyImmediate(value); }
        private void OnDestroy() { Clear(); }
    }
}
