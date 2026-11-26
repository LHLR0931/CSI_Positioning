using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace CsiApp
{
    /// <summary>
    /// Reusable geometry in the owner's local metre coordinates. It never creates a
    /// camera, changes the owner's transform, or decides an AR alignment/scale.
    /// All survey geometry comes from ApartmentMap.Current.
    /// </summary>
    [ExecuteAlways]
    public sealed class ApartmentWorldModel : MonoBehaviour
    {
        [Header("Temporary cutaway display heights, NOT site measurements")]
        [Min(0)] public float wallDisplayHeight = .42f;
        [Min(0)] public float furnitureDisplayHeight = .26f;
        [Header("Presentation")]
        public bool showLabels = true;
        [Range(0, 31)] public int renderLayer = 0;
        [SerializeField, HideInInspector] private Transform generatedRoot, floorsRoot;
        [SerializeField, HideInInspector] private bool bakedInspection;
        [SerializeField, HideInInspector] private Material plainMaterial, nodeMaterial, predictionMaterial, targetMaterial, routeMaterial;
        [SerializeField, HideInInspector] private List<NodeHandle> nodeHandles = new List<NodeHandle>();
        private readonly List<Mesh> ownedMeshes = new List<Mesh>();
        private readonly List<Material> ownedMaterials = new List<Material>();
        private readonly List<GameObject> routeObjects = new List<GameObject>();
        private readonly List<Mesh> routeMeshes = new List<Mesh>();
        private readonly List<TextMesh> textLabels = new List<TextMesh>();
        private SiteMapData data;
        private Font font;
        public bool IsInitialized { get { return generatedRoot != null; } }
        public Transform GeometryRoot { get { return generatedRoot; } }
        public SiteMapData Data { get { return data ?? (data = ApartmentMap.Current); } }
        public int FloorCount { get; private set; }

        [Serializable] private sealed class NodeHandle { public int label; public MeshRenderer renderer; }

        public void Initialize(bool showFloors = true)
        {
            if (generatedRoot != null)
            {
                if (floorsRoot != null) floorsRoot.gameObject.SetActive(showFloors);
                RestoreFontMaterials(); return;
            }
            data = ApartmentMap.Current;
            var shader = Resources.Load<Shader>("Maps/ApartmentMapUnlit");
            if (shader == null) throw new InvalidOperationException("ApartmentMapUnlit resource is missing.");
            font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            plainMaterial = MaterialFor(shader, Color.white, "Map vertex colors");
            nodeMaterial = MaterialFor(shader, Hex("DCE9E0"), "Grid nodes");
            predictionMaterial = MaterialFor(shader, Hex("3678CF"), "Demo prediction");
            targetMaterial = MaterialFor(shader, Hex("D89B40"), "Grid destination");
            routeMaterial = MaterialFor(shader, Hex("008C72"), "Four-neighbor route");
            generatedRoot = Child("Confirmed apartment geometry / local metres", transform);
            floorsRoot = Child("Context floors / optional for AR", generatedRoot);
            FloorCount = 0;
            foreach (var region in data.regions)
            {
                if (region.model_role == "excluded_unknown") continue;
                float x0 = region.min_xy_m[0], y0 = region.min_xy_m[1], x1 = region.max_xy_m[0], y1 = region.max_xy_m[1];
                var color = region.id == "balcony" ? Hex("DCEADD") : region.id == "office" ? Hex("E9E3DC") :
                    region.id == "kitchen" ? Hex("EEE5D2") : Hex("FAFBF6");
                OwnedMesh("Floor / " + region.id, Quad(P(x0, y0), P(x1, y0), P(x1, y1), P(x0, y1), color), floorsRoot);
                FloorCount++;
                float labelY = region.id == "living_room" ? y1 - .44f : (y0 + y1) / 2;
                AddLabel(RoomName(region.id), P((x0 + x1) / 2, labelY, .065f), .30f, Hex("697F73"));
            }
            foreach (var boundary in data.boundaries)
            {
                if (boundary.kind == "open_connection") continue;
                Vector3 a = ApartmentMap.ToWorld(boundary.start_xy_m), b = ApartmentMap.ToWorld(boundary.end_xy_m);
                if (boundary.kind == "region_boundary_unspecified")
                {
                    Dashes(boundary.id + " unspecified boundary", a + Vector3.up * .008f, b + Vector3.up * .008f, .018f, Hex("A8B1AA"));
                    continue;
                }
                bool furniture = boundary.kind == "furniture_simplified_barrier";
                float height = Mathf.Max(0, furniture ? furnitureDisplayHeight : wallDisplayHeight);
                if (height > 0)
                    OwnedMesh(boundary.id + (furniture ? " furniture / temporary height" : " zero-thickness wall / temporary height"),
                        Quad(a, b, b + Vector3.up * height, a + Vector3.up * height, furniture ? Hex("BE9274") : Hex("9AAFA1")));
                OwnedMesh(boundary.id + " plan boundary", Line(a + Vector3.up * .007f, b + Vector3.up * .007f, .018f,
                    furniture ? Hex("9A7157") : Hex("627E6C")));
            }
            foreach (var opening in data.openings)
            {
                Dashes(opening.drawing_id + " opening span / not T01", ApartmentMap.ToWorld(opening.start_xy_m, .012f),
                    ApartmentMap.ToWorld(opening.end_xy_m, .012f), .035f, Hex("63A08D"));
                Vector3 pos = ApartmentMap.ToWorld(opening.center_xy_m, .14f);
                if (opening.id == "balcony_opening") pos.x -= .24f; else pos.z += .22f;
                AddLabel(opening.drawing_id, pos, .27f, Hex("377D6C"));
            }
            foreach (var edge in data.grid.edges)
            {
                var a = data.FindPoint(edge.from); var b = data.FindPoint(edge.to);
                OwnedMesh("Grid edge " + edge.from + "-" + edge.to, Line(P(a.x_m, a.y_m, .014f), P(b.x_m, b.y_m, .014f), .017f, Hex("CDDDD2")));
            }
            foreach (var point in data.grid.points)
            {
                Mesh mesh = Disc(P(point.x_m, point.y_m, .045f), .145f); ownedMeshes.Add(mesh);
                var node = MeshObject("Grid node " + point.label, mesh, nodeMaterial, generatedRoot);
                nodeHandles.Add(new NodeHandle { label = point.label, renderer = node.GetComponent<MeshRenderer>() });
                AddLabel(point.label.ToString(), P(point.x_m, point.y_m, .07f), .34f, Hex("163D33"));
            }
            floorsRoot.gameObject.SetActive(showFloors);
        }

        public void SetRoute(IList<int> route, int prediction, int target)
        {
            if (!IsInitialized) Initialize();
            ClearRoute();
            foreach (var node in nodeHandles)
                if (node.renderer != null) node.renderer.sharedMaterial = node.label == prediction ? predictionMaterial : node.label == target ? targetMaterial : nodeMaterial;
            if (route == null) return;
            for (int i = 1; i < route.Count; i++)
            {
                var a = Data.FindPoint(route[i - 1]); var b = Data.FindPoint(route[i]);
                if (a == null || b == null || !Data.HasEdge(a.label, b.label))
                { Debug.LogWarning("ApartmentWorldModel skipped a non-neighbor route segment."); continue; }
                var mesh = Line(P(a.x_m, a.y_m, .028f), P(b.x_m, b.y_m, .028f), .068f, Color.white);
                routeMeshes.Add(mesh);
                routeObjects.Add(MeshObject("Route " + a.label + " to " + b.label, mesh, routeMaterial, generatedRoot));
            }
        }
        public void SetLabelsFacing(Camera camera)
        {
            if (camera == null || generatedRoot == null) return;
            if (textLabels.Count == 0) textLabels.AddRange(generatedRoot.GetComponentsInChildren<TextMesh>(true));
            foreach (var text in textLabels) if (text != null) text.transform.rotation = camera.transform.rotation;
        }
        public void SetRenderLayer(int layer)
        {
            if (layer < 0 || layer > 31) throw new ArgumentOutOfRangeException("layer");
            renderLayer = layer;
            if (generatedRoot != null) foreach (var child in generatedRoot.GetComponentsInChildren<Transform>(true)) child.gameObject.layer = layer;
        }
        private void OnEnable() { if (bakedInspection) RestoreFontMaterials(); }
        private void RestoreFontMaterials()
        {
            if (generatedRoot == null) return;
            if (font == null) font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            textLabels.Clear(); textLabels.AddRange(generatedRoot.GetComponentsInChildren<TextMesh>(true));
            foreach (var text in textLabels)
            { text.font = font; text.GetComponent<MeshRenderer>().sharedMaterial = font.material; }
        }
        private Transform Child(string name, Transform parent)
        {
            var go = new GameObject(name); go.layer = renderLayer; go.transform.SetParent(parent, false); return go.transform;
        }
        private void AddLabel(string value, Vector3 position, float emSize, Color color)
        {
            if (!showLabels) return;
            var go = new GameObject("Label / " + value, typeof(TextMesh)); go.layer = renderLayer;
            go.transform.SetParent(generatedRoot, false); go.transform.localPosition = position;
            go.transform.localRotation = Quaternion.Euler(90, 0, 0); // Readable from above until the owner supplies its camera.
            var text = go.GetComponent<TextMesh>(); text.font = font; text.text = value; text.fontSize = 64;
            textLabels.Add(text);
            text.characterSize = emSize * 10f / text.fontSize; text.color = color; text.fontStyle = FontStyle.Bold;
            text.anchor = TextAnchor.MiddleCenter; text.alignment = TextAlignment.Center;
            var renderer = go.GetComponent<MeshRenderer>(); renderer.sharedMaterial = font.material;
            renderer.shadowCastingMode = ShadowCastingMode.Off; renderer.receiveShadows = false;
        }
        private void Dashes(string name, Vector3 a, Vector3 b, float width, Color color)
        {
            int count = Mathf.Max(2, Mathf.CeilToInt(Vector3.Distance(a, b) / .12f));
            for (int i = 0; i < count; i += 2)
                OwnedMesh(name, Line(Vector3.Lerp(a, b, (float)i / count), Vector3.Lerp(a, b, (float)Mathf.Min(i + 1, count) / count), width, color));
        }
        private void OwnedMesh(string name, Mesh mesh, Transform parent = null)
        { ownedMeshes.Add(mesh); MeshObject(name, mesh, plainMaterial, parent == null ? generatedRoot : parent); }
        private GameObject MeshObject(string name, Mesh mesh, Material material, Transform parent)
        {
            var go = new GameObject(name, typeof(MeshFilter), typeof(MeshRenderer)); go.layer = renderLayer;
            go.transform.SetParent(parent, false); go.GetComponent<MeshFilter>().sharedMesh = mesh;
            var renderer = go.GetComponent<MeshRenderer>(); renderer.sharedMaterial = material;
            renderer.shadowCastingMode = ShadowCastingMode.Off; renderer.receiveShadows = false; return go;
        }
        private Material MaterialFor(Shader shader, Color color, string name)
        { var material = new Material(shader); material.name = name; material.color = color; ownedMaterials.Add(material); return material; }
        private static Vector3 P(float x, float y, float h = 0) { return ApartmentMap.ToWorld(x, y, h); }
        private static Color Hex(string value) { Color color; ColorUtility.TryParseHtmlString("#" + value, out color); return color; }
        private static Mesh Quad(Vector3 a, Vector3 b, Vector3 c, Vector3 d, Color color)
        {
            var mesh = new Mesh(); mesh.vertices = new[] { a, b, c, d }; mesh.triangles = new[] { 0, 1, 2, 0, 2, 3 };
            mesh.colors = new[] { color, color, color, color }; mesh.RecalculateBounds(); return mesh;
        }
        private static Mesh Line(Vector3 a, Vector3 b, float width, Color color)
        { Vector3 dir = (b - a).normalized, n = new Vector3(-dir.z, 0, dir.x) * width / 2; return Quad(a - n, b - n, b + n, a + n, color); }
        private static Mesh Disc(Vector3 center, float radius)
        {
            const int count = 32; var vertices = new Vector3[count + 1]; var colors = new Color[count + 1]; var indices = new int[count * 3];
            vertices[0] = center; colors[0] = Color.white;
            for (int i = 0; i < count; i++)
            {
                float angle = 2 * Mathf.PI * i / count; vertices[i + 1] = center + new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle)) * radius;
                colors[i + 1] = Color.white; indices[i * 3] = 0; indices[i * 3 + 1] = i + 1; indices[i * 3 + 2] = (i + 1) % count + 1;
            }
            var mesh = new Mesh(); mesh.vertices = vertices; mesh.triangles = indices; mesh.colors = colors; mesh.RecalculateBounds(); return mesh;
        }
        private void ClearRoute()
        {
            foreach (var go in routeObjects) { if (go != null) go.SetActive(false); DestroyOwned(go); }
            foreach (var mesh in routeMeshes) DestroyOwned(mesh);
            routeObjects.Clear(); routeMeshes.Clear();
        }
        private static void DestroyOwned(UnityEngine.Object value)
        { if (value == null) return; if (Application.isPlaying) Destroy(value); else DestroyImmediate(value); }
        public void Dispose()
        {
            ClearRoute();
            // Persistent inspection geometry/assets belong to the saved scene/project,
            // not to this component's runtime allocation lists.
            if (bakedInspection) return;
            if (generatedRoot != null) { generatedRoot.gameObject.SetActive(false); DestroyOwned(generatedRoot.gameObject); }
            foreach (var mesh in ownedMeshes) DestroyOwned(mesh);
            foreach (var material in ownedMaterials) DestroyOwned(material);
            ownedMeshes.Clear(); ownedMaterials.Clear(); nodeHandles.Clear(); textLabels.Clear(); generatedRoot = null; floorsRoot = null; data = null;
        }
        private void OnDestroy() { Dispose(); }
#if UNITY_EDITOR
        public Material[] GetGeneratedMaterialsForPersistence()
        { return new[] { plainMaterial, nodeMaterial, predictionMaterial, targetMaterial, routeMaterial }; }
        public void SealAsInspectionAssets()
        {
            // Call only after the Editor builder persists every generated Mesh and Material.
            bakedInspection = true; ownedMeshes.Clear(); ownedMaterials.Clear(); routeMeshes.Clear(); routeObjects.Clear();
        }
#endif
        private static string RoomName(string id)
        {
            switch (id)
            {
                case "balcony": return "BALCONY"; case "office": return "OFFICE"; case "kitchen": return "KITCHEN";
                case "living_room": return "LIVING ROOM"; case "corridor_extension": return "CORRIDOR"; default: return id;
            }
        }
    }
}
