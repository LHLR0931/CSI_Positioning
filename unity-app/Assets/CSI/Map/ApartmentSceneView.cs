using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace CsiApp
{
    /// <summary>
    /// A virtual survey-model preview rendered into a UI texture, not a camera AR feed.
    /// Wall planes have zero thickness. Display heights are temporary cutaway styling,
    /// never measured wall/furniture heights. Unknown space is deliberately not filled.
    /// </summary>
    public sealed class ApartmentSceneView : MonoBehaviour
    {
        private const int PreviewLayer = 30;
        private const float WallDisplayHeight = .42f, FurnitureDisplayHeight = .26f;
        private readonly List<Mesh> ownedMeshes = new List<Mesh>();
        private readonly List<Material> ownedMaterials = new List<Material>();
        private readonly List<Transform> labels = new List<Transform>();
        private readonly Dictionary<int, Renderer> nodes = new Dictionary<int, Renderer>();
        private readonly List<GameObject> routeObjects = new List<GameObject>();
        private readonly List<Mesh> routeMeshes = new List<Mesh>();
        private GameObject previewRoot;
        private Camera previewCamera;
        private RenderTexture texture;
        private SiteMapData map;
        private Material baseMaterial, nodeMaterial, predictionMaterial, targetMaterial, routeMaterial;
        private Font font;
        private bool topView = true;
        public RenderTexture Texture { get { return texture; } }
        public bool IsInitialized { get { return previewRoot != null && texture != null; } }

        public void Initialize(int width = 768, int height = 768)
        {
            width = Mathf.Clamp(width, 128, 2048); height = Mathf.Clamp(height, 128, 2048);
            if (previewRoot == null)
            {
                map = ApartmentMap.Current;
                var shader = Resources.Load<Shader>("Maps/ApartmentMapUnlit");
                if (shader == null) throw new InvalidOperationException("Apartment preview shader resource is missing.");
                baseMaterial = NewMaterial(shader, Color.white);
                nodeMaterial = NewMaterial(shader, Hex("E0EAE4"));
                predictionMaterial = NewMaterial(shader, Hex("3678CF"));
                targetMaterial = NewMaterial(shader, Hex("D89B40"));
                routeMaterial = NewMaterial(shader, Hex("008C72"));
                font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
                previewRoot = new GameObject("Apartment / virtual preview / temporary cutaway heights");
                previewRoot.hideFlags = HideFlags.DontSave;
                // Keep the preview out of the normal UI scene. It never moves an AR origin.
                previewRoot.transform.position = new Vector3(1000, 0, 1000);
                var cameraObject = new GameObject("Apartment Preview Camera", typeof(Camera));
                cameraObject.transform.SetParent(previewRoot.transform, false);
                previewCamera = cameraObject.GetComponent<Camera>();
                previewCamera.enabled = false; // Render only when data/view changes.
                previewCamera.cullingMask = 1 << PreviewLayer;
                previewCamera.clearFlags = CameraClearFlags.SolidColor;
                previewCamera.backgroundColor = Hex("EFF3EE");
                previewCamera.allowHDR = false; previewCamera.allowMSAA = true;
                previewCamera.nearClipPlane = .05f; previewCamera.farClipPlane = 80;
                BuildGeometry();
            }
            if (texture == null || texture.width != width || texture.height != height)
            {
                if (texture != null) { previewCamera.targetTexture = null; texture.Release(); DestroyOwned(texture); }
                texture = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
                texture.name = "CSI Apartment Virtual Map";
                texture.antiAliasing = 2; texture.filterMode = FilterMode.Bilinear;
                texture.useMipMap = false; texture.Create(); previewCamera.targetTexture = texture;
            }
            PositionCamera(); RenderNow();
        }

        public void SetTopView(bool value)
        {
            topView = value;
            if (!IsInitialized) Initialize();
            PositionCamera(); RenderNow();
        }

        public void SetRoute(IList<int> route, int prediction, int target)
        {
            if (!IsInitialized) Initialize();
            foreach (var go in routeObjects) { go.SetActive(false); DestroyOwned(go); }
            routeObjects.Clear();
            foreach (var mesh in routeMeshes) DestroyOwned(mesh);
            routeMeshes.Clear();
            foreach (var entry in nodes)
                entry.Value.sharedMaterial = entry.Key == prediction ? predictionMaterial : entry.Key == target ? targetMaterial : nodeMaterial;
            if (route != null)
            {
                for (int i = 1; i < route.Count; i++)
                {
                    var a = map.FindPoint(route[i - 1]); var b = map.FindPoint(route[i]);
                    if (a == null || b == null || !map.HasEdge(a.label, b.label))
                    {
                        Debug.LogWarning("Apartment preview skipped a route segment that is not a grid edge.");
                        continue;
                    }
                    var mesh = FlatLine(ApartmentMap.ToWorld(a.x_m, a.y_m, .028f), ApartmentMap.ToWorld(b.x_m, b.y_m, .028f), .068f, Color.white);
                    routeMeshes.Add(mesh);
                    routeObjects.Add(MeshObject("Route " + a.label + " to " + b.label, mesh, routeMaterial));
                }
            }
            RenderNow();
        }

        private void BuildGeometry()
        {
            foreach (var region in map.regions)
            {
                if (region.model_role == "excluded_unknown") continue;
                Color floor = region.id == "balcony" ? Hex("DCEADD") : region.id == "office" ? Hex("E9E3DC") :
                    region.id == "kitchen" ? Hex("EEE5D2") : Hex("FAFBF6");
                float x0 = region.min_xy_m[0], z0 = region.min_xy_m[1], x1 = region.max_xy_m[0], z1 = region.max_xy_m[1];
                AddOwnedMesh(region.id + " floor", Quad(ApartmentMap.ToWorld(x0, z0), ApartmentMap.ToWorld(x1, z0),
                    ApartmentMap.ToWorld(x1, z1), ApartmentMap.ToWorld(x0, z1), floor));
                string title = region.id == "balcony" ? "BALCONY" : region.id == "office" ? "OFFICE" : region.id == "kitchen" ? "KITCHEN" :
                    region.id == "corridor_extension" ? "CORRIDOR" : "LIVING ROOM";
                float labelY = region.id == "living_room" ? 2.36f : (z0 + z1) / 2;
                AddLabel(title, ApartmentMap.ToWorld((x0 + x1) / 2, labelY, .065f), .30f, Hex("697F73"));
            }
            foreach (var boundary in map.boundaries)
            {
                if (boundary.kind == "open_connection") continue;
                var a = ApartmentMap.ToWorld(boundary.start_xy_m); var b = ApartmentMap.ToWorld(boundary.end_xy_m);
                if (boundary.kind == "region_boundary_unspecified")
                {
                    AddDashes(boundary.id + " unspecified edge", a + Vector3.up * .008f, b + Vector3.up * .008f, .018f, Hex("A8B1AA"));
                    continue;
                }
                bool furniture = boundary.kind == "furniture_simplified_barrier";
                float height = furniture ? FurnitureDisplayHeight : WallDisplayHeight;
                var color = furniture ? Hex("BE9274") : Hex("9AAFA1");
                AddOwnedMesh(boundary.id + (furniture ? " furniture barrier / display height only" : " wall plane / display height only"),
                    Quad(a, b, b + Vector3.up * height, a + Vector3.up * height, color));
                // Exact boundary at the floor. No box width or collision slab consumes clear space.
                AddOwnedMesh(boundary.id + " boundary line", FlatLine(a + Vector3.up * .007f, b + Vector3.up * .007f, .018f,
                    furniture ? Hex("9A7157") : Hex("627E6C")));
            }
            foreach (var opening in map.openings)
            {
                AddDashes(opening.drawing_id + " opening / not a route", ApartmentMap.ToWorld(opening.start_xy_m, .012f),
                    ApartmentMap.ToWorld(opening.end_xy_m, .012f), .035f, Hex("63A08D"));
                Vector3 pos = ApartmentMap.ToWorld(opening.center_xy_m, .14f);
                if (opening.id == "balcony_opening") pos.x -= .24f; else pos.z += .22f;
                AddLabel(opening.drawing_id, pos, .27f, Hex("377D6C"));
            }
            foreach (var edge in map.grid.edges)
            {
                var a = map.FindPoint(edge.from); var b = map.FindPoint(edge.to);
                AddOwnedMesh("Grid edge " + edge.from + "-" + edge.to, FlatLine(ApartmentMap.ToWorld(a.x_m, a.y_m, .014f),
                    ApartmentMap.ToWorld(b.x_m, b.y_m, .014f), .017f, Hex("CDDDD2")));
            }
            foreach (var point in map.grid.points)
            {
                var mesh = Disc(ApartmentMap.ToWorld(point.x_m, point.y_m, .045f), .145f);
                ownedMeshes.Add(mesh);
                var node = MeshObject("Grid " + point.label, mesh, nodeMaterial);
                nodes[point.label] = node.GetComponent<Renderer>();
                AddLabel(point.label.ToString(), ApartmentMap.ToWorld(point.x_m, point.y_m, .07f), .34f, Hex("163D33"));
            }
        }

        private void PositionCamera()
        {
            if (previewCamera == null) return;
            Rect bounds = map.PlanBounds;
            var center = ApartmentMap.ToWorld(bounds.center.x, bounds.center.y);
            float aspect = texture == null ? 1 : (float)texture.width / texture.height;
            previewCamera.aspect = aspect;
            if (topView)
            {
                previewCamera.orthographic = true;
                previewCamera.orthographicSize = Mathf.Max(bounds.height / 2 + .52f, (bounds.width / 2 + .52f) / aspect);
                previewCamera.transform.localPosition = center + Vector3.up * 16;
                previewCamera.transform.localRotation = Quaternion.Euler(90, 0, 0);
            }
            else
            {
                previewCamera.orthographic = false;
                previewCamera.fieldOfView = 39;
                // Oblique overview, explicitly not a phone/head pose or a first-person AR view.
                float halfVertical = previewCamera.fieldOfView * Mathf.Deg2Rad / 2;
                float halfHorizontal = Mathf.Atan(Mathf.Tan(halfVertical) * aspect);
                Vector3 direction = new Vector3(.2f, .92f, -.74f).normalized;
                Quaternion rotation = Quaternion.LookRotation(-direction, Vector3.up);
                Vector3 right = rotation * Vector3.right, up = rotation * Vector3.up;
                // Fit the actual projected bounds, rather than a large enclosing sphere.
                // This makes the small portrait UI map readable without clipping corners.
                float distance = 0;
                foreach (float x in new[] { bounds.xMin, bounds.xMax })
                    foreach (float y in new[] { bounds.yMin, bounds.yMax })
                        foreach (float h in new[] { 0f, WallDisplayHeight })
                        {
                            Vector3 point = ApartmentMap.ToWorld(x, y, h) - center;
                            float depthOffset = Vector3.Dot(direction, point);
                            distance = Mathf.Max(distance, depthOffset + (Mathf.Abs(Vector3.Dot(right, point)) + .38f) / Mathf.Tan(halfHorizontal));
                            distance = Mathf.Max(distance, depthOffset + (Mathf.Abs(Vector3.Dot(up, point)) + .38f) / Mathf.Tan(halfVertical));
                        }
                Vector3 offset = direction * distance;
                previewCamera.transform.localPosition = center + offset;
                previewCamera.transform.localRotation = rotation;
            }
            foreach (var label in labels) label.rotation = previewCamera.transform.rotation;
        }

        private void RenderNow()
        {
            if (previewCamera == null || texture == null) return;
            // Mobile backgrounding can discard the native texture while the C#
            // object survives. Recreate it before the UI's explicit resume redraw.
            if (!texture.IsCreated() && !texture.Create())
                throw new InvalidOperationException("Apartment preview render texture could not be restored.");
            previewCamera.targetTexture = texture;
            previewCamera.Render();
        }
        private void AddLabel(string value, Vector3 position, float size, Color color)
        {
            var go = new GameObject(value, typeof(TextMesh));
            go.layer = PreviewLayer; go.transform.SetParent(previewRoot.transform, false); go.transform.localPosition = position;
            var text = go.GetComponent<TextMesh>(); text.text = value; text.font = font; text.fontSize = 64;
            // TextMesh geometry scales with both fontSize and characterSize. Keep a
            // sharp 64 px glyph atlas while expressing the requested em size in metres.
            text.characterSize = size * 10f / text.fontSize;
            text.anchor = TextAnchor.MiddleCenter; text.alignment = TextAlignment.Center;
            text.color = color; text.fontStyle = FontStyle.Bold;
            var renderer = go.GetComponent<MeshRenderer>(); renderer.sharedMaterial = font.material;
            renderer.shadowCastingMode = ShadowCastingMode.Off; renderer.receiveShadows = false;
            labels.Add(go.transform);
        }
        private void AddDashes(string name, Vector3 a, Vector3 b, float width, Color color)
        {
            int pieces = Mathf.Max(2, Mathf.CeilToInt(Vector3.Distance(a, b) / .12f));
            for (int i = 0; i < pieces; i += 2)
                AddOwnedMesh(name, FlatLine(Vector3.Lerp(a, b, (float)i / pieces), Vector3.Lerp(a, b, (float)Mathf.Min(i + 1, pieces) / pieces), width, color));
        }
        private void AddOwnedMesh(string name, Mesh mesh) { ownedMeshes.Add(mesh); MeshObject(name, mesh, baseMaterial); }
        private GameObject MeshObject(string name, Mesh mesh, Material material)
        {
            var go = new GameObject(name, typeof(MeshFilter), typeof(MeshRenderer));
            go.layer = PreviewLayer; go.transform.SetParent(previewRoot.transform, false);
            go.GetComponent<MeshFilter>().sharedMesh = mesh;
            var renderer = go.GetComponent<MeshRenderer>(); renderer.sharedMaterial = material;
            renderer.shadowCastingMode = ShadowCastingMode.Off; renderer.receiveShadows = false;
            return go;
        }
        private Material NewMaterial(Shader shader, Color color)
        {
            var material = new Material(shader); material.color = color; ownedMaterials.Add(material); return material;
        }
        private static Mesh Quad(Vector3 a, Vector3 b, Vector3 c, Vector3 d, Color color)
        {
            var mesh = new Mesh(); mesh.vertices = new[] { a, b, c, d };
            mesh.triangles = new[] { 0, 1, 2, 0, 2, 3 }; mesh.colors = new[] { color, color, color, color };
            mesh.RecalculateBounds(); return mesh;
        }
        private static Mesh FlatLine(Vector3 a, Vector3 b, float width, Color color)
        {
            var direction = (b - a).normalized;
            var offset = new Vector3(-direction.z, 0, direction.x) * width / 2;
            return Quad(a - offset, b - offset, b + offset, a + offset, color);
        }
        private static Mesh Disc(Vector3 center, float radius)
        {
            const int count = 32;
            var vertices = new Vector3[count + 1]; var colors = new Color[count + 1]; var triangles = new int[count * 3];
            vertices[0] = center; colors[0] = Color.white;
            for (int i = 0; i < count; i++)
            {
                float angle = 2 * Mathf.PI * i / count;
                vertices[i + 1] = center + new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle)) * radius; colors[i + 1] = Color.white;
                triangles[i * 3] = 0; triangles[i * 3 + 1] = i + 1; triangles[i * 3 + 2] = (i + 1) % count + 1;
            }
            var mesh = new Mesh(); mesh.vertices = vertices; mesh.triangles = triangles; mesh.colors = colors; mesh.RecalculateBounds(); return mesh;
        }
        private static Color Hex(string value) { Color color; ColorUtility.TryParseHtmlString("#" + value, out color); return color; }
        private static void DestroyOwned(UnityEngine.Object value)
        {
            if (value == null) return;
            if (Application.isPlaying) Destroy(value); else DestroyImmediate(value);
        }
        public void Dispose()
        {
            if (previewCamera != null) previewCamera.targetTexture = null;
            if (texture != null) { texture.Release(); DestroyOwned(texture); texture = null; }
            if (previewRoot != null) { previewRoot.SetActive(false); DestroyOwned(previewRoot); previewRoot = null; }
            foreach (var mesh in routeMeshes) DestroyOwned(mesh);
            foreach (var mesh in ownedMeshes) DestroyOwned(mesh);
            foreach (var material in ownedMaterials) DestroyOwned(material);
            routeMeshes.Clear(); routeObjects.Clear(); ownedMeshes.Clear(); ownedMaterials.Clear(); labels.Clear(); nodes.Clear();
            previewCamera = null; map = null;
        }
        private void OnDestroy() { Dispose(); }
    }
}
