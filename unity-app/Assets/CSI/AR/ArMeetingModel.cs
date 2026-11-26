using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace CsiApp
{
    /// <summary>Marker-anchored meeting content. Never changes the AR camera or a CSI estimate.</summary>
    public sealed class ArMeetingModel : MonoBehaviour
    {
        private ApartmentWorldModel grid;
        private DemoRouteVisual routeVisual;
        private GameObject architecture;
        private readonly List<Material> ownedMaterials = new List<Material>();

        public void Initialize(ArMeetingView view, DemoNavigationRoute route, int startLabel)
        {
            var prefab = Resources.Load<GameObject>("Presentation/ApartmentArchitecture");
            if (prefab == null) throw new InvalidOperationException("Meeting architecture prefab is missing. Rebuild the meeting package.");
            architecture = Instantiate(prefab, transform, false);
            architecture.name = "Blender apartment / measured plan / illustrative heights";
            // The existing presentation shader clips by WORLD height, which is unsuitable
            // for an arbitrary AR anchor. Use independent unlit materials in this scene.
            bool fullSize = view == ArMeetingView.FullSizeOverlay;
            var shader = fullSize ? Resources.Load<Shader>("AR/ArMeetingSurface") : Shader.Find("Unlit/Color");
            if (shader == null) throw new InvalidOperationException("AR meeting surface shader is missing.");
            var replacements = new Dictionary<Material, Material>();
            int visibleSurfaces = 0, hiddenSurfaces = 0;
            foreach (var renderer in architecture.GetComponentsInChildren<Renderer>(true))
            {
                // This is a viewing cutaway, not a change to the surveyed map or geometry.
                // Keep the real ceiling and the view through the balcony unobstructed.
                if (fullSize && HideForFullSize(renderer.gameObject.name))
                {
                    renderer.enabled = false;
                    hiddenSurfaces++;
                    continue;
                }
                var source = renderer.sharedMaterials;
                var materials = new Material[source.Length];
                for (int i = 0; i < source.Length; i++)
                {
                    Material material;
                    if (source[i] != null && replacements.TryGetValue(source[i], out material)) materials[i] = material;
                    else
                    {
                        Color tint = source[i] != null ? source[i].color : Color.white;
                        tint.a = fullSize ? .18f : 1f;
                        material = new Material(shader) { color = tint };
                        material.name = fullSize ? "AR meeting translucent / independent" : "AR meeting unlit / independent";
                        ownedMaterials.Add(material); materials[i] = material;
                        if (source[i] != null) replacements[source[i]] = material;
                    }
                }
                renderer.sharedMaterials = materials;
                renderer.shadowCastingMode = ShadowCastingMode.Off; renderer.receiveShadows = false;
                if (renderer.enabled) visibleSurfaces++;
            }
            architecture.SetActive(true);
            Debug.Log("AR_MEETING_ARCHITECTURE: view=" + view + "; visible_surfaces=" + visibleSurfaces
                + "; hidden_surfaces=" + hiddenSurfaces + "; alpha=" + (fullSize ? .18f : 1f)
                + "; site_alignment=false");
            var gridObject = new GameObject("AR map guides / no opaque floor"); gridObject.transform.SetParent(transform, false);
            grid = gridObject.AddComponent<ApartmentWorldModel>();
            grid.wallDisplayHeight = grid.furnitureDisplayHeight = 0;
            grid.showLabels = true; grid.Initialize(false);
            var routeObject = new GameObject("AR route / illustrative connectors"); routeObject.transform.SetParent(transform, false);
            routeVisual = routeObject.AddComponent<DemoRouteVisual>(); routeVisual.floorOffset = .08f;
            SetRoute(route, startLabel);
        }

        private static bool HideForFullSize(string objectName)
        {
            return objectName.StartsWith("Floor_", StringComparison.OrdinalIgnoreCase)
                || objectName.IndexOf("ceiling", StringComparison.OrdinalIgnoreCase) >= 0
                || objectName.IndexOf("roof", StringComparison.OrdinalIgnoreCase) >= 0
                || string.Equals(objectName, "Wall_B01", StringComparison.OrdinalIgnoreCase)
                || string.Equals(objectName, "Wall_B02", StringComparison.OrdinalIgnoreCase)
                || string.Equals(objectName, "Wall_B03", StringComparison.OrdinalIgnoreCase);
        }

        public void SetRoute(DemoNavigationRoute route, int startLabel)
        {
            if (grid == null || routeVisual == null) return;
            grid.SetRoute(null, startLabel, route.ConnectorLabel);
            routeVisual.Build(route, true);
        }

        public void SetDistance(float distance) { if (routeVisual != null) routeVisual.SetDistance(distance); }
        public void FaceLabels(Camera camera) { if (grid != null) grid.SetLabelsFacing(camera); }

        private void OnDestroy()
        {
            foreach (var material in ownedMaterials) if (material != null) Destroy(material);
            ownedMaterials.Clear();
        }
    }
}
