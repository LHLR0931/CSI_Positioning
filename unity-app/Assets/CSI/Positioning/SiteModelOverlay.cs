using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace CsiApp
{
    // Attach to an identity child of the measured SiteArTracker.MapRoot.
    // The caller gates alignment/visibility. This component only renders existing geometry.
    public sealed class SiteModelOverlay : MonoBehaviour
    {
        public string Status { get; private set; } = "Model overlay not initialized.";
        private GameObject architecture;
        private readonly List<Material> ownedMaterials = new List<Material>();

        public void Initialize()
        {
            if (architecture != null) return;
            try
            {
                var prefab = Resources.Load<GameObject>("Presentation/ApartmentArchitecture");
                if (prefab == null) throw new InvalidOperationException("Apartment architecture resource is missing.");
                var shader = Resources.Load<Shader>("AR/ArMeetingSurface");
                if (shader == null) throw new InvalidOperationException("Transparent AR surface shader is missing.");

                architecture = Instantiate(prefab, transform, false);
                architecture.name = "Site apartment / confirmed plan / illustrative heights";
                architecture.SetActive(false);
                // The resource already contains the Blender-to-map correction. Do not apply it twice.
                architecture.transform.localPosition = Vector3.zero;
                architecture.transform.localRotation = Quaternion.identity;
                architecture.transform.localScale = Vector3.one;

                foreach (var collider in architecture.GetComponentsInChildren<Collider>(true)) collider.enabled = false;
                var replacements = new Dictionary<Material, Material>();
                int visible = 0, hidden = 0;
                foreach (var renderer in architecture.GetComponentsInChildren<Renderer>(true))
                {
                    string objectName = renderer.gameObject.name;
                    if (objectName.StartsWith("Floor", StringComparison.OrdinalIgnoreCase)
                        || objectName.IndexOf("ceiling", StringComparison.OrdinalIgnoreCase) >= 0
                        || objectName.IndexOf("roof", StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        renderer.enabled = false; hidden++; continue;
                    }
                    var originals = renderer.sharedMaterials;
                    var materials = new Material[originals.Length];
                    for (int i = 0; i < originals.Length; i++)
                    {
                        Material material;
                        if (originals[i] != null && replacements.TryGetValue(originals[i], out material))
                        { materials[i] = material; continue; }
                        // Dark, legible survey surfaces against the camera image. The FBX
                        // has two-sided faces, so moderate alpha already gives strong walls.
                        Color tint = objectName.StartsWith("Furniture_", StringComparison.OrdinalIgnoreCase)
                            ? new Color(.20f, .13f, .08f, .62f) : new Color(.055f, .10f, .13f, .62f);
                        material = new Material(shader) { name = "Site overlay / independent translucent surface", color = tint };
                        ownedMaterials.Add(material); materials[i] = material;
                        if (originals[i] != null) replacements[originals[i]] = material;
                    }
                    renderer.sharedMaterials = materials;
                    renderer.shadowCastingMode = ShadowCastingMode.Off;
                    renderer.receiveShadows = false;
                    renderer.enabled = true; visible++;
                }
                AddWallBaseOutlines(shader);
                architecture.SetActive(true);
                Status = "1:1 dark model; outlined wall bases. Heights are illustrative.";
                Debug.Log("CSI_SITE_MODEL_OVERLAY: scale=1; visible_surfaces=" + visible + "; hidden_surfaces=" + hidden
                    + "; wall_B01_B03_retained=true; measured_heights=false");
            }
            catch (Exception error)
            {
                Release();
                Status = "Model overlay unavailable: " + error.Message;
                Debug.LogError(Status);
            }
        }

        private void AddWallBaseOutlines(Shader shader)
        {
            var material = new Material(shader) { name = "Site wall bases / independent outline",
                color = new Color(.01f, .018f, .025f, .98f), renderQueue = 3001 };
            ownedMaterials.Add(material);
            foreach (var boundary in ApartmentMap.Current.boundaries)
            {
                if (boundary.kind == "open_connection" || boundary.kind == "region_boundary_unspecified") continue;
                var lineObject = new GameObject("Wall base / " + boundary.id, typeof(LineRenderer));
                lineObject.transform.SetParent(architecture.transform, false);
                var line = lineObject.GetComponent<LineRenderer>();
                line.useWorldSpace = false;
                line.positionCount = 2;
                // Exact existing map endpoints at floor height: visual strokes only,
                // without changing the source meshes or registered map transform.
                line.SetPosition(0, ApartmentMap.ToWorld(boundary.start_xy_m));
                line.SetPosition(1, ApartmentMap.ToWorld(boundary.end_xy_m));
                line.widthMultiplier = .018f;
                line.numCapVertices = 2;
                line.sharedMaterial = material;
                line.shadowCastingMode = ShadowCastingMode.Off;
                line.receiveShadows = false;
            }
        }

        private void Release()
        {
            if (architecture != null) { architecture.SetActive(false); Destroy(architecture); architecture = null; }
            foreach (var material in ownedMaterials) if (material != null) Destroy(material);
            ownedMaterials.Clear();
        }
        private void OnDestroy() { Release(); }
    }
}
