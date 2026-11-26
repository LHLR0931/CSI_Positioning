using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace CsiApp.Editor
{
    /// <summary>Copy the validated desktop model into an independent portrait touch scene.</summary>
    public static class ApartmentMobilePresentationBuilder
    {
        public const string ScenePath = "Assets/CSI/Scenes/ApartmentMobilePresentation.unity";
        [MenuItem("CSI/Presentation/Prepare mobile model scene")]
        public static void Prepare()
        {
            if (!File.Exists(ApartmentPresentationBuilder.ScenePath))
                throw new InvalidOperationException("Generate and verify the desktop presentation scene before preparing the mobile copy.");
            var existing = SceneManager.GetSceneByPath(ScenePath);
            if (existing.IsValid() && existing.isLoaded && existing.isDirty)
                throw new InvalidOperationException("Save or close the mobile presentation scene before regenerating it.");
            if (existing.IsValid() && existing.isLoaded) EditorSceneManager.CloseScene(existing, true);
            ApartmentInspectionBuilder.ValidateGeometry(ApartmentMap.Current);
            // Copy scene serialization only. Unity gives this separate path its own .meta GUID.
            File.Copy(ApartmentPresentationBuilder.ScenePath, ScenePath, true);
            AssetDatabase.ImportAsset(ScenePath, ImportAssetOptions.ForceSynchronousImport);
            var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Additive);
            SceneManager.SetActiveScene(scene);
            ApartmentPresentationController controller = null;
            foreach (var root in scene.GetRootGameObjects())
            {
                controller = root.GetComponentInChildren<ApartmentPresentationController>();
                if (controller != null) break;
            }
            if (controller == null) throw new InvalidOperationException("Desktop source scene has no presentation controller.");
            controller.mobileLayout = true;
            controller.gameObject.name = "MOBILE MODEL / offline simulation / no live CSI or AR";
            controller.viewCamera.ResetAspect(); controller.interfaceCamera.ResetAspect();

            // Runtime cutaway changes in the phone scene must not change desktop presentation materials.
            string folder = "Assets/CSI/Generated/ApartmentMobilePresentation/run_" + DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff");
            Directory.CreateDirectory(folder); AssetDatabase.Refresh();
            var copies = new Dictionary<Material, Material>(); var materials = new List<Material>();
            foreach (var renderer in controller.architecture.GetComponentsInChildren<MeshRenderer>(true))
            {
                var old = renderer.sharedMaterials; var replaced = new Material[old.Length];
                for (int i = 0; i < old.Length; i++)
                {
                    if (old[i] == null) continue;
                    Material copy;
                    if (!copies.TryGetValue(old[i], out copy))
                    {
                        copy = new Material(old[i]); copy.name = old[i].name + " mobile";
                        AssetDatabase.CreateAsset(copy, folder + "/surface_" + materials.Count.ToString("D3") + ".mat");
                        copies.Add(old[i], copy); materials.Add(copy);
                    }
                    replaced[i] = copy;
                }
                renderer.sharedMaterials = replaced;
            }
            controller.architectureMaterials = materials.ToArray();
            EditorUtility.SetDirty(controller); AssetDatabase.SaveAssets();
            if (!EditorSceneManager.SaveScene(scene, ScenePath)) throw new InvalidOperationException("Could not save mobile presentation scene.");
            Debug.Log("CSI_MOBILE_MODEL_PREPARE_PASS: " + ScenePath + " / portrait touch UI / separate materials / original desktop and AR scene lists unchanged.");
        }
    }
}
