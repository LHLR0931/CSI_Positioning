using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace CsiApp.Editor
{
    public static class UiSkeletonBuilder
    {
        [MenuItem("CSI/Create UI Demo Scene")]
        public static void CreateScene()
        {
            // A new scene is created additively so existing unsaved work is never discarded.
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Additive);
            UnityEngine.SceneManagement.SceneManager.SetActiveScene(scene);
            var root = new GameObject("CSI App UI");
            root.AddComponent<CsiAppUi>();
            Directory.CreateDirectory("Assets/CSI/Scenes");
            string path = AssetDatabase.GenerateUniqueAssetPath("Assets/CSI/Scenes/UiDemo.unity");
            EditorSceneManager.SaveScene(scene, path);
            Selection.activeGameObject = root;
            Debug.Log("CSI UI demo saved to " + path + ". Press Play. All data / AR states are simulated.");
        }

        [MenuItem("CSI/Configure Portrait Demo Build")]
        public static void ConfigureBuild()
        {
            PlayerSettings.companyName = "CSI Capstone";
            PlayerSettings.productName = "CSI Indoor Navigation Demo";
            PlayerSettings.defaultInterfaceOrientation = UIOrientation.Portrait;
            PlayerSettings.SetApplicationIdentifier(UnityEditor.Build.NamedBuildTarget.Android, "edu.capstone.csi.demo");
            var scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
            if (string.IsNullOrEmpty(scene.path)) { Debug.LogWarning("Save the CSI demo scene first."); return; }
            // Append; do not erase existing build scenes.
            var scenes = new System.Collections.Generic.List<EditorBuildSettingsScene>(EditorBuildSettings.scenes);
            if (!scenes.Exists(s => s.path == scene.path)) scenes.Add(new EditorBuildSettingsScene(scene.path, true));
            EditorBuildSettings.scenes = scenes.ToArray();
            Debug.Log("Portrait and application ID set. Install Android modules and verify build settings before building. No AR provider is installed by this UI skeleton.");
        }
    }
}
