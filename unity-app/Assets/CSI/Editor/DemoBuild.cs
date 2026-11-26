using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

namespace CsiApp.Editor
{
    /// <summary>Reproducible offline Demo builds. No AR or live CSI is implied.</summary>
    public static class DemoBuild
    {
        private const string ScenePath = "Assets/CSI/Scenes/UiDemo.unity";

        public static void Validate()
        {
            var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            if (!scene.IsValid() || UnityEngine.Object.FindFirstObjectByType<CsiAppUi>() == null)
                throw new InvalidOperationException("Demo scene must contain CsiAppUi.");
            TextAsset map = Resources.Load<TextAsset>("Maps/apartment_v1_2026-10-10");
            if (map == null) throw new InvalidOperationException("Confirmed apartment map is missing.");
            var data = ApartmentMap.Current;
            if (data.grid.columns != GridMap.Columns || data.grid.rows != GridMap.Rows ||
                Math.Abs(data.grid.spacing_m - GridMap.Spacing) > .0001)
                throw new InvalidOperationException("Map and navigation grid configuration disagree.");
            for (int label = 0; label < 15; label++)
            {
                var point = data.FindPoint(label);
                if (point == null || point.i != label % GridMap.Columns || point.j != label / GridMap.Columns ||
                    Math.Abs(point.x_m - point.i * GridMap.Spacing) > .0001 ||
                    Math.Abs(point.y_m - point.j * GridMap.Spacing) > .0001)
                    throw new InvalidOperationException("Missing or inconsistent grid point " + label);
            }
            for (int start = 0; start < 15; start++)
                for (int target = 0; target < 15; target++)
                {
                    var route = GridMap.Route(start, target);
                    int hops = Math.Abs(start % 5 - target % 5) + Math.Abs(start / 5 - target / 5);
                    if (route.Count != hops + 1 || route[0] != start || route[route.Count - 1] != target)
                        throw new InvalidOperationException("Invalid grid route.");
                    for (int index = 1; index < route.Count; index++)
                        if (!data.HasEdge(route[index - 1], route[index]))
                            throw new InvalidOperationException("Route crosses a missing map edge.");
                }
            AssetDatabase.SaveAssets();
            Debug.Log("CSI_VALIDATE_PASS: scene, confirmed map, 15 nodes, 225 grid routes.");
        }

        private static void Configure()
        {
            PlayerSettings.companyName = "CSI Capstone";
            PlayerSettings.productName = "CSI Navigation Demo";
            PlayerSettings.bundleVersion = "0.2.0";
            PlayerSettings.defaultInterfaceOrientation = UIOrientation.Portrait;
            PlayerSettings.allowedAutorotateToLandscapeLeft = false;
            PlayerSettings.allowedAutorotateToLandscapeRight = false;
            PlayerSettings.allowedAutorotateToPortraitUpsideDown = false;
            PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.Android, "edu.capstone.csi.demo");
            PlayerSettings.defaultScreenWidth = 390;
            PlayerSettings.defaultScreenHeight = 844;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.runInBackground = true;
            // These shaders are created at runtime by the parameterised scene.
            IncludeShader("Unlit/Color");
            IncludeShader("Sprites/Default");
            // LegacyRuntime.ttf supplies its runtime font material. Its editor-only
            // GUI/Text Shader has DontSave flags and must not be serialized here.
            IncludeShader("UI/Default");
        }

        private static void IncludeShader(string name)
        {
            Shader shader = Shader.Find(name);
            if (shader == null) throw new InvalidOperationException("Required shader missing: " + name);
            var assets = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/GraphicsSettings.asset");
            if (assets.Length == 0) return;
            var settings = new SerializedObject(assets[0]);
            var list = settings.FindProperty("m_AlwaysIncludedShaders");
            for (int i = 0; i < list.arraySize; i++)
                if (list.GetArrayElementAtIndex(i).objectReferenceValue == shader) return;
            list.InsertArrayElementAtIndex(list.arraySize);
            list.GetArrayElementAtIndex(list.arraySize - 1).objectReferenceValue = shader;
            settings.ApplyModifiedPropertiesWithoutUndo();
        }

        [MenuItem("CSI/Build Android Demo APK")]
        public static void BuildAndroid()
        {
            if (!BuildPipeline.IsBuildTargetSupported(BuildTargetGroup.Android, BuildTarget.Android))
                throw new InvalidOperationException("Install Android Build Support for this Editor before building.");
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android)
                throw new InvalidOperationException("Select Android first, or launch with -buildTarget Android.");
            Validate(); Configure();
            PlayerSettings.SetScriptingBackend(NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel26;
            PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto;
            PlayerSettings.Android.bundleVersionCode = 2;
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
            PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.OpenGLES3 });
            EditorUserBuildSettings.buildAppBundle = false;
            Build(BuildTarget.Android, "Builds/Android/CSI-Navigation-Demo-v0.2.apk");
        }

        [MenuItem("CSI/Build Windows Demo")]
        public static void BuildWindows()
        {
            Validate(); Configure();
            PlayerSettings.SetScriptingBackend(NamedBuildTarget.Standalone, ScriptingImplementation.Mono2x);
            Build(BuildTarget.StandaloneWindows64, "Builds/Windows/CSI-Navigation-Demo.exe");
        }

        private static void Build(BuildTarget target, string output)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(output));
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { ScenePath }, target = target, locationPathName = output,
                options = BuildOptions.Development
            });
            Debug.Log("CSI_BUILD_RESULT: " + report.summary.result + " / " + output +
                      " / bytes=" + report.summary.totalSize + " / errors=" + report.summary.totalErrors);
            if (report.summary.result != BuildResult.Succeeded)
                throw new InvalidOperationException("Demo build failed; inspect the Unity build log.");
        }
    }
}
