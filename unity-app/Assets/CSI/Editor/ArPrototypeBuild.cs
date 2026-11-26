using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEditor.XR.ARCore;
using UnityEditor.XR.ARSubsystems;
using UnityEditor.XR.Management;
using UnityEditor.XR.Management.Metadata;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.XR.ARSubsystems;
using UnityEngine.XR.Management;

namespace CsiApp.Editor
{
    /// <summary>Separate, opt-in AR calibration prototype alongside the offline Demo.</summary>
    public static class ArPrototypeBuild
    {
        private const string DemoScene = "Assets/CSI/Scenes/UiDemo.unity";
        private const string ArScene = "Assets/CSI/Scenes/ArCalibration.unity";
        private const string XrSettingsPath = "Assets/XR/Settings/XRGeneralSettingsPerBuildTarget.asset";
        private const string ArCoreSettingsPath = "Assets/XR/Settings/ARCoreSettings.asset";
        private const string MarkerPath = "Assets/Resources/AR/CSI_Reference.png";
        private const string LibraryPath = "Assets/Resources/AR/ReferenceLibrary.asset";

        [MenuItem("CSI/Prepare AR Calibration Prototype")]
        public static void Prepare()
        {
            DemoBuild.Validate();
            ConfigureXr();
            PrepareReferenceImage();
            PrepareScene();
            ConfigurePlayer();
            EditorBuildSettings.scenes = new[]
            {
                new EditorBuildSettingsScene(DemoScene, true),
                new EditorBuildSettingsScene(ArScene, true)
            };
            AssetDatabase.SaveAssets();
            Debug.Log("CSI_AR_PREPARE_PASS: AR Optional, manual XR startup, reference library, two scenes.");
        }

        private static void ConfigureXr()
        {
            Directory.CreateDirectory("Assets/XR/Settings");
            AssetDatabase.Refresh();
            EditorBuildSettings.TryGetConfigObject(XRGeneralSettings.k_SettingsKey,
                out XRGeneralSettingsPerBuildTarget settings);
            if (settings == null)
                settings = AssetDatabase.LoadAssetAtPath<XRGeneralSettingsPerBuildTarget>(XrSettingsPath);
            if (settings == null)
            {
                settings = ScriptableObject.CreateInstance<XRGeneralSettingsPerBuildTarget>();
                AssetDatabase.CreateAsset(settings, XrSettingsPath);
            }
            EditorBuildSettings.AddConfigObject(XRGeneralSettings.k_SettingsKey, settings, true);
            if (!settings.HasSettingsForBuildTarget(BuildTargetGroup.Android))
                settings.CreateDefaultSettingsForBuildTarget(BuildTargetGroup.Android);
            if (!settings.HasManagerSettingsForBuildTarget(BuildTargetGroup.Android))
                settings.CreateDefaultManagerSettingsForBuildTarget(BuildTargetGroup.Android);

            var android = settings.SettingsForBuildTarget(BuildTargetGroup.Android);
            android.InitManagerOnStart = false;
            android.Manager.automaticLoading = false;
            android.Manager.automaticRunning = false;
            if (!XRPackageMetadataStore.AssignLoader(android.Manager,
                    "UnityEngine.XR.ARCore.ARCoreLoader", BuildTargetGroup.Android))
                throw new InvalidOperationException("Could not assign the installed ARCore XR loader.");
            EditorUtility.SetDirty(android);
            EditorUtility.SetDirty(android.Manager);
            EditorUtility.SetDirty(settings);

            var arCore = ARCoreSettings.currentSettings;
            if (arCore == null) arCore = AssetDatabase.LoadAssetAtPath<ARCoreSettings>(ArCoreSettingsPath);
            if (arCore == null)
            {
                arCore = ScriptableObject.CreateInstance<ARCoreSettings>();
                AssetDatabase.CreateAsset(arCore, ArCoreSettingsPath);
            }
            arCore.requirement = ARCoreSettings.Requirement.Optional;
            arCore.depth = ARCoreSettings.Requirement.Optional;
            ARCoreSettings.currentSettings = arCore;
            EditorUtility.SetDirty(arCore);
        }

        private static void PrepareReferenceImage()
        {
            var importer = AssetImporter.GetAtPath(MarkerPath) as TextureImporter;
            if (importer == null)
                throw new InvalidOperationException("Missing reference image: " + MarkerPath);
            // Runtime mutable libraries need CPU-readable pixels at the width measured by the user.
            importer.textureType = TextureImporterType.Default;
            importer.isReadable = true;
            importer.mipmapEnabled = false;
            importer.textureCompression = TextureImporterCompression.Uncompressed;
            importer.maxTextureSize = 2048;
            importer.npotScale = TextureImporterNPOTScale.None;
            importer.SaveAndReimport();
            var marker = AssetDatabase.LoadAssetAtPath<Texture2D>(MarkerPath);
            if (marker == null || !marker.isReadable)
                throw new InvalidOperationException("Reference texture could not be imported as readable.");

            var library = AssetDatabase.LoadAssetAtPath<XRReferenceImageLibrary>(LibraryPath);
            if (library == null)
            {
                library = ScriptableObject.CreateInstance<XRReferenceImageLibrary>();
                AssetDatabase.CreateAsset(library, LibraryPath);
            }
            while (library.count > 1) library.RemoveAt(library.count - 1);
            if (library.count == 0) library.Add();
            library.SetName(0, "CSI Reference");
            library.SetTexture(0, marker, true);
            library.SetSpecifySize(0, true);
            // Fallback only. Runtime calibration uses the physical width entered on the phone.
            library.SetSize(0, new Vector2(.20f, .20f * marker.height / marker.width));
            EditorUtility.SetDirty(library);
        }

        private static void PrepareScene()
        {
            var scene = File.Exists(ArScene)
                ? EditorSceneManager.OpenScene(ArScene, OpenSceneMode.Single)
                : EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            if (UnityEngine.Object.FindFirstObjectByType<ArCalibrationController>() == null)
                new GameObject("AR Calibration Controller").AddComponent<ArCalibrationController>();
            EditorSceneManager.SaveScene(scene, ArScene);
        }

        private static void ConfigurePlayer()
        {
            PlayerSettings.companyName = "CSI Capstone";
            PlayerSettings.productName = "CSI Navigation Demo";
            PlayerSettings.bundleVersion = "0.3.1";
            PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.Android, "edu.capstone.csi.demo");
            PlayerSettings.defaultInterfaceOrientation = UIOrientation.Portrait;
            PlayerSettings.allowedAutorotateToLandscapeLeft = false;
            PlayerSettings.allowedAutorotateToLandscapeRight = false;
            PlayerSettings.allowedAutorotateToPortraitUpsideDown = false;
            PlayerSettings.defaultScreenWidth = 390;
            PlayerSettings.defaultScreenHeight = 844;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.runInBackground = true;
            PlayerSettings.SetScriptingBackend(NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel26;
            PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto;
            PlayerSettings.Android.bundleVersionCode = 4;
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
            PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.OpenGLES3 });
            EditorUserBuildSettings.buildAppBundle = false;
            // The official ARCore build processor adds CAMERA and SkipPermissionsDialog.
            // Runtime asks permission only on entry into the AR scene.
            IncludeShader("Unlit/Color");
            IncludeShader("Sprites/Default");
            IncludeShader("UI/Default");
            IncludeShader("Unlit/ARCoreBackground");
            IncludeShader("Unlit/ARCoreBackground/AfterOpaques");
        }

        private static void IncludeShader(string shaderName)
        {
            var shader = Shader.Find(shaderName);
            if (shader == null) throw new InvalidOperationException("Required shader missing: " + shaderName);
            var objects = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/GraphicsSettings.asset");
            if (objects.Length == 0) throw new InvalidOperationException("Graphics settings asset is missing.");
            var serialized = new SerializedObject(objects[0]);
            var list = serialized.FindProperty("m_AlwaysIncludedShaders");
            for (int i = 0; i < list.arraySize; i++)
                if (list.GetArrayElementAtIndex(i).objectReferenceValue == shader) return;
            list.InsertArrayElementAtIndex(list.arraySize);
            list.GetArrayElementAtIndex(list.arraySize - 1).objectReferenceValue = shader;
            serialized.ApplyModifiedPropertiesWithoutUndo();
        }

        public static void GenerateInspectionBatch()
        {
            if (!Application.isBatchMode)
                throw new InvalidOperationException("Use CSI/Generate field inspection model in the interactive Editor.");
            // Batch startup can have an untitled empty scene, which Unity cannot open alongside an additive scene.
            EditorSceneManager.OpenScene(DemoScene, OpenSceneMode.Single);
            ApartmentInspectionBuilder.BuildAll();
        }

        public static void RenderInspectionBatch()
        {
            if (!Application.isBatchMode)
                throw new InvalidOperationException("This rendering helper is for batch QA only.");
            if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.Null)
                throw new InvalidOperationException("Inspection rendering needs a graphics device; omit -nographics.");
            EditorSceneManager.OpenScene(ApartmentInspectionBuilder.ScenePath, OpenSceneMode.Single);
            var camera = UnityEngine.Object.FindAnyObjectByType<Camera>();
            if (camera == null) throw new InvalidOperationException("Inspection camera is missing.");
            var output = Path.GetFullPath(Path.Combine(Application.dataPath,
                "../QA/2026-10-10/model/inspection-overview.png"));
            Directory.CreateDirectory(Path.GetDirectoryName(output));
            var target = new RenderTexture(1600, 1000, 24, RenderTextureFormat.ARGB32);
            var pixels = new Texture2D(1600, 1000, TextureFormat.RGB24, false);
            var previous = RenderTexture.active;
            var previousTarget = camera.targetTexture;
            try
            {
                target.Create(); camera.targetTexture = target; camera.aspect = 1.6f;
                camera.Render(); RenderTexture.active = target;
                pixels.ReadPixels(new Rect(0, 0, 1600, 1000), 0, 0); pixels.Apply();
                File.WriteAllBytes(output, pixels.EncodeToPNG());
                Debug.Log("CSI_INSPECTION_RENDER_PASS: " + output);
            }
            finally
            {
                camera.targetTexture = previousTarget; RenderTexture.active = previous;
                target.Release(); UnityEngine.Object.DestroyImmediate(target);
                UnityEngine.Object.DestroyImmediate(pixels);
            }
        }

        [MenuItem("CSI/Build Android Demo With AR Prototype")]
        public static void BuildAndroid()
        {
            if (!BuildPipeline.IsBuildTargetSupported(BuildTargetGroup.Android, BuildTarget.Android))
                throw new InvalidOperationException("Android Build Support is required.");
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android)
                throw new InvalidOperationException("Launch Unity with -buildTarget Android first.");
            Prepare();
            const string output = "Builds/Android/CSI-Navigation-Demo-v0.3.1.apk";
            Directory.CreateDirectory(Path.GetDirectoryName(output));
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { DemoScene, ArScene },
                target = BuildTarget.Android,
                locationPathName = output,
                options = BuildOptions.Development
            });
            Debug.Log("CSI_AR_BUILD_RESULT: " + report.summary.result + " / " + output +
                      " / bytes=" + report.summary.totalSize + " / errors=" + report.summary.totalErrors);
            if (report.summary.result != BuildResult.Succeeded)
                throw new InvalidOperationException("AR prototype build failed; inspect the Unity build log.");
        }

        [MenuItem("CSI/Build Android Positioning")]
        public static void BuildAndroidWithModel()
        {
            if (!BuildPipeline.IsBuildTargetSupported(BuildTargetGroup.Android, BuildTarget.Android)
                || EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android)
                throw new InvalidOperationException("Launch the Android-supported Editor with -buildTarget Android first.");
            Prepare();
            MeetingDemoBuild.PrepareArchitectureResource();
            ApartmentMobilePresentationBuilder.Prepare();
            const string modelScene = "Assets/CSI/Scenes/ApartmentMobilePresentation.unity";
            const string positioningScene = "Assets/CSI/Scenes/LivePositioning.unity";
            var liveScene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            new GameObject("Live Positioning").AddComponent<LivePositioningController>();
            EditorSceneManager.SaveScene(liveScene, positioningScene);
            var scenes = new[] { DemoScene, positioningScene, ArScene, modelScene };
            EditorBuildSettings.scenes = Array.ConvertAll(scenes, path => new EditorBuildSettingsScene(path, true));
            MeetingDemoBuild.ValidateDemoRoutes();
            IntegrationStateChecks.RunUnity();
            LiveNavigationChecks.RunUnity();
            PositioningSessionChecks.RunUnity();
            RealReplayChecks.RunUnity();
            FrontendV080Checks.RunUnity();
            FrontendV090Checks.RunUnity();
            PlayerSettings.bundleVersion = "0.9.0";
            PlayerSettings.Android.bundleVersionCode = 14;
            PlayerSettings.Android.forceInternetPermission = true;
            PlayerSettings.insecureHttpOption = InsecureHttpOption.AlwaysAllowed;
            AssetDatabase.SaveAssets();
            const string output = "Builds/Android/CSI-Positioning-v0.9.0.apk";
            Directory.CreateDirectory(Path.GetDirectoryName(output));
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = scenes, target = BuildTarget.Android, locationPathName = output,
                options = BuildOptions.None
            });
            Debug.Log("CSI_MODEL_ANDROID_BUILD_RESULT: " + report.summary.result + " / " + output +
                " / bytes=" + report.summary.totalSize + " / errors=" + report.summary.totalErrors);
            if (report.summary.result != BuildResult.Succeeded)
                throw new InvalidOperationException("Android model presentation build failed; inspect the Unity build log.");
        }
    }
}
