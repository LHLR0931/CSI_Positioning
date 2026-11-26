using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace CsiApp
{
    /// <summary>Opt-in standalone integration checks; never runs during ordinary use.</summary>
    public sealed class ApartmentPresentationSmoke : MonoBehaviour
    {
        private string output;
        private readonly List<string> errors = new List<string>();
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Launch()
        {
            var arguments = Environment.GetCommandLineArgs(); int index = Array.IndexOf(arguments, "--presentation-smoke-output");
            if (index < 0 || index + 1 >= arguments.Length) return;
            var runner = new GameObject("Opt-in presentation integration QA").AddComponent<ApartmentPresentationSmoke>();
            runner.output = Path.GetFullPath(arguments[index + 1]); Directory.CreateDirectory(runner.output);
            runner.StartCoroutine(runner.Run());
        }
        private IEnumerator Run()
        {
            Application.logMessageReceived += OnLog;
            var sequence = Exercise();
            while (true)
            {
                bool more = false; object current = null;
                try { more = sequence.MoveNext(); if (more) current = sequence.Current; }
                catch (Exception e) { errors.Add(e.ToString()); }
                if (!more || errors.Count > 0) break;
                yield return current;
            }
            File.WriteAllText(Path.Combine(output, "smoke-result.txt"), errors.Count == 0
                ? "PASS: single isolated scene; 15 nodes; perspective/top-view UI callbacks; route playback reaches selected grid; route replacement removes old segments; Space Submit disabled; no runtime exception. SIMULATION ONLY."
                : "FAIL\n" + string.Join("\n", errors.ToArray()));
            Debug.Log(errors.Count == 0 ? "CSI_PRESENTATION_SMOKE_PASS" : "CSI_PRESENTATION_SMOKE_FAIL");
            Application.logMessageReceived -= OnLog;
            Application.Quit(errors.Count == 0 ? 0 : 1);
        }
        private IEnumerator Exercise()
        {
            yield return new WaitForSecondsRealtime(1);
            var controller = FindAnyObjectByType<ApartmentPresentationController>();
            Require(controller != null, "Presentation controller missing");
            Require(SceneManager.sceneCount == 1 && SceneManager.GetActiveScene().name == "ApartmentPresentation", "Other scenes loaded");
            Require(FindAnyObjectByType<CsiAppUi>() == null, "Original mobile UI must not be running");
            Require(Count(controller.grid.GeometryRoot, "Grid node ") == 15, "Expected 15 grid nodes");
            Require(EventSystem.current != null && !EventSystem.current.sendNavigationEvents, "Space must not submit a selected uGUI button");
            Capture("player-overview.png"); yield return new WaitForSecondsRealtime(.25f);
            Click(controller, "2  TOP VIEW"); yield return null;
            Require(controller.Mode == ApartmentPresentationController.ViewMode.Plan && controller.viewCamera.orthographic, "Top view button failed");
            Capture("player-route-plan.png"); yield return new WaitForSecondsRealtime(.25f);
            Click(controller, "3  FIRST PERSON"); yield return null;
            Require(!controller.viewCamera.orthographic, "First person must be perspective");
            Capture("player-first-person.png"); yield return new WaitForSecondsRealtime(.25f);
            Click(controller, "PLAY / PAUSE"); yield return new WaitForSecondsRealtime(3.8f);
            Require(controller.IsPlaying, "Playback ended too early"); Capture("player-route-turn.png");
            Click(controller, "PLAY / PAUSE"); var paused = controller.walker.position;
            yield return new WaitForSecondsRealtime(.3f);
            Require(!controller.IsPlaying && Vector3.Distance(paused, controller.walker.position) < .0001f, "Paused walker continued moving");
            Click(controller, "PLAY / PAUSE");
            yield return new WaitForSecondsRealtime(5.6f);
            Require(!controller.IsPlaying, "Playback did not complete");
            Require(Vector3.Distance(controller.walker.position, new Vector3(3.2f, .14f, 1.6f)) < .001f, "Playback did not reach grid 14");
            controller.ConfigureRoute(14, 13); yield return null; yield return null;
            Require(Count(controller.grid.GeometryRoot, "Route ") == 1, "Old route geometry remained after changing target");
            Click(controller, "RESET ROUTE"); yield return null;
            Require(!controller.IsPlaying && Vector3.Distance(controller.walker.position, new Vector3(3.2f, .14f, 1.6f)) < .001f, "Reset did not restore the selected start");
            Click(controller, "1  OVERVIEW"); yield return null;
            Require(controller.viewCamera.orthographic, "Overview did not restore orthographic view");
        }
        private static int Count(Transform root, string prefix)
        { int count = 0; foreach (Transform item in root) if (item.gameObject.activeSelf && item.name.StartsWith(prefix)) count++; return count; }
        private static void Click(ApartmentPresentationController controller, string title)
        {
            foreach (var button in controller.GetComponentsInChildren<Button>())
                if (button.name == "Button / " + title) { button.onClick.Invoke(); return; }
            throw new InvalidOperationException("Missing button " + title);
        }
        private void Capture(string name)
        {
            // A hidden desktop window can expose an all-black swapchain. Render the
            // actual running scene and its UI cameras explicitly, independent of window visibility.
            var controller = FindAnyObjectByType<ApartmentPresentationController>();
            var world = controller.viewCamera; var ui = controller.interfaceCamera;
            var oldWorld = world.targetTexture; var oldUi = ui.targetTexture; var active = RenderTexture.active;
            var target = new RenderTexture(1600, 1000, 24, RenderTextureFormat.ARGB32) { antiAliasing = 4 };
            var pixels = new Texture2D(1600, 1000, TextureFormat.RGB24, false);
            try
            {
                target.Create(); world.targetTexture = target; ui.targetTexture = target;
                Canvas.ForceUpdateCanvases(); world.Render(); ui.Render(); RenderTexture.active = target;
                pixels.ReadPixels(new Rect(0, 0, 1600, 1000), 0, 0); pixels.Apply();
                var colors = pixels.GetPixels32(); var unique = new HashSet<int>();
                for (int i = 0; i < colors.Length; i += 301)
                    unique.Add((colors[i].r << 16) | (colors[i].g << 8) | colors[i].b);
                Require(unique.Count > 5, "Player render produced a blank image: " + name);
                File.WriteAllBytes(Path.Combine(output, name), pixels.EncodeToPNG());
            }
            finally
            {
                world.targetTexture = oldWorld; ui.targetTexture = oldUi; RenderTexture.active = active;
                target.Release(); Destroy(target); Destroy(pixels);
            }
        }
        private static void Require(bool value, string message) { if (!value) throw new InvalidOperationException(message); }
        private void OnLog(string condition, string stack, LogType type)
        { if (type == LogType.Exception || type == LogType.Error || type == LogType.Assert) errors.Add(condition + "\n" + stack); }
    }
}
