#if DEVELOPMENT_BUILD || UNITY_EDITOR
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using UnityEngine;
using UnityEngine.UI;

namespace CsiApp
{
    // Opt-in player QA: inactive during normal use; exercises actual UI callbacks.
    public sealed class DemoSmokeRunner : MonoBehaviour
    {
        private string output;
        private NavigationState state;
        private readonly List<string> errors = new List<string>();

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Launch()
        {
            var args = Environment.GetCommandLineArgs();
            int flag = Array.IndexOf(args, "--csi-smoke-output");
            if (flag < 0 || flag + 1 >= args.Length) return;
            var runner = new GameObject("Opt-in Demo QA").AddComponent<DemoSmokeRunner>();
            runner.output = Path.GetFullPath(args[flag + 1]);
            Directory.CreateDirectory(runner.output);
            runner.StartCoroutine(runner.GuardedRun());
        }

        private IEnumerator GuardedRun()
        {
            Application.logMessageReceived += OnLog;
            var steps = Exercise();
            while (true)
            {
                object current = null;
                bool more = false;
                try { more = steps.MoveNext(); if (more) current = steps.Current; }
                catch (Exception e) { errors.Add(e.ToString()); }
                if (!more || errors.Count > 0) break;
                yield return current;
            }
            File.WriteAllText(Path.Combine(output, "smoke-result.txt"), errors.Count == 0
                ? "PASS: UI navigation, measured model rendering, target input, demo route, pause/recovery, arrival, independent ground truth, recording. Screenshots are virtual DEMO data."
                : "FAIL\n" + string.Join("\n", errors.ToArray()));
            Debug.Log(errors.Count == 0 ? "CSI_SMOKE_PASS" : "CSI_SMOKE_FAIL");
            Application.logMessageReceived -= OnLog;
            Application.Quit(errors.Count == 0 ? 0 : 1);
        }

        private IEnumerator Exercise()
        {
            yield return new WaitForSecondsRealtime(1);
            var ui = UnityEngine.Object.FindFirstObjectByType<CsiAppUi>();
            Require(ui != null, "UI component missing");
            state = (NavigationState)typeof(CsiAppUi).GetField("state", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(ui);
            Capture("01-home"); yield return new WaitForSecondsRealtime(.4f);
            Click("Settings"); yield return null;
            var address = ui.GetComponentInChildren<InputField>(); address.text = "invalid-address";
            Click("Save address"); yield return null;
            Require(address.text == "invalid-address" && ContainsText("Enter a valid HTTP"), "Invalid URL must retain its draft and show feedback");
            Click("X"); yield return null;
            Click("Explore navigation  >"); yield return new WaitForSecondsRealtime(.4f);
            Require(UnityEngine.Object.FindFirstObjectByType<RawImage>() != null, "Measured apartment preview missing");
            var apartment = UnityEngine.Object.FindFirstObjectByType<ApartmentSceneView>();
            apartment.Texture.Release(); // Simulate loss of the mobile GPU resource.
            apartment.SetTopView(true);
            Require(apartment.Texture.IsCreated(), "Lost map texture must be recreated before rendering");
            Capture("02-top-map"); yield return new WaitForSecondsRealtime(.4f);
            Click("3D view"); yield return new WaitForSecondsRealtime(.4f);
            Capture("03-3d-map"); yield return new WaitForSecondsRealtime(.4f);
            Click("Top view"); yield return null;
            Click("Edit"); yield return null;
            Require(!FindButton("Balcony Entrance / Unavailable").interactable, "Unconfigured balcony must stay disabled");
            var inputs = ui.GetComponentsInChildren<InputField>();
            Require(inputs.Length == 2, "Grid input fields missing");
            inputs[0].text = "5"; inputs[1].text = "0";
            Click("Preview grid route"); yield return null;
            Require(ContainsText("Out of range."), "Invalid grid coordinate must be rejected");
            inputs = ui.GetComponentsInChildren<InputField>();
            Require(inputs[0].text == "5" && inputs[1].text == "0", "Invalid grid draft must be retained");
            Capture("04-destination-drawer"); yield return new WaitForSecondsRealtime(.4f);
            inputs[0].text = "0"; inputs[1].text = "0";
            Require(!ContainsText("Out of range."), "Editing the draft should clear stale validation feedback");
            Click("Preview grid route"); yield return null;
            Require(state.Target == 0 && state.Path.Count == 5, "Expected four-neighbour route from 12 to 0");
            Click("Start grid demo"); yield return null;
            Capture("05-active-route"); yield return new WaitForSecondsRealtime(.4f);
            Click("Next demo step"); yield return null;
            Require(state.Prediction == 11, "Demo should advance along a grid edge");
            Click("Pause"); yield return null;
            Click("Next demo step"); yield return null;
            Require(state.Paused && state.Prediction == 11, "Paused route must not advance");
            Click("Resume"); yield return null;
            Click("Route details & demo controls   ^"); yield return null;
            Click("Simulate tracking loss"); yield return null;
            Require(state.TrackingLostDemo && state.Paused, "Tracking loss must pause");
            Click("Restore tracking demo"); yield return null;
            Require(state.Paused, "Restore must require explicit resume");
            Capture("06-route-drawer"); yield return new WaitForSecondsRealtime(.4f);
            Click("X"); yield return null;
            Click("Resume"); yield return null;
            for (int n = 0; n < 3; n++) { Click("Next demo step"); yield return null; }
            Require(state.Prediction == 0 && state.Navigating, "Arrival must remain manually confirmed");
            Click("End"); yield return null;
            Capture("07-arrival-drawer"); yield return new WaitForSecondsRealtime(.4f);
            Click("Confirm demo arrival / end"); yield return null;
            Require(!state.Navigating, "End navigation failed");
            Click("<"); yield return null;
            Click("Open positioning lab  >"); yield return null;
            Click("More"); yield return null;
            Click("(0, 0)"); yield return null;
            Click("X"); yield return null;
            Click("Start recording"); yield return null;
            Click("Next demo sample"); yield return null;
            Require(state.RecordCount == 1 && Math.Abs(state.MeanError - .8) < .00001, "Independent ground truth error calculation failed");
            Capture("08-developer"); yield return new WaitForSecondsRealtime(.4f);
            Click("Stop recording"); yield return null;
            Click("Test controls & results   ^"); yield return null;
            Click("Results & CSV export"); yield return null;
            Capture("09-results"); yield return new WaitForSecondsRealtime(.4f);
            File.WriteAllText(Path.Combine(output, "demo-recording.csv"), state.Csv());
        }

        private static void Require(bool condition, string message) { if (!condition) throw new InvalidOperationException(message); }
        private static Button FindButton(string name)
        {
            foreach (var button in UnityEngine.Object.FindObjectsByType<Button>(FindObjectsSortMode.None))
                if (button.gameObject.activeInHierarchy && button.name == name) return button;
            throw new InvalidOperationException("Missing button: " + name);
        }
        private static void Click(string name)
        {
            var button = FindButton(name);
            Require(button.interactable, "Disabled button: " + name);
            button.onClick.Invoke();
        }
        private static bool ContainsText(string value)
        {
            foreach (var label in UnityEngine.Object.FindObjectsByType<Text>(FindObjectsSortMode.None))
                if (label.gameObject.activeInHierarchy && label.text.Contains(value)) return true;
            return false;
        }
        private void Capture(string name)
        {
            // A hidden Windows player has no presented backbuffer. Render the real
            // Canvas into a texture instead of accepting a black desktop screenshot.
            Canvas canvas = UnityEngine.Object.FindFirstObjectByType<Canvas>();
            var cameraObject = new GameObject("QA capture camera", typeof(Camera));
            var camera = cameraObject.GetComponent<Camera>();
            camera.enabled = false; camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = Color.white; camera.cullingMask = 1;
            camera.transform.position = new Vector3(0, 0, -10);
            var target = new RenderTexture(390, 844, 24);
            target.Create(); camera.targetTexture = target;
            canvas.renderMode = RenderMode.ScreenSpaceCamera;
            canvas.worldCamera = camera; canvas.planeDistance = 1;
            Canvas.ForceUpdateCanvases(); camera.Render();
            SaveTexture(target, name);
            canvas.renderMode = RenderMode.ScreenSpaceOverlay; canvas.worldCamera = null;
            Canvas.ForceUpdateCanvases();
            camera.targetTexture = null; target.Release(); Destroy(target); Destroy(cameraObject);
            if (name == "02-top-map" || name == "03-3d-map")
            {
                var map = UnityEngine.Object.FindFirstObjectByType<RawImage>();
                SaveTexture((RenderTexture)map.texture, name + "-detail");
            }
        }
        private void SaveTexture(RenderTexture target, string name)
        {
            var previous = RenderTexture.active; RenderTexture.active = target;
            var pixels = new Texture2D(target.width, target.height, TextureFormat.RGB24, false);
            pixels.ReadPixels(new Rect(0, 0, target.width, target.height), 0, 0); pixels.Apply();
            RenderTexture.active = previous;
            var data = pixels.GetPixels32(); var first = data[0]; int varied = 0;
            for (int n = 0; n < data.Length; n++)
                if (Math.Abs(data[n].r - first.r) + Math.Abs(data[n].g - first.g) + Math.Abs(data[n].b - first.b) > 30) varied++;
            Require(varied > data.Length / 50, "Blank render: " + name);
            File.WriteAllBytes(Path.Combine(output, name + ".png"), pixels.EncodeToPNG());
            Destroy(pixels);
        }
        private void OnLog(string message, string trace, LogType type)
        {
            if (type == LogType.Error || type == LogType.Exception || type == LogType.Assert) errors.Add(message + "\n" + trace);
        }
    }
}
#endif
