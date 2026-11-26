using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.UI;

namespace CsiApp
{
    // Display-only localization. Protocol values, prediction objects, input text and CSV data
    // are never changed. English is the source language and the upgrade default.
    public static class CsiLocalization
    {
        private const string PreferenceKey = "csi.language";
        private static bool initialized, chinese, applying;
        private static Font uiFont;
        private static readonly Dictionary<string, string> Sources = new Dictionary<string, string>(StringComparer.Ordinal);
        private static readonly Dictionary<string, string> Translations = new Dictionary<string, string>(StringComparer.Ordinal);
        private static readonly Dictionary<Text, TextState> Labels = new Dictionary<Text, TextState>();
        private static readonly Dictionary<TextMesh, MeshState> WorldLabels = new Dictionary<TextMesh, MeshState>();
        public static event Action Changed;
        public static bool IsChinese { get { Initialize(); return chinese; } }
        public static string LanguageCode { get { return IsChinese ? "zh-CN" : "en"; } }
        public static Font UiFont
        {
            get
            {
                if (uiFont == null) uiFont = Resources.Load<Font>("Localization/NotoSansSC-Regular");
                if (uiFont == null) uiFont = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
                return uiFont;
            }
        }
        private sealed class TextState
        {
            public string Source, Rendered;
            public bool Chinese, MetricsCaptured, OriginalGeometryAlignment;
            public VerticalWrapMode OriginalVerticalOverflow;
            public Vector2 LastRect;
            public int LastFontSize;
        }
        private sealed class MeshState { public string Source, Rendered; public bool Chinese; }
        private static void Initialize()
        {
            if (initialized) return;
            chinese = PlayerPrefs.GetString(PreferenceKey, "en") == "zh-CN"; initialized = true;
        }
        public static void SetLanguage(bool useChinese, bool persist = true)
        {
            Initialize(); bool changed = chinese != useChinese; chinese = useChinese;
            if (persist) { PlayerPrefs.SetString(PreferenceKey, chinese ? "zh-CN" : "en"); PlayerPrefs.Save(); }
            if (changed && Changed != null) Changed();
            ApplyNow();
        }
        public static string Translate(string english)
        {
            if (string.IsNullOrEmpty(english)) return english;
            Initialize();
            string source;
            if (Sources.TryGetValue(english, out source)) english = source;
            if (!chinese) return english;
            string translated;
            if (!Translations.TryGetValue(english, out translated))
            {
                translated = CsiTranslationCatalog.Chinese(english);
                // Recreated Replay pages share this bounded cache, so static labels and
                // repeated messages never rerun the phrase matcher every 0.2/0.3 seconds.
                if (Translations.Count < 16384) Translations.Add(english, translated);
            }
            // Callers may explicitly assign Translate(source); retain the original for the
            // scanner and for a later switch to English. Cap dynamic-message history.
            if (translated != english)
            {
                if (Sources.Count < 16384 && !Sources.ContainsKey(translated)) Sources.Add(translated, english);
            }
            return translated;
        }
        public static void ApplyNow()
        {
            if (applying) return;
            applying = true;
            try { Discover(); ApplyTracked(); }
            finally { applying = false; }
        }
        private static void Discover()
        {
            foreach (var text in UnityEngine.Object.FindObjectsByType<Text>(FindObjectsInactive.Include))
                if (!Labels.ContainsKey(text) && !IsInputBody(text)) Labels.Add(text, new TextState());
            foreach (var text in UnityEngine.Object.FindObjectsByType<TextMesh>(FindObjectsInactive.Include))
                if (!WorldLabels.ContainsKey(text)) WorldLabels.Add(text, new MeshState());
            var deadText = new List<Text>(); foreach (var pair in Labels) if (pair.Key == null) deadText.Add(pair.Key);
            foreach (var text in deadText) Labels.Remove(text);
            var deadMesh = new List<TextMesh>(); foreach (var pair in WorldLabels) if (pair.Key == null) deadMesh.Add(pair.Key);
            foreach (var text in deadMesh) WorldLabels.Remove(text);
        }
        private static bool IsInputBody(Text text)
        {
            var input = text.GetComponentInParent<InputField>(true);
            return input != null && input.textComponent == text;
        }
        private static string Original(string value)
        { string original; return value != null && Sources.TryGetValue(value, out original) ? original : value; }
        private static void ApplyTracked()
        {
            Font font = UiFont;
            foreach (var pair in Labels)
            {
                var text = pair.Key; if (text == null || IsInputBody(text)) continue;
                var state = pair.Value;
                bool updated = state.Source == null || text.text != state.Rendered;
                if (updated) state.Source = Original(text.text);
                bool languageChanged = state.Chinese != chinese;
                string rendered = updated || languageChanged ? Translate(state.Source) : state.Rendered;
                state.Rendered = rendered; state.Chinese = chinese;
                bool fontChanged = text.font != font;
                if (fontChanged) text.font = font;
                if (text.text != rendered) text.text = rendered;
                var size = text.rectTransform.rect.size;
                if (!state.MetricsCaptured || updated || languageChanged || fontChanged || state.LastRect != size || state.LastFontSize != text.fontSize)
                    ApplyCompactLineMetrics(text, state, size);
            }
            foreach (var pair in WorldLabels)
            {
                var text = pair.Key; if (text == null) continue;
                var state = pair.Value;
                bool updated = state.Source == null || text.text != state.Rendered;
                if (updated) state.Source = Original(text.text);
                string rendered = updated || state.Chinese != chinese ? Translate(state.Source) : state.Rendered;
                state.Rendered = rendered; state.Chinese = chinese;
                // Pure numeric floor labels keep their existing world-space material and metrics.
                if (!Regex.IsMatch(rendered ?? "", @"^\s*[0-9.,+\- ]*\s*$") && text.font != font)
                {
                    text.font = font;
                    var renderer = text.GetComponent<MeshRenderer>(); if (renderer != null) renderer.sharedMaterial = font.material;
                }
                if (text.text != rendered) text.text = rendered;
            }
        }
        private static void ApplyCompactLineMetrics(Text text, TextState state, Vector2 size)
        {
            if (!state.MetricsCaptured)
            {
                state.MetricsCaptured = true;
                state.OriginalVerticalOverflow = text.verticalOverflow;
                state.OriginalGeometryAlignment = text.alignByGeometry;
            }
            state.LastRect = size; state.LastFontSize = text.fontSize;
            // Noto SC's ascent+descent is 1.448 em. Truncate can therefore suppress an
            // entire 23px line in a 32px heading even though the visible glyphs fit.
            // Permit generation only for a genuinely single, width-fitting line whose
            // glyph-size fits the existing box. Align its visible geometry to the same
            // anchor. Paragraphs and wrapping warnings retain the author's overflow.
            bool compactLine = !text.resizeTextForBestFit && !string.IsNullOrEmpty(text.text)
                && text.text.IndexOf('\n') < 0 && text.text.IndexOf('\r') < 0
                && size.x > 0 && size.y >= text.fontSize && text.preferredWidth <= size.x + .5f;
            text.verticalOverflow = compactLine ? VerticalWrapMode.Overflow : state.OriginalVerticalOverflow;
            text.alignByGeometry = compactLine || state.OriginalGeometryAlignment;
        }
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void Install()
        {
            Initialize();
            var owner = new GameObject("CSI display localization");
            UnityEngine.Object.DontDestroyOnLoad(owner); owner.AddComponent<CsiLocalizationRunner>();
        }
        internal static void Tick(bool discover)
        {
            if (applying) return;
            applying = true;
            try { if (discover) Discover(); ApplyTracked(); }
            finally { applying = false; }
        }
    }
    public sealed class CsiLocalizationRunner : MonoBehaviour
    {
        private float nextDiscovery;
        private void LateUpdate()
        {
            bool discover = Time.unscaledTime >= nextDiscovery;
            if (discover) nextDiscovery = Time.unscaledTime + .1f;
            CsiLocalization.Tick(discover);
        }
    }
}
