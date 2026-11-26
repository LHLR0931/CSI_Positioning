using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;

namespace CsiApp
{
    // JsonUtility handles DTOs. This small structural reader preserves missing/null fields,
    // and parses prediction objects separately so sentinel defaults survive deserialization.
    public static class CsiJson
    {
        [Serializable] private sealed class StringValue { public string value; }
        public static CsiMeta ParseMeta(string json)
        {
            var result = new CsiMeta(); JsonUtility.FromJsonOverwrite(json, result);
            string error; if (!CsiValidation.Meta(result, out error)) throw new FormatException(error);
            return result;
        }
        public static CsiStatus ParseStatus(string json)
        {
            var result = new CsiStatus(); JsonUtility.FromJsonOverwrite(json, result);
            if (result.state != "offline" && result.state != "calibrating" && result.state != "training"
                && result.state != "ready" && result.state != "degraded") throw new FormatException("Unknown server state.");
            string progress = Property(json, "progress"); float n;
            if (progress != null && progress != "null")
            {
                if (!float.TryParse(progress, NumberStyles.Float, CultureInfo.InvariantCulture, out n)
                    || !CsiValidation.Finite(n) || n < 0 || n > 1) throw new FormatException("Status progress must be null or 0-1.");
                result.Progress = n;
            }
            if (result.links != null)
                foreach (var link in new[] { result.links.piA_apX, result.links.piA_apY, result.links.piB_apX, result.links.piB_apY })
                    if (link != null && (!CsiValidation.Finite(link.fps) || !CsiValidation.Finite(link.last_frame_age_s)
                        || link.fps < 0 || link.last_frame_age_s < 0)) throw new FormatException("Invalid link frame rate/age.");
            return result;
        }
        public static List<CsiPrediction> ParsePredictions(string json, string fallbackSource = null, string fallbackSession = null)
        {
            string source = StringProperty(json, "source") ?? fallbackSource;
            string session = StringProperty(json, "session_id") ?? fallbackSession;
            string array = Property(json, "predictions");
            var result = new List<CsiPrediction>();
            if (array == null) result.Add(ParsePrediction(json, source, session));
            else foreach (string row in Items(array)) result.Add(ParsePrediction(row, source, session));
            if (result.Count == 0) throw new FormatException("Prediction array is empty.");
            return result;
        }
        private static CsiPrediction ParsePrediction(string json, string source, string session)
        {
            foreach (string field in new[] { "seq", "label", "x", "y", "x_m", "y_m", "confidence" })
            {
                double number; string raw = Property(json, field);
                if (raw == null || !double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out number)
                    || double.IsNaN(number) || double.IsInfinity(number)) throw new FormatException("Missing/invalid prediction number: " + field);
                if ((field == "seq" || field == "label" || field == "x" || field == "y") && Math.Floor(number) != number)
                    throw new FormatException("Prediction integer is fractional: " + field);
            }
            var result = new CsiPrediction(); JsonUtility.FromJsonOverwrite(json, result);
            // Unity's inline-class serializer can materialize default instances for null
            // references. The wire protocol's absent/null optional objects stay absent.
            string window = Property(json, "window"), expected = Property(json, "expected_m");
            if (window == null || window == "null") result.window = null;
            if (expected == null || expected == "null") result.expected_m = null;
            if (string.IsNullOrEmpty(result.source)) result.source = source;
            if (string.IsNullOrEmpty(result.session_id)) result.session_id = session;
            string error; if (!CsiValidation.Prediction(result, out error)) throw new FormatException(error);
            return result;
        }
        public static CsiReplay ParseReplay(string json)
        {
            string meta = Property(json, "meta"), predictions = Property(json, "predictions");
            if (meta == null || predictions == null) throw new FormatException("Replay requires meta and predictions.");
            var result = new CsiReplay { Meta = ParseMeta(meta), Source = StringProperty(json, "source"), SessionId = StringProperty(json, "session_id") };
            var rows = ParsePredictions(json, result.Source ?? result.Meta.source, result.SessionId ?? result.Meta.session_id);
            // Preserve file order and reject decreasing time within one session/model.
            var times = new Dictionary<string, DateTime>();
            foreach (var row in rows)
            {
                DateTime now, previous; CsiValidation.TryTime(row.timestamp, out now);
                string key = row.session_id + "|" + CsiValidation.ModelId(row);
                if (times.TryGetValue(key, out previous) && now < previous) throw new FormatException("Replay rows must be ordered by timestamp.");
                times[key] = now; result.Predictions.Add(row);
            }
            return result;
        }
        public static string StringProperty(string json, string name)
        {
            string raw = Property(json, name);
            if (raw == null || raw == "null") return null;
            if (!raw.StartsWith("\"", StringComparison.Ordinal)) throw new FormatException(name + " must be a string.");
            return JsonUtility.FromJson<StringValue>("{\"value\":" + raw + "}").value;
        }
        public static string Property(string json, string name)
        {
            if (string.IsNullOrWhiteSpace(json)) throw new FormatException("Empty JSON.");
            int p = 0; Skip(json, ref p); Require(json, ref p, '{');
            while (true)
            {
                Skip(json, ref p); if (p >= json.Length) throw new FormatException("Incomplete JSON object.");
                if (json[p] == '}') return null;
                int keyStart = p; SkipString(json, ref p);
                string key = json.Substring(keyStart + 1, p - keyStart - 2);
                Skip(json, ref p); Require(json, ref p, ':'); Skip(json, ref p);
                int start = p; SkipValue(json, ref p);
                if (key == name) return json.Substring(start, p - start).Trim();
                Skip(json, ref p);
                if (p < json.Length && json[p] == ',') { p++; continue; }
                if (p < json.Length && json[p] == '}') return null;
                throw new FormatException("Invalid JSON property separator.");
            }
        }
        private static IEnumerable<string> Items(string json)
        {
            int p = 0; Skip(json, ref p); Require(json, ref p, '[');
            while (true)
            {
                Skip(json, ref p); if (p >= json.Length) throw new FormatException("Incomplete JSON array.");
                if (json[p] == ']') yield break;
                int start = p; SkipValue(json, ref p); yield return json.Substring(start, p - start);
                Skip(json, ref p);
                if (p < json.Length && json[p] == ',') { p++; continue; }
                if (p < json.Length && json[p] == ']') yield break;
                throw new FormatException("Invalid JSON array separator.");
            }
        }
        private static void Skip(string json, ref int p) { while (p < json.Length && char.IsWhiteSpace(json[p])) p++; }
        private static void Require(string json, ref int p, char c)
        { if (p >= json.Length || json[p++] != c) throw new FormatException("Expected JSON " + c); }
        private static void SkipString(string json, ref int p)
        {
            Require(json, ref p, '"');
            while (p < json.Length)
            { char c = json[p++]; if (c == '"') return; if (c == '\\') { if (p >= json.Length) break; p++; } }
            throw new FormatException("Unclosed JSON string.");
        }
        private static void SkipValue(string json, ref int p)
        {
            if (p >= json.Length) throw new FormatException("Missing JSON value.");
            if (json[p] == '"') { SkipString(json, ref p); return; }
            if (json[p] == '{' || json[p] == '[')
            {
                var stack = new Stack<char>(); stack.Push(json[p++] == '{' ? '}' : ']');
                while (p < json.Length && stack.Count > 0)
                {
                    char c = json[p];
                    if (c == '"') { SkipString(json, ref p); continue; }
                    p++;
                    if (c == '{' || c == '[') stack.Push(c == '{' ? '}' : ']');
                    else if (c == '}' || c == ']') { if (stack.Pop() != c) throw new FormatException("Mismatched JSON bracket."); }
                }
                if (stack.Count > 0) throw new FormatException("Unclosed JSON value.");
                return;
            }
            int start = p;
            while (p < json.Length && json[p] != ',' && json[p] != '}' && json[p] != ']' && !char.IsWhiteSpace(json[p])) p++;
            if (start == p) throw new FormatException("Missing JSON primitive.");
        }
    }
}
