using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace CsiApp
{
    [Serializable] public sealed class SiteWaypoint { public float x, y; }
    [Serializable] public sealed class SiteDestinationConfig
    { public string id; public bool surveyed; public SiteWaypoint[] approach; }

    // User-measured installation and approach paths. No demo endpoints are imported.
    [Serializable] public sealed class SiteNavigationConfig
    {
        public bool markerMeasured;
        public float imageWidthM, imageX, imageY, imageHeightM, imageFacingDeg;
        public SiteDestinationConfig[] destinations;
        [NonSerialized] public string LastError;
        public SiteNavigationConfig()
        {
            destinations = new[] { Unmeasured("balcony"), Unmeasured("office"), Unmeasured("kitchen") };
        }
        private static SiteDestinationConfig Unmeasured(string id)
        { return new SiteDestinationConfig { id = id, surveyed = false, approach = new SiteWaypoint[0] }; }
        public static string ConfigPath { get { return Path.Combine(Application.persistentDataPath, "site_navigation.json"); } }
        public static int ConnectorFor(string id)
        { return id == "balcony" ? 5 : id == "office" ? 12 : id == "kitchen" ? 14 : -1; }
        public SiteDestinationConfig FindDestination(string id)
        {
            if (destinations != null) foreach (var d in destinations) if (d != null && d.id == id) return d;
            return null;
        }
        public bool ValidateMarker(out string error)
        {
            error = null;
            if (!markerMeasured) { error = "Reference image installation has not been measured and confirmed."; return false; }
            if (!Finite(imageWidthM) || imageWidthM < .03f || imageWidthM > 2f
                || !Finite(imageHeightM) || imageHeightM < .05f || imageHeightM > 3f)
            { error = "Measured image width must be 0.03-2 m and centre height 0.05-3 m."; return false; }
            if (!Finite(imageX) || !Finite(imageY) || Mathf.Abs(imageX) > 100 || Mathf.Abs(imageY) > 100
                || !Finite(imageFacingDeg) || imageFacingDeg < -360 || imageFacingDeg > 360)
            { error = "Image centre coordinates/yaw must be finite; yaw is -360 to 360 degrees."; return false; }
            return true;
        }
        public Pose MarkerMapPose
        {
            get
            {
                string error; if (!ValidateMarker(out error)) throw new InvalidOperationException(error);
                float yaw = imageFacingDeg * Mathf.Deg2Rad;
                var normal = new Vector3(Mathf.Sin(yaw), 0, Mathf.Cos(yaw));
                // ARTrackedImage local XZ plane: +Y is front normal, +Z is image top.
                // Upright vertical installation makes image top global/map +height.
                return new Pose(new Vector3(imageX, imageHeightM, imageY), Quaternion.LookRotation(Vector3.up, normal));
            }
        }
        public bool ValidateDestination(string id, out string error)
        {
            error = null; int connector = ConnectorFor(id);
            var d = FindDestination(id);
            if (connector < 0 || d == null) { error = "Unknown destination."; return false; }
            if (!d.surveyed) { error = "Measure and confirm this destination's approach before navigation."; return false; }
            if (d.approach == null || d.approach.Length == 0 || d.approach.Length > 64)
            { error = "A surveyed destination needs 1-64 ordered approach waypoints after its connector."; return false; }
            try
            {
                var start = ApartmentMap.Current.FindPoint(connector);
                var path = new List<Vector3> { ApartmentMap.ToWorld(start.x_m, start.y_m) };
                foreach (var point in d.approach)
                {
                    if (point == null || !Finite(point.x) || !Finite(point.y))
                    { error = "Approach contains a missing or nonfinite point."; return false; }
                    Vector3 next = ApartmentMap.ToWorld(point.x, point.y);
                    if ((next - path[path.Count - 1]).sqrMagnitude < .0001f)
                    { error = "Do not repeat the connector or a waypoint in the approach."; return false; }
                    path.Add(next);
                }
                return SiteNavigationGeometry.ValidatePath(path, ApartmentMap.Current, out error);
            }
            catch (Exception e) { error = "Could not validate surveyed approach: " + e.Message; return false; }
        }
        private bool ValidateForSave(out string error)
        {
            error = null;
            if (!Finite(imageWidthM) || !Finite(imageX) || !Finite(imageY) || !Finite(imageHeightM) || !Finite(imageFacingDeg))
            { error = "Configuration contains a nonfinite marker number."; return false; }
            if (imageWidthM < 0 || imageWidthM > 2 || imageHeightM < 0 || imageHeightM > 3)
            { error = "Marker dimensions/height must be unset (0) or within supported meter ranges."; return false; }
            if (markerMeasured && !ValidateMarker(out error)) return false;
            if (destinations == null || destinations.Length != 3) { error = "Configuration requires balcony, office and kitchen entries, even when unmeasured."; return false; }
            var seen = new HashSet<string>();
            foreach (var d in destinations)
            {
                if (d == null || ConnectorFor(d.id) < 0 || !seen.Add(d.id))
                { error = "Destination IDs must be unique balcony/office/kitchen."; return false; }
                if (d.approach != null) foreach (var point in d.approach)
                    if (point == null || !Finite(point.x) || !Finite(point.y))
                    { error = "Configuration contains a nonfinite approach point."; return false; }
                if (d.surveyed && !ValidateDestination(d.id, out error)) return false;
            }
            return true;
        }
        public bool Save()
        {
            string error;
            if (!ValidateForSave(out error)) { LastError = error; return false; }
            try
            {
                string pending = ConfigPath + ".new";
                File.WriteAllText(pending, JsonUtility.ToJson(this, true));
                if (File.Exists(ConfigPath)) File.Replace(pending, ConfigPath, null); else File.Move(pending, ConfigPath);
                LastError = null; return true;
            }
            catch (Exception e) { LastError = "Configuration was not saved: " + e.Message; return false; }
        }
        public static SiteNavigationConfig Load()
        {
            var value = new SiteNavigationConfig();
            try
            {
                if (!File.Exists(ConfigPath)) return value;
                if (new FileInfo(ConfigPath).Length > 1024 * 1024) throw new FormatException("Configuration exceeds 1 MB.");
                JsonUtility.FromJsonOverwrite(File.ReadAllText(ConfigPath), value);
                string error; if (!value.ValidateForSave(out error)) throw new FormatException(error);
                return value;
            }
            catch (Exception e) { return new SiteNavigationConfig { LastError = "Saved configuration rejected: " + e.Message }; }
        }
        internal static bool Finite(float v) { return !float.IsNaN(v) && !float.IsInfinity(v); }
    }

    // Tests confirmed planar geometry only; physical clearances/door states still need survey.
    public static class SiteNavigationGeometry
    {
        public static bool InGrid(Vector3 p)
        { return p.x >= -.001f && p.x <= 3.201f && p.z >= -.001f && p.z <= 1.601f; }
        public static bool ValidatePath(IList<Vector3> path, SiteMapData map, out string error)
        {
            error = null;
            if (path == null || path.Count == 0 || map == null || map.regions == null || map.boundaries == null)
            { error = "Route or map geometry is missing."; return false; }
            foreach (var p in path)
                if (!SiteNavigationConfig.Finite(p.x) || !SiteNavigationConfig.Finite(p.z))
                { error = "Route contains a nonfinite coordinate."; return false; }
            for (int i = 0; i < path.Count; i++)
            {
                Vector2 a = new Vector2(path[Mathf.Max(0, i - 1)].x, path[Mathf.Max(0, i - 1)].z);
                Vector2 b = new Vector2(path[i].x, path[i].z);
                if (i > 0)
                    foreach (var wall in map.boundaries)
                    {
                        if (wall.kind == "open_connection" || wall.kind == "region_boundary_unspecified") continue;
                        if (SegmentsTouch(a, b, new Vector2(wall.start_xy_m[0], wall.start_xy_m[1]), new Vector2(wall.end_xy_m[0], wall.end_xy_m[1])))
                        { error = "Route crosses or touches known barrier " + wall.id + "."; return false; }
                    }
                int samples = Mathf.Max(1, Mathf.CeilToInt(Vector2.Distance(a, b) / .025f));
                for (int k = 0; k <= samples; k++)
                {
                    Vector2 point = Vector2.Lerp(a, b, (float)k / samples); bool known = false;
                    foreach (var region in map.regions)
                        if (region.model_role != "excluded_unknown" && point.x >= region.min_xy_m[0] - .0001f
                            && point.x <= region.max_xy_m[0] + .0001f && point.y >= region.min_xy_m[1] - .0001f && point.y <= region.max_xy_m[1] + .0001f)
                        { known = true; break; }
                    if (!known) { error = "Route leaves confirmed floor regions."; return false; }
                }
            }
            return true;
        }
        private static float Cross(Vector2 a, Vector2 b) { return a.x * b.y - a.y * b.x; }
        private static bool SegmentsTouch(Vector2 a, Vector2 b, Vector2 c, Vector2 d)
        {
            var r = b - a; var s = d - c; float denominator = Cross(r, s);
            if (r.sqrMagnitude < .0000001f) return false;
            if (Mathf.Abs(denominator) < .000001f)
            {
                if (Mathf.Abs(Cross(c - a, r)) > .00001f) return false;
                float t0 = Vector2.Dot(c - a, r) / r.sqrMagnitude, t1 = Vector2.Dot(d - a, r) / r.sqrMagnitude;
                return Mathf.Max(Mathf.Min(t0, t1), 0) <= Mathf.Min(Mathf.Max(t0, t1), 1);
            }
            float t = Cross(c - a, s) / denominator, u = Cross(c - a, r) / denominator;
            return t >= -.00001f && t <= 1.00001f && u >= -.00001f && u <= 1.00001f;
        }
    }
}
