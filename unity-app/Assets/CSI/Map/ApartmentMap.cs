using System;
using UnityEngine;

namespace CsiApp
{
    // DTOs intentionally read the flat fields supported by JsonUtility. The original
    // source JSON also retains evidence, unknowns and polygons for the survey record.
    [Serializable] public sealed class SiteMapData
    {
        public string schema_version;
        public MapMetadata metadata;
        public MapRegion[] regions;
        public MapBoundary[] boundaries;
        public MapOpening[] openings;
        public MapGrid grid;

        public MapPoint FindPoint(int label)
        {
            if (grid == null || grid.points == null) return null;
            foreach (var point in grid.points) if (point.label == label) return point;
            return null;
        }
        public MapRegion FindRegion(string id)
        {
            if (regions != null) foreach (var region in regions) if (region.id == id) return region;
            return null;
        }
        public MapOpening FindOpening(string id)
        {
            if (openings != null) foreach (var opening in openings) if (opening.id == id) return opening;
            return null;
        }
        public bool HasEdge(int from, int to)
        {
            if (grid == null || grid.edges == null) return false;
            foreach (var edge in grid.edges)
                if ((edge.from == from && edge.to == to) || (edge.from == to && edge.to == from)) return true;
            return false;
        }
        public Rect PlanBounds
        {
            get
            {
                float left = float.PositiveInfinity, bottom = float.PositiveInfinity;
                float right = float.NegativeInfinity, top = float.NegativeInfinity;
                foreach (var region in regions)
                {
                    if (region.model_role == "excluded_unknown") continue;
                    left = Mathf.Min(left, region.min_xy_m[0]); bottom = Mathf.Min(bottom, region.min_xy_m[1]);
                    right = Mathf.Max(right, region.max_xy_m[0]); top = Mathf.Max(top, region.max_xy_m[1]);
                }
                return Rect.MinMaxRect(left, bottom, right, top);
            }
        }
    }
    [Serializable] public sealed class MapMetadata { public string map_id, status, title_zh, approximation_zh; }
    [Serializable] public sealed class MapRegion
    {
        public string id, label_zh, model_role;
        public float[] min_xy_m, max_xy_m;
        public bool navigation_enabled_by_default;
    }
    [Serializable] public sealed class MapBoundary
    {
        public string id, kind, note_zh;
        public float[] start_xy_m, end_xy_m;
    }
    [Serializable] public sealed class MapOpening
    {
        public string id, drawing_id, kind, door_leaf_state;
        public float[] start_xy_m, end_xy_m, center_xy_m;
        public float span_m;
    }
    [Serializable] public sealed class MapGrid
    {
        public int columns, rows;
        public float spacing_m;
        public MapPoint[] points;
        public MapEdge[] edges;
    }
    [Serializable] public sealed class MapPoint { public int label, i, j; public float x_m, y_m; }
    [Serializable] public sealed class MapEdge { public int from, to; public float length_m; }

    public static class ApartmentMap
    {
        public const string ResourcePath = "Maps/apartment_v1_2026-10-10";
        private static SiteMapData current;
        public static SiteMapData Current
        {
            get
            {
                if (current == null)
                {
                    var asset = Resources.Load<TextAsset>(ResourcePath);
                    if (asset == null) throw new InvalidOperationException("Apartment map resource is missing: " + ResourcePath);
                    var loaded = JsonUtility.FromJson<SiteMapData>(asset.text);
                    Validate(loaded);
                    current = loaded;
                }
                return current;
            }
        }
        public static Vector3 ToWorld(float x, float y, float height = 0) { return new Vector3(x, height, y); }
        public static Vector3 ToWorld(float[] xy, float height = 0) { return ToWorld(xy[0], xy[1], height); }

        private static void Validate(SiteMapData data)
        {
            if (data == null || data.regions == null || data.boundaries == null || data.openings == null ||
                data.grid == null || data.grid.points == null || data.grid.edges == null)
                throw new InvalidOperationException("Apartment map has missing required fields.");
            if (data.grid.points.Length != 15 || data.grid.edges.Length != 22)
                throw new InvalidOperationException("This prototype requires the confirmed 15-node, 22-edge grid.");
            foreach (var region in data.regions)
            {
                RequirePair(region.min_xy_m, region.id); RequirePair(region.max_xy_m, region.id);
                if (region.max_xy_m[0] <= region.min_xy_m[0] || region.max_xy_m[1] <= region.min_xy_m[1])
                    throw new InvalidOperationException("Invalid region bounds: " + region.id);
            }
            foreach (var boundary in data.boundaries) { RequirePair(boundary.start_xy_m, boundary.id); RequirePair(boundary.end_xy_m, boundary.id); }
            foreach (var opening in data.openings) { RequirePair(opening.start_xy_m, opening.id); RequirePair(opening.end_xy_m, opening.id); RequirePair(opening.center_xy_m, opening.id); }
            foreach (var edge in data.grid.edges)
                if (data.FindPoint(edge.from) == null || data.FindPoint(edge.to) == null)
                    throw new InvalidOperationException("Grid edge references a missing point.");
        }
        private static void RequirePair(float[] pair, string id)
        {
            if (pair == null || pair.Length != 2 || float.IsNaN(pair[0]) || float.IsNaN(pair[1]) || float.IsInfinity(pair[0]) || float.IsInfinity(pair[1]))
                throw new InvalidOperationException("Invalid map coordinate: " + id);
        }
    }
}
