using System;
using System.Collections.Generic;
using UnityEngine;

namespace CsiApp
{
    /// <summary>
    /// A presentation-only route. Confirmed grid topology plus explicitly illustrative
    /// approach points, never written back to the survey or used as measured T01.
    /// Points are local map metres: X = map x, Z = map y, Y = zero.
    /// </summary>
    public sealed class DemoNavigationRoute
    {
        public List<Vector3> Points { get; private set; }
        public List<int> GridLabels { get; private set; }
        public string DestinationId { get; private set; }
        public string DestinationName { get; private set; }
        public int ConnectorLabel { get; private set; }
        public float Length { get; private set; }
        public bool UsesIllustrativeConnector { get; private set; }
        public float GridLength { get; private set; }
        public string StartName { get; private set; }
        private readonly List<bool> illustrativeSegments = new List<bool>();
        public bool IsIllustrativeSegment(int endPointIndex)
        { return endPointIndex > 0 && endPointIndex <= illustrativeSegments.Count && illustrativeSegments[endPointIndex - 1]; }

        public static DemoNavigationRoute BuildFromGridPath(IList<int> labels)
        {
            if (labels == null || labels.Count < 2) throw new ArgumentException("A preview needs at least two grid points.");
            var result = new DemoNavigationRoute { Points = new List<Vector3>(), GridLabels = new List<int>(labels),
                DestinationId = "grid", DestinationName = "Grid " + labels[labels.Count - 1], ConnectorLabel = labels[labels.Count - 1], StartName = "Grid " + labels[0] };
            for (int i = 0; i < labels.Count; i++)
            {
                if (!GridMap.Valid(labels[i])) throw new ArgumentException("Invalid grid label.");
                if (i > 0 && Mathf.Abs(labels[i] % 5 - labels[i - 1] % 5) + Mathf.Abs(labels[i] / 5 - labels[i - 1] / 5) != 1)
                    throw new ArgumentException("Only adjacent four-neighbor grid edges are supported.");
                var point = ApartmentMap.Current.FindPoint(labels[i]); result.Add(ApartmentMap.ToWorld(point.x_m, point.y_m));
            }
            result.GridLength = result.Length; result.ValidateKnownGeometry(); return result;
        }

        public static DemoNavigationRoute Build(int startLabel, string destinationId, int gridTarget = 14)
        {
            if (!GridMap.Valid(startLabel)) throw new ArgumentOutOfRangeException("startLabel");
            string id = (destinationId ?? "").Trim().ToLowerInvariant();
            int connector;
            string name;
            switch (id)
            {
                case "balcony": connector = 5; name = "Balcony entrance"; break;
                case "office": connector = 12; name = "Office entrance"; break;
                case "kitchen": connector = 14; name = "Kitchen entrance"; break;
                case "grid":
                    if (!GridMap.Valid(gridTarget)) throw new ArgumentOutOfRangeException("gridTarget");
                    connector = gridTarget; name = "Grid " + gridTarget; break;
                default: throw new ArgumentException("Unknown demo destination: " + destinationId, "destinationId");
            }
            var result = new DemoNavigationRoute
            {
                Points = new List<Vector3>(), GridLabels = new List<int>(GridMap.Route(startLabel, connector)),
                DestinationId = id, DestinationName = name, ConnectorLabel = connector,
                UsesIllustrativeConnector = id != "grid", StartName = "Grid " + startLabel
            };
            var map = ApartmentMap.Current;
            foreach (int label in result.GridLabels)
            {
                var point = map.FindPoint(label);
                if (point == null) throw new InvalidOperationException("Missing grid point " + label);
                result.Add(ApartmentMap.ToWorld(point.x_m, point.y_m));
            }
            result.GridLength = result.Length;
            Vector3 last = result.Points[result.Points.Count - 1];
            // The 0.22 m stand-off and 0.25 m corridor offset are DEMO choices,
            // not surveyed endpoints, clearances or body-accessibility evidence.
            if (id == "balcony")
            {
                Vector3 doorway = ApartmentMap.ToWorld(map.FindOpening("balcony_opening").center_xy_m);
                Vector3 target = doorway + Vector3.right * .22f;
                result.Add(new Vector3(last.x, 0, target.z), true);
                result.Add(target, true);
            }
            else if (id == "office")
            {
                Vector3 target = ApartmentMap.ToWorld(map.FindOpening("office_opening").center_xy_m) - Vector3.forward * .22f;
                result.Add(new Vector3(target.x, 0, last.z), true);
                result.Add(target, true);
            }
            else if (id == "kitchen")
            {
                Vector3 target = ApartmentMap.ToWorld(map.FindOpening("kitchen_opening").center_xy_m) - Vector3.forward * .22f;
                var corridor = map.FindRegion("corridor_extension");
                float approachZ = corridor.min_xy_m[1] + .25f;
                result.Add(new Vector3(last.x, 0, approachZ), true);
                result.Add(new Vector3(target.x, 0, approachZ), true);
                result.Add(target, true);
            }
            result.ValidateKnownGeometry();
            return result;
        }

        public static DemoNavigationRoute BuildFromPlace(string startPlace, string destinationId, int gridTarget = 14)
        {
            string id = (startPlace ?? "").Trim().ToLowerInvariant();
            int connector = id == "balcony" ? 5 : id == "office" ? 12 : id == "kitchen" ? 14 : -1;
            if (connector < 0) throw new ArgumentException("Start place must be balcony, office or kitchen.", "startPlace");
            var departure = Build(connector, id);
            var arrival = Build(connector, destinationId, gridTarget);
            var result = new DemoNavigationRoute
            {
                Points = new List<Vector3>(), GridLabels = new List<int>(arrival.GridLabels),
                DestinationId = arrival.DestinationId, DestinationName = arrival.DestinationName,
                StartName = departure.DestinationName, ConnectorLabel = arrival.ConnectorLabel,
                UsesIllustrativeConnector = true, GridLength = arrival.GridLength
            };
            if (id == arrival.DestinationId)
            {
                result.Add(departure.Points[departure.Points.Count - 1]);
                return result;
            }
            for (int i = departure.Points.Count - 1; i >= 0; i--) result.Add(departure.Points[i], true);
            for (int i = 1; i < arrival.Points.Count; i++) result.Add(arrival.Points[i], arrival.IsIllustrativeSegment(i));
            result.ValidateKnownGeometry();
            return result;
        }

        private void Add(Vector3 point, bool illustrative = false)
        {
            if (Points.Count > 0)
            {
                float length = Vector3.Distance(Points[Points.Count - 1], point);
                if (length < .0001f) return;
                Length += length;
                illustrativeSegments.Add(illustrative);
            }
            Points.Add(point);
        }

        public Vector3 PointAtDistance(float metres, out Vector3 direction)
        {
            direction = Vector3.forward;
            if (Points.Count == 1) return Points[0];
            float remaining = Mathf.Clamp(metres, 0, Length);
            for (int i = 1; i < Points.Count; i++)
            {
                Vector3 step = Points[i] - Points[i - 1];
                float length = step.magnitude;
                direction = step / length;
                if (remaining <= length || i == Points.Count - 1)
                    return Points[i - 1] + direction * Mathf.Min(remaining, length);
                remaining -= length;
            }
            return Points[Points.Count - 1];
        }

        public string InstructionAtDistance(float metres)
        {
            float distance = Mathf.Clamp(metres, 0, Length);
            if (Length - distance < .025f)
                return DestinationId != "grid" ? "At demo entrance — end when ready" : "At grid target — end when ready";
            float cumulative = 0;
            for (int i = 1; i < Points.Count; i++)
            {
                Vector3 current = Points[i] - Points[i - 1];
                cumulative += current.magnitude;
                if (cumulative < distance - .001f) continue;
                if (i < Points.Count - 1)
                {
                    Vector3 next = Points[i + 1] - Points[i];
                    float turn = Vector3.SignedAngle(current, next, Vector3.up);
                    if (Mathf.Abs(turn) > 12f)
                    {
                        string action = Mathf.Abs(turn) >= 150f ? "turn back" : "turn " + (turn > 0 ? "right" : "left");
                        float toTurn = Mathf.Max(0, cumulative - distance);
                        if (toTurn <= .15f) return char.ToUpperInvariant(action[0]) + action.Substring(1) + " now";
                        return "In " + toTurn.ToString("F1") + " m, " + action;
                    }
                }
            }
            return "Continue " + (Length - distance).ToString("F1") + " m to " + (DestinationId != "grid" ? "demo entrance" : "grid target");
        }

        private void ValidateKnownGeometry()
        {
            var map = ApartmentMap.Current;
            for (int i = 1; i < GridLabels.Count; i++)
                if (!map.HasEdge(GridLabels[i - 1], GridLabels[i]))
                    throw new InvalidOperationException("Demo grid route contains a non-neighbor edge.");
            for (int i = 1; i < Points.Count; i++)
            {
                Vector2 a = new Vector2(Points[i - 1].x, Points[i - 1].z), b = new Vector2(Points[i].x, Points[i].z);
                foreach (var boundary in map.boundaries)
                {
                    if (boundary.kind == "open_connection" || boundary.kind == "region_boundary_unspecified") continue;
                    var c = new Vector2(boundary.start_xy_m[0], boundary.start_xy_m[1]);
                    var d = new Vector2(boundary.end_xy_m[0], boundary.end_xy_m[1]);
                    if (Intersects(a, b, c, d)) throw new InvalidOperationException("Demo route crosses known boundary " + boundary.id);
                }
                int samples = Mathf.Max(1, Mathf.CeilToInt(Vector2.Distance(a, b) / .025f));
                for (int k = 0; k <= samples; k++)
                {
                    Vector2 p = Vector2.Lerp(a, b, (float)k / samples);
                    bool known = false;
                    foreach (var region in map.regions)
                        if (region.model_role != "excluded_unknown" && p.x >= region.min_xy_m[0] - .0001f && p.x <= region.max_xy_m[0] + .0001f &&
                            p.y >= region.min_xy_m[1] - .0001f && p.y <= region.max_xy_m[1] + .0001f) { known = true; break; }
                    if (!known) throw new InvalidOperationException("Demo route leaves confirmed floor regions.");
                }
            }
        }
        private static float Cross(Vector2 a, Vector2 b) { return a.x * b.y - a.y * b.x; }
        private static bool Intersects(Vector2 a, Vector2 b, Vector2 c, Vector2 d)
        {
            var r = b - a; var s = d - c; float denominator = Cross(r, s);
            if (Mathf.Abs(denominator) < .000001f)
            {
                if (Mathf.Abs(Cross(c - a, r)) > .00001f) return false;
                float rr = Vector2.Dot(r, r);
                float t0 = Vector2.Dot(c - a, r) / rr, t1 = Vector2.Dot(d - a, r) / rr;
                return Mathf.Max(Mathf.Min(t0, t1), 0) <= Mathf.Min(Mathf.Max(t0, t1), 1);
            }
            float t = Cross(c - a, s) / denominator, u = Cross(c - a, r) / denominator;
            return t >= -.00001f && t <= 1.00001f && u >= -.00001f && u <= 1.00001f;
        }
    }
}
