using System;
using System.Collections.Generic;
using UnityEngine;

namespace CsiApp
{
    // Recommend at most two shortest paths with fewest turns; this is a UI limit.
    public static class GridRoutePlanner
    {
        public static List<List<int>> Candidates(int start, int target)
        {
            var result = new List<List<int>>();
            if (!GridMap.Valid(start) || !GridMap.Valid(target) || start == target) return result;
            var distance = new Dictionary<int, int>(); var queue = new Queue<int>(); distance[target] = 0; queue.Enqueue(target);
            while (queue.Count > 0)
            {
                int node = queue.Dequeue();
                foreach (int next in GridMap.Neighbours(node))
                    if (ApartmentMap.Current.HasEdge(node, next) && !distance.ContainsKey(next))
                    { distance[next] = distance[node] + 1; queue.Enqueue(next); }
            }
            if (!distance.ContainsKey(start)) return result;
            var all = new List<List<int>>(); Enumerate(start, target, distance, new List<int> { start }, all);
            all.Sort((a, b) => {
                int compare = Turns(a).CompareTo(Turns(b)); if (compare != 0) return compare;
                for (int i = 0; i < a.Count; i++) { compare = a[i].CompareTo(b[i]); if (compare != 0) return compare; }
                return 0;
            });
            foreach (var path in all)
            {
                var geometry = new List<Vector3>();
                foreach (int label in path) { var p = ApartmentMap.Current.FindPoint(label); geometry.Add(ApartmentMap.ToWorld(p.x_m, p.y_m)); }
                string error;
                if (SiteNavigationGeometry.ValidatePath(geometry, ApartmentMap.Current, out error)) result.Add(path);
                if (result.Count == 2) break;
            }
            return result;
        }
        private static void Enumerate(int node, int target, Dictionary<int, int> distance, List<int> path, List<List<int>> result)
        {
            if (node == target) { result.Add(new List<int>(path)); return; }
            foreach (int next in GridMap.Neighbours(node))
                if (ApartmentMap.Current.HasEdge(node, next) && distance.ContainsKey(next) && distance[next] == distance[node] - 1)
                { path.Add(next); Enumerate(next, target, distance, path, result); path.RemoveAt(path.Count - 1); }
        }
        public static int Turns(IList<int> path)
        { int turns = 0; for (int i = 2; i < path.Count; i++) if (path[i] - path[i - 1] != path[i - 1] - path[i - 2]) turns++; return turns; }
    }
}
