using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace CsiApp
{
    // Developer-mode grid diagram. The apartment is rendered by ApartmentSceneView.
    [RequireComponent(typeof(CanvasRenderer))]
    public sealed class GridGraphic : MaskableGraphic
    {
        public NavigationState State;
        public DeveloperDemoData SampleData;
        public bool ShowRoute;
        public bool UseResult;
        public int ResultPrediction = -1, ResultTruth = -1;
        public static readonly Color Ink = new Color32(31, 58, 55, 255);
        public static readonly Color Teal = new Color32(0, 119, 102, 255);
        public Vector2 Point(int label)
        {
            float x = label % 5, y = label / 5;
            Rect r = rectTransform.rect;
            return new Vector2(r.xMin + r.width * (.12f + x * .19f), r.yMin + r.height * (.18f + y * .32f));
        }
        protected override void OnPopulateMesh(VertexHelper vh)
        {
            vh.Clear(); if (State == null && SampleData == null) return;
            int prediction = UseResult ? ResultPrediction : SampleData != null ? SampleData.Prediction : State.Prediction;
            int truth = UseResult ? ResultTruth : SampleData != null ? SampleData.GroundTruth : State.GroundTruth;
            Color grid = new Color32(199, 217, 210, 255);
            for (int n = 0; n < 15; n++)
                foreach (int next in GridMap.Neighbours(n)) if (next > n) Line(vh, Point(n), Point(next), 1.5f, grid);
            if (ShowRoute && State != null)
            {
                for (int i = 1; i < State.Path.Count; i++) Line(vh, Point(State.Path[i - 1]), Point(State.Path[i]), 6, Teal);
            }
            for (int n = 0; n < 15; n++)
            {
                Disc(vh, Point(n), 15, grid); Disc(vh, Point(n), 12.5f, new Color32(247, 250, 246, 255));
            }
            if (truth >= 0 && !ShowRoute)
            {
                if (prediction >= 0) Line(vh, Point(truth), Point(prediction), 3, new Color32(218, 138, 41, 255));
                Disc(vh, Point(truth), 24, new Color32(218, 138, 41, 255));
                Disc(vh, Point(truth), 19, new Color32(247, 250, 246, 255));
            }
            if (prediction >= 0) {
                Disc(vh, Point(prediction), 19, new Color32(54, 111, 218, 255));
                Disc(vh, Point(prediction), 14, new Color32(237, 244, 255, 255));
            }
            if (ShowRoute && State != null && State.DestinationAvailable) { Disc(vh, Point(State.Target), 17, Teal); Disc(vh, Point(State.Target), 12, new Color32(224, 245, 235, 255)); }
        }
        private static void Line(VertexHelper vh, Vector2 a, Vector2 b, float width, Color color)
        {
            Vector2 d = (b - a).normalized, normal = new Vector2(-d.y, d.x) * width / 2;
            int i = vh.currentVertCount;
            vh.AddVert(a + normal, color, Vector2.zero); vh.AddVert(b + normal, color, Vector2.zero);
            vh.AddVert(b - normal, color, Vector2.zero); vh.AddVert(a - normal, color, Vector2.zero);
            vh.AddTriangle(i, i + 1, i + 2); vh.AddTriangle(i, i + 2, i + 3);
        }
        private static void Disc(VertexHelper vh, Vector2 center, float radius, Color color)
        {
            int first = vh.currentVertCount; vh.AddVert(center, color, Vector2.zero);
            for (int i = 0; i <= 32; i++)
            {
                float angle = i * Mathf.PI * 2 / 32;
                vh.AddVert(center + new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)) * radius, color, Vector2.zero);
                if (i > 0) vh.AddTriangle(first, first + i, first + i + 1);
            }
        }
    }
}
