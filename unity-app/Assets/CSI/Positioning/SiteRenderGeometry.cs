using UnityEngine;
using UnityEngine.Rendering;

namespace CsiApp
{
    // Visual markers do not need physics. Avoid CreatePrimitive's implicit collider,
    // which can be stripped from a Player even when the visual sphere is requested.
    public sealed class SiteRenderGeometry : MonoBehaviour
    {
        private Mesh ownedMesh;

        // Same unit diameter as Unity's sphere; existing marker scales remain unchanged.
        public static GameObject CreateSphere(string objectName)
        {
            var result = new GameObject(objectName, typeof(MeshFilter), typeof(MeshRenderer), typeof(SiteRenderGeometry));
            var owner = result.GetComponent<SiteRenderGeometry>();
            owner.ownedMesh = MakeUnitSphere();
            result.GetComponent<MeshFilter>().sharedMesh = owner.ownedMesh;
            var renderer = result.GetComponent<MeshRenderer>();
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            return result;
        }

        private static Mesh MakeUnitSphere()
        {
            const int latitude = 8, longitude = 16;
            var vertices = new Vector3[(latitude + 1) * (longitude + 1)];
            var normals = new Vector3[vertices.Length];
            var uv = new Vector2[vertices.Length];
            for (int row = 0; row <= latitude; row++)
            {
                float polar = Mathf.PI * row / latitude;
                for (int column = 0; column <= longitude; column++)
                {
                    float azimuth = 2 * Mathf.PI * column / longitude;
                    int index = row * (longitude + 1) + column;
                    normals[index] = new Vector3(Mathf.Sin(polar) * Mathf.Cos(azimuth), Mathf.Cos(polar),
                        Mathf.Sin(polar) * Mathf.Sin(azimuth));
                    vertices[index] = normals[index] * .5f;
                    uv[index] = new Vector2((float)column / longitude, 1 - (float)row / latitude);
                }
            }
            var triangles = new int[longitude * (latitude - 1) * 6];
            int cursor = 0;
            for (int row = 0; row < latitude; row++)
                for (int column = 0; column < longitude; column++)
                {
                    int a = row * (longitude + 1) + column, b = a + longitude + 1;
                    if (row > 0)
                    { triangles[cursor++] = a; triangles[cursor++] = a + 1; triangles[cursor++] = b; }
                    if (row < latitude - 1)
                    { triangles[cursor++] = a + 1; triangles[cursor++] = b + 1; triangles[cursor++] = b; }
                }
            var mesh = new Mesh { name = "Site marker sphere / no collider" };
            mesh.vertices = vertices; mesh.normals = normals; mesh.uv = uv; mesh.triangles = triangles;
            mesh.RecalculateBounds();
            return mesh;
        }

        private void OnDestroy() { if (ownedMesh != null) Destroy(ownedMesh); }
    }
}
