using UnityEngine;

namespace OFM
{
    /// <summary>Point of interest, landmark or cave entrance placed by the OFM importer.</summary>
    public class OFMMarker : MonoBehaviour
    {
        public string markerId;
        public string category;   // poi | landmark | cave
        public string kind;       // waterfall, summit, ruin, spire, sinkhole ...
        [TextArea] public string description;
        public float importance;
        public string region;
        public string source;     // procedural | ai | simulation | karst | sea ...

        static Color ColorFor(string category)
        {
            switch (category)
            {
                case "landmark": return new Color(1f, 0.82f, 0.25f, 0.85f);
                case "cave": return new Color(0.15f, 0.15f, 0.18f, 0.9f);
                default: return new Color(0.92f, 0.25f, 0.25f, 0.85f);
            }
        }

        void OnDrawGizmos()
        {
            Gizmos.color = ColorFor(category);
            float r = 2f + 4f * Mathf.Clamp01(importance);
            Gizmos.DrawSphere(transform.position + Vector3.up * (r + 1f), r);
            Gizmos.DrawLine(transform.position, transform.position + Vector3.up * (r + 1f));
#if UNITY_EDITOR
            UnityEditor.Handles.Label(transform.position + Vector3.up * (2f * r + 4f), name);
#endif
        }
    }
}
