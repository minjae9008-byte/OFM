using UnityEngine;

namespace OFM
{
    /// <summary>Root object of an imported OFM world (lets tools find and replace it).</summary>
    public class OFMWorldRoot : MonoBehaviour
    {
        public string worldName;
        public int seed;
        public string manifestPath;
        public float extent;
        public string meshMode;
    }
}
