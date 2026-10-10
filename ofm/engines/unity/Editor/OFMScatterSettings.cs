// Optional mapping from OFM scatter types (conifer, broadleaf, bush, boulder, ...) to project prefabs.
// Create via Assets > Create > OFM > Scatter Settings. Without it the importer spawns placeholders.
using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace OFM.EditorTools
{
    [CreateAssetMenu(menuName = "OFM/Scatter Settings", fileName = "OFMScatterSettings")]
    public class OFMScatterSettings : ScriptableObject
    {
        [Serializable]
        public class Entry
        {
            public string type = "conifer";
            public GameObject prefab;
        }

        [Tooltip("Place mapped prefabs as Terrain trees (fast, GPU instanced) instead of GameObjects.")]
        public bool useTerrainTrees = true;

        public List<Entry> prefabs = new List<Entry>();

        public GameObject PrefabFor(string type)
        {
            foreach (var e in prefabs)
                if (e != null && e.type == type && e.prefab != null) return e.prefab;
            return null;
        }

        public static OFMScatterSettings Load()
        {
            string[] guids = AssetDatabase.FindAssets("t:OFMScatterSettings");
            if (guids.Length == 0) return null;
            return AssetDatabase.LoadAssetAtPath<OFMScatterSettings>(AssetDatabase.GUIDToAssetPath(guids[0]));
        }
    }
}
