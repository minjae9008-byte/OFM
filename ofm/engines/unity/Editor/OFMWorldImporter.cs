// OFM world importer for the Unity Editor (2021.3+; Built-in, URP or HDRP).
//
// Reads an OFM world.json package and builds:
//   * a Terrain (heights, terrain holes, splat layers) for ordinary ground,
//   * volumetric meshes (caves, overhangs, arches, spires) from .ofmmesh chunks,
//   * lakes, river ribbons and an ocean plane,
//   * POI / landmark / cave markers (OFMMarker components),
//   * vegetation and rock scatter (terrain trees when prefabs are mapped, placeholders otherwise).
//
// Menu: Tools > OFM > Import World...
// Batch: Unity -batchmode -projectPath <proj> -executeMethod OFM.EditorTools.OFMWorldImporter.ImportFromCommandLine
//        -ofmManifest <path/to/world.json> -quit
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

namespace OFM.EditorTools
{
    [Serializable] public class OFMLayer { public string name; public int[] color; }

    [Serializable]
    public class OFMTerrain
    {
        public int heightmap_resolution;
        public string heightmap_raw16;
        public float height_min;
        public float height_max;
        public float size_m;
        public int holes_resolution;
        public string holes_raw8;
        public int splat_resolution;
        public string splat_raw;
        public OFMLayer[] layers;
    }

    [Serializable]
    public class OFMMeshEntry
    {
        public string name;
        public string[] kinds;
        public float[] origin_unity;
        public string unity;
        public int triangles;
    }

    [Serializable]
    public class OFMLake
    {
        public string id;
        public string name;
        public string description;
        public float level;
        public bool saline;
        public float[] origin_unity;
        public string unity;
    }

    [Serializable]
    public class OFMRiver
    {
        public string id;
        public string name;
        public float[] origin_unity;
        public string unity;
        public float[] points;
        public float[] widths;
    }

    [Serializable] public class OFMOcean { public bool enabled; public float level; }

    [Serializable]
    public class OFMWater
    {
        public OFMOcean ocean;
        public OFMLake[] lakes;
        public OFMRiver[] rivers;
    }

    [Serializable]
    public class OFMPoi
    {
        public string id;
        public string kind;
        public string name;
        public string description;
        public string region;
        public string source;
        public float importance;
        public float[] position_unity;
    }

    [Serializable]
    public class OFMLandmark
    {
        public string id;
        public string kind;
        public string name;
        public string description;
        public float height;
        public float salience;
        public float[] position_unity;
    }

    [Serializable] public class OFMEntrance { public string kind; public float[] position_unity; }

    [Serializable]
    public class OFMCave
    {
        public string id;
        public string kind;
        public string name;
        public string description;
        public float length_m;
        public OFMEntrance[] entrances;
    }

    [Serializable] public class OFMScatter { public string type; public int count; public string file; }

    [Serializable]
    public class OFMManifest
    {
        public string format;
        public int version;
        public string name;
        public int seed;
        public float extent_m;
        public float sea_level;
        public string mesh_mode;
        public OFMTerrain terrain;
        public OFMMeshEntry[] meshes;
        public OFMWater water;
        public OFMPoi[] pois;
        public OFMLandmark[] landmarks;
        public OFMCave[] caves;
        public OFMScatter[] scatter;
    }

    [Serializable]
    public class OFMImportOptions
    {
        public bool terrain = true;
        public bool meshes = true;
        public bool water = true;
        public bool markers = true;
        public bool scatter = true;
        public int maxScatterPerType = 3000;
        public string assetRoot = "Assets/OFM_Worlds";
        public bool replaceExisting = true;
    }

    public static class OFMWorldImporter
    {
        [MenuItem("Tools/OFM/Import World...", priority = 1)]
        static void ImportMenu()
        {
            string path = EditorUtility.OpenFilePanel("Select OFM world.json", "", "json");
            if (string.IsNullOrEmpty(path)) return;
            try
            {
                var root = Import(path, new OFMImportOptions());
                Selection.activeGameObject = root;
            }
            catch (Exception e)
            {
                Debug.LogException(e);
                EditorUtility.DisplayDialog("OFM import failed", e.Message, "OK");
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }
        }

        public static void ImportFromCommandLine()
        {
            string[] args = Environment.GetCommandLineArgs();
            string manifest = null;
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == "-ofmManifest") manifest = args[i + 1];
            if (manifest == null) throw new ArgumentException("missing -ofmManifest <path>");
            var root = Import(manifest, new OFMImportOptions());
            var scene = EditorSceneManager.GetActiveScene();
            string scenePath = string.IsNullOrEmpty(scene.path)
                ? $"{new OFMImportOptions().assetRoot}/{root.name.Replace(' ', '_')}.unity"
                : scene.path;
            EnsureFolder(Path.GetDirectoryName(scenePath).Replace('\\', '/'));
            EditorSceneManager.SaveScene(scene, scenePath);
            Debug.Log($"[OFM] imported {manifest} into {scenePath}");
        }

        public static GameObject Import(string manifestPath, OFMImportOptions opt)
        {
            manifestPath = Path.GetFullPath(manifestPath);
            var m = new OFMManifest();
            JsonUtility.FromJsonOverwrite(File.ReadAllText(manifestPath), m);
            if (m.format != "ofm-world") throw new Exception($"{manifestPath} is not an OFM world manifest");
            string dir = Path.GetDirectoryName(manifestPath);
            string assetDir = $"{opt.assetRoot}/{Sanitize(m.name)}_{m.seed}";
            EnsureFolder(assetDir);
            EnsureFolder(assetDir + "/Meshes");
            EnsureFolder(assetDir + "/Water");
            EnsureFolder(assetDir + "/Materials");

            string rootName = $"OFM {m.name} ({m.seed})";
            if (opt.replaceExisting)
            {
                var old = GameObject.Find(rootName);
                if (old != null) UnityEngine.Object.DestroyImmediate(old);
            }
            var root = new GameObject(rootName);
            var info = root.AddComponent<OFM.OFMWorldRoot>();
            info.worldName = m.name;
            info.seed = m.seed;
            info.manifestPath = manifestPath;
            info.extent = m.extent_m;
            info.meshMode = m.mesh_mode;
            Undo.RegisterCreatedObjectUndo(root, "Import OFM world");

            var rockMat = GetOrCreateMaterial($"{assetDir}/Materials/OFM_Rock.mat", "OFM/VertexColorLit", new Color(0.62f, 0.6f, 0.56f), false);
            var waterMat = GetOrCreateMaterial($"{assetDir}/Materials/OFM_Water.mat", "OFM/Water", new Color(0.2f, 0.42f, 0.6f, 0.75f), true);

            if (opt.terrain && m.terrain != null && m.mesh_mode != "full")
                BuildTerrain(m, dir, assetDir, root.transform);
            if (opt.meshes && m.meshes != null)
                BuildMeshes(m, dir, assetDir, root.transform, rockMat);
            if (opt.water && m.water != null)
                BuildWater(m, dir, assetDir, root.transform, waterMat);
            if (opt.markers)
                BuildMarkers(m, root.transform);
            if (opt.scatter && m.scatter != null)
                BuildScatter(m, dir, root.transform, opt);

            EditorUtility.ClearProgressBar();
            AssetDatabase.SaveAssets();
            EditorSceneManager.MarkSceneDirty(root.scene);
            Debug.Log($"[OFM] world '{m.name}' imported ({(m.meshes != null ? m.meshes.Length : 0)} volume chunks)");
            return root;
        }

        // ------------------------------------------------------------------ terrain
        static void BuildTerrain(OFMManifest m, string dir, string assetDir, Transform parent)
        {
            var t = m.terrain;
            int res = t.heightmap_resolution;
            EditorUtility.DisplayProgressBar("OFM", "Terrain heights", 0.05f);
            var td = new TerrainData();
            td.heightmapResolution = res;
            td.size = new Vector3(t.size_m, Mathf.Max(1f, t.height_max - t.height_min), t.size_m);

            byte[] raw = File.ReadAllBytes(Path.Combine(dir, t.heightmap_raw16));
            if (raw.Length != res * res * 2) throw new Exception("heightmap size mismatch");
            var heights = new float[res, res];
            for (int y = 0; y < res; y++)
                for (int x = 0; x < res; x++)
                {
                    int i = (y * res + x) * 2;
                    heights[y, x] = (raw[i] | (raw[i + 1] << 8)) / 65535f;
                }
            td.SetHeights(0, 0, heights);

            if (!string.IsNullOrEmpty(t.holes_raw8))
            {
                byte[] hb = File.ReadAllBytes(Path.Combine(dir, t.holes_raw8));
                int hr = td.holesResolution;
                if (hb.Length == hr * hr)
                {
                    var surface = new bool[hr, hr];
                    for (int y = 0; y < hr; y++)
                        for (int x = 0; x < hr; x++)
                            surface[y, x] = hb[y * hr + x] == 0;   // Unity: true = terrain present
                    td.SetHoles(0, 0, surface);
                }
                else Debug.LogWarning($"[OFM] holes resolution {Mathf.Sqrt(hb.Length)} != terrain {hr}, skipped");
            }

            EditorUtility.DisplayProgressBar("OFM", "Terrain layers", 0.15f);
            int nl = t.layers.Length;
            var layers = new TerrainLayer[nl];
            for (int k = 0; k < nl; k++)
            {
                string lpath = $"{assetDir}/Materials/Layer_{t.layers[k].name}.terrainlayer";
                var layer = AssetDatabase.LoadAssetAtPath<TerrainLayer>(lpath);
                if (layer == null)
                {
                    var c = t.layers[k].color;
                    var tex = MakeNoiseTexture(new Color32((byte)c[0], (byte)c[1], (byte)c[2], 255), k);
                    AssetDatabase.CreateAsset(tex, $"{assetDir}/Materials/Layer_{t.layers[k].name}_albedo.asset");
                    layer = new TerrainLayer { diffuseTexture = tex, tileSize = new Vector2(6f, 6f) };
                    AssetDatabase.CreateAsset(layer, lpath);
                }
                layers[k] = layer;
            }
            td.terrainLayers = layers;

            int sr = t.splat_resolution;
            td.alphamapResolution = sr;
            byte[] sb = File.ReadAllBytes(Path.Combine(dir, t.splat_raw));
            if (sb.Length == sr * sr * nl && td.alphamapResolution == sr)
            {
                var maps = new float[sr, sr, nl];
                for (int y = 0; y < sr; y++)
                    for (int x = 0; x < sr; x++)
                        for (int l = 0; l < nl; l++)
                            maps[y, x, l] = sb[(y * sr + x) * nl + l] / 255f;
                td.SetAlphamaps(0, 0, maps);
            }
            else Debug.LogWarning("[OFM] splat map size mismatch, skipped");

            string tdPath = $"{assetDir}/TerrainData.asset";
            AssetDatabase.DeleteAsset(tdPath);
            AssetDatabase.CreateAsset(td, tdPath);
            var go = Terrain.CreateTerrainGameObject(td);
            go.name = "Terrain";
            go.transform.SetParent(parent, false);
            go.transform.position = new Vector3(0f, t.height_min, 0f);
            var terrain = go.GetComponent<Terrain>();
            terrain.heightmapPixelError = 3f;
            terrain.basemapDistance = 1500f;
        }

        static Texture2D MakeNoiseTexture(Color32 baseColor, int seed)
        {
            const int n = 64;
            var tex = new Texture2D(n, n, TextureFormat.RGBA32, true) { wrapMode = TextureWrapMode.Repeat };
            var rnd = new System.Random(1234 + seed);
            var px = new Color32[n * n];
            for (int i = 0; i < px.Length; i++)
            {
                float f = 0.82f + 0.36f * (float)rnd.NextDouble();
                px[i] = new Color32((byte)Mathf.Clamp(baseColor.r * f, 0, 255), (byte)Mathf.Clamp(baseColor.g * f, 0, 255),
                                    (byte)Mathf.Clamp(baseColor.b * f, 0, 255), 255);
            }
            tex.SetPixels32(px);
            tex.Apply(true);
            return tex;
        }

        // ------------------------------------------------------------------ meshes
        static void BuildMeshes(OFMManifest m, string dir, string assetDir, Transform parent, Material mat)
        {
            var group = new GameObject("Volume").transform;
            group.SetParent(parent, false);
            for (int i = 0; i < m.meshes.Length; i++)
            {
                var e = m.meshes[i];
                if (i % 8 == 0)
                    EditorUtility.DisplayProgressBar("OFM", $"Volume meshes ({i}/{m.meshes.Length})", 0.2f + 0.5f * i / Mathf.Max(1, m.meshes.Length));
                Vector3 origin;
                var mesh = OFMMeshReader.Read(Path.Combine(dir, e.unity), out origin);
                mesh.name = e.name;
                string mpath = $"{assetDir}/Meshes/{e.name}.asset";
                AssetDatabase.DeleteAsset(mpath);
                AssetDatabase.CreateAsset(mesh, mpath);
                var go = new GameObject(e.name + (e.kinds != null && e.kinds.Length > 0 ? " [" + string.Join(",", e.kinds) + "]" : ""));
                go.transform.SetParent(group, false);
                go.transform.position = origin;
                go.AddComponent<MeshFilter>().sharedMesh = mesh;
                go.AddComponent<MeshRenderer>().sharedMaterial = mat;
                go.AddComponent<MeshCollider>().sharedMesh = mesh;
                GameObjectUtility.SetStaticEditorFlags(go, StaticEditorFlags.BatchingStatic | StaticEditorFlags.ContributeGI | StaticEditorFlags.OccluderStatic);
            }
        }

        // ------------------------------------------------------------------ water
        static void BuildWater(OFMManifest m, string dir, string assetDir, Transform parent, Material mat)
        {
            EditorUtility.DisplayProgressBar("OFM", "Water", 0.75f);
            var group = new GameObject("Water").transform;
            group.SetParent(parent, false);
            if (m.water.ocean != null && m.water.ocean.enabled)
            {
                var ocean = GameObject.CreatePrimitive(PrimitiveType.Plane);
                ocean.name = "Ocean";
                UnityEngine.Object.DestroyImmediate(ocean.GetComponent<Collider>());
                ocean.transform.SetParent(group, false);
                ocean.transform.position = new Vector3(m.extent_m * 0.5f, m.water.ocean.level, m.extent_m * 0.5f);
                float s = m.extent_m * 3f / 10f;   // the built-in plane is 10 x 10 units
                ocean.transform.localScale = new Vector3(s, 1f, s);
                ocean.GetComponent<MeshRenderer>().sharedMaterial = mat;
            }
            if (m.water.lakes != null)
                foreach (var l in m.water.lakes) AddWaterMesh(dir, assetDir, group, mat, l.id, l.name, l.unity);
            if (m.water.rivers != null)
                foreach (var r in m.water.rivers) AddWaterMesh(dir, assetDir, group, mat, r.id, r.name, r.unity);
        }

        static void AddWaterMesh(string dir, string assetDir, Transform group, Material mat, string id, string label, string file)
        {
            Vector3 origin;
            var mesh = OFMMeshReader.Read(Path.Combine(dir, file), out origin);
            mesh.name = id;
            string p = $"{assetDir}/Water/{id}.asset";
            AssetDatabase.DeleteAsset(p);
            AssetDatabase.CreateAsset(mesh, p);
            var go = new GameObject(string.IsNullOrEmpty(label) ? id : $"{label} ({id})");
            go.transform.SetParent(group, false);
            go.transform.position = origin;
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            go.AddComponent<MeshRenderer>().sharedMaterial = mat;
        }

        // ------------------------------------------------------------------ markers
        static void BuildMarkers(OFMManifest m, Transform parent)
        {
            EditorUtility.DisplayProgressBar("OFM", "Markers", 0.8f);
            if (m.pois != null)
            {
                var g = MakeGroup(parent, "POIs");
                foreach (var p in m.pois)
                    Marker(g, p.name, p.id, "poi", p.kind, p.description, p.importance, p.region, p.source, p.position_unity);
            }
            if (m.landmarks != null)
            {
                var g = MakeGroup(parent, "Landmarks");
                foreach (var l in m.landmarks)
                    Marker(g, l.name, l.id, "landmark", l.kind, l.description, l.salience, "", "simulation", l.position_unity);
            }
            if (m.caves != null)
            {
                var g = MakeGroup(parent, "Caves");
                foreach (var c in m.caves)
                    if (c.entrances != null)
                        for (int i = 0; i < c.entrances.Length; i++)
                            Marker(g, $"{(string.IsNullOrEmpty(c.name) ? c.id : c.name)} #{i + 1}", c.id, "cave",
                                   c.entrances[i].kind, c.description, 0.9f, "", c.kind, c.entrances[i].position_unity);
            }
        }

        static Transform MakeGroup(Transform parent, string n)
        {
            var g = new GameObject(n).transform;
            g.SetParent(parent, false);
            return g;
        }

        static void Marker(Transform parent, string label, string id, string category, string kind, string desc,
                           float importance, string region, string source, float[] pos)
        {
            if (pos == null || pos.Length < 3) return;
            var go = new GameObject(string.IsNullOrEmpty(label) ? id : label);
            go.transform.SetParent(parent, false);
            go.transform.position = new Vector3(pos[0], pos[1], pos[2]);
            var mk = go.AddComponent<OFM.OFMMarker>();
            mk.markerId = id;
            mk.category = category;
            mk.kind = kind;
            mk.description = desc;
            mk.importance = importance;
            mk.region = region;
            mk.source = source;
        }

        // ------------------------------------------------------------------ scatter
        static void BuildScatter(OFMManifest m, string dir, Transform parent, OFMImportOptions opt)
        {
            var settings = OFMScatterSettings.Load();
            var group = new GameObject("Scatter").transform;
            group.SetParent(parent, false);
            var terrain = parent.GetComponentInChildren<Terrain>();
            var treeProtos = new List<TreePrototype>();
            var trees = new List<TreeInstance>();
            int k = 0;
            foreach (var s in m.scatter)
            {
                EditorUtility.DisplayProgressBar("OFM", $"Scatter: {s.type}", 0.85f + 0.1f * k++ / Mathf.Max(1, m.scatter.Length));
                byte[] b = File.ReadAllBytes(Path.Combine(dir, s.file));
                int n = b.Length / 20;
                var f = new float[n * 5];
                Buffer.BlockCopy(b, 0, f, 0, n * 20);
                var prefab = settings != null ? settings.PrefabFor(s.type) : null;
                if (prefab != null && terrain != null && settings.useTerrainTrees)
                {
                    int proto = treeProtos.Count;
                    treeProtos.Add(new TreePrototype { prefab = prefab });
                    var size = terrain.terrainData.size;
                    var tpos = terrain.transform.position;
                    for (int i = 0; i < n; i++)
                    {
                        var wp = new Vector3(f[i * 5], f[i * 5 + 2], f[i * 5 + 1]);
                        trees.Add(new TreeInstance
                        {
                            prototypeIndex = proto,
                            position = new Vector3(wp.x / size.x, (wp.y - tpos.y) / size.y, wp.z / size.z),
                            rotation = (90f - f[i * 5 + 3]) * Mathf.Deg2Rad,
                            widthScale = f[i * 5 + 4],
                            heightScale = f[i * 5 + 4],
                            color = Color.white,
                            lightmapColor = Color.white,
                        });
                    }
                    continue;
                }
                var tg = new GameObject(s.type).transform;
                tg.SetParent(group, false);
                int stride = Mathf.Max(1, Mathf.CeilToInt(n / (float)Mathf.Max(1, opt.maxScatterPerType)));
                for (int i = 0; i < n; i += stride)
                {
                    var wp = new Vector3(f[i * 5], f[i * 5 + 2], f[i * 5 + 1]);
                    GameObject go = prefab != null ? (GameObject)PrefabUtility.InstantiatePrefab(prefab) : Placeholder(s.type);
                    go.transform.SetParent(tg, false);
                    go.transform.position = wp;
                    go.transform.rotation = Quaternion.Euler(0f, 90f - f[i * 5 + 3], 0f);
                    go.transform.localScale *= f[i * 5 + 4];
                }
            }
            if (terrain != null && treeProtos.Count > 0)
            {
                terrain.terrainData.treePrototypes = treeProtos.ToArray();
                terrain.terrainData.SetTreeInstances(trees.ToArray(), true);
            }
        }

        static readonly Dictionary<string, Material> placeholderMats = new Dictionary<string, Material>();

        static GameObject Placeholder(string type)
        {
            PrimitiveType prim = PrimitiveType.Sphere;
            Vector3 scale = Vector3.one * 2f;
            Color col = new Color(0.25f, 0.45f, 0.2f);
            switch (type)
            {
                case "conifer": prim = PrimitiveType.Capsule; scale = new Vector3(2.5f, 7f, 2.5f); col = new Color(0.15f, 0.32f, 0.2f); break;
                case "broadleaf": prim = PrimitiveType.Sphere; scale = new Vector3(6f, 6f, 6f); col = new Color(0.25f, 0.45f, 0.18f); break;
                case "palm": prim = PrimitiveType.Capsule; scale = new Vector3(1.5f, 6f, 1.5f); col = new Color(0.4f, 0.5f, 0.2f); break;
                case "bush": prim = PrimitiveType.Sphere; scale = new Vector3(2f, 1.4f, 2f); col = new Color(0.3f, 0.42f, 0.2f); break;
                case "boulder": prim = PrimitiveType.Sphere; scale = new Vector3(3f, 2f, 2.6f); col = new Color(0.45f, 0.44f, 0.42f); break;
                case "cactus": prim = PrimitiveType.Capsule; scale = new Vector3(0.8f, 2.5f, 0.8f); col = new Color(0.35f, 0.5f, 0.3f); break;
                case "reed": prim = PrimitiveType.Cylinder; scale = new Vector3(0.6f, 1f, 0.6f); col = new Color(0.5f, 0.55f, 0.3f); break;
                case "dead_tree": prim = PrimitiveType.Cylinder; scale = new Vector3(0.6f, 4f, 0.6f); col = new Color(0.35f, 0.3f, 0.25f); break;
            }
            var go = GameObject.CreatePrimitive(prim);
            go.name = type;
            UnityEngine.Object.DestroyImmediate(go.GetComponent<Collider>());
            go.transform.localScale = scale;
            Material mat;
            if (!placeholderMats.TryGetValue(type, out mat) || mat == null)
            {
                mat = new Material(DefaultLitShader()) { color = col };
                placeholderMats[type] = mat;
            }
            go.GetComponent<MeshRenderer>().sharedMaterial = mat;
            return go;
        }

        // ------------------------------------------------------------------ helpers
        static Shader DefaultLitShader()
        {
            var rp = GraphicsSettings.currentRenderPipeline;
            if (rp != null && rp.defaultMaterial != null) return rp.defaultMaterial.shader;
            return Shader.Find("Standard");
        }

        static Material GetOrCreateMaterial(string path, string shaderName, Color color, bool transparent)
        {
            var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (mat != null) return mat;
            var shader = Shader.Find(shaderName);
            bool srp = GraphicsSettings.currentRenderPipeline != null;
            if (shader == null || !shader.isSupported || srp)
            {
                // Scriptable pipelines: use the pipeline's lit shader (vertex colours need a Shader Graph, see README).
                shader = DefaultLitShader();
            }
            mat = new Material(shader) { color = color };
            if (transparent && srp)
            {
                if (mat.HasProperty("_Surface")) mat.SetFloat("_Surface", 1f);        // URP: transparent
                if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", color);
                mat.renderQueue = (int)RenderQueue.Transparent;
            }
            AssetDatabase.CreateAsset(mat, path);
            return mat;
        }

        static string Sanitize(string s)
        {
            foreach (char c in Path.GetInvalidFileNameChars()) s = s.Replace(c, '_');
            return s.Replace(' ', '_');
        }

        public static void EnsureFolder(string path)
        {
            path = path.Replace('\\', '/').TrimEnd('/');
            if (AssetDatabase.IsValidFolder(path)) return;
            string parent = Path.GetDirectoryName(path).Replace('\\', '/');
            if (!AssetDatabase.IsValidFolder(parent)) EnsureFolder(parent);
            AssetDatabase.CreateFolder(parent, Path.GetFileName(path));
        }
    }

    /// <summary>Reader for the .ofmmesh format written by ofm/export/meshio.py.</summary>
    public static class OFMMeshReader
    {
        public static Mesh Read(string path, out Vector3 origin)
        {
            using (var br = new BinaryReader(File.OpenRead(path)))
            {
                byte[] magic = br.ReadBytes(4);
                if (magic[0] != 'O' || magic[1] != 'F' || magic[2] != 'M' || magic[3] != 'M')
                    throw new Exception($"{path}: not an .ofmmesh file");
                uint version = br.ReadUInt32();
                if (version != 1) throw new Exception($"{path}: unsupported version {version}");
                int nv = (int)br.ReadUInt32();
                int ni = (int)br.ReadUInt32();
                uint flags = br.ReadUInt32();
                origin = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());

                var pf = new float[nv * 3];
                Buffer.BlockCopy(br.ReadBytes(nv * 12), 0, pf, 0, nv * 12);
                var nf = new float[nv * 3];
                Buffer.BlockCopy(br.ReadBytes(nv * 12), 0, nf, 0, nv * 12);
                var pos = new Vector3[nv];
                var nrm = new Vector3[nv];
                for (int i = 0; i < nv; i++)
                {
                    pos[i] = new Vector3(pf[i * 3], pf[i * 3 + 1], pf[i * 3 + 2]);
                    nrm[i] = new Vector3(nf[i * 3], nf[i * 3 + 1], nf[i * 3 + 2]);
                }
                Color32[] cols = null;
                if ((flags & 1) != 0)
                {
                    byte[] cb = br.ReadBytes(nv * 4);
                    cols = new Color32[nv];
                    for (int i = 0; i < nv; i++) cols[i] = new Color32(cb[i * 4], cb[i * 4 + 1], cb[i * 4 + 2], cb[i * 4 + 3]);
                }
                Vector2[] uv = null;
                if ((flags & 2) != 0)
                {
                    var uf = new float[nv * 2];
                    Buffer.BlockCopy(br.ReadBytes(nv * 8), 0, uf, 0, nv * 8);
                    uv = new Vector2[nv];
                    for (int i = 0; i < nv; i++) uv[i] = new Vector2(uf[i * 2], uf[i * 2 + 1]);
                }
                var idx = new int[ni];
                Buffer.BlockCopy(br.ReadBytes(ni * 4), 0, idx, 0, ni * 4);

                var mesh = new Mesh { indexFormat = nv > 65000 ? IndexFormat.UInt32 : IndexFormat.UInt16 };
                mesh.vertices = pos;
                mesh.normals = nrm;
                if (cols != null) mesh.colors32 = cols;
                if (uv != null) mesh.uv = uv;
                mesh.triangles = idx;
                mesh.RecalculateBounds();
                return mesh;
            }
        }
    }
}
