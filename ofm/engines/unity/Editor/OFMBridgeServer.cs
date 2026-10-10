// Local TCP bridge so external OFM tooling (python -m ofm push unity) can drive the Unity Editor.
//
// Protocol: one JSON object per line, one response line per request, loopback only.
//   {"cmd":"ping"}
//   {"cmd":"import","manifest":"/abs/path/world.json","scatter":true,"maxScatterPerType":3000}
//   {"cmd":"clear"}                 removes every imported OFM world from the open scene
//   {"cmd":"save"}                  saves the open scenes
//   {"cmd":"frame","x":..,"y":..,"z":..}   moves the Scene view camera to a Unity-space point
// Commands run on the main thread (EditorApplication.update); the listener thread only queues them.
using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace OFM.EditorTools
{
    [InitializeOnLoad]
    public static class OFMBridgeServer
    {
        const string PortKey = "OFM.Bridge.Port";
        const string AutoKey = "OFM.Bridge.AutoStart";

        class Job
        {
            public string request;
            public TcpClient client;
        }

        [Serializable]
        class Request
        {
            public string cmd = "";
            public string manifest = "";
            public bool terrain = true;
            public bool meshes = true;
            public bool water = true;
            public bool markers = true;
            public bool scatter = true;
            public int maxScatterPerType = 3000;
            public float x, y, z;
        }

        [Serializable]
        class Response
        {
            public bool ok;
            public string message;
            public string unity;
        }

        static TcpListener listener;
        static Thread thread;
        static volatile bool running;
        static readonly ConcurrentQueue<Job> jobs = new ConcurrentQueue<Job>();

        public static int Port
        {
            get { return EditorPrefs.GetInt(PortKey, 8765); }
            set { EditorPrefs.SetInt(PortKey, value); }
        }

        static OFMBridgeServer()
        {
            EditorApplication.update += Pump;
            AssemblyReloadEvents.beforeAssemblyReload += Stop;
            EditorApplication.quitting += Stop;
            if (EditorPrefs.GetBool(AutoKey, true) && !Application.isBatchMode) Start();
        }

        [MenuItem("Tools/OFM/Bridge/Start", priority = 20)]
        public static void Start()
        {
            if (running) return;
            try
            {
                listener = new TcpListener(IPAddress.Loopback, Port);
                listener.Start();
                running = true;
                thread = new Thread(Loop) { IsBackground = true, Name = "OFM Bridge" };
                thread.Start();
                EditorPrefs.SetBool(AutoKey, true);
                Debug.Log($"[OFM] bridge listening on 127.0.0.1:{Port}");
            }
            catch (Exception e)
            {
                running = false;
                Debug.LogWarning($"[OFM] bridge could not start on port {Port}: {e.Message}");
            }
        }

        [MenuItem("Tools/OFM/Bridge/Stop", priority = 21)]
        public static void StopMenu()
        {
            EditorPrefs.SetBool(AutoKey, false);
            Stop();
        }

        static void Stop()
        {
            running = false;
            try { listener?.Stop(); } catch (Exception) { }
            listener = null;
        }

        static void Loop()
        {
            while (running)
            {
                try
                {
                    var client = listener.AcceptTcpClient();
                    client.ReceiveTimeout = 10000;
                    var reader = new StreamReader(client.GetStream(), new UTF8Encoding(false));
                    string line = reader.ReadLine();
                    if (string.IsNullOrEmpty(line)) { client.Close(); continue; }
                    jobs.Enqueue(new Job { request = line, client = client });
                }
                catch (SocketException) { if (!running) break; }
                catch (ObjectDisposedException) { break; }
                catch (InvalidOperationException) { break; }
                catch (IOException) { }
            }
        }

        static void Pump()
        {
            Job job;
            while (jobs.TryDequeue(out job))
            {
                Response resp;
                try { resp = Handle(job.request); }
                catch (Exception e)
                {
                    Debug.LogException(e);
                    resp = new Response { ok = false, message = e.Message };
                }
                finally { EditorUtility.ClearProgressBar(); }
                resp.unity = Application.unityVersion;
                try
                {
                    byte[] bytes = new UTF8Encoding(false).GetBytes(JsonUtility.ToJson(resp) + "\n");
                    var stream = job.client.GetStream();
                    stream.Write(bytes, 0, bytes.Length);
                    stream.Flush();
                }
                catch (Exception) { }
                finally { job.client.Close(); }
            }
        }

        static Response Handle(string json)
        {
            var req = new Request();
            JsonUtility.FromJsonOverwrite(json, req);
            switch (req.cmd)
            {
                case "ping":
                    return new Response { ok = true, message = "pong" };
                case "import":
                {
                    if (!File.Exists(req.manifest)) return new Response { ok = false, message = $"manifest not found: {req.manifest}" };
                    var opt = new OFMImportOptions
                    {
                        terrain = req.terrain, meshes = req.meshes, water = req.water,
                        markers = req.markers, scatter = req.scatter, maxScatterPerType = req.maxScatterPerType,
                    };
                    var root = OFMWorldImporter.Import(req.manifest, opt);
                    Selection.activeGameObject = root;
                    return new Response { ok = true, message = root.name };
                }
                case "clear":
                {
                    int n = 0;
                    foreach (var r in UnityEngine.Object.FindObjectsOfType<OFM.OFMWorldRoot>())
                    {
                        UnityEngine.Object.DestroyImmediate(r.gameObject);
                        n++;
                    }
                    return new Response { ok = true, message = $"removed {n} world(s)" };
                }
                case "save":
                    EditorSceneManager.SaveOpenScenes();
                    return new Response { ok = true, message = "saved" };
                case "frame":
                {
                    var view = SceneView.lastActiveSceneView;
                    if (view == null) return new Response { ok = false, message = "no scene view open" };
                    view.LookAt(new Vector3(req.x, req.y, req.z), view.rotation, 120f);
                    return new Response { ok = true, message = "framed" };
                }
                default:
                    return new Response { ok = false, message = $"unknown command '{req.cmd}'" };
            }
        }
    }
}
