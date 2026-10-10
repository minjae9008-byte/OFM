// Built-in render pipeline shader for OFM volumetric meshes.
// Vertex colour = rock stratum / ground colour, vertex alpha = sky exposure (used as occlusion).
// A small world-space value noise breaks up flat colours. For URP/HDRP build the same in Shader Graph
// (Vertex Color node -> Base Color, alpha -> Ambient Occlusion).
Shader "OFM/VertexColorLit"
{
    Properties
    {
        _Color ("Tint", Color) = (1, 1, 1, 1)
        _Glossiness ("Smoothness", Range(0, 1)) = 0.12
        _DetailScale ("Detail Scale", Float) = 0.35
        _DetailStrength ("Detail Strength", Range(0, 1)) = 0.35
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" }
        LOD 200

        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows vertex:vert addshadow
        #pragma target 3.0

        struct Input
        {
            float4 vcolor;
            float3 worldPos;
        };

        fixed4 _Color;
        half _Glossiness;
        float _DetailScale;
        float _DetailStrength;

        float hash3(float3 p)
        {
            return frac(sin(dot(p, float3(12.9898, 78.233, 37.719))) * 43758.5453);
        }

        float valueNoise(float3 p)
        {
            float3 i = floor(p);
            float3 f = frac(p);
            f = f * f * (3.0 - 2.0 * f);
            float n000 = hash3(i);
            float n100 = hash3(i + float3(1, 0, 0));
            float n010 = hash3(i + float3(0, 1, 0));
            float n110 = hash3(i + float3(1, 1, 0));
            float n001 = hash3(i + float3(0, 0, 1));
            float n101 = hash3(i + float3(1, 0, 1));
            float n011 = hash3(i + float3(0, 1, 1));
            float n111 = hash3(i + float3(1, 1, 1));
            float x00 = lerp(n000, n100, f.x);
            float x10 = lerp(n010, n110, f.x);
            float x01 = lerp(n001, n101, f.x);
            float x11 = lerp(n011, n111, f.x);
            return lerp(lerp(x00, x10, f.y), lerp(x01, x11, f.y), f.z);
        }

        void vert(inout appdata_full v, out Input o)
        {
            UNITY_INITIALIZE_OUTPUT(Input, o);
            o.vcolor = v.color;
        }

        void surf(Input IN, inout SurfaceOutputStandard o)
        {
            float n = valueNoise(IN.worldPos * _DetailScale) * 0.65 + valueNoise(IN.worldPos * _DetailScale * 4.1) * 0.35;
            float detail = lerp(1.0, 0.7 + 0.6 * n, _DetailStrength);
            o.Albedo = IN.vcolor.rgb * _Color.rgb * detail;
            o.Smoothness = _Glossiness;
            o.Metallic = 0;
            o.Occlusion = lerp(0.45, 1.0, IN.vcolor.a);
        }
        ENDCG
    }
    FallBack "Diffuse"
}
