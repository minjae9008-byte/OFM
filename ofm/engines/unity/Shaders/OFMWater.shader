// Minimal transparent water for the built-in render pipeline (lakes, rivers, ocean plane).
Shader "OFM/Water"
{
    Properties
    {
        _Color ("Color", Color) = (0.2, 0.42, 0.6, 0.75)
        _Glossiness ("Smoothness", Range(0, 1)) = 0.92
    }
    SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType" = "Transparent" }
        LOD 200

        CGPROGRAM
        #pragma surface surf Standard alpha:fade
        #pragma target 3.0

        struct Input
        {
            float3 worldPos;
        };

        fixed4 _Color;
        half _Glossiness;

        void surf(Input IN, inout SurfaceOutputStandard o)
        {
            float ripple = sin(IN.worldPos.x * 0.21 + _Time.y * 1.3) * sin(IN.worldPos.z * 0.17 + _Time.y * 0.9);
            o.Albedo = _Color.rgb * (0.95 + 0.05 * ripple);
            o.Alpha = _Color.a;
            o.Smoothness = _Glossiness;
            o.Metallic = 0;
        }
        ENDCG
    }
    FallBack "Transparent/Diffuse"
}
