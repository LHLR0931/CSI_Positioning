Shader "CSI/Apartment Presentation Surface"
{
    Properties
    {
        _Color ("Color", Color) = (0.8,0.8,0.8,1)
        _CutHeight ("Presentation cutaway height", Float) = 20
        _Glossiness ("Smoothness", Range(0,1)) = 0.15
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Cull Back
        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows addshadow
        #pragma target 3.0
        fixed4 _Color;
        float _CutHeight;
        half _Glossiness;
        struct Input { float3 worldPos; };
        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            clip(_CutHeight - IN.worldPos.y);
            o.Albedo = _Color.rgb;
            o.Metallic = 0;
            o.Smoothness = _Glossiness;
            o.Alpha = 1;
        }
        ENDCG
    }
    FallBack "Diffuse"
}
