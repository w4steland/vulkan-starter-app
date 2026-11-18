#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragWorldNormal;
layout(location = 2) out vec2 fragUV;
layout(location = 3) out vec4 fragShadowPos;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;
    mat4 view;
    mat4 proj;
    mat4 shadowViewProj; // Матрица источника света для теней
    vec4 cameraPos;
    vec4 ambientLight;   // rgb = color, w = intensity
    vec4 dirLightDir;    // xyz = direction, w = intensity
    vec4 dirLightColor;  // rgb = color, w = enabled (1.0 or 0.0)
    vec4 shadowParams;   // x = depthBiasConstant, y = depthBiasSlope, z = orthoSize, w = shaderBias
    int numPointLights;
    int numSpotLights;
    int pad0;
    int pad1;
} scene;

// Set 1: Material / Model (Dynamic Uniform Buffer)
layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    mat4 normalMatrix; // mat4 for std140 alignment, upper 3x3 used for normals
    vec4 albedo;       // rgb = tint/base color, w = unused
    vec4 specular;     // rgb = specular color, w = shininess
} modelData;

void main() {
    vec4 worldPos = modelData.model * vec4(inPosition, 1.0);
    fragWorldPos = worldPos.xyz;
    fragWorldNormal = normalize(mat3(modelData.normalMatrix) * inNormal);
    fragUV = inUV;
    fragShadowPos = scene.shadowViewProj * worldPos;
    gl_Position = scene.viewProj * worldPos;
}
