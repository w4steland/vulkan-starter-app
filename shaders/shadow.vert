#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

// Set 0: Scene
layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;
    mat4 view;
    mat4 proj;
    mat4 shadowViewProj; // Матрица вида-проекции для источника света (Солнца)
    vec4 cameraPos;
    vec4 ambientLight;
    vec4 dirLightDir;
    vec4 dirLightColor;
    vec4 shadowParams;   // x = depthBiasConstant, y = depthBiasSlope, z = orthoSize, w = shaderBias
    int numPointLights;
    int numSpotLights;
    int pad0;
    int pad1;
} scene;

// Set 1: Material / Model (Dynamic Uniform Buffer)
layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specular;
} modelData;

void main() {
    gl_Position = scene.shadowViewProj * modelData.model * vec4(inPosition, 1.0);
}
