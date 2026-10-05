#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 uiColor;
    vec4 edgeColor;
    vec4 cornerColor;
} ubo;

layout(push_constant) uniform PushConsts {
    int drawMode; // 0 = грани, 1 = ребра, 2 = углы
} pc;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    gl_PointSize = 12.0; // размер точек для углов
    fragColor = inColor;
}
