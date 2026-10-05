#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

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
    if (pc.drawMode == 1) {
        // Ребра: контрастный сплошной цвет
        outColor = ubo.edgeColor;
    } else if (pc.drawMode == 2) {
        // Углы: рисуем красивую круглую точку (маркер вершины)
        vec2 coord = gl_PointCoord - vec2(0.5);
        if (dot(coord, coord) > 0.25) {
            discard; // обрезаем квадрат до круга
        }
        outColor = ubo.cornerColor;
    } else {
        // Грани: процедурный цвет вершин * цвет UI
        outColor = vec4(fragColor, 1.0) * ubo.uiColor;
    }
}
