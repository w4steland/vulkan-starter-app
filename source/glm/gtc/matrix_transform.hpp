#pragma once

#include "../glm.hpp"

namespace glm {

// Translation
inline mat4 translate(const mat4& m, const vec3& v) {
    mat4 res = m;
    res[3] = m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3];
    return res;
}

// Scale
inline mat4 scale(const mat4& m, const vec3& v) {
    mat4 res = m;
    res[0] = m[0] * v.x;
    res[1] = m[1] * v.y;
    res[2] = m[2] * v.z;
    return res;
}

// Rotation around arbitrary axis (Rodrigues' formula in matrix form)
inline mat4 rotate(const mat4& m, float angle_rad, const vec3& axis) {
    float a = angle_rad;
    float c = std::cos(a);
    float s = std::sin(a);

    vec3 u = normalize(axis);

    mat4 rot(0.0f);
    rot[0][0] = c + (1.0f - c) * u.x * u.x;
    rot[0][1] = (1.0f - c) * u.x * u.y + s * u.z;
    rot[0][2] = (1.0f - c) * u.x * u.z - s * u.y;
    rot[0][3] = 0.0f;

    rot[1][0] = (1.0f - c) * u.y * u.x - s * u.z;
    rot[1][1] = c + (1.0f - c) * u.y * u.y;
    rot[1][2] = (1.0f - c) * u.y * u.z + s * u.x;
    rot[1][3] = 0.0f;

    rot[2][0] = (1.0f - c) * u.z * u.x + s * u.y;
    rot[2][1] = (1.0f - c) * u.z * u.y - s * u.x;
    rot[2][2] = c + (1.0f - c) * u.z * u.z;
    rot[2][3] = 0.0f;

    rot[3][3] = 1.0f;

    return m * rot;
}

// LookAt View Matrix
inline mat4 lookAt(const vec3& eye, const vec3& center, const vec3& up) {
    vec3 f = normalize(center - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);

    mat4 res(1.0f);
    res[0][0] =  s.x;
    res[1][0] =  s.y;
    res[2][0] =  s.z;
    res[0][1] =  u.x;
    res[1][1] =  u.y;
    res[2][1] =  u.z;
    res[0][2] = -f.x;
    res[1][2] = -f.y;
    res[2][2] = -f.z;
    res[3][0] = -dot(s, eye);
    res[3][1] = -dot(u, eye);
    res[3][2] =  dot(f, eye);
    return res;
}

// Perspective Projection Matrix for Vulkan (Right-Handed, Depth in [0, 1])
inline mat4 perspective(float fovy_rad, float aspect, float zNear, float zFar) {
    float tanHalfFovy = std::tan(fovy_rad / 2.0f);

    mat4 res(0.0f);
    res[0][0] = 1.0f / (aspect * tanHalfFovy);
    res[1][1] = 1.0f / (tanHalfFovy);
    res[2][2] = zFar / (zNear - zFar);
    res[2][3] = -1.0f;
    res[3][2] = (zFar * zNear) / (zNear - zFar);
    return res;
}

// Orthographic Projection Matrix for Vulkan (Right-Handed, Depth in [0, 1])
inline mat4 ortho(float left, float right, float bottom, float top, float zNear, float zFar) {
    mat4 res(1.0f);
    res[0][0] = 2.0f / (right - left);
    res[1][1] = 2.0f / (top - bottom);
    res[2][2] = 1.0f / (zNear - zFar);
    res[3][0] = -(right + left) / (right - left);
    res[3][1] = -(top + bottom) / (top - bottom);
    res[3][2] = zNear / (zNear - zFar);
    return res;
}

} // namespace glm
