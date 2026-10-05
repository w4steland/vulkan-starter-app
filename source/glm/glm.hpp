#pragma once

#include <cmath>
#include <cstring>
#include <iostream>

namespace glm {

constexpr float pi = 3.14159265358979323846f;

inline float radians(float degrees) {
    return degrees * (pi / 180.0f);
}

inline float degrees(float rad) {
    return rad * (180.0f / pi);
}

// -------------------------------------------------------------
// vec3
// -------------------------------------------------------------
struct vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    vec3() = default;
    vec3(float val) : x(val), y(val), z(val) {}
    vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

    float& operator[](int i) { return (&x)[i]; }
    const float& operator[](int i) const { return (&x)[i]; }

    vec3 operator+(const vec3& b) const { return {x + b.x, y + b.y, z + b.z}; }
    vec3 operator-(const vec3& b) const { return {x - b.x, y - b.y, z - b.z}; }
    vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    vec3 operator/(float s) const { return {x / s, y / s, z / s}; }

    vec3& operator+=(const vec3& b) { x += b.x; y += b.y; z += b.z; return *this; }
    vec3& operator-=(const vec3& b) { x -= b.x; y -= b.y; z -= b.z; return *this; }
    vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
};

inline vec3 operator*(float s, const vec3& v) { return v * s; }
inline vec3 operator-(const vec3& v) { return {-v.x, -v.y, -v.z}; }

inline float dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline vec3 cross(const vec3& a, const vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

inline float length(const vec3& v) {
    return std::sqrt(dot(v, v));
}

inline vec3 normalize(const vec3& v) {
    float len = length(v);
    if (len > 0.00001f) {
        return v / len;
    }
    return v;
}

// -------------------------------------------------------------
// vec4
// -------------------------------------------------------------
struct vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;

    vec4() = default;
    vec4(float val) : x(val), y(val), z(val), w(val) {}
    vec4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}
    vec4(const vec3& v, float _w) : x(v.x), y(v.y), z(v.z), w(_w) {}

    float& operator[](int i) { return (&x)[i]; }
    const float& operator[](int i) const { return (&x)[i]; }

    vec4 operator+(const vec4& b) const { return {x + b.x, y + b.y, z + b.z, w + b.w}; }
    vec4 operator*(float s) const { return {x * s, y * s, z * s, w * s}; }
};

// -------------------------------------------------------------
// mat4 (Column-Major, matches OpenGL & Vulkan & GLSL memory layout)
// -------------------------------------------------------------
struct mat4 {
    vec4 cols[4];

    mat4() {
        cols[0] = vec4(1.0f, 0.0f, 0.0f, 0.0f);
        cols[1] = vec4(0.0f, 1.0f, 0.0f, 0.0f);
        cols[2] = vec4(0.0f, 0.0f, 1.0f, 0.0f);
        cols[3] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    explicit mat4(float diag) {
        cols[0] = vec4(diag, 0.0f, 0.0f, 0.0f);
        cols[1] = vec4(0.0f, diag, 0.0f, 0.0f);
        cols[2] = vec4(0.0f, 0.0f, diag, 0.0f);
        cols[3] = vec4(0.0f, 0.0f, 0.0f, diag);
    }

    vec4& operator[](int col) { return cols[col]; }
    const vec4& operator[](int col) const { return cols[col]; }

    mat4 operator*(const mat4& b) const {
        mat4 res(0.0f);
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                res[c][r] = cols[0][r] * b[c][0] +
                            cols[1][r] * b[c][1] +
                            cols[2][r] * b[c][2] +
                            cols[3][r] * b[c][3];
            }
        }
        return res;
    }
};

} // namespace glm
