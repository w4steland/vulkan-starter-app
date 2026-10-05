#pragma once

#include "../glm.hpp"

namespace glm {

inline float* value_ptr(vec3& v) {
    return &v.x;
}

inline const float* value_ptr(const vec3& v) {
    return &v.x;
}

inline float* value_ptr(vec4& v) {
    return &v.x;
}

inline const float* value_ptr(const vec4& v) {
    return &v.x;
}

inline float* value_ptr(mat4& m) {
    return &m[0][0];
}

inline const float* value_ptr(const mat4& m) {
    return &m[0][0];
}

inline vec4 make_vec4(const float* ptr) {
    return vec4(ptr[0], ptr[1], ptr[2], ptr[3]);
}

} // namespace glm
