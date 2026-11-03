#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragWorldNormal;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 ambientLight;   // rgb = color, w = intensity
    vec4 dirLightDir;    // xyz = direction, w = intensity
    vec4 dirLightColor;  // rgb = color, w = enabled (1.0 or 0.0)
    int numPointLights;
    int numSpotLights;
    int pad0;
    int pad1;
} scene;

struct PointLight {
    vec4 position; // xyz = position, w = intensity
    vec4 color;    // rgb = color, w = unused
};

layout(std430, set = 0, binding = 1) readonly buffer PointLightBuffer {
    PointLight pointLights[];
};

struct SpotLight {
    vec4 position;   // xyz = position, w = intensity
    vec4 direction;  // xyz = direction, w = unused
    vec4 color;      // rgb = color, w = unused
    vec4 spotParams; // x = cos(innerAngle), y = cos(outerAngle), z = unused, w = unused
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLightBuffer {
    SpotLight spotLights[];
};

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;       // rgb = diffuse color, w = opacity
    vec4 specular;     // rgb = specular color, w = shininess
} modelData;

// Модель освещения Блинн-Фонга для одного источника света
vec3 calcBlinnPhong(vec3 L, vec3 N, vec3 V, vec3 lightColor, float intensity, vec3 albedo, vec3 specColor, float shininess) {
    // 1. Диффузная составляющая (закон Ламберта)
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = albedo * diff;

    // 2. Зеркальная составляющая (Блинн-Фонг с половинчатым вектором H)
    vec3 H = normalize(L + V);
    float spec = 0.0;
    if (diff > 0.0) {
        spec = pow(max(dot(N, H), 0.0), shininess);
    }
    vec3 specular = specColor * spec;

    return (diffuse + specular) * lightColor * intensity;
}

void main() {
    vec3 N = normalize(fragWorldNormal);
    vec3 V = normalize(scene.cameraPos.xyz - fragWorldPos);

    vec3 albedo = modelData.albedo.rgb;
    vec3 specColor = modelData.specular.rgb;
    float shininess = max(modelData.specular.w, 1.0);

    // 1. Рассеянный свет (Ambient)
    vec3 resultColor = albedo * scene.ambientLight.rgb * scene.ambientLight.w;

    // 2. Направленный источник света (Directional Light)
    if (scene.dirLightColor.w > 0.5) {
        vec3 L = normalize(-scene.dirLightDir.xyz);
        resultColor += calcBlinnPhong(L, N, V, scene.dirLightColor.rgb, scene.dirLightDir.w, albedo, specColor, shininess);
    }

    // 3. Массив точечных источников света (Point Lights, закон обратных квадратов)
    for (int i = 0; i < scene.numPointLights; ++i) {
        vec3 lightPos = pointLights[i].position.xyz;
        vec3 toLight = lightPos - fragWorldPos;
        float dist2 = dot(toLight, toLight);
        float dist = sqrt(dist2);
        vec3 L = toLight / max(dist, 0.0001);

        float attenuation = 1.0 / max(dist2, 0.01);
        float intensity = pointLights[i].position.w;

        resultColor += calcBlinnPhong(L, N, V, pointLights[i].color.rgb, intensity * attenuation, albedo, specColor, shininess);
    }

    // 4. Массив прожекторных источников света (Spot Lights, затухание + мягкие края)
    for (int i = 0; i < scene.numSpotLights; ++i) {
        vec3 lightPos = spotLights[i].position.xyz;
        vec3 toLight = lightPos - fragWorldPos;
        float dist2 = dot(toLight, toLight);
        float dist = sqrt(dist2);
        vec3 L = toLight / max(dist, 0.0001);

        float attenuation = 1.0 / max(dist2, 0.01);
        float intensity = spotLights[i].position.w;

        vec3 spotDir = normalize(spotLights[i].direction.xyz);
        float cosTheta = dot(-L, spotDir);
        float cosInner = spotLights[i].spotParams.x;
        float cosOuter = spotLights[i].spotParams.y;

        float epsilon = max(cosInner - cosOuter, 0.0001);
        float spotIntensity = clamp((cosTheta - cosOuter) / epsilon, 0.0, 1.0);

        if (spotIntensity > 0.0) {
            resultColor += calcBlinnPhong(L, N, V, spotLights[i].color.rgb, intensity * attenuation * spotIntensity, albedo, specColor, shininess);
        }
    }

    outColor = vec4(resultColor, 1.0);
}
