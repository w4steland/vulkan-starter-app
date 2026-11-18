#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragWorldNormal;
layout(location = 2) in vec2 fragUV;
layout(location = 3) in vec4 fragShadowPos;

layout(location = 0) out vec4 outColor;

// Set 0: Scene
layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;
    mat4 view;
    mat4 proj;
    mat4 shadowViewProj; // Матрица источника света для теней
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

// Set 0, Binding 3: Карта теней с аппаратным сэмплером сравнения (sampler2DShadow)
layout(set = 0, binding = 3) uniform sampler2DShadow shadowMap;

// Set 1: Material (Dynamic Model UBO + Combined Image Sampler)
layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;       // rgb = tint factor
    vec4 specular;     // rgb = specular color, w = shininess
} modelData;

layout(set = 1, binding = 1) uniform sampler2D albedoTexture;

// Push Constants для нетривиального сэмплирования (Доп. задание 2)
layout(push_constant) uniform PushConsts {
    int enableEffects;     // 0 = стандартное сэмплирование, 1 = нетривиальное
    float time;            // непрерывное время анимации
    float waveDistortion;  // амплитуда волнового искажения UV
    float chromaticAberr;  // интенсивность хроматической аберрации
    float colorModulation; // интенсивность динамической пульсации цвета
} pc;

vec3 calcBlinnPhong(vec3 L, vec3 N, vec3 V, vec3 lightColor, float intensity, vec3 albedo, vec3 specColor, float shininess) {
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = albedo * diff;

    vec3 H = normalize(L + V);
    float spec = 0.0;
    if (diff > 0.0) {
        spec = pow(max(dot(N, H), 0.0), shininess);
    }
    vec3 specular = specColor * spec;

    return (diffuse + specular) * lightColor * intensity;
}

// Сэмплирование карты теней через sampler2DShadow (аппаратное сравнение и PCF)
float calcShadow(vec4 shadowPos, float bias) {
    // 1. Перспективное деление в пространство источника света (NDC)
    vec3 projCoords = shadowPos.xyz / shadowPos.w;

    // 2. Преобразование координат XY из NDC [-1, 1] в текстурные координаты [0, 1]
    projCoords.xy = projCoords.xy * 0.5 + 0.5;

    // 3. За пределами видимости источника света тени нет (1.0 = освещено)
    if (projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0) {
        return 1.0;
    }

    // 4. texture(sampler2DShadow, vec3(u, v, refZ)) возвращает 1.0 (освещено) или 0.0 (в тени)
    float shadow = texture(shadowMap, vec3(projCoords.xy, projCoords.z - bias));
    return shadow;
}

void main() {
    vec2 uv = fragUV;
    vec4 texColor = vec4(1.0);

    if (pc.enableEffects == 1) {
        // --- Доп. задание 2: Нетривиальное сэмплирование ---
        // 1. Волновое искажение UV (параметризовано от времени)
        vec2 waveOffset = vec2(
            sin(uv.y * 12.0 + pc.time * 3.0) * 0.03 * pc.waveDistortion,
            cos(uv.x * 12.0 + pc.time * 2.5) * 0.03 * pc.waveDistortion
        );
        vec2 distortedUV = uv + waveOffset;

        // 2. Множественное сэмплирование (хроматическая аберрация с расщеплением каналов R и B)
        vec2 rOffset = vec2( 0.015 * pc.chromaticAberr, 0.0);
        vec2 bOffset = vec2(-0.015 * pc.chromaticAberr, 0.0);

        float r = texture(albedoTexture, distortedUV + rOffset).r;
        float g = texture(albedoTexture, distortedUV).g;
        float b = texture(albedoTexture, distortedUV + bOffset).b;
        float a = texture(albedoTexture, distortedUV).a;

        // 3. Динамическая модуляция цвета во времени
        vec3 colorPulse = vec3(
            1.0 + 0.25 * sin(pc.time * 2.0) * pc.colorModulation,
            1.0 + 0.25 * sin(pc.time * 2.0 + 2.094) * pc.colorModulation,
            1.0 + 0.25 * sin(pc.time * 2.0 + 4.188) * pc.colorModulation
        );

        texColor = vec4(vec3(r, g, b) * colorPulse, a);
    } else {
        // Стандартное сэмплирование
        texColor = texture(albedoTexture, uv);
    }

    // Текстура выступает в роли Albedo
    vec3 albedo = texColor.rgb * modelData.albedo.rgb;
    vec3 specColor = modelData.specular.rgb;
    float shininess = max(modelData.specular.w, 1.0);

    vec3 N = normalize(fragWorldNormal);
    vec3 V = normalize(scene.cameraPos.xyz - fragWorldPos);

    // 1. Ambient
    vec3 resultColor = albedo * scene.ambientLight.rgb * scene.ambientLight.w;

    // 2. Directional Light (Солнце) с наложением теней
    if (scene.dirLightColor.w > 0.5) {
        vec3 L = normalize(-scene.dirLightDir.xyz);
        float diff = max(dot(N, L), 0.0);
        float shadow = 1.0;
        if (diff > 0.0) {
            shadow = calcShadow(fragShadowPos, scene.shadowParams.w);
        } else {
            shadow = 0.0;
        }

        vec3 dirLighting = calcBlinnPhong(L, N, V, scene.dirLightColor.rgb, scene.dirLightDir.w, albedo, specColor, shininess);
        resultColor += dirLighting * shadow;
    }

    // 3. Point Lights (обратные квадраты)
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

    // 4. Spot Lights (обратные квадраты + мягкие края)
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

    outColor = vec4(resultColor, texColor.a);
}
