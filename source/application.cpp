#include "application.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace application {

// -------------------------------------------------------------
// Структуры данных
// -------------------------------------------------------------

// Вершина: позиция и процедурно вычисленная нормаль
struct Vertex {
  glm::vec3 pos;
  glm::vec3 normal;
};

// Точечный источник света для Storage Buffer (std430: выравнивание 16 байт)
struct PointLight {
  alignas(16) glm::vec4 position; // xyz = мировая позиция, w = интенсивность
  alignas(16) glm::vec4 color;    // rgb = цвет, w = не используется
};

// Прожекторный источник света для Storage Buffer (std430: выравнивание 16 байт)
struct SpotLight {
  alignas(16) glm::vec4 position;   // xyz = мировая позиция, w = интенсивность
  alignas(16) glm::vec4 direction;  // xyz = направление, w = не используется
  alignas(16) glm::vec4 color;      // rgb = цвет, w = не используется
  alignas(16) glm::vec4 spotParams; // x = cos(innerAngle), y = cos(outerAngle)
};

// UBO сцены (Set 0, Binding 0)
struct SceneUniforms {
  alignas(16) glm::mat4 viewProj;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
  alignas(16) glm::vec4 cameraPos;
  alignas(16) glm::vec4 ambientLight;  // rgb = цвет, w = интенсивность
  alignas(16) glm::vec4 dirLightDir;   // xyz = направление к сцене, w = интенсивность
  alignas(16) glm::vec4 dirLightColor; // rgb = цвет, w = 1.0 (вкл) / 0.0 (выкл)
  alignas(4) int numPointLights;
  alignas(4) int numSpotLights;
  alignas(4) int pad0;
  alignas(4) int pad1;
};

// UBO модели (Set 1, Binding 0)
struct ModelUniforms {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 normalMatrix; // mat4 для std140, используется верхний 3x3
  alignas(16) glm::vec4 albedo;       // rgb = диффузный цвет, w = непрозрачность
  alignas(16) glm::vec4 specular;     // rgb = цвет блика, w = коэффициент блеска (shininess)
};

// -------------------------------------------------------------
// Геометрия параллелепипеда (Вариант 3: a = 1.6, b = 1.0, c = 0.8)
// -------------------------------------------------------------
const float hx = 0.8f;
const float hy = 0.5f;
const float hz = 0.4f;

static std::vector<Vertex> vertices;
static std::vector<uint16_t> indices;

// Процедурная генерация вершин и нормалей для 6 граней
static void generateParallelepiped(float halfX, float halfY, float halfZ) {
  vertices.clear();
  indices.clear();

  struct Quad {
    glm::vec3 p0, p1, p2, p3;
  };

  // 6 граней, каждая задается 4 точками в порядке обхода CCW
  std::array<Quad, 6> quads = {{
      // Перед (+Z)
      {{-halfX, -halfY, halfZ}, {halfX, -halfY, halfZ}, {halfX, halfY, halfZ}, {-halfX, halfY, halfZ}},
      // Зад (-Z)
      {{halfX, -halfY, -halfZ}, {-halfX, -halfY, -halfZ}, {-halfX, halfY, -halfZ}, {halfX, halfY, -halfZ}},
      // Лево (-X)
      {{-halfX, -halfY, -halfZ}, {-halfX, -halfY, halfZ}, {-halfX, halfY, halfZ}, {-halfX, halfY, -halfZ}},
      // Право (+X)
      {{halfX, -halfY, halfZ}, {halfX, -halfY, -halfZ}, {halfX, halfY, -halfZ}, {halfX, halfY, halfZ}},
      // Верх (+Y)
      {{-halfX, halfY, halfZ}, {halfX, halfY, halfZ}, {halfX, halfY, -halfZ}, {-halfX, halfY, -halfZ}},
      // Низ (-Y)
      {{-halfX, -halfY, -halfZ}, {halfX, -halfY, -halfZ}, {halfX, -halfY, halfZ}, {-halfX, -halfY, halfZ}},
  }};

  for (size_t i = 0; i < quads.size(); ++i) {
    const auto &q = quads[i];
    // Процедурный расчет нормали через векторное произведение: normal = normalize(cross(p1 - p0, p2 - p0))
    glm::vec3 normal = glm::normalize(glm::cross(q.p1 - q.p0, q.p2 - q.p0));

    uint16_t baseIdx = static_cast<uint16_t>(vertices.size());
    vertices.push_back({q.p0, normal});
    vertices.push_back({q.p1, normal});
    vertices.push_back({q.p2, normal});
    vertices.push_back({q.p3, normal});

    // Два треугольника на грань
    indices.push_back(baseIdx + 0);
    indices.push_back(baseIdx + 1);
    indices.push_back(baseIdx + 2);

    indices.push_back(baseIdx + 0);
    indices.push_back(baseIdx + 2);
    indices.push_back(baseIdx + 3);
  }
}

// -------------------------------------------------------------
// Vulkan ресурсы
// -------------------------------------------------------------
static VkBuffer vertBuffer = VK_NULL_HANDLE;
static VmaAllocation vertAlloc = VK_NULL_HANDLE;

static VkBuffer idxBuffer = VK_NULL_HANDLE;
static VmaAllocation idxAlloc = VK_NULL_HANDLE;

// UBO и SSBO буферы
static VkBuffer sceneUboBuffer = VK_NULL_HANDLE;
static VmaAllocation sceneUboAlloc = VK_NULL_HANDLE;

static VkBuffer modelUboBuffer = VK_NULL_HANDLE;
static VmaAllocation modelUboAlloc = VK_NULL_HANDLE;

static const size_t MAX_LIGHTS = 16;
static VkBuffer pointLightsBuffer = VK_NULL_HANDLE;
static VmaAllocation pointLightsAlloc = VK_NULL_HANDLE;

static VkBuffer spotLightsBuffer = VK_NULL_HANDLE;
static VmaAllocation spotLightsAlloc = VK_NULL_HANDLE;

// Дескрипторы для двух наборов (Set 0: сцена, Set 1: модель)
static VkDescriptorSetLayout descLayoutScene = VK_NULL_HANDLE;
static VkDescriptorSetLayout descLayoutModel = VK_NULL_HANDLE;
static VkDescriptorPool descPool = VK_NULL_HANDLE;
static VkDescriptorSet descSetScene = VK_NULL_HANDLE;
static VkDescriptorSet descSetModel = VK_NULL_HANDLE;

static VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
static VkPipeline pipeline = VK_NULL_HANDLE;

// -------------------------------------------------------------
// Параметры сцены, камеры, модели и освещения
// -------------------------------------------------------------

// Камера
static glm::vec3 camPos = {0.0f, 1.8f, 4.0f};
static float camYaw = 0.0f;     // угол поворота вокруг Y в градусах
static float camPitch = -15.0f; // угол наклона в градусах
static float camFov = 45.0f;

// Модель
static glm::vec3 modelPos = {0.0f, 0.0f, 0.0f};
static glm::vec3 modelRot = {15.0f, 30.0f, 0.0f};
static glm::vec3 modelScale = {1.0f, 1.0f, 1.0f};

// Анимация траектории
static bool playAnim = false;
static float animSpeed = 1.0f;
static float trajRadius = 1.8f;
static float trajHeight = 0.6f;
static double animTime = 0.0;
static double prevTime = 0.0;

// Материал (Блинн-Фонг)
static float matAlbedo[3] = {0.25f, 0.65f, 0.95f};   // базовый цвет
static float matSpecular[3] = {1.0f, 1.0f, 1.0f};    // цвет блика
static float matShininess = 32.0f;                   // коэффициент блеска

// Рассеянный свет (Ambient)
static float ambientColor[3] = {1.0f, 1.0f, 1.0f};
static float ambientIntensity = 0.15f;

// Направленный свет (Directional)
static bool dirLightEnabled = true;
static float dirLightDir[3] = {-0.6f, -1.0f, -0.8f};
static float dirLightColor[3] = {1.0f, 0.96f, 0.88f};
static float dirLightIntensity = 1.0f;

// Структуры для UI источников света
struct PointLightUI {
  glm::vec3 pos;
  glm::vec3 color;
  float intensity;
};

struct SpotLightUI {
  glm::vec3 pos;
  glm::vec3 dir;
  glm::vec3 color;
  float intensity;
  float innerAngle; // в градусах
  float outerAngle; // в градусах
};

static std::vector<PointLightUI> pointLightsList = {
    {{1.8f, 1.2f, 1.2f}, {1.0f, 0.35f, 0.35f}, 5.0f},  // красный точечный
    {{-1.8f, 1.0f, -0.8f}, {0.35f, 1.0f, 0.45f}, 4.0f} // зеленый точечный
};

static std::vector<SpotLightUI> spotLightsList = {
    {{0.0f, 3.0f, 2.0f}, {0.0f, -1.0f, -0.7f}, {1.0f, 0.95f, 0.4f}, 12.0f, 15.0f, 28.0f} // желтый прожектор
};

// Вычисление обратной транспонированной 3x3 матрицы для трансформации нормалей
static glm::mat4 computeNormalMatrix(const glm::mat4 &m) {
  float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
  float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
  float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

  float det = a00 * (a11 * a22 - a12 * a21) -
              a01 * (a10 * a22 - a12 * a20) +
              a02 * (a10 * a21 - a11 * a20);

  float invDet = (std::abs(det) > 1e-6f) ? (1.0f / det) : 1.0f;

  glm::mat4 res(1.0f);
  res[0][0] =  (a11 * a22 - a12 * a21) * invDet;
  res[0][1] = -(a10 * a22 - a12 * a20) * invDet;
  res[0][2] =  (a10 * a21 - a11 * a20) * invDet;

  res[1][0] = -(a01 * a22 - a02 * a21) * invDet;
  res[1][1] =  (a00 * a22 - a02 * a20) * invDet;
  res[1][2] = -(a00 * a21 - a01 * a20) * invDet;

  res[2][0] =  (a01 * a12 - a02 * a11) * invDet;
  res[2][1] = -(a00 * a12 - a02 * a10) * invDet;
  res[2][2] =  (a00 * a11 - a01 * a10) * invDet;

  return res;
}

// -------------------------------------------------------------
// Вспомогательные функции Vulkan
// -------------------------------------------------------------

static std::vector<char> loadShader(const std::string &name) {
  const std::string paths[] = {name, "shaders/" + name, "../shaders/" + name,
                               "../../shaders/" + name};
  for (const auto &p : paths) {
    std::ifstream f(p, std::ios::ate | std::ios::binary);
    if (f.is_open()) {
      size_t size = (size_t)f.tellg();
      std::vector<char> buf(size);
      f.seekg(0);
      f.read(buf.data(), size);
      return buf;
    }
  }
  std::cerr << "Cannot open shader: " << name << "\n";
  return {};
}

static VkShaderModule makeShaderModule(const std::vector<char> &code) {
  if (code.empty())
    return VK_NULL_HANDLE;

  VkShaderModuleCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = code.size();
  ci.pCode = reinterpret_cast<const uint32_t *>(code.data());

  VkShaderModule mod = VK_NULL_HANDLE;
  vkCreateShaderModule(graphics::internal::context.device, &ci, nullptr, &mod);
  return mod;
}

static bool makeBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                       VkBuffer &buf, VmaAllocation &alloc) {
  VkBufferCreateInfo bi = {};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = size;
  bi.usage = usage;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo ai = {};
  ai.usage = VMA_MEMORY_USAGE_AUTO;
  ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

  return vmaCreateBuffer(graphics::internal::context.allocator, &bi, &ai, &buf,
                         &alloc, nullptr) == VK_SUCCESS;
}

static void copyToBuffer(VmaAllocation alloc, const void *data,
                         VkDeviceSize size) {
  void *mapped = nullptr;
  vmaMapMemory(graphics::internal::context.allocator, alloc, &mapped);
  std::memcpy(mapped, data, size);
  vmaUnmapMemory(graphics::internal::context.allocator, alloc);
}

// -------------------------------------------------------------
// Инициализация
// -------------------------------------------------------------

bool initialize() {
  auto &ctx = graphics::internal::context;

  // 1. Процедурная генерация вершин и нормалей
  generateParallelepiped(hx, hy, hz);

  // 2. Буфер вершин
  VkDeviceSize vertSize = sizeof(vertices[0]) * vertices.size();
  if (!makeBuffer(vertSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertBuffer, vertAlloc))
    return false;
  copyToBuffer(vertAlloc, vertices.data(), vertSize);

  // 3. Буфер индексов
  VkDeviceSize idxSize = sizeof(indices[0]) * indices.size();
  if (!makeBuffer(idxSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, idxBuffer, idxAlloc))
    return false;
  copyToBuffer(idxAlloc, indices.data(), idxSize);

  // 4. Буферы UBO и SSBO
  if (!makeBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneUboBuffer, sceneUboAlloc))
    return false;
  if (!makeBuffer(sizeof(ModelUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, modelUboBuffer, modelUboAlloc))
    return false;

  VkDeviceSize pointLightBufSize = sizeof(PointLight) * MAX_LIGHTS;
  if (!makeBuffer(pointLightBufSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, pointLightsBuffer, pointLightsAlloc))
    return false;

  VkDeviceSize spotLightBufSize = sizeof(SpotLight) * MAX_LIGHTS;
  if (!makeBuffer(spotLightBufSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spotLightsBuffer, spotLightsAlloc))
    return false;

  // 5. Описание дескрипторов (Доп. задание 1: два набора дескрипторов)

  // Набор 0: Сцена (Binding 0: Scene UBO, Binding 1: Point Lights SSBO, Binding 2: Spot Lights SSBO)
  VkDescriptorSetLayoutBinding sceneBindings[3] = {};

  // Binding 0: Scene UBO
  sceneBindings[0].binding = 0;
  sceneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  sceneBindings[0].descriptorCount = 1;
  sceneBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  // Binding 1: Point Lights SSBO
  sceneBindings[1].binding = 1;
  sceneBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sceneBindings[1].descriptorCount = 1;
  sceneBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  // Binding 2: Spot Lights SSBO
  sceneBindings[2].binding = 2;
  sceneBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sceneBindings[2].descriptorCount = 1;
  sceneBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo sceneLayoutInfo = {};
  sceneLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  sceneLayoutInfo.bindingCount = 3;
  sceneLayoutInfo.pBindings = sceneBindings;

  if (vkCreateDescriptorSetLayout(ctx.device, &sceneLayoutInfo, nullptr, &descLayoutScene) != VK_SUCCESS) {
    return false;
  }

  // Набор 1: Модель (Binding 0: Model UBO)
  VkDescriptorSetLayoutBinding modelBinding = {};
  modelBinding.binding = 0;
  modelBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  modelBinding.descriptorCount = 1;
  modelBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo modelLayoutInfo = {};
  modelLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  modelLayoutInfo.bindingCount = 1;
  modelLayoutInfo.pBindings = &modelBinding;

  if (vkCreateDescriptorSetLayout(ctx.device, &modelLayoutInfo, nullptr, &descLayoutModel) != VK_SUCCESS) {
    return false;
  }

  // 6. Descriptor Pool (2 UBO + 2 SSBO, maxSets = 2)
  VkDescriptorPoolSize poolSizes[2] = {};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[0].descriptorCount = 2; // scene UBO + model UBO
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSizes[1].descriptorCount = 2; // point lights SSBO + spot lights SSBO

  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.maxSets = 2;
  poolInfo.poolSizeCount = 2;
  poolInfo.pPoolSizes = poolSizes;

  if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descPool) != VK_SUCCESS) {
    return false;
  }

  // 7. Выделение Descriptor Sets
  VkDescriptorSetLayout layouts[2] = {descLayoutScene, descLayoutModel};
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &descLayoutScene;
  if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &descSetScene) != VK_SUCCESS)
    return false;

  allocInfo.pSetLayouts = &descLayoutModel;
  if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &descSetModel) != VK_SUCCESS)
    return false;

  // Привязка буферов к Set 0 (Сцена)
  VkDescriptorBufferInfo sceneBufInfo = {};
  sceneBufInfo.buffer = sceneUboBuffer;
  sceneBufInfo.offset = 0;
  sceneBufInfo.range = sizeof(SceneUniforms);

  VkDescriptorBufferInfo pointLightBufInfo = {};
  pointLightBufInfo.buffer = pointLightsBuffer;
  pointLightBufInfo.offset = 0;
  pointLightBufInfo.range = pointLightBufSize;

  VkDescriptorBufferInfo spotLightBufInfo = {};
  spotLightBufInfo.buffer = spotLightsBuffer;
  spotLightBufInfo.offset = 0;
  spotLightBufInfo.range = spotLightBufSize;

  VkWriteDescriptorSet sceneWrites[3] = {};
  sceneWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  sceneWrites[0].dstSet = descSetScene;
  sceneWrites[0].dstBinding = 0;
  sceneWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  sceneWrites[0].descriptorCount = 1;
  sceneWrites[0].pBufferInfo = &sceneBufInfo;

  sceneWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  sceneWrites[1].dstSet = descSetScene;
  sceneWrites[1].dstBinding = 1;
  sceneWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sceneWrites[1].descriptorCount = 1;
  sceneWrites[1].pBufferInfo = &pointLightBufInfo;

  sceneWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  sceneWrites[2].dstSet = descSetScene;
  sceneWrites[2].dstBinding = 2;
  sceneWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sceneWrites[2].descriptorCount = 1;
  sceneWrites[2].pBufferInfo = &spotLightBufInfo;

  vkUpdateDescriptorSets(ctx.device, 3, sceneWrites, 0, nullptr);

  // Привязка буфера к Set 1 (Модель)
  VkDescriptorBufferInfo modelBufInfo = {};
  modelBufInfo.buffer = modelUboBuffer;
  modelBufInfo.offset = 0;
  modelBufInfo.range = sizeof(ModelUniforms);

  VkWriteDescriptorSet modelWrite = {};
  modelWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  modelWrite.dstSet = descSetModel;
  modelWrite.dstBinding = 0;
  modelWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  modelWrite.descriptorCount = 1;
  modelWrite.pBufferInfo = &modelBufInfo;

  vkUpdateDescriptorSets(ctx.device, 1, &modelWrite, 0, nullptr);

  // 8. Загрузка шейдеров
  auto vCode = loadShader("shader.vert.spv");
  auto fCode = loadShader("shader.frag.spv");

  VkShaderModule vMod = makeShaderModule(vCode);
  VkShaderModule fMod = makeShaderModule(fCode);
  if (!vMod || !fMod)
    return false;

  VkPipelineShaderStageCreateInfo stages[2] = {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vMod;
  stages[0].pName = "main";

  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fMod;
  stages[1].pName = "main";

  // 9. Формат вершин (Binding 0: pos = location 0, normal = location 1)
  VkVertexInputBindingDescription bindDesc = {};
  bindDesc.binding = 0;
  bindDesc.stride = sizeof(Vertex);
  bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  VkVertexInputAttributeDescription attrDesc[2] = {};
  attrDesc[0].binding = 0;
  attrDesc[0].location = 0;
  attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attrDesc[0].offset = offsetof(Vertex, pos);

  attrDesc[1].binding = 0;
  attrDesc[1].location = 1;
  attrDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attrDesc[1].offset = offsetof(Vertex, normal);

  VkPipelineVertexInputStateCreateInfo vInput = {};
  vInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vInput.vertexBindingDescriptionCount = 1;
  vInput.pVertexBindingDescriptions = &bindDesc;
  vInput.vertexAttributeDescriptionCount = 2;
  vInput.pVertexAttributeDescriptions = attrDesc;

  // Динамические состояния
  VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynState = {};
  dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynState.dynamicStateCount = 2;
  dynState.pDynamicStates = dynStates;

  VkPipelineViewportStateCreateInfo vpState = {};
  vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  vpState.viewportCount = 1;
  vpState.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo raster = {};
  raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.lineWidth = 1.0f;
  raster.cullMode = VK_CULL_MODE_NONE; // обе стороны для надежности
  raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

  VkPipelineMultisampleStateCreateInfo ms = {};
  ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo ds = {};
  ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  ds.depthTestEnable = VK_TRUE;
  ds.depthWriteEnable = VK_TRUE;
  ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

  VkPipelineColorBlendAttachmentState blendAtt = {};
  blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

  VkPipelineColorBlendStateCreateInfo blend = {};
  blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  blend.attachmentCount = 1;
  blend.pAttachments = &blendAtt;

  // Pipeline Layout со ссылкой на 2 набора дескрипторов (Set 0 и Set 1)
  VkPipelineLayoutCreateInfo plInfo = {};
  plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plInfo.setLayoutCount = 2;
  plInfo.pSetLayouts = layouts;
  plInfo.pushConstantRangeCount = 0;
  plInfo.pPushConstantRanges = nullptr;

  if (vkCreatePipelineLayout(ctx.device, &plInfo, nullptr, &pipeLayout) != VK_SUCCESS) {
    return false;
  }

  // Создание графического конвейера
  VkPipelineInputAssemblyStateCreateInfo triAssembly = {};
  triAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  triAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkGraphicsPipelineCreateInfo pipeInfo = {};
  pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeInfo.stageCount = 2;
  pipeInfo.pStages = stages;
  pipeInfo.pVertexInputState = &vInput;
  pipeInfo.pInputAssemblyState = &triAssembly;
  pipeInfo.pViewportState = &vpState;
  pipeInfo.pRasterizationState = &raster;
  pipeInfo.pMultisampleState = &ms;
  pipeInfo.pDepthStencilState = &ds;
  pipeInfo.pColorBlendState = &blend;
  pipeInfo.pDynamicState = &dynState;
  pipeInfo.layout = pipeLayout;
  pipeInfo.renderPass = ctx.render_pass;

  if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline) != VK_SUCCESS) {
    return false;
  }

  vkDestroyShaderModule(ctx.device, fMod, nullptr);
  vkDestroyShaderModule(ctx.device, vMod, nullptr);

  return true;
}

// -------------------------------------------------------------
// Очистка ресурсов
// -------------------------------------------------------------

void shutdown() {
  auto &ctx = graphics::internal::context;
  vkQueueWaitIdle(ctx.graphics_queue);

  if (pipeline)
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
  if (pipeLayout)
    vkDestroyPipelineLayout(ctx.device, pipeLayout, nullptr);

  if (descPool)
    vkDestroyDescriptorPool(ctx.device, descPool, nullptr);
  if (descLayoutScene)
    vkDestroyDescriptorSetLayout(ctx.device, descLayoutScene, nullptr);
  if (descLayoutModel)
    vkDestroyDescriptorSetLayout(ctx.device, descLayoutModel, nullptr);

  if (spotLightsBuffer)
    vmaDestroyBuffer(ctx.allocator, spotLightsBuffer, spotLightsAlloc);
  if (pointLightsBuffer)
    vmaDestroyBuffer(ctx.allocator, pointLightsBuffer, pointLightsAlloc);
  if (modelUboBuffer)
    vmaDestroyBuffer(ctx.allocator, modelUboBuffer, modelUboAlloc);
  if (sceneUboBuffer)
    vmaDestroyBuffer(ctx.allocator, sceneUboBuffer, sceneUboAlloc);

  if (idxBuffer)
    vmaDestroyBuffer(ctx.allocator, idxBuffer, idxAlloc);
  if (vertBuffer)
    vmaDestroyBuffer(ctx.allocator, vertBuffer, vertAlloc);
}

// -------------------------------------------------------------
// Обновление состояния и интерфейс
// -------------------------------------------------------------

void update(double time) {
  if (prevTime == 0.0)
    prevTime = time;
  double dt = time - prevTime;
  prevTime = time;

  if (playAnim) {
    animTime += dt * animSpeed;
  }

  // --- ImGui Интерфейс ---
  ImGui::Begin("Lab 2: Camera & Blinn-Phong Lighting (Grade 4)");

  ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.4f, 1.0f), "Tasks: Base + Add. 1, 2, 3 (Grade 4)");
  ImGui::Separator();

  // 1. Камера
  if (ImGui::CollapsingHeader("1. Camera Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::DragFloat3("Camera Position", glm::value_ptr(camPos), 0.05f);
    ImGui::SliderFloat("Yaw (Rotation Y)", &camYaw, -180.0f, 180.0f, "%.1f deg");
    ImGui::SliderFloat("Pitch (Tilt)", &camPitch, -89.0f, 89.0f, "%.1f deg");
    ImGui::SliderFloat("Field of View (FOV)", &camFov, 20.0f, 120.0f, "%.1f deg");
    if (ImGui::Button("Reset Camera")) {
      camPos = {0.0f, 1.8f, 4.0f};
      camYaw = 0.0f;
      camPitch = -15.0f;
      camFov = 45.0f;
    }
  }

  // 2. Модель и материал
  if (ImGui::CollapsingHeader("2. Model & Material (Blinn-Phong)", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::TextDisabled("Model Affine Transforms:");
    ImGui::DragFloat3("Position", glm::value_ptr(modelPos), 0.05f);
    ImGui::SliderFloat3("Rotation", glm::value_ptr(modelRot), -180.0f, 180.0f, "%.1f deg");
    ImGui::DragFloat3("Scale", glm::value_ptr(modelScale), 0.05f, 0.1f, 4.0f);

    ImGui::Separator();
    ImGui::TextDisabled("Material Properties:");
    ImGui::ColorEdit3("Albedo (Diffuse Color)", matAlbedo);
    ImGui::ColorEdit3("Specular Color", matSpecular);
    ImGui::SliderFloat("Shininess", &matShininess, 1.0f, 128.0f, "%.1f");

    ImGui::Separator();
    if (playAnim) {
      if (ImGui::Button("Pause Animation ||")) playAnim = false;
    } else {
      if (ImGui::Button("Play Animation >")) playAnim = true;
    }
    ImGui::SameLine();
    ImGui::SliderFloat("Speed", &animSpeed, 0.0f, 3.0f, "%.2fx");
  }

  // 3. Рассеянный и направленный свет
  if (ImGui::CollapsingHeader("3. Ambient & Directional Light", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::TextDisabled("Ambient Light (Scattered):");
    ImGui::ColorEdit3("Ambient Color", ambientColor);
    ImGui::SliderFloat("Ambient Intensity", &ambientIntensity, 0.0f, 1.0f, "%.2f");

    ImGui::Separator();
    ImGui::TextDisabled("Directional Light (Sun):");
    ImGui::Checkbox("Enable Directional Light", &dirLightEnabled);
    if (dirLightEnabled) {
      ImGui::SliderFloat3("Direction", dirLightDir, -1.0f, 1.0f);
      ImGui::ColorEdit3("Sun Color", dirLightColor);
      ImGui::SliderFloat("Sun Intensity", &dirLightIntensity, 0.0f, 5.0f, "%.2f");
    }
  }

  // 4. Точечные источники света (Доп. задание 2: Storage Buffer, обратные квадраты)
  if (ImGui::CollapsingHeader("4. Point Lights (SSBO Array)", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("Active count: %d / %zu", (int)pointLightsList.size(), MAX_LIGHTS);
    if (pointLightsList.size() < MAX_LIGHTS) {
      if (ImGui::Button("+ Add Point Light")) {
        pointLightsList.push_back({{0.0f, 1.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, 4.0f});
      }
    }

    int toRemovePoint = -1;
    for (size_t i = 0; i < pointLightsList.size(); ++i) {
      ImGui::PushID((int)i);
      char header[32];
      snprintf(header, sizeof(header), "Point Light #%zu", i + 1);
      if (ImGui::TreeNode(header)) {
        ImGui::DragFloat3("Position", glm::value_ptr(pointLightsList[i].pos), 0.05f);
        ImGui::ColorEdit3("Color", glm::value_ptr(pointLightsList[i].color));
        ImGui::SliderFloat("Intensity", &pointLightsList[i].intensity, 0.0f, 20.0f, "%.1f");
        if (ImGui::Button("Remove Light")) {
          toRemovePoint = (int)i;
        }
        ImGui::TreePop();
      }
      ImGui::PopID();
    }
    if (toRemovePoint >= 0) {
      pointLightsList.erase(pointLightsList.begin() + toRemovePoint);
    }
  }

  // 5. Прожекторные источники света (Доп. задание 3: Storage Buffer, мягкие края)
  if (ImGui::CollapsingHeader("5. Spot Lights (SSBO Array + Soft Edges)", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("Active count: %d / %zu", (int)spotLightsList.size(), MAX_LIGHTS);
    if (spotLightsList.size() < MAX_LIGHTS) {
      if (ImGui::Button("+ Add Spot Light")) {
        spotLightsList.push_back({{0.0f, 2.5f, 1.5f}, {0.0f, -1.0f, -0.5f}, {1.0f, 0.9f, 0.5f}, 10.0f, 15.0f, 30.0f});
      }
    }

    int toRemoveSpot = -1;
    for (size_t i = 0; i < spotLightsList.size(); ++i) {
      ImGui::PushID(100 + (int)i);
      char header[32];
      snprintf(header, sizeof(header), "Spot Light #%zu", i + 1);
      if (ImGui::TreeNode(header)) {
        ImGui::DragFloat3("Position", glm::value_ptr(spotLightsList[i].pos), 0.05f);
        ImGui::SliderFloat3("Direction", glm::value_ptr(spotLightsList[i].dir), -1.0f, 1.0f);
        ImGui::ColorEdit3("Color", glm::value_ptr(spotLightsList[i].color));
        ImGui::SliderFloat("Intensity", &spotLightsList[i].intensity, 0.0f, 30.0f, "%.1f");
        ImGui::SliderFloat("Inner Angle", &spotLightsList[i].innerAngle, 1.0f, 60.0f, "%.1f deg");
        ImGui::SliderFloat("Outer Angle", &spotLightsList[i].outerAngle, spotLightsList[i].innerAngle, 85.0f, "%.1f deg");
        if (ImGui::Button("Remove Spot Light")) {
          toRemoveSpot = (int)i;
        }
        ImGui::TreePop();
      }
      ImGui::PopID();
    }
    if (toRemoveSpot >= 0) {
      spotLightsList.erase(spotLightsList.begin() + toRemoveSpot);
    }
  }

  ImGui::End();

  // -------------------------------------------------------------
  // Математические расчеты матриц
  // -------------------------------------------------------------
  auto &ctx = graphics::internal::context;
  float aspect = (float)ctx.swapchain_extent.width /
                 (float)(ctx.swapchain_extent.height > 0 ? ctx.swapchain_extent.height : 1);

  // Матрица проекции
  glm::mat4 projMat = glm::perspective(glm::radians(camFov), aspect, 0.1f, 100.0f);
  projMat[1][1] *= -1.0f; // инверсия Y под Vulkan

  // Матрица вида (вычисляем вектор взгляда из углов Yaw/Pitch)
  float yawRad = glm::radians(camYaw);
  float pitchRad = glm::radians(camPitch);
  glm::vec3 front;
  front.x = std::sin(yawRad) * std::cos(pitchRad);
  front.y = std::sin(pitchRad);
  front.z = -std::cos(yawRad) * std::cos(pitchRad);
  front = glm::normalize(front);

  glm::mat4 viewMat = glm::lookAt(camPos, camPos + front, glm::vec3(0.0f, 1.0f, 0.0f));

  // Анимация траектории модели (3D Helix)
  float t = (float)animTime;
  glm::vec3 animPos = {0.0f, 0.0f, 0.0f};
  float animRotY = 0.0f;
  if (playAnim) {
    animPos.x = trajRadius * std::cos(t);
    animPos.z = trajRadius * std::sin(t);
    animPos.y = trajHeight * std::sin(2.0f * t);
    animRotY = t * 0.8f;
  }

  // Матрица модели: M = T * R * S
  glm::mat4 modelMat = glm::mat4(1.0f);
  modelMat = glm::translate(modelMat, modelPos + animPos);
  modelMat = glm::rotate(modelMat, glm::radians(modelRot.x), glm::vec3(1.0f, 0.0f, 0.0f));
  modelMat = glm::rotate(modelMat, glm::radians(modelRot.y) + animRotY, glm::vec3(0.0f, 1.0f, 0.0f));
  modelMat = glm::rotate(modelMat, glm::radians(modelRot.z), glm::vec3(0.0f, 0.0f, 1.0f));
  modelMat = glm::scale(modelMat, modelScale);

  // Нормальная матрица: обратная транспонированная 3x3
  glm::mat4 normalMatrix = computeNormalMatrix(modelMat);

  // -------------------------------------------------------------
  // Заполнение и отправка UBO и Storage буферов
  // -------------------------------------------------------------

  // 1. Scene UBO
  SceneUniforms sceneUbo = {};
  sceneUbo.view = viewMat;
  sceneUbo.proj = projMat;
  sceneUbo.viewProj = projMat * viewMat;
  sceneUbo.cameraPos = glm::vec4(camPos, 1.0f);
  sceneUbo.ambientLight = glm::vec4(ambientColor[0], ambientColor[1], ambientColor[2], ambientIntensity);
  sceneUbo.dirLightDir = glm::vec4(glm::normalize(glm::vec3(dirLightDir[0], dirLightDir[1], dirLightDir[2])), dirLightIntensity);
  sceneUbo.dirLightColor = glm::vec4(dirLightColor[0], dirLightColor[1], dirLightColor[2], dirLightEnabled ? 1.0f : 0.0f);
  sceneUbo.numPointLights = (int)pointLightsList.size();
  sceneUbo.numSpotLights = (int)spotLightsList.size();
  copyToBuffer(sceneUboAlloc, &sceneUbo, sizeof(sceneUbo));

  // 2. Model UBO
  ModelUniforms modelUbo = {};
  modelUbo.model = modelMat;
  modelUbo.normalMatrix = normalMatrix;
  modelUbo.albedo = glm::vec4(matAlbedo[0], matAlbedo[1], matAlbedo[2], 1.0f);
  modelUbo.specular = glm::vec4(matSpecular[0], matSpecular[1], matSpecular[2], matShininess);
  copyToBuffer(modelUboAlloc, &modelUbo, sizeof(modelUbo));

  // 3. Point Lights Storage Buffer
  if (!pointLightsList.empty()) {
    std::vector<PointLight> gpuPointLights(pointLightsList.size());
    for (size_t i = 0; i < pointLightsList.size(); ++i) {
      gpuPointLights[i].position = glm::vec4(pointLightsList[i].pos, pointLightsList[i].intensity);
      gpuPointLights[i].color = glm::vec4(pointLightsList[i].color, 1.0f);
    }
    copyToBuffer(pointLightsAlloc, gpuPointLights.data(), sizeof(PointLight) * gpuPointLights.size());
  }

  // 4. Spot Lights Storage Buffer
  if (!spotLightsList.empty()) {
    std::vector<SpotLight> gpuSpotLights(spotLightsList.size());
    for (size_t i = 0; i < spotLightsList.size(); ++i) {
      gpuSpotLights[i].position = glm::vec4(spotLightsList[i].pos, spotLightsList[i].intensity);
      gpuSpotLights[i].direction = glm::vec4(glm::normalize(spotLightsList[i].dir), 0.0f);
      gpuSpotLights[i].color = glm::vec4(spotLightsList[i].color, 1.0f);
      gpuSpotLights[i].spotParams = glm::vec4(
          std::cos(glm::radians(spotLightsList[i].innerAngle)),
          std::cos(glm::radians(spotLightsList[i].outerAngle)),
          0.0f, 0.0f);
    }
    copyToBuffer(spotLightsAlloc, gpuSpotLights.data(), sizeof(SpotLight) * gpuSpotLights.size());
  }
}

// -------------------------------------------------------------
// Рендеринг
// -------------------------------------------------------------

void render(const graphics::internal::FrameData &frame) {
  auto &ctx = graphics::internal::context;
  VkCommandBuffer cmd = frame.command_buffer;

  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &beginInfo);

  VkClearValue clears[2] = {};
  clears[0].color = {{0.08f, 0.09f, 0.11f, 1.0f}};
  clears[1].depthStencil = {1.0f, 0};

  VkRenderPassBeginInfo rpInfo = {};
  rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpInfo.renderPass = ctx.render_pass;
  rpInfo.framebuffer = frame.framebuffer;
  rpInfo.renderArea.offset = {0, 0};
  rpInfo.renderArea.extent = ctx.swapchain_extent;
  rpInfo.clearValueCount = 2;
  rpInfo.pClearValues = clears;

  vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

  VkViewport vp = {};
  vp.width = (float)ctx.swapchain_extent.width;
  vp.height = (float)ctx.swapchain_extent.height;
  vp.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &vp);

  VkRect2D sc = {};
  sc.extent = ctx.swapchain_extent;
  vkCmdSetScissor(cmd, 0, 1, &sc);

  // Привязываем буфер вершин
  VkBuffer vbs[] = {vertBuffer};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);

  // Привязываем буфер индексов
  vkCmdBindIndexBuffer(cmd, idxBuffer, 0, VK_INDEX_TYPE_UINT16);

  // Привязываем пайплайн
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

  // Привязываем оба набора дескрипторов (Set 0 = Сцена, Set 1 = Модель)
  VkDescriptorSet sets[2] = {descSetScene, descSetModel};
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 0, 2, sets, 0, nullptr);

  // Отрисовка параллелепипеда
  vkCmdDrawIndexed(cmd, (uint32_t)indices.size(), 1, 0, 0, 0);

  vkCmdEndRenderPass(cmd);
  vkEndCommandBuffer(cmd);
}

} // namespace application
