#include "application.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace application {

// -------------------------------------------------------------
// Структуры данных
// -------------------------------------------------------------

// Вершина: позиция, нормаль и текстурные координаты UV
struct Vertex {
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec2 uv;
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

// UBO модели (Set 1, Binding 0: Dynamic Uniform Buffer)
struct ModelUniforms {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 normalMatrix; // mat4 для std140, используется верхний 3x3
  alignas(16) glm::vec4 albedo;       // rgb = модуляция цвета (tint), w = непрозрачность
  alignas(16) glm::vec4 specular;     // rgb = цвет блика, w = коэффициент блеска (shininess)
};

// Push Constants для нетривиального сэмплирования (Доп. задание 2)
struct PushConstants {
  int enableEffects;     // 0 = выкл, 1 = вкл
  float time;            // непрерывное время анимации
  float waveDistortion;  // амплитуда волнового искажения UV
  float chromaticAberr;  // интенсивность хроматической аберрации
  float colorModulation; // интенсивность динамической модуляции цвета
};

// Структура текстуры
struct Texture {
  VkImage image = VK_NULL_HANDLE;
  VmaAllocation alloc = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkSampler sampler = VK_NULL_HANDLE;
  std::string name;
};

// -------------------------------------------------------------
// Геометрия параллелепипеда (Вариант 3: a = 1.6, b = 1.0, c = 0.8)
// -------------------------------------------------------------
const float hx = 0.8f;
const float hy = 0.5f;
const float hz = 0.4f;

static std::vector<Vertex> vertices;
static std::vector<uint16_t> indices;

// Процедурная генерация вершин, нормалей и UV для 6 граней
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

  // UV-координаты для четырех углов грани
  const glm::vec2 uvs[4] = {
      {0.0f, 1.0f}, // нижний левый
      {1.0f, 1.0f}, // нижний правый
      {1.0f, 0.0f}, // верхний правый
      {0.0f, 0.0f}  // верхний левый
  };

  for (size_t i = 0; i < quads.size(); ++i) {
    const auto &q = quads[i];
    // Процедурный расчет нормали через векторное произведение
    glm::vec3 normal = glm::normalize(glm::cross(q.p1 - q.p0, q.p2 - q.p0));

    uint16_t baseIdx = static_cast<uint16_t>(vertices.size());
    vertices.push_back({q.p0, normal, uvs[0]});
    vertices.push_back({q.p1, normal, uvs[1]});
    vertices.push_back({q.p2, normal, uvs[2]});
    vertices.push_back({q.p3, normal, uvs[3]});

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
// Вершинный и индексный буферы в чистой VRAM (Device Local)
static VkBuffer vertBuffer = VK_NULL_HANDLE;
static VmaAllocation vertAlloc = VK_NULL_HANDLE;

static VkBuffer idxBuffer = VK_NULL_HANDLE;
static VmaAllocation idxAlloc = VK_NULL_HANDLE;

// Командный пул для одноразовых операций копирования (staging)
static VkCommandPool transferCmdPool = VK_NULL_HANDLE;

// UBO сцены
static VkBuffer sceneUboBuffer = VK_NULL_HANDLE;
static VmaAllocation sceneUboAlloc = VK_NULL_HANDLE;

// Динамический UBO модели
static VkBuffer modelUboBuffer = VK_NULL_HANDLE;
static VmaAllocation modelUboAlloc = VK_NULL_HANDLE;
static VkDeviceSize dynamicAlignment = 0;

// Storage Buffers для источников света
static const size_t MAX_LIGHTS = 16;
static VkBuffer pointLightsBuffer = VK_NULL_HANDLE;
static VmaAllocation pointLightsAlloc = VK_NULL_HANDLE;

static VkBuffer spotLightsBuffer = VK_NULL_HANDLE;
static VmaAllocation spotLightsAlloc = VK_NULL_HANDLE;

// Текстуры и дескрипторы материалов
static std::vector<Texture> textures;
static int selectedTexture = 0;

static VkDescriptorSetLayout descLayoutScene = VK_NULL_HANDLE;
static VkDescriptorSetLayout descLayoutModel = VK_NULL_HANDLE;
static VkDescriptorPool descPool = VK_NULL_HANDLE;
static VkDescriptorSet descSetScene = VK_NULL_HANDLE;
static std::vector<VkDescriptorSet> descSetsModel; // отдельный descriptor set для каждого материала

static VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
static VkPipeline pipeline = VK_NULL_HANDLE;

// -------------------------------------------------------------
// Параметры сцены, камеры, модели и освещения
// -------------------------------------------------------------

// Камера
static glm::vec3 camPos = {0.0f, 1.8f, 4.0f};
static float camYaw = 0.0f;
static float camPitch = -15.0f;
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
static float matAlbedoTint[3] = {1.0f, 1.0f, 1.0f}; // модуляция цвета текстуры
static float matSpecular[3] = {1.0f, 1.0f, 1.0f};    // цвет блика
static float matShininess = 32.0f;                   // коэффициент блеска

// Нетривиальное сэмплирование (Доп. задание 2)
static bool enableNonTrivialSampling = true;
static float waveDistortion = 1.0f;
static float chromaticAberr = 1.0f;
static float colorModulation = 0.8f;

// Рассеянный свет (Ambient)
static float ambientColor[3] = {1.0f, 1.0f, 1.0f};
static float ambientIntensity = 0.20f;

// Направленный свет (Directional)
static bool dirLightEnabled = true;
static float dirLightDir[3] = {-0.6f, -1.0f, -0.8f};
static float dirLightColor[3] = {1.0f, 0.96f, 0.88f};
static float dirLightIntensity = 1.2f;

// Списки источников света
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
  float innerAngle;
  float outerAngle;
};

static std::vector<PointLightUI> pointLightsList = {
    {{1.8f, 1.2f, 1.2f}, {1.0f, 0.4f, 0.4f}, 5.0f},
    {{-1.8f, 1.0f, -0.8f}, {0.4f, 1.0f, 0.5f}, 4.0f}
};

static std::vector<SpotLightUI> spotLightsList = {
    {{0.0f, 3.0f, 2.0f}, {0.0f, -1.0f, -0.7f}, {1.0f, 0.95f, 0.5f}, 12.0f, 15.0f, 28.0f}
};

// -------------------------------------------------------------
// Вспомогательные функции Vulkan
// -------------------------------------------------------------

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

// Одноразовый буфер команд для staging операций
static VkCommandBuffer beginSingleTimeCommands() {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = transferCmdPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(graphics::internal::context.device, &allocInfo, &cmd);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &beginInfo);

  return cmd;
}

static void endSingleTimeCommands(VkCommandBuffer cmd) {
  auto &ctx = graphics::internal::context;
  vkEndCommandBuffer(cmd);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd;

  vkQueueSubmit(ctx.graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(ctx.graphics_queue);

  vkFreeCommandBuffers(ctx.device, transferCmdPool, 1, &cmd);
}

// Создание хост-буфера (для UBO/SSBO)
static bool makeBufferHost(VkDeviceSize size, VkBufferUsageFlags usage,
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

static void copyToBuffer(VmaAllocation alloc, const void *data, VkDeviceSize size, VkDeviceSize offset = 0) {
  void *mapped = nullptr;
  vmaMapMemory(graphics::internal::context.allocator, alloc, &mapped);
  std::memcpy(static_cast<char*>(mapped) + offset, data, size);
  vmaUnmapMemory(graphics::internal::context.allocator, alloc);
}

// Создание буфера в чистой видеопамяти (VRAM / Device Local) через Staging Buffer
static bool createBufferVRAM(VkDeviceSize size, VkBufferUsageFlags usage, const void *data,
                            VkBuffer &deviceBuf, VmaAllocation &deviceAlloc) {
  auto &ctx = graphics::internal::context;

  // 1. Создаем промежуточный Staging Buffer в оперативной памяти (Host Visible)
  VkBuffer stagingBuf = VK_NULL_HANDLE;
  VmaAllocation stagingAlloc = VK_NULL_HANDLE;

  VkBufferCreateInfo stagingBi = {};
  stagingBi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  stagingBi.size = size;
  stagingBi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  stagingBi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo stagingAi = {};
  stagingAi.usage = VMA_MEMORY_USAGE_AUTO;
  stagingAi.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

  if (vmaCreateBuffer(ctx.allocator, &stagingBi, &stagingAi, &stagingBuf, &stagingAlloc, nullptr) != VK_SUCCESS) {
    return false;
  }

  // Копируем данные с CPU в staging buffer
  copyToBuffer(stagingAlloc, data, size);

  // 2. Создаем постоянный буфер в VRAM (Device Local)
  VkBufferCreateInfo deviceBi = {};
  deviceBi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  deviceBi.size = size;
  deviceBi.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  deviceBi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo deviceAi = {};
  deviceAi.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE; // Быстрая видеопамять VRAM!

  if (vmaCreateBuffer(ctx.allocator, &deviceBi, &deviceAi, &deviceBuf, &deviceAlloc, nullptr) != VK_SUCCESS) {
    vmaDestroyBuffer(ctx.allocator, stagingBuf, stagingAlloc);
    return false;
  }

  // 3. Аппаратное копирование силами GPU: Staging Buffer (RAM) -> Device Buffer (VRAM)
  VkCommandBuffer cmd = beginSingleTimeCommands();
  VkBufferCopy copyRegion{};
  copyRegion.size = size;
  vkCmdCopyBuffer(cmd, stagingBuf, deviceBuf, 1, &copyRegion);
  endSingleTimeCommands(cmd);

  // 4. Освобождаем промежуточный буфер
  vmaDestroyBuffer(ctx.allocator, stagingBuf, stagingAlloc);
  return true;
}

// -------------------------------------------------------------
// Загрузка и создание текстур
// -------------------------------------------------------------

static bool createTextureFromRGBA(const void *pixels, int width, int height,
                                 const std::string &name, Texture &outTex) {
  auto &ctx = graphics::internal::context;
  VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4;

  // 1. Создаем Staging Buffer для пикселей
  VkBuffer stagingBuf = VK_NULL_HANDLE;
  VmaAllocation stagingAlloc = VK_NULL_HANDLE;

  VkBufferCreateInfo stagingBi = {};
  stagingBi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  stagingBi.size = imageSize;
  stagingBi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  stagingBi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo stagingAi = {};
  stagingAi.usage = VMA_MEMORY_USAGE_AUTO;
  stagingAi.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

  if (vmaCreateBuffer(ctx.allocator, &stagingBi, &stagingAi, &stagingBuf, &stagingAlloc, nullptr) != VK_SUCCESS) {
    return false;
  }
  copyToBuffer(stagingAlloc, pixels, imageSize);

  // 2. Создаем объект VkImage в VRAM (Оптимальный для GPU формат)
  VkImageCreateInfo imageInfo = {};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = static_cast<uint32_t>(width);
  imageInfo.extent.height = static_cast<uint32_t>(height);
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

  VmaAllocationCreateInfo imageAllocInfo = {};
  imageAllocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE; // В чистую видеопамять VRAM

  if (vmaCreateImage(ctx.allocator, &imageInfo, &imageAllocInfo, &outTex.image, &outTex.alloc, nullptr) != VK_SUCCESS) {
    vmaDestroyBuffer(ctx.allocator, stagingBuf, stagingAlloc);
    return false;
  }

  // 3. Копирование и смена layout изображения через барьеры конвейера
  VkCommandBuffer cmd = beginSingleTimeCommands();

  // Переход: UNDEFINED -> TRANSFER_DST_OPTIMAL
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = outTex.image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                       0, nullptr, 0, nullptr, 1, &barrier);

  // Копирование пикселей из Staging Buffer в VkImage
  VkBufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};

  vkCmdCopyBufferToImage(cmd, stagingBuf, outTex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  // Переход: TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                       0, nullptr, 0, nullptr, 1, &barrier);

  endSingleTimeCommands(cmd);
  vmaDestroyBuffer(ctx.allocator, stagingBuf, stagingAlloc);

  // 4. Создаем VkImageView
  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = outTex.image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  if (vkCreateImageView(ctx.device, &viewInfo, nullptr, &outTex.view) != VK_SUCCESS) {
    return false;
  }

  // 5. Создаем VkSampler
  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.anisotropyEnable = VK_FALSE;
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  if (vkCreateSampler(ctx.device, &samplerInfo, nullptr, &outTex.sampler) != VK_SUCCESS) {
    return false;
  }

  outTex.name = name;
  return true;
}

// Загрузка изображения с диска
static bool loadTextureFromFile(const std::string &filename, const std::string &displayName, Texture &outTex) {
  const std::string paths[] = {
      filename,
      "textures/" + filename,
      "../textures/" + filename,
      "../../textures/" + filename,
      "LectionsLab/" + filename,
      "../LectionsLab/" + filename,
      "../../LectionsLab/" + filename
  };

  int texWidth = 0, texHeight = 0, texChannels = 0;
  stbi_uc *pixels = nullptr;

  for (const auto &p : paths) {
    pixels = stbi_load(p.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (pixels) {
      std::cout << "Loaded texture: " << p << " (" << texWidth << "x" << texHeight << ")\n";
      break;
    }
  }

  if (!pixels) {
    std::cerr << "Failed to load texture image: " << filename << "\n";
    return false;
  }

  bool ok = createTextureFromRGBA(pixels, texWidth, texHeight, displayName, outTex);
  stbi_image_free(pixels);
  return ok;
}

// Генерация процедурной текстуры «Шахматка»
static bool generateCheckerboardTexture(Texture &outTex) {
  const int size = 512;
  const int tileSize = 64;
  std::vector<uint32_t> pixels(size * size);

  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      bool isDark = ((x / tileSize) ^ (y / tileSize)) & 1;
      bool isBorder = (x % tileSize == 0) || (y % tileSize == 0);

      uint32_t color;
      if (isBorder) {
        color = 0xFF00E5FF; // яркая бирюзовая рамка (ABGR: R=FF, G=E5, B=00)
      } else if (isDark) {
        color = 0xFF2A1635; // глубокий фиолетовый
      } else {
        color = 0xFFE0E5EB; // светлый серебристый
      }
      pixels[y * size + x] = color;
    }
  }

  return createTextureFromRGBA(pixels.data(), size, size, "Procedural Checkerboard", outTex);
}

// Генерация процедурной текстуры «Неоновая кибер-сетка»
static bool generateCyberGridTexture(Texture &outTex) {
  const int size = 512;
  std::vector<uint32_t> pixels(size * size);

  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      int gx = x % 64;
      int gy = y % 64;
      float distToLine = static_cast<float>(std::min({gx, 64 - gx, gy, 64 - gy}));

      // Центр ячейки
      float cx = (x % 128) - 64.0f;
      float cy = (y % 128) - 64.0f;
      float dCenter = std::sqrt(cx * cx + cy * cy);

      uint8_t r = 20, g = 25, b = 35; // темный фон
      if (distToLine < 3.0f) {
        r = 255; g = 140; b = 30; // неоновые оранжевые линии сетки
      } else if (dCenter < 20.0f) {
        r = 40; g = 220; b = 255; // неоновые круги в узлах сетки
      }

      uint32_t color = (0xFF << 24) | (b << 16) | (g << 8) | r;
      pixels[y * size + x] = color;
    }
  }

  return createTextureFromRGBA(pixels.data(), size, size, "Neon Cyber Grid", outTex);
}

// -------------------------------------------------------------
// Инициализация
// -------------------------------------------------------------

bool initialize() {
  auto &ctx = graphics::internal::context;

  // 0. Создаем командный пул для staging-операций
  VkCommandPoolCreateInfo cmdPoolInfo{};
  cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cmdPoolInfo.queueFamilyIndex = ctx.graphics_queue_index;
  if (vkCreateCommandPool(ctx.device, &cmdPoolInfo, nullptr, &transferCmdPool) != VK_SUCCESS) {
    return false;
  }

  // 1. Процедурная генерация вершин, нормалей и UV
  generateParallelepiped(hx, hy, hz);

  // 2. Создание буферов вершин и индексов в VRAM (Device Local через Staging)
  VkDeviceSize vertSize = sizeof(vertices[0]) * vertices.size();
  if (!createBufferVRAM(vertSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices.data(), vertBuffer, vertAlloc))
    return false;

  VkDeviceSize idxSize = sizeof(indices[0]) * indices.size();
  if (!createBufferVRAM(idxSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indices.data(), idxBuffer, idxAlloc))
    return false;

  // 3. Вычисление шага динамического UBO (minUniformBufferOffsetAlignment)
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(ctx.physical_device, &props);
  VkDeviceSize minAlign = props.limits.minUniformBufferOffsetAlignment;

  dynamicAlignment = sizeof(ModelUniforms);
  if (minAlign > 0) {
    dynamicAlignment = (dynamicAlignment + minAlign - 1) & ~(minAlign - 1);
  }

  // Выделяем буфер под несколько моделей с динамическим сдвигом
  const size_t MAX_MODELS = 4;
  VkDeviceSize modelUboSize = dynamicAlignment * MAX_MODELS;
  if (!makeBufferHost(modelUboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, modelUboBuffer, modelUboAlloc))
    return false;

  // UBO сцены
  if (!makeBufferHost(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneUboBuffer, sceneUboAlloc))
    return false;

  // Storage Buffers под источники света
  VkDeviceSize pointLightBufSize = sizeof(PointLight) * MAX_LIGHTS;
  if (!makeBufferHost(pointLightBufSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, pointLightsBuffer, pointLightsAlloc))
    return false;

  VkDeviceSize spotLightBufSize = sizeof(SpotLight) * MAX_LIGHTS;
  if (!makeBufferHost(spotLightBufSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spotLightsBuffer, spotLightsAlloc))
    return false;

  // 4. Загрузка текстур (Доп. задание 1)
  Texture tex1, tex2, tex3;
  if (loadTextureFromFile("pictureforlab.jpg", "1. User Photo (pictureforlab.jpg)", tex1)) {
    textures.push_back(tex1);
  }
  if (generateCheckerboardTexture(tex2)) {
    textures.push_back(tex2);
  }
  if (generateCyberGridTexture(tex3)) {
    textures.push_back(tex3);
  }

  if (textures.empty()) {
    std::cerr << "No textures could be created!\n";
    return false;
  }

  // 5. Описание дескрипторов

  // Набор 0: Сцена (Binding 0: Scene UBO, Binding 1: Point Lights SSBO, Binding 2: Spot Lights SSBO)
  VkDescriptorSetLayoutBinding sceneBindings[3] = {};

  sceneBindings[0].binding = 0;
  sceneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  sceneBindings[0].descriptorCount = 1;
  sceneBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  sceneBindings[1].binding = 1;
  sceneBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sceneBindings[1].descriptorCount = 1;
  sceneBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

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

  // Набор 1: Материал/Модель (Binding 0: Dynamic Model UBO, Binding 1: Combined Image Sampler)
  VkDescriptorSetLayoutBinding modelBindings[2] = {};

  modelBindings[0].binding = 0;
  modelBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; // Динамический UBO!
  modelBindings[0].descriptorCount = 1;
  modelBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  modelBindings[1].binding = 1;
  modelBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; // Текстура + сэмплер!
  modelBindings[1].descriptorCount = 1;
  modelBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo modelLayoutInfo = {};
  modelLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  modelLayoutInfo.bindingCount = 2;
  modelLayoutInfo.pBindings = modelBindings;

  if (vkCreateDescriptorSetLayout(ctx.device, &modelLayoutInfo, nullptr, &descLayoutModel) != VK_SUCCESS) {
    return false;
  }

  // 6. Descriptor Pool (Scene UBO + SSBOs + Dynamic UBOs + Texture Samplers)
  VkDescriptorPoolSize poolSizes[4] = {};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[0].descriptorCount = 1;
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSizes[1].descriptorCount = 2;
  poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  poolSizes[2].descriptorCount = static_cast<uint32_t>(textures.size());
  poolSizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSizes[3].descriptorCount = static_cast<uint32_t>(textures.size());

  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.maxSets = 1 + static_cast<uint32_t>(textures.size());
  poolInfo.poolSizeCount = 4;
  poolInfo.pPoolSizes = poolSizes;

  if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descPool) != VK_SUCCESS) {
    return false;
  }

  // 7. Выделение Descriptor Set для сцены (Set 0)
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &descLayoutScene;
  if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &descSetScene) != VK_SUCCESS)
    return false;

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

  // 8. Выделение Descriptor Sets для каждого материала / текстуры (Set 1)
  descSetsModel.resize(textures.size());
  for (size_t i = 0; i < textures.size(); ++i) {
    allocInfo.pSetLayouts = &descLayoutModel;
    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &descSetsModel[i]) != VK_SUCCESS)
      return false;

    // Binding 0: Динамический UBO модели
    VkDescriptorBufferInfo modelBufInfo = {};
    modelBufInfo.buffer = modelUboBuffer;
    modelBufInfo.offset = 0;
    modelBufInfo.range = sizeof(ModelUniforms);

    // Binding 1: Текстура и сэмплер материала
    VkDescriptorImageInfo imageInfo = {};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = textures[i].view;
    imageInfo.sampler = textures[i].sampler;

    VkWriteDescriptorSet modelWrites[2] = {};
    modelWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    modelWrites[0].dstSet = descSetsModel[i];
    modelWrites[0].dstBinding = 0;
    modelWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    modelWrites[0].descriptorCount = 1;
    modelWrites[0].pBufferInfo = &modelBufInfo;

    modelWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    modelWrites[1].dstSet = descSetsModel[i];
    modelWrites[1].dstBinding = 1;
    modelWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    modelWrites[1].descriptorCount = 1;
    modelWrites[1].pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(ctx.device, 2, modelWrites, 0, nullptr);
  }

  // 9. Загрузка шейдеров
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

  // 10. Формат вершин (pos = 0, normal = 1, uv = 2)
  VkVertexInputBindingDescription bindDesc = {};
  bindDesc.binding = 0;
  bindDesc.stride = sizeof(Vertex);
  bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  VkVertexInputAttributeDescription attrDesc[3] = {};
  attrDesc[0].binding = 0;
  attrDesc[0].location = 0;
  attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attrDesc[0].offset = offsetof(Vertex, pos);

  attrDesc[1].binding = 0;
  attrDesc[1].location = 1;
  attrDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attrDesc[1].offset = offsetof(Vertex, normal);

  attrDesc[2].binding = 0;
  attrDesc[2].location = 2;
  attrDesc[2].format = VK_FORMAT_R32G32_SFLOAT;
  attrDesc[2].offset = offsetof(Vertex, uv);

  VkPipelineVertexInputStateCreateInfo vInput = {};
  vInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vInput.vertexBindingDescriptionCount = 1;
  vInput.pVertexBindingDescriptions = &bindDesc;
  vInput.vertexAttributeDescriptionCount = 3;
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
  raster.cullMode = VK_CULL_MODE_NONE;
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

  // Push Constants для параметров нетривиального сэмплирования
  VkPushConstantRange pcRange = {};
  pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  pcRange.offset = 0;
  pcRange.size = sizeof(PushConstants);

  // Pipeline Layout со ссылкой на 2 набора дескрипторов (Set 0 и Set 1) и Push Constants
  VkDescriptorSetLayout layouts[2] = {descLayoutScene, descLayoutModel};
  VkPipelineLayoutCreateInfo plInfo = {};
  plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plInfo.setLayoutCount = 2;
  plInfo.pSetLayouts = layouts;
  plInfo.pushConstantRangeCount = 1;
  plInfo.pPushConstantRanges = &pcRange;

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

  // Удаляем текстуры
  for (auto &t : textures) {
    if (t.sampler) vkDestroySampler(ctx.device, t.sampler, nullptr);
    if (t.view) vkDestroyImageView(ctx.device, t.view, nullptr);
    if (t.image) vmaDestroyImage(ctx.allocator, t.image, t.alloc);
  }
  textures.clear();

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

  if (transferCmdPool)
    vkDestroyCommandPool(ctx.device, transferCmdPool, nullptr);
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
  ImGui::Begin("Lab 3: Texture Mapping & Sampling (Grade 4)");

  ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.4f, 1.0f), "Tasks: Base + Add. 1 & 2 (Grade 4)");
  ImGui::Separator();

  // 1. Выбор текстуры и материала (Доп. задание 1)
  if (ImGui::CollapsingHeader("1. Texture Selection (Task 1)", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (!textures.empty()) {
      if (ImGui::BeginCombo("Active Texture", textures[selectedTexture].name.c_str())) {
        for (int i = 0; i < static_cast<int>(textures.size()); ++i) {
          bool isSelected = (selectedTexture == i);
          if (ImGui::Selectable(textures[i].name.c_str(), isSelected)) {
            selectedTexture = i;
          }
          if (isSelected) {
            ImGui::SetItemDefaultFocus();
          }
        }
        ImGui::EndCombo();
      }
    }
    ImGui::ColorEdit3("Albedo Tint", matAlbedoTint);
    ImGui::ColorEdit3("Specular Color", matSpecular);
    ImGui::SliderFloat("Shininess", &matShininess, 1.0f, 128.0f, "%.1f");
  }

  // 2. Нетривиальное сэмплирование (Доп. задание 2)
  if (ImGui::CollapsingHeader("2. Non-Trivial Sampling & Effects (Task 2)", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("Enable Non-Trivial Sampling", &enableNonTrivialSampling);
    if (enableNonTrivialSampling) {
      ImGui::SliderFloat("Wave Distortion (UV)", &waveDistortion, 0.0f, 3.0f, "%.2fx");
      ImGui::SliderFloat("Chromatic Aberration", &chromaticAberr, 0.0f, 3.0f, "%.2fx");
      ImGui::SliderFloat("Color Pulsation", &colorModulation, 0.0f, 2.0f, "%.2fx");
    }
  }

  // 3. Камера
  if (ImGui::CollapsingHeader("3. Camera Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
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

  // 4. Модель и трансформация
  if (ImGui::CollapsingHeader("4. Model Transforms", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::DragFloat3("Position", glm::value_ptr(modelPos), 0.05f);
    ImGui::SliderFloat3("Rotation", glm::value_ptr(modelRot), -180.0f, 180.0f, "%.1f deg");
    ImGui::DragFloat3("Scale", glm::value_ptr(modelScale), 0.05f, 0.1f, 4.0f);

    ImGui::Separator();
    if (playAnim) {
      if (ImGui::Button("Pause Animation ||")) playAnim = false;
    } else {
      if (ImGui::Button("Play Animation >")) playAnim = true;
    }
    ImGui::SameLine();
    ImGui::SliderFloat("Speed", &animSpeed, 0.0f, 3.0f, "%.2fx");
  }

  // 5. Освещение (Ambient, Directional, Point, Spot)
  if (ImGui::CollapsingHeader("5. Lighting (Blinn-Phong)", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::TextDisabled("Ambient Light:");
    ImGui::ColorEdit3("Ambient Color", ambientColor);
    ImGui::SliderFloat("Ambient Intensity", &ambientIntensity, 0.0f, 1.0f, "%.2f");

    ImGui::Separator();
    ImGui::TextDisabled("Directional Light:");
    ImGui::Checkbox("Enable Directional Light", &dirLightEnabled);
    if (dirLightEnabled) {
      ImGui::SliderFloat3("Direction", dirLightDir, -1.0f, 1.0f);
      ImGui::ColorEdit3("Sun Color", dirLightColor);
      ImGui::SliderFloat("Sun Intensity", &dirLightIntensity, 0.0f, 5.0f, "%.2f");
    }

    ImGui::Separator();
    ImGui::Text("Point Lights: %zu / %zu", pointLightsList.size(), MAX_LIGHTS);
    if (pointLightsList.size() < MAX_LIGHTS && ImGui::Button("+ Add Point Light")) {
      pointLightsList.push_back({{0.0f, 1.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, 4.0f});
    }

    int toRemovePoint = -1;
    for (size_t i = 0; i < pointLightsList.size(); ++i) {
      ImGui::PushID((int)i);
      char header[32];
      snprintf(header, sizeof(header), "Point Light #%zu", i + 1);
      if (ImGui::TreeNode(header)) {
        ImGui::DragFloat3("Pos", glm::value_ptr(pointLightsList[i].pos), 0.05f);
        ImGui::ColorEdit3("Color", glm::value_ptr(pointLightsList[i].color));
        ImGui::SliderFloat("Intensity", &pointLightsList[i].intensity, 0.0f, 20.0f, "%.1f");
        if (ImGui::Button("Remove")) toRemovePoint = (int)i;
        ImGui::TreePop();
      }
      ImGui::PopID();
    }
    if (toRemovePoint >= 0) pointLightsList.erase(pointLightsList.begin() + toRemovePoint);

    ImGui::Separator();
    ImGui::Text("Spot Lights: %zu / %zu", spotLightsList.size(), MAX_LIGHTS);
    if (spotLightsList.size() < MAX_LIGHTS && ImGui::Button("+ Add Spot Light")) {
      spotLightsList.push_back({{0.0f, 2.5f, 1.5f}, {0.0f, -1.0f, -0.5f}, {1.0f, 0.9f, 0.5f}, 10.0f, 15.0f, 30.0f});
    }

    int toRemoveSpot = -1;
    for (size_t i = 0; i < spotLightsList.size(); ++i) {
      ImGui::PushID(100 + (int)i);
      char header[32];
      snprintf(header, sizeof(header), "Spot Light #%zu", i + 1);
      if (ImGui::TreeNode(header)) {
        ImGui::DragFloat3("Pos", glm::value_ptr(spotLightsList[i].pos), 0.05f);
        ImGui::SliderFloat3("Dir", glm::value_ptr(spotLightsList[i].dir), -1.0f, 1.0f);
        ImGui::ColorEdit3("Color", glm::value_ptr(spotLightsList[i].color));
        ImGui::SliderFloat("Intensity", &spotLightsList[i].intensity, 0.0f, 30.0f, "%.1f");
        ImGui::SliderFloat("Inner Angle", &spotLightsList[i].innerAngle, 1.0f, 60.0f, "%.1f deg");
        ImGui::SliderFloat("Outer Angle", &spotLightsList[i].outerAngle, spotLightsList[i].innerAngle, 85.0f, "%.1f deg");
        if (ImGui::Button("Remove")) toRemoveSpot = (int)i;
        ImGui::TreePop();
      }
      ImGui::PopID();
    }
    if (toRemoveSpot >= 0) spotLightsList.erase(spotLightsList.begin() + toRemoveSpot);
  }

  ImGui::End();

  // -------------------------------------------------------------
  // Математические расчеты матриц
  // -------------------------------------------------------------
  auto &ctx = graphics::internal::context;
  float aspect = (float)ctx.swapchain_extent.width /
                 (float)(ctx.swapchain_extent.height > 0 ? ctx.swapchain_extent.height : 1);

  // Матрица проекции (инверсия Y под Vulkan)
  glm::mat4 projMat = glm::perspective(glm::radians(camFov), aspect, 0.1f, 100.0f);
  projMat[1][1] *= -1.0f;

  // Матрица вида
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

  glm::mat4 normalMatrix = computeNormalMatrix(modelMat);

  // -------------------------------------------------------------
  // Отправка данных в буферы GPU
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

  // 2. Dynamic Model UBO (смещение 0 для модели 0)
  ModelUniforms modelUbo = {};
  modelUbo.model = modelMat;
  modelUbo.normalMatrix = normalMatrix;
  modelUbo.albedo = glm::vec4(matAlbedoTint[0], matAlbedoTint[1], matAlbedoTint[2], 1.0f);
  modelUbo.specular = glm::vec4(matSpecular[0], matSpecular[1], matSpecular[2], matShininess);
  copyToBuffer(modelUboAlloc, &modelUbo, sizeof(modelUbo), 0);

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

  // Привязываем буфер вершин из чистой VRAM
  VkBuffer vbs[] = {vertBuffer};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);

  // Привязываем буфер индексов из чистой VRAM
  vkCmdBindIndexBuffer(cmd, idxBuffer, 0, VK_INDEX_TYPE_UINT16);

  // Привязываем графический конвейер
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

  // Привязываем Set 0 (Сцена: без динамических смещений)
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 0, 1, &descSetScene, 0, nullptr);

  // Привязываем Set 1 (Материал с выбранной текстурой + динамический UBO модели со смещением)
  uint32_t dynamicOffset = 0; // Смещение модели 0 в Dynamic Uniform Buffer
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 1, 1,
                          &descSetsModel[selectedTexture], 1, &dynamicOffset);

  // Передаем параметры нетривиального сэмплирования через Push Constants (Доп. задание 2)
  PushConstants pc{};
  pc.enableEffects = enableNonTrivialSampling ? 1 : 0;
  pc.time = static_cast<float>(animTime);
  pc.waveDistortion = waveDistortion;
  pc.chromaticAberr = chromaticAberr;
  pc.colorModulation = colorModulation;

  vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);

  // Отрисовка параллелепипеда
  vkCmdDrawIndexed(cmd, (uint32_t)indices.size(), 1, 0, 0, 0);

  vkCmdEndRenderPass(cmd);
  vkEndCommandBuffer(cmd);
}

} // namespace application
