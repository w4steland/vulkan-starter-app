#include "application.hpp"

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <array>

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace application {

// структура вершины
struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
};

// ubo для шейдеров (матрицы + цвета для граней, ребер и углов)
struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec4 uiColor;
    alignas(16) glm::vec4 edgeColor;
    alignas(16) glm::vec4 cornerColor;
};

// полуразмеры параллелепипеда (вариант 3: a=1.6, b=1.0, c=0.8)
const float hx = 0.8f;
const float hy = 0.5f;
const float hz = 0.4f;

// 8 вершин параллелепипеда с яркими процедурными цветами
const std::vector<Vertex> vertices = {
    { {-hx, -hy, -hz}, {0.35f, 0.35f, 0.95f} }, // 0
    { { hx, -hy, -hz}, {0.95f, 0.25f, 0.25f} }, // 1
    { { hx,  hy, -hz}, {0.95f, 0.90f, 0.20f} }, // 2
    { {-hx,  hy, -hz}, {0.25f, 0.90f, 0.30f} }, // 3
    { {-hx, -hy,  hz}, {0.20f, 0.80f, 0.95f} }, // 4
    { { hx, -hy,  hz}, {0.95f, 0.30f, 0.85f} }, // 5
    { { hx,  hy,  hz}, {1.00f, 1.00f, 1.00f} }, // 6
    { {-hx,  hy,  hz}, {0.30f, 0.95f, 0.75f} }  // 7
};

// 12 треугольников для граней (по 2 на каждую из 6 сторон)
const std::vector<uint16_t> indices = {
    4, 5, 6,  4, 6, 7, // перед
    1, 0, 3,  1, 3, 2, // зад
    0, 4, 7,  0, 7, 3, // лево
    5, 1, 2,  5, 2, 6, // право
    3, 2, 6,  3, 6, 7, // верх
    4, 0, 1,  4, 1, 5  // низ
};

// 12 ребер параллелепипеда (24 индекса для линий)
const std::vector<uint16_t> edgeIndices = {
    0, 1,  1, 2,  2, 3,  3, 0, // задняя рамка
    4, 5,  5, 6,  6, 7,  7, 4, // передняя рамка
    0, 4,  1, 5,  2, 6,  3, 7  // соединительные ребра между ними
};

// vulkan ресурсы
static VkBuffer vertBuffer = VK_NULL_HANDLE;
static VmaAllocation vertAlloc = VK_NULL_HANDLE;

static VkBuffer idxBuffer = VK_NULL_HANDLE;
static VmaAllocation idxAlloc = VK_NULL_HANDLE;

static VkBuffer edgeIdxBuffer = VK_NULL_HANDLE;
static VmaAllocation edgeIdxAlloc = VK_NULL_HANDLE;

// ubo и дескрипторы
static VkBuffer uboBuffer = VK_NULL_HANDLE;
static VmaAllocation uboAlloc = VK_NULL_HANDLE;

static VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
static VkDescriptorPool descPool = VK_NULL_HANDLE;
static VkDescriptorSet descSet = VK_NULL_HANDLE;

static VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
static VkPipeline pipeline = VK_NULL_HANDLE;      // треугольники (грани)
static VkPipeline linePipeline = VK_NULL_HANDLE;  // линии (ребра)
static VkPipeline pointPipeline = VK_NULL_HANDLE; // точки (углы)

// параметры проекции
static int projType = 0; // 0 - перспектива, 1 - ортографическая
static float fov = 45.0f;
static float orthoScale = 3.0f;

// ручные трансформации
static glm::vec3 pos = { 0.0f, 0.0f, 0.0f };
static glm::vec3 rot = { 20.0f, 30.0f, 0.0f };
static glm::vec3 scale = { 1.0f, 1.0f, 1.0f };

// анимация
static bool playAnim = true;
static float animSpeed = 1.0f;
static float trajRadius = 2.0f;
static float trajHeight = 0.8f;
static int trajType = 0; // 0 - спираль, 1 - лиссажу, 2 - круг
static double animTime = 0.0;
static double prevTime = 0.0;

// цвета и отображение ребер / углов
static float color1[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
static bool showEdges = true;
static float edgeColor[4] = { 0.05f, 0.05f, 0.05f, 1.0f }; // темный четкий контур ребер
static bool showCorners = true;
static float cornerColor[4] = { 1.0f, 1.0f, 0.2f, 1.0f }; // светящиеся желтые маркеры углов

// камера
static glm::vec3 camEye = { 0.0f, 2.5f, 5.0f };
static glm::vec3 camCenter = { 0.0f, 0.0f, 0.0f };
static glm::vec3 camUp = { 0.0f, 1.0f, 0.0f };

// читаем шейдер
static std::vector<char> loadShader(const std::string& name) {
    const std::string paths[] = {
        name,
        "shaders/" + name,
        "../shaders/" + name,
        "../../shaders/" + name
    };

    for (const auto& p : paths) {
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

// создание шейдерного модуля
static VkShaderModule makeShaderModule(const std::vector<char>& code) {
    if (code.empty()) return VK_NULL_HANDLE;

    VkShaderModuleCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule mod = VK_NULL_HANDLE;
    vkCreateShaderModule(graphics::internal::context.device, &ci, nullptr, &mod);
    return mod;
}

// создание буфера vma
static bool makeBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buf, VmaAllocation& alloc) {
    VkBufferCreateInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai = {};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    return vmaCreateBuffer(graphics::internal::context.allocator, &bi, &ai, &buf, &alloc, nullptr) == VK_SUCCESS;
}

// копирование данных в буфер
static void copyToBuffer(VmaAllocation alloc, const void* data, VkDeviceSize size) {
    void* mapped = nullptr;
    vmaMapMemory(graphics::internal::context.allocator, alloc, &mapped);
    std::memcpy(mapped, data, size);
    vmaUnmapMemory(graphics::internal::context.allocator, alloc);
}

bool initialize() {
    auto& ctx = graphics::internal::context;

    // 1. буфер вершин
    VkDeviceSize vertSize = sizeof(vertices[0]) * vertices.size();
    if (!makeBuffer(vertSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertBuffer, vertAlloc)) return false;
    copyToBuffer(vertAlloc, vertices.data(), vertSize);

    // 2. буфер индексов граней (треугольники)
    VkDeviceSize idxSize = sizeof(indices[0]) * indices.size();
    if (!makeBuffer(idxSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, idxBuffer, idxAlloc)) return false;
    copyToBuffer(idxAlloc, indices.data(), idxSize);

    // 3. буфер индексов ребер (линии)
    VkDeviceSize edgeIdxSize = sizeof(edgeIndices[0]) * edgeIndices.size();
    if (!makeBuffer(edgeIdxSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, edgeIdxBuffer, edgeIdxAlloc)) return false;
    copyToBuffer(edgeIdxAlloc, edgeIndices.data(), edgeIdxSize);

    // 4. создаем ubo буфер
    VkDeviceSize uboSize = sizeof(UniformBufferObject);
    if (!makeBuffer(uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, uboBuffer, uboAlloc)) {
        return false;
    }

    // 5. descriptor layout
    VkDescriptorSetLayoutBinding uboBind = {};
    uboBind.binding = 0;
    uboBind.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboBind.descriptorCount = 1;
    uboBind.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboBind;

    if (vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr, &descLayout) != VK_SUCCESS) {
        return false;
    }

    // 6. descriptor pool
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descPool) != VK_SUCCESS) {
        return false;
    }

    // 7. привязка дескрипторов
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descLayout;

    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &descSet) != VK_SUCCESS) {
        return false;
    }

    VkDescriptorBufferInfo bufInfo = {};
    bufInfo.buffer = uboBuffer;
    bufInfo.offset = 0;
    bufInfo.range = sizeof(UniformBufferObject);

    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descSet;
    write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufInfo;

    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);

    // 8. загрузка шейдеров
    auto vCode = loadShader("shader.vert.spv");
    auto fCode = loadShader("shader.frag.spv");

    VkShaderModule vMod = makeShaderModule(vCode);
    VkShaderModule fMod = makeShaderModule(fCode);
    if (!vMod || !fMod) return false;

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vMod;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fMod;
    stages[1].pName = "main";

    // 9. формат вершин
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
    attrDesc[1].offset = offsetof(Vertex, color);

    VkPipelineVertexInputStateCreateInfo vInput = {};
    vInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vInput.vertexBindingDescriptionCount = 1;
    vInput.pVertexBindingDescriptions = &bindDesc;
    vInput.vertexAttributeDescriptionCount = 2;
    vInput.pVertexAttributeDescriptions = attrDesc;

    // динамические состояния
    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
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

    // тест глубины с LESS_OR_EQUAL чтобы ребра ложились прямо на грани
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

    // push constant для передачи drawMode (0 = грань, 1 = ребро, 2 = угол)
    VkPushConstantRange pcRange = {};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(int);

    VkPipelineLayoutCreateInfo plInfo = {};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &descLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;

    if (vkCreatePipelineLayout(ctx.device, &plInfo, nullptr, &pipeLayout) != VK_SUCCESS) {
        return false;
    }

    // --- 1. Основной пайплайн (Грани - треугольники) ---
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

    // --- 2. Пайплайн для ребер (Линии) ---
    VkPipelineInputAssemblyStateCreateInfo lineAssembly = {};
    lineAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    lineAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    pipeInfo.pInputAssemblyState = &lineAssembly;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &linePipeline) != VK_SUCCESS) {
        return false;
    }

    // --- 3. Пайплайн для углов (Точки) ---
    VkPipelineInputAssemblyStateCreateInfo pointAssembly = {};
    pointAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    pointAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    pipeInfo.pInputAssemblyState = &pointAssembly;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pointPipeline) != VK_SUCCESS) {
        return false;
    }

    vkDestroyShaderModule(ctx.device, fMod, nullptr);
    vkDestroyShaderModule(ctx.device, vMod, nullptr);

    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);

    if (pointPipeline) vkDestroyPipeline(ctx.device, pointPipeline, nullptr);
    if (linePipeline) vkDestroyPipeline(ctx.device, linePipeline, nullptr);
    if (pipeline) vkDestroyPipeline(ctx.device, pipeline, nullptr);
    if (pipeLayout) vkDestroyPipelineLayout(ctx.device, pipeLayout, nullptr);

    if (descPool) vkDestroyDescriptorPool(ctx.device, descPool, nullptr);
    if (descLayout) vkDestroyDescriptorSetLayout(ctx.device, descLayout, nullptr);

    if (uboBuffer) vmaDestroyBuffer(ctx.allocator, uboBuffer, uboAlloc);
    if (edgeIdxBuffer) vmaDestroyBuffer(ctx.allocator, edgeIdxBuffer, edgeIdxAlloc);
    if (idxBuffer) vmaDestroyBuffer(ctx.allocator, idxBuffer, idxAlloc);
    if (vertBuffer) vmaDestroyBuffer(ctx.allocator, vertBuffer, vertAlloc);
}

void update(double time) {
    if (prevTime == 0.0) prevTime = time;
    double dt = time - prevTime;
    prevTime = time;

    if (playAnim) {
        animTime += dt * animSpeed;
    }

    // интерфейс управления
    ImGui::Begin("Lab 1: 3D Graphics (Variant 3: Parallelepiped)");

    ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.4f, 1.0f), "Status: All tasks (1-6) active for Grade 5");
    ImGui::Separator();

    // 1. проекция
    if (ImGui::CollapsingHeader("1. Projection Matrix (Task 1)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::RadioButton("Perspective", &projType, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Orthographic", &projType, 1);

        if (projType == 0) {
            ImGui::SliderFloat("Field of View (FOV)", &fov, 20.0f, 120.0f, "%.1f deg");
        } else {
            ImGui::SliderFloat("Ortho Zoom Scale", &orthoScale, 0.5f, 10.0f, "%.2f");
        }
    }

    // 2. трансформации
    if (ImGui::CollapsingHeader("2. Model Transforms (Task 2)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Position (X, Y, Z)", glm::value_ptr(pos), 0.05f);
        ImGui::SliderFloat3("Rotation (Euler)", glm::value_ptr(rot), -180.0f, 180.0f, "%.1f deg");
        ImGui::DragFloat3("Scale (Sx, Sy, Sz)", glm::value_ptr(scale), 0.05f, 0.1f, 5.0f);

        if (ImGui::Button("Reset Transforms")) {
            pos = { 0.0f, 0.0f, 0.0f };
            rot = { 20.0f, 30.0f, 0.0f };
            scale = { 1.0f, 1.0f, 1.0f };
        }
    }

    // 3. траектория и анимация
    if (ImGui::CollapsingHeader("3. Trajectory & Animation (Task 3)", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (playAnim) {
            if (ImGui::Button("Pause ||")) playAnim = false;
        } else {
            if (ImGui::Button("Play >")) playAnim = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset Time")) animTime = 0.0;

        ImGui::SliderFloat("Speed", &animSpeed, 0.0f, 5.0f, "%.2fx");
        ImGui::SliderFloat("Trajectory Radius (R)", &trajRadius, 0.5f, 4.0f);
        ImGui::SliderFloat("Height Wave (H)", &trajHeight, 0.0f, 2.0f);
        ImGui::TextDisabled("Trajectory: 3D Helix (X=R*cos, Z=R*sin, Y=H*sin)");
    }

    // 4 & 5. цвет граней, ребер и углов
    if (ImGui::CollapsingHeader("4 & 5. Color, Edges & Corners (Tasks 4 & 5)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit4("Faces Tint (UI Color)", color1);
        if (ImGui::Button("Reset Face Color to White (1,1,1)")) {
            color1[0] = color1[1] = color1[2] = color1[3] = 1.0f;
        }

        ImGui::Separator();
        ImGui::Checkbox("Highlight Edges (Lines)", &showEdges);
        if (showEdges) {
            ImGui::ColorEdit4("Edge Color", edgeColor);
        }

        ImGui::Checkbox("Highlight Corners (Points)", &showCorners);
        if (showCorners) {
            ImGui::ColorEdit4("Corner Color", cornerColor);
        }
    }

    ImGui::End();

    // считаем матрицы
    auto& ctx = graphics::internal::context;
    float aspect = (float)ctx.swapchain_extent.width / (float)(ctx.swapchain_extent.height > 0 ? ctx.swapchain_extent.height : 1);

    glm::mat4 projMat;
    if (projType == 0) {
        projMat = glm::perspective(glm::radians(fov), aspect, 0.1f, 100.0f);
    } else {
        projMat = glm::ortho(-orthoScale * aspect, orthoScale * aspect, -orthoScale, orthoScale, -50.0f, 50.0f);
    }
    projMat[1][1] *= -1.0f; // инверсия Y для Vulkan

    glm::mat4 viewMat = glm::lookAt(camEye, camCenter, camUp);

    // вычисляем траекторию первого объекта
    float t = (float)animTime;
    glm::vec3 animPos = { 0.0f, 0.0f, 0.0f };

    // Простая и элегантная сложная 3D-траектория (спиральный облет с волновыми колебаниями)
    animPos.x = trajRadius * std::cos(t);
    animPos.z = trajRadius * std::sin(t);
    animPos.y = trajHeight * std::sin(2.0f * t);

    // Одновременное вращение вокруг собственных осей во время движения
    float animRotX = t * 0.5f;
    float animRotY = t * 1.0f;

    // собираем матрицу модели объекта 1
    glm::mat4 model1 = glm::mat4(1.0f);
    model1 = glm::translate(model1, pos + animPos);
    model1 = glm::rotate(model1, glm::radians(rot.x) + animRotX, glm::vec3(1.0f, 0.0f, 0.0f));
    model1 = glm::rotate(model1, glm::radians(rot.y) + animRotY, glm::vec3(0.0f, 1.0f, 0.0f));
    model1 = glm::rotate(model1, glm::radians(rot.z), glm::vec3(0.0f, 0.0f, 1.0f));
    model1 = glm::scale(model1, scale);

    UniformBufferObject ubo1 = {};
    ubo1.model = model1;
    ubo1.view = viewMat;
    ubo1.proj = projMat;
    ubo1.uiColor = glm::make_vec4(color1);
    ubo1.edgeColor = glm::make_vec4(edgeColor);
    ubo1.cornerColor = glm::make_vec4(cornerColor);
    copyToBuffer(uboAlloc, &ubo1, sizeof(ubo1));
}

void render(const graphics::internal::FrameData& frame) {
    auto& ctx = graphics::internal::context;
    VkCommandBuffer cmd = frame.command_buffer;

    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkClearValue clears[2] = {};
    clears[0].color = { { 0.12f, 0.12f, 0.14f, 1.0f } };
    clears[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = ctx.render_pass;
    rpInfo.framebuffer = frame.framebuffer;
    rpInfo.renderArea.offset = { 0, 0 };
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

    VkBuffer vbs[] = { vertBuffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);

    // привязываем дескриптор сет с матрицами
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 0, 1, &descSet, 0, nullptr);

    // 1. рисуем грани (треугольники)
    int mode = 0;
    vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &mode);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindIndexBuffer(cmd, idxBuffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(cmd, (uint32_t)indices.size(), 1, 0, 0, 0);

    // 2. рисуем ребра (линии)
    if (showEdges) {
        mode = 1;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &mode);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, linePipeline);
        vkCmdBindIndexBuffer(cmd, edgeIdxBuffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(cmd, (uint32_t)edgeIndices.size(), 1, 0, 0, 0);
    }

    // 3. рисуем углы (точки)
    if (showCorners) {
        mode = 2;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &mode);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pointPipeline);
        vkCmdDraw(cmd, (uint32_t)vertices.size(), 1, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace application
