// main.cpp
#include <vulkan/vulkan.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// FidelityFX FSR2 (host) + Vulkan backend
#include <FidelityFX/host/ffx_fsr2.h>
#include <FidelityFX/host/backends/vk/ffx_fsr2_vk.h>

namespace fs = std::filesystem;

// ------------ quick helpers ------------
static void checkVk(VkResult r, const char* where){
    if(r != VK_SUCCESS){ std::cerr << "Vulkan error at " << where << " code=" << r << "\n"; std::exit(1); }
}
static std::string fmt5(int i){ std::ostringstream os; os << std::setw(5) << std::setfill('0') << i; return os.str(); }

static std::vector<uint8_t> readBin(const fs::path& p){
    std::ifstream f(p, std::ios::binary);
    if(!f) { throw std::runtime_error("Could not open " + p.string()); }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}

// ------------ Vulkan minimal bootstrap ------------
struct VkCtx {
    VkInstance instance{};
    VkPhysicalDevice pdev{};
    VkDevice device{};
    uint32_t queueFamily{};
    VkQueue queue{};
    VkCommandPool cmdPool{};
};

static VkCtx makeVulkan() {
    VkCtx C{};

    // Instance
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "fsr2_vulkan_min";
    ai.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    checkVk(vkCreateInstance(&ici, nullptr, &C.instance), "vkCreateInstance");

    // Physical device
    uint32_t ndev=0; vkEnumeratePhysicalDevices(C.instance, &ndev, nullptr);
    if(ndev==0) { std::cerr << "No Vulkan devices.\n"; std::exit(1); }
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(C.instance, &ndev, devs.data());
    // pick first with compute queue
    for(auto d: devs){
        uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qprops.data());
        for(uint32_t i=0;i<nq;++i){
            if(qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT){ C.pdev=d; C.queueFamily=i; break; }
        }
        if(C.pdev) break;
    }
    if(!C.pdev){ std::cerr << "No compute-capable device.\n"; std::exit(1); }

    // Device + queue
    float prio=1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = C.queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    // FSR2 needs some features (storage images, etc.) — most drivers support them by default
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    checkVk(vkCreateDevice(C.pdev, &dci, nullptr, &C.device), "vkCreateDevice");
    vkGetDeviceQueue(C.device, C.queueFamily, 0, &C.queue);

    // Command pool
    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cp.queueFamilyIndex = C.queueFamily;
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    checkVk(vkCreateCommandPool(C.device, &cp, nullptr, &C.cmdPool), "vkCreateCommandPool");

    return C;
}

static VkCommandBuffer beginCmd(VkCtx& C){
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = C.cmdPool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cmd; checkVk(vkAllocateCommandBuffers(C.device, &ai, &cmd), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    checkVk(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
    return cmd;
}
static void endSubmitWait(VkCtx& C, VkCommandBuffer cmd){
    checkVk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
    checkVk(vkQueueSubmit(C.queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
    checkVk(vkQueueWaitIdle(C.queue), "vkQueueWaitIdle");
    vkFreeCommandBuffers(C.device, C.cmdPool, 1, &cmd);
}

// ------------ images & uploads ------------
struct Image {
    VkImage img{}; VkDeviceMemory mem{}; VkImageView view{}; VkFormat fmt{}; uint32_t w{}, h{};
};

static uint32_t findMemType(VkPhysicalDevice pdev, uint32_t bits, VkMemoryPropertyFlags req){
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;++i)
        if( (bits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & req)==req ) return i;
    std::cerr<<"No memtype.\n"; std::exit(1);
}

static Image makeImage(VkCtx& C, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, VkImageLayout initLay=VK_IMAGE_LAYOUT_UNDEFINED){
    Image I; I.w=w; I.h=h; I.fmt=fmt;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType=VK_IMAGE_TYPE_2D; ci.extent={w,h,1}; ci.mipLevels=1; ci.arrayLayers=1;
    ci.format=fmt; ci.tiling=VK_IMAGE_TILING_OPTIMAL; ci.initialLayout=initLay;
    ci.samples=VK_SAMPLE_COUNT_1_BIT; ci.usage=usage;
    checkVk(vkCreateImage(C.device, &ci, nullptr, &I.img), "vkCreateImage");
    VkMemoryRequirements mr{}; vkGetImageMemoryRequirements(C.device, I.img, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize=mr.size; ai.memoryTypeIndex=findMemType(C.pdev, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    checkVk(vkAllocateMemory(C.device, &ai, nullptr, &I.mem), "vkAllocateMemory");
    vkBindImageMemory(C.device, I.img, I.mem, 0);

    // view
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image=I.img; vi.viewType=VK_IMAGE_VIEW_TYPE_2D; vi.format=fmt;
    vi.subresourceRange.aspectMask = (fmt==VK_FORMAT_R16_UNORM? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
    vi.subresourceRange.levelCount=1; vi.subresourceRange.layerCount=1;
    checkVk(vkCreateImageView(C.device, &vi, nullptr, &I.view), "vkCreateImageView");
    return I;
}

static void trans(VkCommandBuffer cmd, VkImage img, VkImageLayout oldL, VkImageLayout newL, VkPipelineStageFlags src=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VkPipelineStageFlags dst=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT){
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout=oldL; b.newLayout=newL;
    b.image=img; b.subresourceRange.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount=1; b.subresourceRange.layerCount=1;
    b.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier(cmd, src, dst, 0, 0,nullptr, 0,nullptr, 1,&b);
}

static void uploadToImageRGBA8(VkCtx& C, Image& I, const uint8_t* rgba, size_t bytes){
    // staging buffer
    VkBuffer buf{}; VkDeviceMemory mem{};
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    checkVk(vkCreateBuffer(C.device, &bi, nullptr, &buf), "vkCreateBuffer");
    VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(C.device, buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize=mr.size; ai.memoryTypeIndex=findMemType(C.pdev, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    checkVk(vkAllocateMemory(C.device, &ai, nullptr, &mem), "vkAllocateMemory");
    vkBindBufferMemory(C.device, buf, mem, 0);
    void* map; vkMapMemory(C.device, mem, 0, bytes, 0, &map); std::memcpy(map, rgba, bytes); vkUnmapMemory(C.device, mem);

    auto cmd = beginCmd(C);
    // transition
    trans(cmd, I.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    // copy
    VkBufferImageCopy r{};
    r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    r.imageSubresource.layerCount = 1;
    r.imageExtent = { I.w, I.h, 1 };
    vkCmdCopyBufferToImage(cmd, buf, I.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    // to shader read
    trans(cmd, I.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    endSubmitWait(C, cmd);

    vkDestroyBuffer(C.device, buf, nullptr);
    vkFreeMemory(C.device, mem, nullptr);
}

// Load 16-bit single-channel PNG (depth) and upload as R16_UNORM (we treat as color aspect for simplicity)
static void uploadToImageR16(VkCtx& C, Image& I, const uint16_t* r16, size_t pxCount){
    size_t bytes = pxCount * sizeof(uint16_t);
    // Reuse same staging+copy as RGBA8 (layout logic identical, just byte size differs)
    VkBuffer buf{}; VkDeviceMemory mem{};
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    checkVk(vkCreateBuffer(C.device, &bi, nullptr, &buf), "vkCreateBuffer");
    VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(C.device, buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize=mr.size; ai.memoryTypeIndex=findMemType(C.pdev, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    checkVk(vkAllocateMemory(C.device, &ai, nullptr, &mem), "vkAllocateMemory");
    vkBindBufferMemory(C.device, buf, mem, 0);
    void* map; vkMapMemory(C.device, mem, 0, bytes, 0, &map); std::memcpy(map, r16, bytes); vkUnmapMemory(C.device, mem);

    auto cmd = beginCmd(C);
    trans(cmd, I.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy r{};
    r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // using color aspect for simplicity
    r.imageSubresource.layerCount = 1;
    r.imageExtent = { I.w, I.h, 1 };
    vkCmdCopyBufferToImage(cmd, buf, I.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    trans(cmd, I.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    endSubmitWait(C, cmd);

    vkDestroyBuffer(C.device, buf, nullptr);
    vkFreeMemory(C.device, mem, nullptr);
}

// Load RG16F motion (two fp16 per pixel) and upload to R16G16_SFLOAT
static void uploadToImageRG16F(VkCtx& C, Image& I, const uint8_t* raw, size_t bytes){
    // Same staging approach
    VkBuffer buf{}; VkDeviceMemory mem{};
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    checkVk(vkCreateBuffer(C.device, &bi, nullptr, &buf), "vkCreateBuffer");
    VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(C.device, buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize=mr.size; ai.memoryTypeIndex=findMemType(C.pdev, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    checkVk(vkAllocateMemory(C.device, &ai, nullptr, &mem), "vkAllocateMemory");
    vkBindBufferMemory(C.device, buf, mem, 0);
    void* map; vkMapMemory(C.device, mem, 0, bytes, 0, &map); std::memcpy(map, raw, bytes); vkUnmapMemory(C.device, mem);

    auto cmd = beginCmd(C);
    trans(cmd, I.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy r{};
    r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    r.imageSubresource.layerCount = 1;
    r.imageExtent = { I.w, I.h, 1 };
    vkCmdCopyBufferToImage(cmd, buf, I.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    trans(cmd, I.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    endSubmitWait(C, cmd);

    vkDestroyBuffer(C.device, buf, nullptr);
    vkFreeMemory(C.device, mem, nullptr);
}

// Download RGBA8 output
static std::vector<uint8_t> downloadRGBA8(VkCtx& C, Image& I){
    size_t bytes = I.w * I.h * 4;
    // Create a staging buffer and copy image → buffer, then map
    // For brevity, we do a simple blit via a linear-tiled image is more complex; here we use copy with an intermediate buffer

    // Create a temp image in TRANSFER_SRC layout is already our I.img; we copy image to buffer:
    VkBuffer buf{}; VkDeviceMemory mem{};
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    checkVk(vkCreateBuffer(C.device, &bi, nullptr, &buf), "vkCreateBuffer");
    VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(C.device, buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize=mr.size; ai.memoryTypeIndex=findMemType(C.pdev, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    checkVk(vkAllocateMemory(C.device, &ai, nullptr, &mem), "vkAllocateMemory");
    vkBindBufferMemory(C.device, buf, mem, 0);

    auto cmd = beginCmd(C);
    // Transition to TRANSFER_SRC
    trans(cmd, I.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy r{};
    r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    r.imageSubresource.layerCount = 1;
    r.imageExtent = { I.w, I.h, 1 };
    vkCmdCopyImageToBuffer(cmd, I.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &r);
    // Back to GENERAL for next frame use if desired
    trans(cmd, I.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    endSubmitWait(C, cmd);

    std::vector<uint8_t> out(bytes);
    void* map; vkMapMemory(C.device, mem, 0, bytes, 0, &map); std::memcpy(out.data(), map, bytes); vkUnmapMemory(C.device, mem);

    vkDestroyBuffer(C.device, buf, nullptr);
    vkFreeMemory(C.device, mem, nullptr);
    return out;
}

// ------------ FSR2 run ------------
int main(int argc, char** argv){
    // Paths
    std::string inRoot = argc>1 ? argv[1] : "outputs_fsr2";
    std::string outRoot = argc>2 ? argv[2] : "upscaled";
    fs::create_directories(outRoot);

    // Read meta of frame 0
    auto meta0 = fs::path(inRoot)/"meta"/"00000.json";
    if(!fs::exists(meta0)){ std::cerr<<"meta/00000.json missing\n"; return 1; }
    // super light JSON parse (width/height only)
    int renderW=0, renderH=0;
    {
        std::ifstream f(meta0); std::string s((std::istreambuf_iterator<char>(f)), {});
        auto wpos = s.find("\"width\""); auto hpos = s.find("\"height\"");
        renderW = std::stoi(s.substr(s.find(':', wpos)+1));
        renderH = std::stoi(s.substr(s.find(':', hpos)+1));
    }

    // Choose an upscale (e.g., 2x)
    int displayW = renderW*2;
    int displayH = renderH*2;

    // Vulkan + FSR2 context
    VkCtx C = makeVulkan();

    FfxFsr2Context fsr{};
    {
        FfxFsr2ContextDescription desc{};
        desc.flags = 0;
        desc.maxRenderSize.width  = renderW;
        desc.maxRenderSize.height = renderH;
        desc.displaySize.width    = displayW;
        desc.displaySize.height   = displayH;

        // Backend interface (provided by FSR2 Vulkan backend helper)
        FfxInterface backend = ffxFsr2GetInterfaceVK(C.instance, C.pdev, C.device, C.queue, C.queueFamily);
        desc.backendInterface = backend;

        FfxErrorCode ec = ffxFsr2ContextCreate(&fsr, &desc);
        if(ec != FFX_OK){ std::cerr<<"ffxFsr2ContextCreate failed "<<ec<<"\n"; return 1; }
    }

    // Output image (upscaled)
    Image outImg = makeImage(C, displayW, displayH,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED);

    // List frames by counting color PNGs
    int numFrames = 0;
    for(; ; ++numFrames){
        auto p = fs::path(inRoot)/"color"/(fmt5(numFrames)+".png");
        if(!fs::exists(p)) break;
    }
    if(numFrames==0){ std::cerr<<"No frames under "<<inRoot<<"/color\n"; return 1; }

    // Per-frame loop
    for(int i=0;i<numFrames;++i){
        std::string idx = fmt5(i);

        // ---- Load COLOR (RGBA8) ----
        int cw,ch,cn;
        std::string cpath = (fs::path(inRoot)/"color"/(idx+".png")).string();
        uint8_t* cimg = stbi_load(cpath.c_str(), &cw, &ch, &cn, 4);
        if(!cimg){ std::cerr<<"stbi_load failed: "<<cpath<<"\n"; return 1; }
        if(cw!=renderW || ch!=renderH){ std::cerr<<"Size mismatch color\n"; return 1; }

        Image color = makeImage(C, renderW, renderH,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        uploadToImageRGBA8(C, color, cimg, renderW*renderH*4);
        stbi_image_free(cimg);

        // ---- Load DEPTH (R16 PNG) ----
        int dw,dh,dc;
        std::string dpath = (fs::path(inRoot)/"depth_r16"/(idx+".png")).string();
        stbi_uc* dimg_raw = stbi_load_16(dpath.c_str(), &dw, &dh, &dc, 1); // 16-bit single channel
        if(!dimg_raw){ std::cerr<<"stbi_load_16 failed: "<<dpath<<"\n"; return 1; }
        if(dw!=renderW || dh!=renderH){ std::cerr<<"Size mismatch depth\n"; return 1; }
        Image depth = makeImage(C, renderW, renderH,
            VK_FORMAT_R16_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        uploadToImageR16(C, depth, reinterpret_cast<uint16_t*>(dimg_raw), size_t(renderW)*renderH);
        stbi_image_free(dimg_raw);

        // ---- Load MOTION (RG16F .bin) ----
        auto mpath = fs::path(inRoot)/"motion_rg16f"/(idx+".bin");
        std::vector<uint8_t> motionData = readBin(mpath);
        // Expect 2 half floats per pixel
        size_t expectedBytes = size_t(renderW)*renderH*2*sizeof(uint16_t);
        if(motionData.size()!=expectedBytes){ std::cerr<<"Motion size mismatch\n"; return 1; }
        Image motion = makeImage(C, renderW, renderH,
            VK_FORMAT_R16G16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        uploadToImageRG16F(C, motion, motionData.data(), motionData.size());

        // ---- Prepare output layout GENERAL for storage write ----
        {
            auto cmd = beginCmd(C);
            trans(cmd, outImg.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            endSubmitWait(C, cmd);
        }

        // ---- Dispatch FSR2 ----
        FfxFsr2DispatchDescription dd{};
        dd.commandList = ffxGetCommandListVK(C.device, C.queue, C.cmdPool); // backend helper provides CL wrapper
        dd.color         = ffxGetResourceVK(color.img,  color.view,  renderW, renderH, L"Color");
        dd.depth         = ffxGetResourceVK(depth.img,  depth.view,  renderW, renderH, L"Depth");
        dd.motionVectors = ffxGetResourceVK(motion.img, motion.view, renderW, renderH, L"Motion");
        dd.output        = ffxGetResourceVK(outImg.img, outImg.view, displayW, displayH, L"Output");

        dd.jitterOffset.x = 0.f; dd.jitterOffset.y = 0.f;
        dd.motionVectorScale.x = 1.f; dd.motionVectorScale.y = 1.f; // pixels, current->previous
        dd.renderSize.width = renderW; dd.renderSize.height = renderH;
        dd.preExposure = 1.f;
        dd.frameTimeDelta = 1.0f/30.0f;
        dd.reset = (i==0) ? FFX_TRUE : FFX_FALSE;
        dd.enableSharpening = FFX_TRUE;
        dd.sharpness = 0.2f;

        FfxErrorCode ec = ffxFsr2ContextDispatch(&fsr, &dd);
        if(ec != FFX_OK){ std::cerr<<"FSR2 dispatch failed "<<ec<<"\n"; return 1; }

        // ---- Read back & save PNG ----
        std::vector<uint8_t> rgba = downloadRGBA8(C, outImg);
        std::string outPath = (fs::path(outRoot)/(idx+".png")).string();
        stbi_write_png(outPath.c_str(), displayW, displayH, 4, rgba.data(), displayW*4);

        // cleanup frame-local images
        vkDestroyImageView(C.device, color.view, nullptr);
        vkFreeMemory(C.device, color.mem, nullptr);
        vkDestroyImage(C.device, color.img, nullptr);

        vkDestroyImageView(C.device, depth.view, nullptr);
        vkFreeMemory(C.device, depth.mem, nullptr);
        vkDestroyImage(C.device, depth.img, nullptr);

        vkDestroyImageView(C.device, motion.view, nullptr);
        vkFreeMemory(C.device, motion.mem, nullptr);
        vkDestroyImage(C.device, motion.img, nullptr);

        std::cout << "\rFrame " << i+1 << "/" << numFrames << std::flush;
    }

    std::cout << "\nDone. Writing video with ffmpeg:\n  ffmpeg -y -framerate 30 -i " << outRoot << "/%05d.png -c:v libx264 -crf 16 -pix_fmt yuv420p upscaled_output.mp4\n";

    ffxFsr2ContextDestroy(&fsr);

    // destroy Vulkan (outImg)
    vkDestroyImageView(C.device, outImg.view, nullptr);
    vkFreeMemory(C.device, outImg.mem, nullptr);
    vkDestroyImage(C.device, outImg.img, nullptr);

    vkDestroyCommandPool(C.device, C.cmdPool, nullptr);
    vkDestroyDevice(C.device, nullptr);
    vkDestroyInstance(C.instance, nullptr);
    return 0;
}
