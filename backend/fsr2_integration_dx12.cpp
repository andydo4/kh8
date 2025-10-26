// fsr2_integration_dx12.cpp (excerpt)
#include <FidelityFX/host/ffx_fsr2.h>

// 1) Create context (once)
FfxFsr2Context fsr2Context = {};
{
    FfxFsr2ContextDescription desc = {};
    desc.flags = FFX_FSR2_ENABLE_AUTO_EXPOSURE | FFX_FSR2_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS; // optional flags
    desc.maxRenderSize = { renderWidth, renderHeight };     // input (color/depth/motion) size
    desc.displaySize = { outputWidth, outputHeight };       // upscaled output size
    // Fill backendInterface, device, etc. per FSR2 sample
    ffxFsr2ContextCreate(&fsr2Context, &desc);
}

// 2) Per-frame: load your produced textures/buffers
// - color:   SRV of color/00042.png uploaded to a DX12 texture (R8G8B8A8_UNORM)
// - depth:   SRV of depth_r16/00042.png uploaded to R16_UNORM (or R32_FLOAT)
// - motion:  SRV of motion_rg16f/00042.bin uploaded to R16G16_FLOAT (pixels, current->previous)

struct FrameIO {
    ID3D12Resource* color;
    ID3D12Resource* depth;     // R16_UNORM
    ID3D12Resource* motion;    // R16G16_FLOAT
    ID3D12Resource* outUpscaled; // UAV
} io;

// 3) Dispatch
void RunFsr2Frame(ID3D12GraphicsCommandList* cmd, int frameIdx, int w, int h) {
    FfxFsr2DispatchDescription d = {};
    d.commandList = ffxGetCommandListDX12(cmd);

    d.color                = ffxGetResourceDX12(io.color,  L"Color");
    d.depth                = ffxGetResourceDX12(io.depth,  L"Depth");
    d.motionVectors        = ffxGetResourceDX12(io.motion, L"Motion");
    d.output               = ffxGetResourceDX12(io.outUpscaled, L"Output");

    d.jitterOffset.x = 0.0f; d.jitterOffset.y = 0.0f;      // no TAA jitter for video
    d.motionVectorScale.x = 1.0f;                          // pixels, matches Python meta
    d.motionVectorScale.y = 1.0f;

    d.reset              = FFX_FALSE;                      // set TRUE on scene cuts
    d.enableSharpening   = FFX_TRUE;
    d.sharpness          = 0.2f;                           // taste
    d.frameTimeDelta     = 1.0f / 30.0f;                   // or real dt
    d.preExposure        = 1.0f;                           // video already exposed
    d.renderSize.width   = w; d.renderSize.height = h;
    d.cameraNear         = 0.01f;                          // used if depth is non-linear; leave reasonable
    d.cameraFar          = 1000.0f;

    ffxFsr2ContextDispatch(&fsr2Context, &d);
}
