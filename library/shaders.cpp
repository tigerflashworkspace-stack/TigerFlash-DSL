#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <link.h>
#include <elf.h>
#include <sys/mman.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sstream>
#include <vector>
#include <unordered_map>

using std::size_t;
using std::string;

// This ABI intentionally duplicates only the small public TigerFlash
// library interface. The library itself does not depend on raylib or the
// TigerFlash interpreter source.
using TFSay3DOptionsHook = bool (*)(const char *options,
                                    char *output,
                                    size_t outputCapacity);
using TFSay3DHook = bool (*)(const char *objectName,
                             const char *fullLine);
using TFLibraryCommandHook = bool (*)(const char *line);
using TFLibraryConditionHook = bool (*)(const char *condition,
                                        bool *handled);

struct TFLibraryAPI
{
    unsigned int version = 1;

    bool (*registerSay3DOptionsHook)(TFSay3DOptionsHook hook) = nullptr;
    bool (*registerSay3DHook)(TFSay3DHook hook) = nullptr;
    bool (*registerCommandHook)(TFLibraryCommandHook hook) = nullptr;
    bool (*registerConditionHook)(TFLibraryConditionHook hook) = nullptr;

    bool (*setObjectMotionBlur)(const char *objectName, float value) = nullptr;
    float (*getObjectMotionBlur)(const char *objectName) = nullptr;
    bool (*object3DExists)(const char *objectName) = nullptr;

    void (*log)(const char *message) = nullptr;
};

namespace
{
TFLibraryAPI gApi;

// -----------------------------------------------------------------------------
// IN-LIBRARY MOTION EFFECTS
// -----------------------------------------------------------------------------
// The host's existing `motionBlur` field is intentionally kept for the old
// transparent trail effect. The new realistic blur lives entirely in this
// library so the host IDE source/ABI does not need another field or hook.
struct TFRealMotionBlurState
{
    float strength = 0.0f;
    float previousX = 0.0f;
    float previousY = 0.0f;
    float previousZ = 0.0f;
    bool previousInitialized = false;
};

static std::unordered_map<std::string, TFRealMotionBlurState> gTfRealMotionBlur;
static std::string gTfLastCreatedObjectName;


// -----------------------------------------------------------------------------
// RUNTIME ADVANCED-LIGHTING BRIDGE
// -----------------------------------------------------------------------------
// The public TigerFlash library ABI exposes language hooks, but the host IDE
// owns the 3D render pass. To keep this feature entirely inside shaders.cpp,
// this module installs a small Linux/x86-64 runtime detour on the host's
// tfPrepareSurfaceShader() function and replaces the surface shader with a
// single-pass, pre-baked environment + PBR lighting shader.
//
// The normal optimized mesh/LOD/frustum path remains untouched, so the added
// lighting does not create extra per-object render passes. The baked lighting
// LUT is generated once on the CPU and compiled into the shader as constants.
// If a host binary is stripped enough that its internal symbols cannot be
// located, the library safely falls back to its public motion-blur feature.

struct TFCompatShader
{
    unsigned int id = 0;
    int *locs = nullptr;
};

static void *gTfSurfaceShaderAddress = nullptr;
static void *gTfSurfaceShaderReadyAddress = nullptr;
static void *gTfPrepareSurfaceShaderAddress = nullptr;
static bool gTfRendererHookInstalled = false;
static bool gTfAdvancedShaderBuilt = false;
static bool gTfAdvancedShaderFailed = false;
static std::string gTfAdvancedShaderError;

using TFGLCreateShader = unsigned int (*)(unsigned int);
using TFGLShaderSource = void (*)(unsigned int, int, const char *const *, const int *);
using TFGLCompileShader = void (*)(unsigned int);
using TFGLGetShaderiv = void (*)(unsigned int, unsigned int, int *);
using TFGLGetShaderInfoLog = void (*)(unsigned int, int, int *, char *);
using TFGLDeleteShader = void (*)(unsigned int);
using TFGLCreateProgram = unsigned int (*)();
using TFGLAttachShader = void (*)(unsigned int, unsigned int);
using TFGLBindAttribLocation = void (*)(unsigned int, unsigned int, const char *);
using TFGLLinkProgram = void (*)(unsigned int);
using TFGLGetProgramiv = void (*)(unsigned int, unsigned int, int *);
using TFGLGetProgramInfoLog = void (*)(unsigned int, int, int *, char *);
using TFGLDeleteProgram = void (*)(unsigned int);
using TFGLGetUniformLocation = int (*)(unsigned int, const char *);

static TFGLCreateShader tfGLCreateShader = nullptr;
static TFGLShaderSource tfGLShaderSource = nullptr;
static TFGLCompileShader tfGLCompileShader = nullptr;
static TFGLGetShaderiv tfGLGetShaderiv = nullptr;
static TFGLGetShaderInfoLog tfGLGetShaderInfoLog = nullptr;
static TFGLDeleteShader tfGLDeleteShader = nullptr;
static TFGLCreateProgram tfGLCreateProgram = nullptr;
static TFGLAttachShader tfGLAttachShader = nullptr;
static TFGLBindAttribLocation tfGLBindAttribLocation = nullptr;
static TFGLLinkProgram tfGLLinkProgram = nullptr;
static TFGLGetProgramiv tfGLGetProgramiv = nullptr;
static TFGLGetProgramInfoLog tfGLGetProgramInfoLog = nullptr;
static TFGLDeleteProgram tfGLDeleteProgram = nullptr;
static TFGLGetUniformLocation tfGLGetUniformLocation = nullptr;
using TFGLUniform1f = void (*)(int, float);
using TFGLUniform4fv = void (*)(int, int, const float *);
using TFGLUseProgram = void (*)(unsigned int);
using TFGLGetIntegerv = void (*)(unsigned int, int *);
static TFGLUniform1f tfGLUniform1f = nullptr;
static TFGLUniform4fv tfGLUniform4fv = nullptr;
static TFGLUseProgram tfGLUseProgram = nullptr;
static TFGLGetIntegerv tfGLGetIntegerv = nullptr;

using TFGLUniform1i = void (*)(int, int);
using TFGLUniform2fv = void (*)(int, int, const float *);
using TFGLUniform3fv = void (*)(int, int, const float *);
using TFGLGenTextures = void (*)(int, unsigned int *);
using TFGLDeleteTextures = void (*)(int, const unsigned int *);
using TFGLBindTexture = void (*)(unsigned int, unsigned int);
using TFGLTexParameteri = void (*)(unsigned int, unsigned int, int);
using TFGLTexImage2D = void (*)(unsigned int, int, int, int, int, int, unsigned int, unsigned int, const void *);
using TFGLCopyTexSubImage2D = void (*)(unsigned int, int, int, int, int, int, int, int);
using TFGLActiveTexture = void (*)(unsigned int);
static TFGLUniform1i tfGLUniform1i = nullptr;
static TFGLUniform2fv tfGLUniform2fv = nullptr;
static TFGLUniform3fv tfGLUniform3fv = nullptr;
static TFGLGenTextures tfGLGenTextures = nullptr;
static TFGLDeleteTextures tfGLDeleteTextures = nullptr;
static TFGLBindTexture tfGLBindTexture = nullptr;
static TFGLTexParameteri tfGLTexParameteri = nullptr;
static TFGLTexImage2D tfGLTexImage2D = nullptr;
static TFGLCopyTexSubImage2D tfGLCopyTexSubImage2D = nullptr;
static TFGLActiveTexture tfGLActiveTexture = nullptr;

static int gTfLightStrengthLocation = -1;
static int gTfDynamicLightCountLocation = -1;
static int gTfDynamicLightsLocation = -1;
static int gTfDynamicColorsLocation = -1;
static int gTfDynamicLightParamsLocation = -1;
static int gTfDynamicOccluderCountLocation = -1;
// Exact analytic occluder data used by the hard shadow ray test:
// center + shape id, half-extents/radii, and Euler rotation (radians).
static int gTfDynamicOccludersLocation = -1;
static int gTfDynamicOccluderSizesLocation = -1;
static int gTfDynamicOccluderRotationsLocation = -1;
// Live soft-body deformation state used by the shadow proxy. The proxy follows
// the same compression/stretch/bend/flow state as the host mesh, so shadows
// deform with the actual runtime shape instead of remaining spherical.
static int gTfDynamicOccluderSoftALocation = -1;
static int gTfDynamicOccluderSoftBLocation = -1;
static int gTfDynamicOccluderSoftCLocation = -1;
static int gTfDynamicOccluderSoftDLocation = -1;
static int gTfGroundYLocation = -1;
static int gTfGroundValidLocation = -1;
static int gTfGroundColorLocation = -1;

// Global lighting pipeline. The shader keeps lighting in linear HDR-range
// values until this final exposure/tonemap stage. GI and atmosphere are
// intentionally low-frequency so their cost is stable as scene complexity grows.
static int gTfExposureLocation = -1;
static int gTfGIIntensityLocation = -1;
static int gTfGIDirectionsLocation = -1;
static int gTfAtmosphereParamsLocation = -1;
static int gTfAtmosphereColorLocation = -1;
static int gTfViewPosLocation = -1;
static int gTfSwapShadingLocation = -1;

static unsigned int gTfAdvancedProgram = 0;

static constexpr int TF_MAX_DYNAMIC_LIGHTS = 4;
static constexpr int TF_MAX_DYNAMIC_OCCLUDERS = 24;
static constexpr int TF_GI_DIRECTION_COUNT = 6;

struct TFGlobalLightingState
{
    bool initialized = false;
    double lastUpdateSeconds = 0.0;
    float exposure = 1.0f;
    float targetExposure = 1.0f;
    float measuredLuminance = 0.18f;
    float updateAccumulator = 0.0f;
    uint64_t serial = 0;

    // Low-frequency indirect lighting lobes for +X,-X,+Y,-Y,+Z,-Z.
    std::array<float, TF_GI_DIRECTION_COUNT * 4> giDirections{};
    float giIntensity = 0.72f;
};

static TFGlobalLightingState gTfGlobalLighting;
static uint64_t gTfGlobalLightingUploadedSerial = ~static_cast<uint64_t>(0);

struct TFObjectLightState
{
    float strength = 0.0f;
    // Plain light() illuminates but does not cast realistic shadows.
    // Realistic soft shadows are globally enabled by the standalone
    // `advanced light` command.
    bool advanced = false;
};
static std::unordered_map<std::string, TFObjectLightState> gTfObjectLights;

// Global opt-in for physically softer, more realistic dynamic shadows.
// It is enabled by the standalone TigerFlash command `advanced light`.
// The light(n) declaration itself remains responsible only for light energy.
static bool gTfAdvancedLightingEnabled = false;

// -----------------------------------------------------------------------------
// ADVANCED LIGHT x REFRACTIVE MATERIALS
// -----------------------------------------------------------------------------
// crystal / diamond / prism and soft bodies (water-like) do NOT block light in
// `advanced light`. The light is bent through them (Snell + dispersion) and its
// energy is preserved, so the receiver is distorted but never darker.
// Occluder material ids sent to the GPU (0 = ordinary opaque object):
//   1 = quartz crystal, 2 = diamond, 3 = BK7 hex prism, 4 = water / soft body
static int gTfDynamicOccluderTintsLocation = -1;
static int gTfOpticalAdvancedLocation = -1;
static std::unordered_map<std::string, double> gTfSoftRefractiveSeen;

static double tfNowSeconds()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct TFTrackedObject
{
    std::string name;
    std::string shape;
    std::string color;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    float scaleZ = 1.0f;
    float rotationX = 0.0f;
    float rotationY = 0.0f;
    float rotationZ = 0.0f;
    float lightStrength = 0.0f;
    bool isLight = false;
    bool advancedLight = false;
};

static std::unordered_map<std::string, TFTrackedObject> gTfTrackedObjects;

// Runtime hooks for the existing host parser/renderer. They let the library
// accept `say 3d light(n) ...` and pass the light intensity to the shader
// without changing TigerFlash_Basics_IDE.cpp on disk.
static void *gTfExecuteLineAddress = nullptr;
static void *gTfExecuteLineTrampoline = nullptr;
static void *gTfDrawObjectSingleAddress = nullptr;
static void *gTfDrawObjectSingleTrampoline = nullptr;
static void *gTfResolvePhysicalCollisionsAddress = nullptr;
static void *gTfResolvePhysicalCollisionsTrampoline = nullptr;
static bool gTfExecuteLineHookInstalled = false;
static bool gTfDrawObjectHookInstalled = false;
static bool gTfResolvePhysicalCollisionsHookInstalled = false;

// Host optical-material bridge. The IDE already has a dedicated optical pass
// for crystal/diamond/prism; shaders.cpp upgrades that pass in-place so the
// IDE source remains untouched.
static void *gTfOpticalShaderAddress = nullptr;
static void *gTfOpticalShaderReadyAddress = nullptr;
static void *gTfOpticalLocViewPosAddress = nullptr;
static void *gTfOpticalLocEnvironmentAddress = nullptr;
static void *gTfOpticalLocParamsAddress = nullptr;
static void *gTfOpticalLocAlphaAddress = nullptr;
static void *gTfOpticalLocTimeAddress = nullptr;
static void *gTfOpticalLocSoftFluidAddress = nullptr;
static void *gTfPrepareOpticalShaderAddress = nullptr;
static bool gTfPhysicalOpticalShaderInstalled = false;
static bool gTfPhysicalOpticalShaderFailed = false;
static std::string gTfPhysicalOpticalError;
static int gTfOpticalMaterialLocation = -1;
static int gTfOpticalLightCountLocation = -1;
static int gTfOpticalLightsLocation = -1;
static int gTfOpticalLightColorsLocation = -1;

// Screen-space transmission source. The opaque framebuffer is copied once at
// the start of the optical pass and sampled through the physically computed
// refracted ray, so a real light/background behind crystal is visible through
// the material rather than being replaced by a dim environment approximation.
static int gTfOpticalScreenTextureLocation = -1;
static int gTfOpticalScreenSizeLocation = -1;
static int gTfOpticalCameraRightLocation = -1;
static int gTfOpticalCameraUpLocation = -1;
static int gTfOpticalCameraForwardLocation = -1;
static int gTfOpticalCameraTanHalfFovLocation = -1;
static int gTfOpticalCameraAspectLocation = -1;
static int gTfOpticalRefractionScaleLocation = -1;
static int gTfOpticalScreenEnabledLocation = -1;
static int gTfOpticalWaterCenterLocation = -1;
static int gTfOpticalWaterRadiusLocation = -1;
static int gTfOpticalWaterMagnificationLocation = -1;
static int gTfOpticalWaterScatterLocation = -1;
static unsigned int gTfOpticalScreenTexture = 0;
static int gTfOpticalScreenWidth = 0;
static int gTfOpticalScreenHeight = 0;
static bool gTfOpticalScreenCaptured = false;
static bool gTfOpticalScreenFunctionsReady = false;
static int gTfOpticalCurrentScreenWidth = 0;
static int gTfOpticalCurrentScreenHeight = 0;
static const void *gTfOpticalCurrentCameraPtr = nullptr;

// Global lighting pipeline forward declarations.
static void tfUpdateGlobalLightingState();
static void tfUploadGlobalLightingUniformsIfNeeded();

using TFExecuteLineFunction = bool (*)(std::string);
using TFDrawObjectSingleFunction = void (*)(const void *, const void *, int, int, float);
using TFResolvePhysicalCollisionsFunction = void (*)();
static TFExecuteLineFunction gTfExecuteLineOriginal = nullptr;
static TFDrawObjectSingleFunction gTfDrawObjectSingleOriginal = nullptr;
static TFResolvePhysicalCollisionsFunction gTfResolvePhysicalCollisionsOriginal = nullptr;

using TFPrepareOpticalShaderFunction = void (*)();
using TFSetOpticalParamsFunction = void (*)(const void *, const void *);
static TFPrepareOpticalShaderFunction gTfPrepareOpticalShaderOriginal = nullptr;
static TFSetOpticalParamsFunction gTfSetOpticalParamsOriginal = nullptr;
static void *gTfSetOpticalParamsAddress = nullptr;
static void *gTfSetOpticalParamsTrampoline = nullptr;
static bool gTfSetOpticalParamsHookInstalled = false;

static constexpr unsigned int TF_GL_VERTEX_SHADER = 0x8B31;
static constexpr unsigned int TF_GL_FRAGMENT_SHADER = 0x8B30;
static constexpr unsigned int TF_GL_COMPILE_STATUS = 0x8B81;
static constexpr unsigned int TF_GL_LINK_STATUS = 0x8B82;
static constexpr unsigned int TF_GL_INFO_LOG_LENGTH = 0x8B84;
static constexpr unsigned int TF_RL_MAX_SHADER_LOCATIONS = 32;
static constexpr unsigned int TF_GL_CURRENT_PROGRAM = 0x8B8D;
static constexpr unsigned int TF_GL_ACTIVE_TEXTURE = 0x84E0;
static constexpr unsigned int TF_GL_TEXTURE0 = 0x84C0;
static constexpr unsigned int TF_GL_TEXTURE1 = 0x84C1;
static constexpr unsigned int TF_GL_TEXTURE_2D = 0x0DE1;
static constexpr unsigned int TF_GL_TEXTURE_MIN_FILTER = 0x2801;
static constexpr unsigned int TF_GL_TEXTURE_MAG_FILTER = 0x2800;
static constexpr unsigned int TF_GL_TEXTURE_WRAP_S = 0x2802;
static constexpr unsigned int TF_GL_TEXTURE_WRAP_T = 0x2803;
static constexpr unsigned int TF_GL_LINEAR = 0x2601;
static constexpr unsigned int TF_GL_CLAMP_TO_EDGE = 0x812F;
static constexpr unsigned int TF_GL_RGBA = 0x1908;
static constexpr unsigned int TF_GL_RGBA8 = 0x8058;
static constexpr unsigned int TF_GL_UNSIGNED_BYTE = 0x1401;

static std::string tfGetGLInfoLog(bool shader, unsigned int object)
{
    int logLength = 0;
    if (shader)
        tfGLGetShaderiv(object, TF_GL_INFO_LOG_LENGTH, &logLength);
    else
        tfGLGetProgramiv(object, TF_GL_INFO_LOG_LENGTH, &logLength);

    if (logLength <= 1)
        return {};

    std::vector<char> buffer(static_cast<size_t>(logLength) + 1u, '\0');
    if (shader)
        tfGLGetShaderInfoLog(object, logLength, nullptr, buffer.data());
    else
        tfGLGetProgramInfoLog(object, logLength, nullptr, buffer.data());

    return std::string(buffer.data());
}

template <typename T>
static bool tfLoadGLProc(T &target, const char *name)
{
    target = reinterpret_cast<T>(dlsym(RTLD_DEFAULT, name));
    if (target)
        return true;

    static void *glHandle = nullptr;
    if (!glHandle)
        glHandle = dlopen("libGL.so.1", RTLD_LAZY | RTLD_GLOBAL);

    if (glHandle)
        target = reinterpret_cast<T>(dlsym(glHandle, name));

    return target != nullptr;
}

static bool tfLoadGLFunctions()
{
    return
        tfLoadGLProc(tfGLCreateShader, "glCreateShader") &&
        tfLoadGLProc(tfGLShaderSource, "glShaderSource") &&
        tfLoadGLProc(tfGLCompileShader, "glCompileShader") &&
        tfLoadGLProc(tfGLGetShaderiv, "glGetShaderiv") &&
        tfLoadGLProc(tfGLGetShaderInfoLog, "glGetShaderInfoLog") &&
        tfLoadGLProc(tfGLDeleteShader, "glDeleteShader") &&
        tfLoadGLProc(tfGLCreateProgram, "glCreateProgram") &&
        tfLoadGLProc(tfGLAttachShader, "glAttachShader") &&
        tfLoadGLProc(tfGLBindAttribLocation, "glBindAttribLocation") &&
        tfLoadGLProc(tfGLLinkProgram, "glLinkProgram") &&
        tfLoadGLProc(tfGLGetProgramiv, "glGetProgramiv") &&
        tfLoadGLProc(tfGLGetProgramInfoLog, "glGetProgramInfoLog") &&
        tfLoadGLProc(tfGLDeleteProgram, "glDeleteProgram") &&
        tfLoadGLProc(tfGLGetUniformLocation, "glGetUniformLocation") &&
        tfLoadGLProc(tfGLUniform1f, "glUniform1f") &&
        tfLoadGLProc(tfGLUniform4fv, "glUniform4fv") &&
        tfLoadGLProc(tfGLUseProgram, "glUseProgram") &&
        tfLoadGLProc(tfGLGetIntegerv, "glGetIntegerv");
}

static bool tfLoadOpticalScreenFunctions()
{
    if (gTfOpticalScreenFunctionsReady)
        return true;

    const bool ok =
        tfLoadGLProc(tfGLUniform1i, "glUniform1i") &&
        tfLoadGLProc(tfGLUniform2fv, "glUniform2fv") &&
        tfLoadGLProc(tfGLUniform3fv, "glUniform3fv") &&
        tfLoadGLProc(tfGLGenTextures, "glGenTextures") &&
        tfLoadGLProc(tfGLDeleteTextures, "glDeleteTextures") &&
        tfLoadGLProc(tfGLBindTexture, "glBindTexture") &&
        tfLoadGLProc(tfGLTexParameteri, "glTexParameteri") &&
        tfLoadGLProc(tfGLTexImage2D, "glTexImage2D") &&
        tfLoadGLProc(tfGLCopyTexSubImage2D, "glCopyTexSubImage2D") &&
        tfLoadGLProc(tfGLActiveTexture, "glActiveTexture");

    gTfOpticalScreenFunctionsReady = ok;
    return ok;
}


static uintptr_t tfMainExecutableBase()
{
    struct Context { uintptr_t base = 0; } context;

    dl_iterate_phdr(
        [](struct dl_phdr_info *info, size_t, void *opaque) -> int
        {
            Context *ctx = static_cast<Context *>(opaque);
            if (!info->dlpi_name || info->dlpi_name[0] == '\0')
            {
                ctx->base = static_cast<uintptr_t>(info->dlpi_addr);
                return 1;
            }
            return 0;
        },
        &context
    );

    return context.base;
}

static void *tfFindELFSymbolInMainExecutable(const char *symbolName)
{
    if (!symbolName || !*symbolName)
        return nullptr;

    void *dynamicSymbol = dlsym(RTLD_DEFAULT, symbolName);
    if (dynamicSymbol)
        return dynamicSymbol;

    std::ifstream file("/proc/self/exe", std::ios::binary);
    if (!file)
        return nullptr;

    std::vector<unsigned char> data(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );

    if (data.size() < sizeof(Elf64_Ehdr))
        return nullptr;

    const Elf64_Ehdr *eh = reinterpret_cast<const Elf64_Ehdr *>(data.data());
    if (eh->e_ident[EI_MAG0] != ELFMAG0 ||
        eh->e_ident[EI_MAG1] != ELFMAG1 ||
        eh->e_ident[EI_MAG2] != ELFMAG2 ||
        eh->e_ident[EI_MAG3] != ELFMAG3 ||
        eh->e_ident[EI_CLASS] != ELFCLASS64)
        return nullptr;

    if (eh->e_shoff == 0 || eh->e_shnum == 0)
        return nullptr;

    const size_t sectionTableEnd =
        static_cast<size_t>(eh->e_shoff) +
        static_cast<size_t>(eh->e_shnum) * sizeof(Elf64_Shdr);
    if (sectionTableEnd > data.size())
        return nullptr;

    const Elf64_Shdr *sections =
        reinterpret_cast<const Elf64_Shdr *>(data.data() + eh->e_shoff);

    for (int pass = 0; pass < 2; ++pass)
    {
        const uint32_t wantedType = (pass == 0) ? SHT_SYMTAB : SHT_DYNSYM;
        for (Elf64_Half i = 0; i < eh->e_shnum; ++i)
        {
            const Elf64_Shdr &symSection = sections[i];
            if (symSection.sh_type != wantedType ||
                symSection.sh_link >= eh->e_shnum ||
                symSection.sh_entsize == 0)
                continue;

            const Elf64_Shdr &strSection = sections[symSection.sh_link];
            const size_t symEnd = static_cast<size_t>(symSection.sh_offset) +
                                  static_cast<size_t>(symSection.sh_size);
            const size_t strEnd = static_cast<size_t>(strSection.sh_offset) +
                                  static_cast<size_t>(strSection.sh_size);
            if (symEnd > data.size() || strEnd > data.size())
                continue;

            const char *strings = reinterpret_cast<const char *>(
                data.data() + strSection.sh_offset);
            const size_t symbolCount = static_cast<size_t>(
                symSection.sh_size / symSection.sh_entsize);

            for (size_t s = 0; s < symbolCount; ++s)
            {
                const size_t offset = static_cast<size_t>(symSection.sh_offset) +
                    s * static_cast<size_t>(symSection.sh_entsize);
                if (offset + sizeof(Elf64_Sym) > symEnd)
                    break;

                const Elf64_Sym *symbol = reinterpret_cast<const Elf64_Sym *>(
                    data.data() + offset);
                if (symbol->st_shndx == SHN_UNDEF || symbol->st_value == 0 ||
                    symbol->st_name >= strSection.sh_size)
                    continue;

                if (std::strcmp(strings + symbol->st_name, symbolName) != 0)
                    continue;

                const uintptr_t base = tfMainExecutableBase();
                const uintptr_t address = (eh->e_type == ET_DYN)
                    ? base + static_cast<uintptr_t>(symbol->st_value)
                    : static_cast<uintptr_t>(symbol->st_value);
                return reinterpret_cast<void *>(address);
            }
        }
    }

    return nullptr;
}

static bool tfPatchAbsoluteJump(void *target, void *replacement)
{
    if (!target || !replacement)
        return false;

    constexpr size_t jumpSize = 12;
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0)
        return false;

    const uintptr_t targetAddress = reinterpret_cast<uintptr_t>(target);
    const uintptr_t pageStart = targetAddress &
        ~static_cast<uintptr_t>(pageSize - 1);

    if (mprotect(reinterpret_cast<void *>(pageStart),
                 static_cast<size_t>(pageSize),
                 PROT_READ | PROT_WRITE | PROT_EXEC) != 0)
        return false;

    unsigned char jump[jumpSize] =
    {
        0x48, 0xB8,
        0,0,0,0,0,0,0,0,
        0xFF, 0xE0
    };

    const uintptr_t destination = reinterpret_cast<uintptr_t>(replacement);
    std::memcpy(jump + 2, &destination, sizeof(destination));
    std::memcpy(target, jump, sizeof(jump));
    __builtin___clear_cache(
        reinterpret_cast<char *>(target),
        reinterpret_cast<char *>(target) + sizeof(jump)
    );

    mprotect(reinterpret_cast<void *>(pageStart),
             static_cast<size_t>(pageSize),
             PROT_READ | PROT_EXEC);
    return true;
}

static std::vector<std::array<float, 3>> tfBuildBakedEnvironmentLUT(int width, int height)
{
    std::vector<std::array<float, 3>> lut(static_cast<size_t>(width * height));

    // Baked directional irradiance. The LUT is intentionally low-frequency:
    // it gives every material a stable, inexpensive global illumination base,
    // while the dynamic light/shadow layer below handles moving lights.
    const float sunX = -0.48f, sunY = 0.82f, sunZ = 0.31f;
    const float sunLen = std::sqrt(sunX*sunX + sunY*sunY + sunZ*sunZ);
    const float sx = sunX / sunLen, sy = sunY / sunLen, sz = sunZ / sunLen;

    for (int y = 0; y < height; ++y)
    {
        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
        const float phi = v * static_cast<float>(3.14159265358979323846);
        for (int x = 0; x < width; ++x)
        {
            const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
            const float theta = u * 2.0f * static_cast<float>(3.14159265358979323846);

            const float nx = std::sin(phi) * std::cos(theta);
            const float ny = std::cos(phi);
            const float nz = std::sin(phi) * std::sin(theta);
            const float sunDot = std::max(0.0f, nx*sx + ny*sy + nz*sz);
            const float skyAmount = std::max(0.0f, ny);
            const float groundAmount = std::max(0.0f, -ny);
            const float horizon = std::exp(-std::pow((ny - 0.02f) / 0.20f, 2.0f));
            const float horizonWarm = std::exp(-std::pow((ny + 0.02f) / 0.12f, 2.0f));
            const float warmSun = std::pow(sunDot, 220.0f);
            const float broadSun = std::pow(sunDot, 11.0f);
            const float skyBounce = skyAmount * skyAmount;
            const float groundBounce = groundAmount * groundAmount;

            std::array<float, 3> c =
            {
                0.012f + 0.070f*skyAmount + 0.028f*skyBounce + 0.024f*horizon + 0.022f*horizonWarm + 3.0f*broadSun + 5.4f*warmSun,
                0.018f + 0.095f*skyAmount + 0.040f*skyBounce + 0.048f*horizon + 0.034f*horizonWarm + 2.55f*broadSun + 4.8f*warmSun,
                0.040f + 0.180f*skyAmount + 0.085f*skyBounce + 0.082f*horizon + 0.052f*horizonWarm + 2.0f*broadSun + 3.5f*warmSun
            };
            // Cool sky bounce and warm earth bounce are baked into the base
            // irradiance so shaded sides do not fall to a dead black.
            c[0] += groundBounce * 0.055f + skyAmount * 0.010f;
            c[1] += groundBounce * 0.042f + skyAmount * 0.012f;
            c[2] += groundBounce * 0.030f + skyAmount * 0.018f;
            lut[static_cast<size_t>(y * width + x)] = c;
        }
    }
    return lut;
}

static std::string tfVec3Literal(const std::array<float, 3> &v)
{
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(6);
    out << "vec3(" << v[0] << "," << v[1] << "," << v[2] << ")";
    return out.str();
}

static std::string tfBuildAdvancedVertexShader()
{
    return R"GLSL(#version 330
in vec3 vertexPosition;
in vec3 vertexNormal;
uniform mat4 mvp;
uniform mat4 matModel;
uniform mat4 matNormal;
uniform mat4 matView;
out vec3 worldPosition;
out vec3 worldNormal;
out vec3 fragCameraPos;
void main()
{
    fragCameraPos = -(transpose(mat3(matView)) * matView[3].xyz);
    worldPosition = (matModel * vec4(vertexPosition, 1.0)).xyz;
    worldNormal = normalize((matNormal * vec4(vertexNormal, 0.0)).xyz);
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
)GLSL";
}

static std::string tfBuildAdvancedFragmentShader()
{
    constexpr int W = 24;
    constexpr int H = 12;
    const auto lut = tfBuildBakedEnvironmentLUT(W, H);

    std::ostringstream baked;
    baked << "const vec3 TF_BAKED_ENV[" << lut.size() << "] = vec3[" << lut.size() << "](\n";
    for (size_t i = 0; i < lut.size(); ++i)
    {
        baked << "    " << tfVec3Literal(lut[i]);
        if (i + 1 != lut.size()) baked << ',';
        baked << '\n';
    }
    baked << ");\n";

    std::ostringstream glsl;
    glsl << R"GLSL(#version 330
in vec3 worldPosition;
in vec3 worldNormal;
in vec3 fragCameraPos;
uniform vec3 viewPos;
// Posicao real da camera: vem da matriz de view enviada pelo raylib; se nao
// chegar, usa o uniform viewPos.
vec3 tfCamPos()
{
    return length(fragCameraPos) > 0.0001 ? fragCameraPos : viewPos;
}
uniform vec4 colDiffuse;
uniform float tfLightStrength;
// 1 = objetos (degrade trocado), 0 = chao (sombreamento normal).
uniform float tfSwapShading;
uniform float tfDynamicLightCount;
uniform vec4 tfDynamicLights[4];
// alpha = 1 when global `advanced light` is enabled;
// alpha = 0 for normal `light(...)`, which intentionally has no realistic shadows.
uniform vec4 tfDynamicColors[4];
// x = effective source radius in world units. This turns a mathematical point
// light into a finite emitter so penumbra can be computed by sampling its area.
uniform vec4 tfDynamicLightParams[4];
uniform float tfDynamicOccluderCount;
// xyz=center, w=analytic shape id.
// 0=cube, 1=sphere, 2=cylinder, 3=capsule, 4=cone, 5=generic box.
uniform vec4 tfDynamicOccluders[24];
// xyz = half-extents/radii in the object's local frame.
uniform vec4 tfDynamicOccluderSizes[24];
// xyz = Euler rotation in radians (inverse rotation is applied to rays).
// w = refractive material: 0 opaque, 1 quartz, 2 diamond, 3 BK7 hex prism, 4 water/soft body.
uniform vec4 tfDynamicOccluderRotations[24];
// Soft-body state packets mirrored from the host physics solver:
// A = compression, stretch, bendX, bendZ
// B = flowX, flowZ, liquidSpread, impactPulse
// C = softness, soft-body-active flag, contactStrength, fluidWaveTime
// D = contactPoint local XYZ, grounded-support flag
uniform vec4 tfDynamicOccluderSoftA[24];
uniform vec4 tfDynamicOccluderSoftB[24];
uniform vec4 tfDynamicOccluderSoftC[24];
uniform vec4 tfDynamicOccluderSoftD[24];
// rgb = luminance-normalised light tint of the refractive body (never darkens).
uniform vec4 tfDynamicOccluderTints[24];
uniform float tfGroundY;
uniform float tfGroundValid;
uniform vec4 tfGroundBounceColor;

// Global lighting controls. tfExposure is camera/eye adaptation, while the
// GI lobes contain a compact low-frequency approximation of one-bounce color
// bleeding. Atmosphere is exponential height fog + directional in-scattering.
uniform float tfExposure;
uniform float tfGIIntensity;
uniform vec4 tfGIDirections[6];
uniform vec4 tfAtmosphereParams; // x=density, y=height scale, z=Rayleigh weight, w=Mie weight
uniform vec4 tfAtmosphereColor;  // rgb=scattering color, a=sun in-scatter strength
out vec4 finalColor;
)GLSL";
    glsl << baked.str();
    glsl << R"GLSL(
const float TF_PI = 3.14159265359;
const vec3 TF_SUN_DIR = vec3(-0.42050491, 0.84100982, 0.34040874);
const vec3 TF_SUN_COLOR = vec3(5.25, 4.85, 4.15);

// Light transport remains physically distance-based. These constants affect
// ONLY the rendered source itself, not illumination received by other objects.
// They provide a display-space floor so a valid light never becomes numerically
// indistinguishable from black just because the camera exposure is conservative.
const float TF_SOURCE_MIN_LINEAR_LUMA = 0.0065;
const float TF_SOURCE_MAX_LIFT = 0.26;

vec3 tfEnvLookup(vec3 n)
{
    n = normalize(n);
    float phi = acos(clamp(n.y, -1.0, 1.0));
    float theta = (abs(n.x) + abs(n.z) > 0.000001) ? atan(n.z, n.x) : 0.0;
    if (theta < 0.0) theta += 2.0 * TF_PI;
    float fx = theta / (2.0 * TF_PI) * 24.0 - 0.5;
    float fy = phi / TF_PI * 12.0 - 0.5;
    int x0 = int(floor(fx));
    int y0 = int(floor(fy));
    float tx = fract(fx);
    float ty = fract(fy);
    x0 = x0 < 0 ? x0 + 24 : x0;
    int x1 = (x0 + 1) % 24;
    y0 = clamp(y0, 0, 11);
    int y1 = min(y0 + 1, 11);
    vec3 a = mix(TF_BAKED_ENV[y0 * 24 + x0], TF_BAKED_ENV[y0 * 24 + x1], tx);
    vec3 b = mix(TF_BAKED_ENV[y1 * 24 + x0], TF_BAKED_ENV[y1 * 24 + x1], tx);
    return mix(a, b, ty);
}

float tfD(float NdotH, float r)
{
    float a = r*r;
    float a2 = a*a;
    float d = (NdotH*NdotH)*(a2-1.0)+1.0;
    return a2 / max(TF_PI*d*d, 0.0001);
}

float tfG1(float NdotV, float r)
{
    float k = ((r+1.0)*(r+1.0))/8.0;
    return NdotV / max(NdotV*(1.0-k)+k, 0.0001);
}

vec3 tfF(float c, vec3 F0)
{
    return F0 + (1.0-F0)*pow(max(1.0-c, 0.0), 5.0);
}

vec3 tfInverseRotateXYZ(vec3 p, vec3 r)
{
    // Host transform is built as X/Y/Z Euler rotations. Undo them in reverse order.
    float cz = cos(r.z), sz = sin(r.z);
    vec2 yz = vec2(p.x * cz + p.y * sz,
                   -p.x * sz + p.y * cz);
    p.x = yz.x;
    p.y = yz.y;

    float cy = cos(r.y), sy = sin(r.y);
    vec2 xz = vec2(p.x * cy - p.z * sy,
                   p.x * sy + p.z * cy);
    p.x = xz.x;
    p.z = xz.y;

    float cx = cos(r.x), sx = sin(r.x);
    vec2 yy = vec2(p.y * cx + p.z * sx,
                   -p.y * sx + p.z * cx);
    p.y = yy.x;
    p.z = yy.y;
    return p;
}

bool tfSegmentBox(vec3 p0, vec3 p1)
{
    vec3 d = p1 - p0;
    float t0 = 0.0;
    float t1 = 1.0;

    for (int axis = 0; axis < 3; ++axis)
    {
        float p = axis == 0 ? p0.x : (axis == 1 ? p0.y : p0.z);
        float q = axis == 0 ? d.x  : (axis == 1 ? d.y  : d.z);
        if (abs(q) < 0.000001)
        {
            if (abs(p) > 1.0)
                return false;
            continue;
        }

        float a = (-1.0 - p) / q;
        float b = ( 1.0 - p) / q;
        if (a > b) { float tmp = a; a = b; b = tmp; }
        t0 = max(t0, a);
        t1 = min(t1, b);
        if (t0 > t1)
            return false;
    }

    return t1 > 0.0005 && t0 < 0.9995;
}

bool tfSegmentSphere(vec3 p0, vec3 p1)
{
    vec3 d = p1 - p0;
    float a = dot(d,d);
    if (a < 0.0000001)
        return false;
    float b = 2.0 * dot(p0,d);
    float c = dot(p0,p0) - 1.0;
    float disc = b*b - 4.0*a*c;
    if (disc < 0.0)
        return false;
    float root = sqrt(max(disc,0.0));
    float u0 = (-b-root)/(2.0*a);
    float u1 = (-b+root)/(2.0*a);
    return (u1 > 0.0005 && u0 < 0.9995 && u1 > u0) ||
           (u0 > 0.0005 && u0 < 0.9995);
}

bool tfSegmentCylinder(vec3 p0, vec3 p1)
{
    vec3 d = p1 - p0;
    float a = d.x*d.x + d.z*d.z;
    float b = 2.0*(p0.x*d.x + p0.z*d.z);
    float c = p0.x*p0.x + p0.z*p0.z - 1.0;

    if (abs(a) > 0.000001)
    {
        float disc = b*b - 4.0*a*c;
        if (disc >= 0.0)
        {
            float root = sqrt(max(disc,0.0));
            float u0 = (-b-root)/(2.0*a);
            float u1 = (-b+root)/(2.0*a);
            if (u0 > u1) { float tmp=u0; u0=u1; u1=tmp; }
            float y0 = p0.y + d.y*u0;
            float y1 = p0.y + d.y*u1;
            if (u1 > 0.0005 && u0 < 0.9995 &&
                max(y0,y1) >= -1.0 && min(y0,y1) <= 1.0)
                return true;
        }
    }

    if (abs(d.y) > 0.000001)
    {
        float uBottom = (-1.0-p0.y)/d.y;
        if (uBottom > 0.0005 && uBottom < 0.9995)
        {
            vec2 q = p0.xz + d.xz*uBottom;
            if (dot(q,q) <= 1.000001)
                return true;
        }
        float uTop = (1.0-p0.y)/d.y;
        if (uTop > 0.0005 && uTop < 0.9995)
        {
            vec2 q = p0.xz + d.xz*uTop;
            if (dot(q,q) <= 1.000001)
                return true;
        }
    }
    return false;
}

bool tfSegmentCone(vec3 p0, vec3 p1)
{
    // Closed right cone: base y=-1, tip y=+1, radius falls linearly from 1 to 0.
    vec3 d = p1-p0;
    float a = d.x*d.x + d.z*d.z - 0.25*d.y*d.y;
    float b = 2.0*(p0.x*d.x + p0.z*d.z) + 0.5*d.y*(1.0-p0.y);
    float c = p0.x*p0.x + p0.z*p0.z - 0.25*(1.0-p0.y)*(1.0-p0.y);

    if (abs(a) < 0.000001)
    {
        if (abs(b) > 0.000001)
        {
            float u = -c/b;
            if (u > 0.0005 && u < 0.9995)
            {
                float y = p0.y + d.y*u;
                if (y > -1.0001 && y < 1.0001)
                    return true;
            }
        }
    }
    else
    {
        float disc = b*b - 4.0*a*c;
        if (disc >= 0.0)
        {
            float root = sqrt(max(disc,0.0));
            float u0 = (-b-root)/(2.0*a);
            float u1 = (-b+root)/(2.0*a);
            if (u0 > u1) { float tmp=u0; u0=u1; u1=tmp; }
            for (int k=0; k<2; ++k)
            {
                float u = (k==0) ? u0 : u1;
                if (u > 0.0005 && u < 0.9995)
                {
                    float y = p0.y + d.y*u;
                    if (y > -1.0001 && y < 1.0001)
                        return true;
                }
            }
        }
    }

    // Base cap at y=-1.
    if (abs(d.y) > 0.000001)
    {
        float u = (-1.0-p0.y)/d.y;
        if (u > 0.0005 && u < 0.9995)
        {
            vec2 q = p0.xz + d.xz*u;
            if (dot(q,q) <= 1.000001)
                return true;
        }
    }
    return false;
}

bool tfSegmentCapsule(vec3 p0, vec3 p1)
{
    // Analytic capsule approximation in normalized local space.
    // Cylinder body is [-0.62,0.62], caps are unit spheres centered there.
    vec3 d = p1-p0;
    float best = 2.0;

    vec3 centers[2] = vec3[2](vec3(0.0,-0.62,0.0), vec3(0.0,0.62,0.0));
    for (int i=0;i<2;++i)
    {
        vec3 q0 = p0-centers[i];
        float a = dot(d,d);
        float b = 2.0*dot(q0,d);
        float c = dot(q0,q0)-1.0;
        float disc = b*b-4.0*a*c;
        if (disc >= 0.0 && a > 0.0000001)
        {
            float root=sqrt(max(disc,0.0));
            float u=(-b-root)/(2.0*a);
            float v=(-b+root)/(2.0*a);
            if (v > 0.0005 && u < 0.9995)
                best=min(best,max(u,0.0005));
        }
    }

    // Segment against the cylindrical middle.
    float a = d.x*d.x+d.z*d.z;
    float b = 2.0*(p0.x*d.x+p0.z*d.z);
    float c = p0.x*p0.x+p0.z*p0.z-1.0;
    if (abs(a)>0.000001)
    {
        float disc=b*b-4.0*a*c;
        if (disc>=0.0)
        {
            float root=sqrt(max(disc,0.0));
            float u0=(-b-root)/(2.0*a);
            float u1=(-b+root)/(2.0*a);
            if (u0>u1){float tmp=u0;u0=u1;u1=tmp;}
            float y0=p0.y+d.y*u0, y1=p0.y+d.y*u1;
            if (u1>0.0005 && u0<0.9995 && max(y0,y1)>=-0.62 && min(y0,y1)<=0.62)
                return true;
        }
    }
    return best < 1.999;
}

vec3 tfSoftShadowForward(vec3 p, vec4 softA, vec4 softB, vec4 softC, vec4 softD)
{
    // Reproduce the important silhouette terms from applySoftBodyDeformation()
    // in the host. This is intentionally a shadow-space proxy, not a second
    // copy of the full CPU mesh solver. It tracks the same live state so the
    // shadow changes immediately as the soft body squashes, stretches, bends
    // and flows.
    float fluid = clamp(softC.x, 0.0, 1.0);
    float compression = clamp(softA.x, 0.0, 0.96);
    float stretch = clamp(softA.y, 0.0, 0.80);
    float bendX = clamp(softA.z, -1.05, 1.05);
    float bendZ = clamp(softA.w, -1.05, 1.05);
    float flowX = clamp(softB.x, -0.80, 0.80);
    float flowZ = clamp(softB.y, -0.80, 0.80);
    float liquidSpread = clamp(softB.z, 0.0, 0.90);
    float impactPulse = clamp(softB.w, 0.0, 1.0);

    float groundedFlatten = fluid * (0.065 + 0.180 * fluid);
    float persistentFlatten = fluid * fluid * 0.038 + groundedFlatten;
    float impactFlatten = clamp(
        compression * (0.18 + 0.08 * fluid) +
        impactPulse * (0.16 + 0.10 * fluid), 0.0, 0.55);
    float verticalScale = clamp(1.0 - persistentFlatten -
                                impactFlatten * (0.16 + 0.16 * fluid),
                                0.54, 1.18);

    float areaScale = 1.0 / sqrt(max(0.08, verticalScale));
    areaScale *= 1.0 + fluid * (liquidSpread * (1.05 + 0.35 * fluid) + 0.0);

    float nx = clamp(p.x, -1.0, 1.0);
    float ny = clamp(p.y, -1.0, 1.0);
    float nz = clamp(p.z, -1.0, 1.0);

    // The same lower-contact rim logic that makes the host liquid spread.
    float bottom = clamp((1.0 - ny) * 0.5, 0.0, 1.0);
    float contact = bottom * bottom;
    float rimSpread = 1.0 + (compression * 0.62 + impactPulse * 0.92) *
                       (0.35 + 0.65 * fluid) * contact;

    p.y *= verticalScale;
    p.x *= areaScale * rimSpread;
    p.z *= areaScale * rimSpread;

    float bendProfile = ny * (0.75 + 0.25 * abs(ny));
    p.x += bendX * bendProfile * (0.65 + 0.85 * fluid);
    p.z += bendZ * bendProfile * (0.65 + 0.85 * fluid);

    float middle = 1.0 - ny * ny;
    p.x += flowX * middle * (0.55 + fluid);
    p.z += flowZ * middle * (0.55 + fluid);

    float stretchAxis = 1.0 + stretch * (0.75 + 0.90 * fluid);
    float verticalStretch = clamp(stretchAxis, 0.55, 1.70);
    p.y *= verticalStretch;
    float horizontalCompensation = 1.0 / sqrt(max(0.20, verticalStretch));
    p.x *= horizontalCompensation;
    p.z *= horizontalCompensation;

    // Contact-aware molding follows the same remembered support point used by
    // the host soft-body solver. The shadow proxy therefore changes shape when
    // the blob rolls from one support point to another, rather than using one
    // globally flattened silhouette.
    if (softD.w > 0.5 && softC.z > 0.01)
    {
        vec3 contactPoint = softD.xyz;
        vec2 patchDelta = p.xz - contactPoint.xz;
        float patchRadius = 0.72 + 0.42 * fluid;
        float patchWeight = 1.0 - length(patchDelta) / max(0.05, patchRadius);
        patchWeight = clamp(patchWeight, 0.0, 1.0);
        patchWeight = patchWeight * patchWeight * (3.0 - 2.0 * patchWeight);

        float signedDistance = p.y - contactPoint.y;
        float band = 0.48 + 0.62 * fluid;
        float nearSurface = 1.0 - signedDistance / max(0.08, band);
        nearSurface = clamp(nearSurface, 0.0, 1.0);
        nearSurface = nearSurface * nearSurface * (3.0 - 2.0 * nearSurface);

        float strength = (0.22 + 0.68 * fluid) *
                         patchWeight * nearSurface * softC.z;
        float targetY = contactPoint.y + (0.012 +
                         (1.0 - nearSurface * patchWeight) * band *
                         (0.34 + 0.30 * (1.0 - fluid)));
        p.y += clamp(targetY - p.y, -0.75, 0.0) * strength;
    }

    if (fluid > 0.45)
    {
        float angle = atan(p.z, p.x);
        float waveTime = softC.w;
        float lobeA = sin(angle * 4.0 + waveTime * (0.8 + 1.7 * fluid));
        float lobeB = sin(angle * 7.0 - waveTime * 0.9 + 1.3);
        float lobeMask = 0.35 + 0.65 * (1.0 - ny * ny);
        float lobeStrength = fluid * (0.018 + 0.055 * liquidSpread);
        float lobe = (lobeA * 0.62 + lobeB * 0.38) * lobeMask;
        p.x *= 1.0 + lobe * lobeStrength;
        p.z *= 1.0 - lobe * lobeStrength * 0.72;
    }

    return p;
}

vec3 tfSoftShadowInverse(vec3 target, vec4 softA, vec4 softB, vec4 softC, vec4 softD)
{
    // Fixed-point inversion keeps the ray test tied to the deformed silhouette
    // while remaining cheap enough for realtime soft shadows.
    vec3 p = target;
    for (int i = 0; i < 4; ++i)
    {
        vec3 deformed = tfSoftShadowForward(p, softA, softB, softC, softD);
        p += (target - deformed) * 0.78;
    }
    return p;
}

bool tfRayHitsExactObjectSegment(vec3 p0, vec3 p1, int shape)
{
    if (shape == 0) return tfSegmentBox(p0,p1);
    if (shape == 1) return tfSegmentSphere(p0,p1);
    if (shape == 2) return tfSegmentCylinder(p0,p1);
    if (shape == 3) return tfSegmentCapsule(p0,p1);
    if (shape == 4) return tfSegmentCone(p0,p1);
    return tfSegmentBox(p0,p1);
}

bool tfRayHitsExactObject(vec3 lightPosition, vec3 point,
                          vec3 center, vec4 sizeData, vec4 rotationData,
                          vec4 softA, vec4 softB, vec4 softC, vec4 softD)
{
    vec3 p0 = tfInverseRotateXYZ(lightPosition-center, rotationData.xyz);
    vec3 p1 = tfInverseRotateXYZ(point-center, rotationData.xyz);

    vec3 size = max(abs(sizeData.xyz), vec3(0.0001));
    p0 /= size;
    p1 /= size;

    int shape = int(sizeData.w + 0.5);
    bool soft = softC.y > 0.5;

    if (!soft)
        return tfRayHitsExactObjectSegment(p0,p1,shape);

    // The deformed ray is curved in base-object space. Subdivide it into four
    // short segments, invert each endpoint, then test against the real base
    // primitive. This tracks bend/flow instead of forcing everything through an
    // undeformed sphere.
    vec3 qPrev = tfSoftShadowInverse(p0, softA, softB, softC, softD);
    for (int k = 1; k <= 4; ++k)
    {
        float t = float(k) / 4.0;
        vec3 qTarget = mix(p0, p1, t);
        vec3 qNow = tfSoftShadowInverse(qTarget, softA, softB, softC, softD);
        if (tfRayHitsExactObjectSegment(qPrev, qNow, shape))
            return true;
        qPrev = qNow;
    }
    return false;
}

float tfHash21(vec2 p)
{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 34.45);
    return fract(p.x * p.y);
}

vec3 tfBuildTangent(vec3 n)
{
    vec3 a = abs(n.y) < 0.999 ? vec3(0.0,1.0,0.0) : vec3(1.0,0.0,0.0);
    return normalize(cross(a,n));
}

bool tfPointLightShadowExact(vec3 lightPosition, vec3 point)
{
    // Exact visibility of a single point source: any opaque primitive crossing
    // the finite segment removes the direct path entirely.
    for (int i=0; i<24; ++i)
    {
        if (float(i) >= tfDynamicOccluderCount) break;
        if (tfDynamicOccluderRotations[i].w > 0.5) continue; // refractive: bends light, never blocks it
        vec4 centerShape = tfDynamicOccluders[i];
        if (tfRayHitsExactObject(lightPosition, point,
                                 centerShape.xyz,
                                 tfDynamicOccluderSizes[i],
                                 tfDynamicOccluderRotations[i],
                                 tfDynamicOccluderSoftA[i],
                                 tfDynamicOccluderSoftB[i],
                                 tfDynamicOccluderSoftC[i],
                                 tfDynamicOccluderSoftD[i]))
            return true;
    }
    return false;
}

float tfPointLightVisibilitySoft(vec3 lightPosition, vec3 point,
                                 float sourceRadius, float seed)
{
    vec3 toPoint = point - lightPosition;
    float distanceToPoint = length(toPoint);
    if (distanceToPoint <= 0.0005) return 1.0;

    vec3 L = normalize(toPoint);
    vec3 T = tfBuildTangent(L);
    vec3 B = normalize(cross(L,T));
    float rotation = 6.28318530718 * tfHash21(point.xz + vec2(seed*17.13, seed*7.91));

    // Finite emitter sampling produces the physically expected umbra,
    // penumbra and fully lit regions.  A point light is the zero-radius limit.
    const int SAMPLE_COUNT = 12;
    float visible = 0.0;

    for (int i=0; i<SAMPLE_COUNT; ++i)
    {
        float fi = float(i) + 0.5;
        float radial = sqrt(fi / float(SAMPLE_COUNT));
        float angle = rotation + fi * 2.39996323;
        vec2 disk = vec2(cos(angle), sin(angle)) * radial * sourceRadius;
        vec3 sampleLight = lightPosition + T*disk.x + B*disk.y;
        if (!tfPointLightShadowExact(sampleLight, point))
            visible += 1.0;
    }

    return visible / float(SAMPLE_COUNT);
}

float tfDirectionalVisibilitySoft(vec3 point, vec3 lightDirection)
{
    vec3 D = normalize(lightDirection);
    vec3 T = tfBuildTangent(D);
    vec3 B = normalize(cross(D,T));

    // Small solar disk: the angular radius is tiny at Earth-like scales but
    // still creates a measurable penumbra instead of an infinitely sharp edge.
    const float angularRadius = 0.014;
    const int SAMPLE_COUNT = 8;
    float visible = 0.0;

    for (int i=0; i<SAMPLE_COUNT; ++i)
    {
        float fi = float(i) + 0.5;
        float r = sqrt(fi/float(SAMPLE_COUNT)) * angularRadius;
        float a = fi * 2.39996323;
        vec3 Dn = normalize(D + T*(cos(a)*r) + B*(sin(a)*r));
        vec3 rayEnd = point + Dn * 240.0;
        bool blocked = false;
        for (int j=0; j<24; ++j)
        {
            if (float(j) >= tfDynamicOccluderCount) break;
            if (tfDynamicOccluderRotations[j].w > 0.5) continue; // refractive: no opaque occlusion
            vec4 centerShape = tfDynamicOccluders[j];
            if (tfRayHitsExactObject(point, rayEnd,
                                     centerShape.xyz,
                                     tfDynamicOccluderSizes[j],
                                     tfDynamicOccluderRotations[j],
                                     tfDynamicOccluderSoftA[j],
                                     tfDynamicOccluderSoftB[j],
                                     tfDynamicOccluderSoftC[j],
                                     tfDynamicOccluderSoftD[j]))
            {
                blocked = true;
                break;
            }
        }
        if (!blocked) visible += 1.0;
    }
    return visible / float(SAMPLE_COUNT);
}

float tfContactOcclusion(vec3 point, vec3 N)
{
    // Very short secondary rays capture contact darkening where surfaces are
    // close to one another. This is the real-time approximation of near-field
    // occlusion and is deliberately weaker than direct-light shadowing.
    vec3 T = tfBuildTangent(N);
    vec3 B = normalize(cross(N,T));
    const int SAMPLE_COUNT = 6;
    float blocked = 0.0;

    const vec3 dirs[6] = vec3[6](
        vec3(0.00,1.00,0.00), vec3(0.00,0.55,0.84), vec3(0.74,0.55,0.38),
        vec3(-0.74,0.55,0.38), vec3(0.38,0.55,-0.74), vec3(-0.38,0.55,-0.74));

    for (int i=0; i<SAMPLE_COUNT; ++i)
    {
        vec3 d = normalize(N*dirs[i].y + T*dirs[i].x + B*dirs[i].z);
        vec3 origin = point + N*0.025;
        vec3 end = origin + d*1.15;
        for (int j=0; j<24; ++j)
        {
            if (float(j) >= tfDynamicOccluderCount) break;
            if (tfDynamicOccluderRotations[j].w > 0.5) continue; // refractive: no opaque occlusion
            if (tfRayHitsExactObject(origin, end,
                                     tfDynamicOccluders[j].xyz,
                                     tfDynamicOccluderSizes[j],
                                     tfDynamicOccluderRotations[j],
                                     tfDynamicOccluderSoftA[j],
                                     tfDynamicOccluderSoftB[j],
                                     tfDynamicOccluderSoftC[j],
                                     tfDynamicOccluderSoftD[j]))
            {
                blocked += 1.0;
                break;
            }
        }
    }

    // Never replace direct shadows with AO. This is only a soft local factor.
    return mix(1.0, 0.42, blocked/float(SAMPLE_COUNT));
}

vec3 tfGroundBouncePhysical(vec3 N, vec3 point,
                            vec3 lightPosition, float lightIntensity,
                            float sourceRadius, vec3 lightColor,
                            float groundY, vec3 groundAlbedo)
{
    if (groundY > point.y + 0.001 || point.y-groundY > 6.0 ||
        N.y <= 0.02 || lightIntensity <= 0.0001)
        return vec3(0.0);

    const vec3 groundNormal = vec3(0.0,1.0,0.0);
    float lightHeight = lightPosition.y-groundY;
    if (lightHeight <= 0.02) return vec3(0.0);

    // Integrate a small disk of differential ground patches. Each patch first
    // receives light from the source, reflects only its albedo/PI fraction, and
    // then contributes a geometrically foreshortened amount to the receiver.
    const int PATCH_COUNT = 6;
    float footprintRadius = max(0.35, lightHeight*0.34 + sourceRadius*1.5);
    vec3 patchCenter = vec3(lightPosition.x, groundY, lightPosition.z);
    float areaPerPatch = (TF_PI*footprintRadius*footprintRadius) / float(PATCH_COUNT);
    vec3 reflected = vec3(0.0);

    float rotation = 6.28318530718 * tfHash21(point.xz + lightPosition.xz*0.17);
    for (int i=0; i<PATCH_COUNT; ++i)
    {
        float fi = float(i)+0.5;
        float rr = sqrt(fi/float(PATCH_COUNT))*footprintRadius;
        float aa = rotation + fi*2.39996323;
        vec3 patch = patchCenter + vec3(cos(aa)*rr, 0.0, sin(aa)*rr);

        vec3 fromPatchToLight = lightPosition-patch;
        float d1 = length(fromPatchToLight);
        if (d1 <= 0.01) continue;
        vec3 Lp = fromPatchToLight/d1;
        float cosIn = max(dot(groundNormal,Lp),0.0);
        if (cosIn <= 0.0) continue;

        float sourceVis = tfPointLightVisibilitySoft(lightPosition,patch,
                                                       sourceRadius,fi+31.0);
        if (sourceVis <= 0.001) continue;

        // Point-source irradiance: I/r^2 multiplied by incident cosine.
        float incident = lightIntensity/(d1*d1) * cosIn * sourceVis;
        // Lambertian reflection: outgoing diffuse radiance is rho/pi times irradiance.
        vec3 patchRadiance = lightColor * incident * groundAlbedo / TF_PI;

        vec3 toReceiver = point-patch;
        float d2 = length(toReceiver);
        if (d2 <= 0.08) continue;
        vec3 Lr = toReceiver/d2;
        float cosOut = max(dot(groundNormal,Lr),0.0);
        float cosReceiver = max(dot(N,-Lr),0.0);
        if (cosOut <= 0.001 || cosReceiver <= 0.001) continue;

        bool blocked = false;
        vec3 rayStart = patch + groundNormal*0.025;
        for (int j=0; j<24; ++j)
        {
            if (float(j) >= tfDynamicOccluderCount) break;
            if (tfDynamicOccluderRotations[j].w > 0.5) continue; // refractive: no opaque occlusion
            if (tfRayHitsExactObject(rayStart,point,
                                     tfDynamicOccluders[j].xyz,
                                     tfDynamicOccluderSizes[j],
                                     tfDynamicOccluderRotations[j],
                                     tfDynamicOccluderSoftA[j],
                                     tfDynamicOccluderSoftB[j],
                                     tfDynamicOccluderSoftC[j],
                                     tfDynamicOccluderSoftD[j]))
            {
                blocked=true;
                break;
            }
        }
        if (blocked) continue;

        // Differential-area transfer. The second cosine and inverse-square term
        // keep the reflected contribution much weaker than direct illumination.
        float transfer = areaPerPatch * cosOut * cosReceiver / (d2*d2);
        reflected += patchRadiance * transfer;
    }

    // The tiny multiplier accounts for the fact that our six patches represent
    // an entire unresolved hemisphere. It remains deliberately low to conserve
    // energy visually instead of turning the bounce into a second bright light.
    return reflected * 0.72;
}


// =============================================================================
// REFRACTIVE LIGHT TRANSPORT (used only by `advanced light`)
// =============================================================================
// A light ray that crosses a crystal / diamond / prism / soft body is bent by
// Snell's law at the entry and exit facets, separately for R, G and B (Cauchy
// dispersion). Radiance is conserved through a lossless dielectric (L/n^2 is
// invariant and the n^2 factors cancel on exit), so the light is DISTORTED but
// not darkened. Fresnel and Beer-Lambert losses are intentionally not applied.
const float TF_REFRACT_MIN_GAIN = 1.0;  // never darker than the unobstructed light
const float TF_REFRACT_MAX_GAIN = 3.2;  // cap for ball-lens focusing (caustic)

vec3 tfRotateXYZ(vec3 p, vec3 r)
{
    float cx = cos(r.x), sx = sin(r.x);
    p = vec3(p.x, p.y*cx - p.z*sx, p.y*sx + p.z*cx);
    float cy = cos(r.y), sy = sin(r.y);
    p = vec3(p.x*cy + p.z*sy, p.y, -p.x*sy + p.z*cy);
    float cz = cos(r.z), sz = sin(r.z);
    return vec3(p.x*cz - p.y*sz, p.x*sz + p.y*cz, p.z);
}

vec3 tfRefractiveIOR(float m)
{
    if (m > 3.5) return vec3(1.3325, 1.3330, 1.3335);      // water / soft body
    if (m > 2.5) return vec3(1.51392, 1.51633, 1.52198);   // BK7 prism
    if (m > 1.5) return vec3(2.415, 2.420, 2.430);         // diamond
    return vec3(1.542, 1.545, 1.548);                      // quartz crystal
}

int tfRefractivePlaneCount(float m)
{
    if (m > 3.5) return 6;   // cube-like water body
    if (m > 2.5) return 8;   // hexagonal prism: 6 sides + 2 caps
    if (m > 1.5) return 8;   // diamond: octahedron
    return 18;               // crystal: 6 sides + 6 top + 6 bottom facets
}

// Convex half-spaces n.p <= dd in the object's normalised local space.
void tfRefractivePlane(float m, int i, out vec3 n, out float dd)
{
    if (m > 3.5)
    {
        int ax = i % 3;
        float sg = (i < 3) ? 1.0 : -1.0;
        n = vec3(ax == 0 ? sg : 0.0, ax == 1 ? sg : 0.0, ax == 2 ? sg : 0.0);
        dd = 1.0;
        return;
    }
    if (m > 2.5)
    {
        if (i < 6)
        {
            float a = 0.5235988 + 1.0471976 * float(i);
            n = vec3(cos(a), 0.0, sin(a));
            dd = 0.8660254;
        }
        else
        {
            n = vec3(0.0, i == 6 ? 1.0 : -1.0, 0.0);
            dd = 1.0;
        }
        return;
    }
    if (m > 1.5)
    {
        float sx = (i % 2 == 0) ? 1.0 : -1.0;
        float sy = ((i / 2) % 2 == 0) ? 1.0 : -1.0;
        float sz = (i < 4) ? 1.0 : -1.0;
        n = vec3(sx, sy, sz) * 0.57735027;
        dd = 0.57735027;
        return;
    }
    int k = i % 6;
    float a = 0.5235988 + 1.0471976 * float(k);
    float ca = cos(a), sa = sin(a);
    if (i < 6)       { n = vec3(ca, 0.0, sa);                 dd = 0.624;  }
    else if (i < 12) { n = vec3(0.5666*ca,  0.824, 0.5666*sa); dd = 0.6288; }
    else             { n = vec3(0.5666*ca, -0.824, 0.5666*sa); dd = 0.6288; }
}

bool tfChordConvex(float m, int shape, vec3 p0, vec3 p1,
                   out float t0, out float t1, out vec3 n0, out vec3 n1)
{
    vec3 d = p1 - p0;
    t0 = 0.0; t1 = 1.0;
    n0 = vec3(0.0, 1.0, 0.0); n1 = vec3(0.0, 1.0, 0.0);

    // Water / soft bodies that are not cube-like are ellipsoids.
    if (m > 3.5 && shape != 0 && shape != 5)
    {
        float a = dot(d, d);
        float b = 2.0 * dot(p0, d);
        float c = dot(p0, p0) - 1.0;
        float disc = b*b - 4.0*a*c;
        if (a < 1e-8 || disc <= 0.0) return false;
        float sq = sqrt(disc);
        t0 = (-b - sq) / (2.0*a);
        t1 = (-b + sq) / (2.0*a);
        n0 = normalize(p0 + d*t0);
        n1 = normalize(p0 + d*t1);
        return true;
    }

    float tin = -1e9, tout = 1e9;
    vec3 nin = vec3(0.0), nout = vec3(0.0);
    int count = tfRefractivePlaneCount(m);
    for (int i = 0; i < 18; ++i)
    {
        if (i >= count) break;
        vec3 n; float dd;
        tfRefractivePlane(m, i, n, dd);
        float denom = dot(n, d);
        float dist = dd - dot(n, p0);
        if (abs(denom) < 1e-7)
        {
            if (dist < 0.0) return false;
            continue;
        }
        float t = dist / denom;
        if (denom < 0.0) { if (t > tin)  { tin = t;  nin = n;  } }
        else             { if (t < tout) { tout = t; nout = n; } }
        if (tin > tout) return false;
    }
    if (dot(nin, nin) < 0.5 || dot(nout, nout) < 0.5) return false;
    t0 = tin; t1 = tout; n0 = nin; n1 = nout;
    return true;
}

vec3 tfBendEnter(vec3 I, vec3 N, float n)
{
    vec3 r = refract(I, N, 1.0 / n);
    return dot(r, r) < 1e-6 ? reflect(I, N) : normalize(r);
}

// N faces against I (points back into the medium). Total internal reflection is
// replaced by a grazing exit so the energy still leaves the crystal ("fire").
vec3 tfBendExit(vec3 I, vec3 N, float n)
{
    vec3 r = refract(I, N, n);
    if (dot(r, r) > 1e-6) return normalize(r);
    vec3 tang = I - N * dot(I, N);
    float tl = length(tang);
    tang = tl > 1e-5 ? tang / tl : vec3(0.0);
    return normalize(tang * 0.98 - N * 0.20);
}

// One wavelength through one body: refract in, follow the BENT ray to its real exit
// facet (not the exit of the straight light->point line), refract out.
vec3 tfBendChannel(float m, int shape, vec3 rot, vec3 size, vec3 pin,
                   vec3 din, vec3 wn0, vec3 fallbackWn1, float n)
{
    vec3 d1 = tfBendEnter(din, wn0, n);
    vec3 ld = tfInverseRotateXYZ(d1, rot) / size;
    float ll = max(length(ld), 1e-5);
    vec3 wn1 = fallbackWn1;
    float e0, e1; vec3 m0, m1;
    if (tfChordConvex(m, shape, pin, pin + ld / ll * 4.0, e0, e1, m0, m1) && e1 > 1e-4)
    {
        wn1 = normalize(tfRotateXYZ(m1 / size, rot));
        if (dot(wn1, d1) < 0.0) wn1 = -wn1;
    }
    return tfBendExit(d1, -wn1, n);
}

// Returns true when at least one refractive body lies on the light->point segment.
// dirR/G/B = propagation direction arriving at `point` for each wavelength.
bool tfRefractLightPath(vec3 lightPos, vec3 point,
                        out vec3 dirR, out vec3 dirG, out vec3 dirB,
                        out float gain, out vec3 tint)
{
    vec3 seg = point - lightPos;
    float dist = length(seg);
    vec3 d0 = seg / max(dist, 1e-5);
    dirR = d0; dirG = d0; dirB = d0;
    gain = 1.0; tint = vec3(1.0);
    bool hit = false;

    for (int i = 0; i < 24; ++i)
    {
        if (float(i) >= tfDynamicOccluderCount) break;
        float m = tfDynamicOccluderRotations[i].w;
        if (m < 0.5) continue;

        vec3 center = tfDynamicOccluders[i].xyz;
        vec3 size = max(abs(tfDynamicOccluderSizes[i].xyz), vec3(0.0001));
        float rad = max(size.x, max(size.y, size.z));
        float tc = clamp(dot(center - lightPos, seg) / max(dot(seg, seg), 1e-6), 0.0, 1.0);
        if (length(lightPos + seg * tc - center) > rad) continue;

        vec3 rot = tfDynamicOccluderRotations[i].xyz;
        int shape = int(tfDynamicOccluders[i].w + 0.5);
        vec3 p0 = tfInverseRotateXYZ(lightPos - center, rot) / size;
        vec3 p1 = tfInverseRotateXYZ(point - center, rot) / size;

        float t0, t1; vec3 n0, n1;
        if (!tfChordConvex(m, shape, p0, p1, t0, t1, n0, n1)) continue;
        if (t0 < 0.0005 || t1 > 0.9995 || t1 <= t0) continue;

        vec3 wn0 = normalize(tfRotateXYZ(n0 / size, rot));
        vec3 wn1 = normalize(tfRotateXYZ(n1 / size, rot));
        if (dot(wn0, d0) > 0.0) wn0 = -wn0;   // entry normal faces the light
        if (dot(wn1, d0) < 0.0) wn1 = -wn1;   // exit normal faces away

        // Fade the bending near the silhouette so the light field stays continuous.
        float chord = (t1 - t0) * dist;
        float w = smoothstep(0.0, 0.30 * rad, chord);

        vec3 nn = tfRefractiveIOR(m);
        vec3 pin = p0 + (p1 - p0) * t0;   // entry point in normalised space
        vec3 din0 = dirR, din1 = dirG, din2 = dirB;
        vec3 o0 = tfBendChannel(m, shape, rot, size, pin, din0, wn0, wn1, nn.x);
        vec3 o1 = tfBendChannel(m, shape, rot, size, pin, din1, wn0, wn1, nn.y);
        vec3 o2 = tfBendChannel(m, shape, rot, size, pin, din2, wn0, wn1, nn.z);
        dirR = normalize(mix(din0, o0, w));
        dirG = normalize(mix(din1, o1, w));
        dirB = normalize(mix(din2, o2, w));

        // Ball-lens focusing for water / soft bodies: f = n R / (2 (n - 1)).
        // Rays near the rim focus closer (spherical aberration), which gives the
        // bright caustic ring. The gain never goes below 1 (no darkening).
        if (m > 3.5 && shape != 0 && shape != 5)
        {
            float R = (size.x + size.y + size.z) * 0.33333;
            float f = nn.y * R / (2.0 * (nn.y - 1.0));
            float lateral = length(cross(seg / dist, center - lightPos)) / max(R, 1e-4);
            float feff = f * (1.0 - 0.35 * clamp(lateral * lateral, 0.0, 1.0));
            float z = length(point - center);
            float mag = max(abs(1.0 - z / feff), 0.38);
            gain *= mix(1.0, clamp(1.0 / (mag * mag), TF_REFRACT_MIN_GAIN, TF_REFRACT_MAX_GAIN), w);
        }

        tint *= mix(vec3(1.0), max(tfDynamicOccluderTints[i].rgb, vec3(0.05)), w);
        hit = true;
    }
    return hit;
}

vec3 tfBRDF(vec3 N, vec3 V, vec3 L, vec3 light,
            vec3 albedo, float roughness, vec3 F0)
{
    float NL = max(dot(N,L), 0.0);
    float NV = max(abs(dot(N,V)), 0.001);
    if (NL <= 0.0) return vec3(0.0);
    vec3 Hsum = V+L;
    vec3 H = dot(Hsum,Hsum) > 0.000001 ? normalize(Hsum) : N;
    float NH = max(dot(N,H), 0.0);
    float HV = max(dot(H,V), 0.0);
    vec3 F = tfF(HV, F0);
    float D = tfD(NH, roughness);
    float G = tfG1(NV, roughness) * tfG1(NL, roughness);
    vec3 spec = (D*G*F) / max(4.0*NV*NL, 0.0001);
    vec3 diff = (1.0-F) * albedo / TF_PI;
    return (diff + spec) * light * NL;
}

vec3 tfTonemap(vec3 x)
{
    const float A = 2.51;
    const float B = 0.03;
    const float C = 2.43;
    const float D = 0.59;
    const float E = 0.14;
    return clamp((x*(A*x+B))/(x*(C*x+D)+E), 0.0, 1.0);
}

vec3 tfEvaluateLowFrequencyGI(vec3 N)
{
    // Six compact irradiance lobes. This behaves like a tiny irradiance-probe
    // field: cheap per pixel, but directional enough to carry colored bounce.
    const vec3 dirs[6] = vec3[6](
        vec3( 1.0,0.0,0.0), vec3(-1.0,0.0,0.0),
        vec3( 0.0,1.0,0.0), vec3( 0.0,-1.0,0.0),
        vec3( 0.0,0.0,1.0), vec3( 0.0,0.0,-1.0));

    vec3 result = vec3(0.0);
    for (int i=0; i<6; ++i)
    {
        float w = max(dot(N, dirs[i]), 0.0);
        result += tfGIDirections[i].rgb * (w*w * 0.82 + w*0.18);
    }
    return result * tfGIIntensity;
}

vec3 tfApplyAtmosphere(vec3 sceneColor, vec3 P, vec3 viewDirection)
{
    float density = max(tfAtmosphereParams.x, 0.0);
    float heightScale = max(tfAtmosphereParams.y, 0.5);
    float rayleighWeight = max(tfAtmosphereParams.z, 0.0);
    float mieWeight = max(tfAtmosphereParams.w, 0.0);

    float distanceToCamera = length(tfCamPos() - P);
    float cameraHeight = max(tfCamPos().y, 0.0);
    float pointHeight = max(P.y, 0.0);
    float avgHeight = 0.5 * (cameraHeight + pointHeight);

    // Exponential atmosphere: denser near the ground and progressively clearer
    // at altitude. The height term is clamped to keep the effect stable for
    // ordinary TigerFlash scene coordinates.
    float heightDensity = exp(-avgHeight / heightScale);
    float opticalDepth = density * distanceToCamera * heightDensity;
    float transmittance = exp(-opticalDepth);

    vec3 ray = normalize(viewDirection);
    float sunForward = max(dot(-ray, TF_SUN_DIR), 0.0);
    float rayleighPhase = 0.75 * (1.0 + sunForward*sunForward);
    float miePhase = pow(sunForward, 24.0);

    float inscatterAmount = (1.0 - transmittance);
    vec3 skyScatter = tfAtmosphereColor.rgb *
                       inscatterAmount *
                       (0.18 * rayleighWeight * rayleighPhase +
                        tfAtmosphereColor.a * mieWeight * miePhase);

    // A cooler horizon creates aerial perspective without replacing the actual
    // surface color. Near objects are almost unchanged; distant objects blend
    // toward the atmosphere color naturally.
    float horizon = pow(max(1.0 - abs(dot(ray, vec3(0.0,1.0,0.0))), 0.0), 2.0);
    skyScatter += tfAtmosphereColor.rgb * inscatterAmount * horizon * 0.12;

    return sceneColor * transmittance + skyScatter;
}

void main()
{
    bool advancedLighting = (tfDynamicLightCount > 0.5 && tfDynamicColors[0].a > 0.5);
    vec3 N = normalize(worldNormal);
    vec3 V = normalize(tfCamPos()-worldPosition);
    if (dot(N,V) < 0.0) N = -N;

    // Troca luz <-> sombra do degrade dos objetos. O chao nao e trocado, para
    // continuar recebendo a luz e a sombra projetada. Somente o sombreamento
    // usa Ns; visibilidade e oclusao de contato usam os calculos originais.
    bool tfSwapActive = tfSwapShading > 0.5 && length(tfCamPos()) < 0.0001;
    vec3 Ns = tfSwapActive ? -N : N;
    // Sem camera conhecida o vetor de visao e invalido (causa do ponto preto
    // no brilho); usa a propria normal como visao para o brilho ficar no ponto
    // mais forte da luz.
    if (tfSwapActive) V = Ns;

    vec3 albedo = pow(clamp(colDiffuse.rgb, 0.0, 1.0), vec3(2.2));
    float lum = dot(albedo, vec3(0.2126,0.7152,0.0722));
    float roughness = clamp(0.30 + 0.18*(1.0-lum), 0.25, 0.58);
    vec3 F0 = mix(vec3(0.045), albedo, 0.06);

    vec3 envDiffuse = tfEnvLookup(Ns);
    // O reflexo (pontinho de luz) tambem e trocado nos objetos; o chao nao.
    vec3 reflectDir = reflect(-V,N);
    vec3 envSpecular = tfEnvLookup(tfSwapActive ? -reflectDir : reflectDir);

    // In `advanced light` mode we deliberately remove the fake baked fill.
    // This makes an occluded point genuinely black when no other light ray
    // reaches it, instead of hiding the shadow with ambient illumination.
    vec3 color = advancedLighting
        ? vec3(0.0)
        : envDiffuse * albedo * 0.78;
    if (!advancedLighting)
        color += envSpecular * (0.12 + 0.42*(1.0-roughness));

    float sunVisibility = tfDirectionalVisibilitySoft(worldPosition, TF_SUN_DIR);
    vec3 sunEnergy = TF_SUN_COLOR * sunVisibility;
    if (!advancedLighting)
        color += tfBRDF(Ns,V,TF_SUN_DIR,sunEnergy,albedo,roughness,F0);

    if (!advancedLighting)
    {
        // Legacy/PBR environment fill remains unchanged outside the explicit
        // hard-shadow mode.
        float upness = max(dot(Ns, vec3(0.0,1.0,0.0)), 0.0);
        float downness = max(dot(Ns, vec3(0.0,-1.0,0.0)), 0.0);
        color += albedo * vec3(0.035,0.050,0.075) * upness;
        color += albedo * vec3(0.020,0.013,0.009) * downness;
    }

    if (!advancedLighting)
    {
        // Legacy fill lights stay available outside `advanced light`. They are
        // deliberately disabled in the hard-shadow mode so they cannot wash
        // out a shadow cast by an explicit light(n) source.
        const vec3 LP0 = vec3(-7.0,9.0,4.0);
        const vec3 LP1 = vec3( 5.0,5.0,-6.0);
        const vec3 LP2 = vec3( 0.0,2.5,8.0);
        const vec3 LC0 = vec3(1.35,1.18,0.98);
        const vec3 LC1 = vec3(0.48,0.64,1.00);
        const vec3 LC2 = vec3(0.60,0.72,0.86);

        vec3 L0 = LP0-worldPosition;
        float d0 = length(L0);
        L0 = d0 > 0.0001 ? L0/d0 : vec3(0,1,0);
        color += tfBRDF(Ns,V,L0,LC0/(1.0+0.08*d0*d0),albedo,roughness,F0);

        vec3 L1 = LP1-worldPosition;
        float d1 = length(L1);
        L1 = d1 > 0.0001 ? L1/d1 : vec3(0,1,0);
        color += tfBRDF(Ns,V,L1,LC1/(1.0+0.10*d1*d1),albedo,roughness,F0);

        vec3 L2 = LP2-worldPosition;
        float d2 = length(L2);
        L2 = d2 > 0.0001 ? L2/d2 : vec3(0,1,0);
        color += tfBRDF(Ns,V,L2,LC2/(1.0+0.07*d2*d2),albedo,roughness,F0);
    }

    // Explicit point lights.  In advanced mode each emitter has a finite area,
    // so visibility is averaged over its disk: hard umbra + smooth penumbra.
    for (int i = 0; i < 4; ++i)
    {
        if (float(i) >= tfDynamicLightCount)
            break;

        vec3 lightPosition = tfDynamicLights[i].xyz;
        float lightStrength = max(tfDynamicLights[i].w, 0.0);
        vec3 lightColor = max(tfDynamicColors[i].rgb, vec3(0.0));
        float sourceRadius = max(tfDynamicLightParams[i].x, 0.025);

        vec3 toLight = lightPosition - worldPosition;
        float d = length(toLight);
        if (d <= 0.0001)
            continue;
        vec3 L = toLight / d;

        // Light intensity is arbitrary in TF units, but its propagation is
        // physically motivated: inverse-square falloff in distance.
        float luminousIntensity = 22.0 * lightStrength;
        float attenuation = 1.0 / max(d*d, 0.0025);

        float advancedMode = tfDynamicColors[i].a;

        // Refractive bodies (crystal / diamond / prism / soft body) bend the
        // light instead of blocking it. Opaque objects are still tested against
        // the bent path, so their shadows are displaced by the refraction too.
        bool refracted = false;
        vec3 dR = -L, dG = -L, dB = -L;
        float refractGain = 1.0;
        vec3 refractTint = vec3(1.0);
        vec3 shadowLight = lightPosition;
        if (advancedMode > 0.5 && tfDynamicOccluderCount > 0.5)
        {
            refracted = tfRefractLightPath(lightPosition, worldPosition,
                                           dR, dG, dB, refractGain, refractTint);
            if (refracted)
                shadowLight = worldPosition - dG * d;
        }

        float visibility = advancedMode > 0.5
            ? tfPointLightVisibilitySoft(shadowLight, worldPosition,
                                         sourceRadius, float(i)+1.0)
            : 1.0;

        vec3 directRadiance = lightColor * luminousIntensity * attenuation * visibility;
        // Direct light obeys the Cook-Torrance BRDF with energy-preserving
        // diffuse/specular partitioning rather than multiplying full source
        // power onto the surface.
        vec3 plainLight = tfBRDF(Ns,V,L,directRadiance,albedo,roughness,F0);
        if (refracted)
        {
            vec3 bentRadiance = directRadiance * refractGain * refractTint;
            vec3 bent = vec3(tfBRDF(Ns,V,-dR,bentRadiance,albedo,roughness,F0).r,
                             tfBRDF(Ns,V,-dG,bentRadiance,albedo,roughness,F0).g,
                             tfBRDF(Ns,V,-dB,bentRadiance,albedo,roughness,F0).b);
            // Distortion without darkening: keep the hue of the bent light, but
            // never let its luminance fall below the unobstructed light.
            const vec3 W = vec3(0.2126, 0.7152, 0.0722);
            float lp = dot(plainLight, W);
            float lb = dot(bent, W);
            if (lb > 1e-5) bent *= max(1.0, lp / lb);
            color += bent;
        }
        else
        {
            color += plainLight;
        }

        // One-bounce colored diffuse interreflection.  A white lamp hitting a
        // blue floor therefore returns weak blue light, not full-strength white.
        if (advancedMode > 0.5 && tfGroundValid > 0.5)
        {
            float receiverToGround = worldPosition.y - tfGroundY;
            if (receiverToGround > 0.03 && receiverToGround < 6.0)
            {
                vec3 bounce = tfGroundBouncePhysical(
                    Ns, worldPosition, lightPosition, luminousIntensity,
                    sourceRadius, lightColor, tfGroundY,
                    clamp(tfGroundBounceColor.rgb, vec3(0.0), vec3(0.92)));
                color += bounce;
            }
        }
    }

    if (advancedLighting)
    {
        // Near-field contact shadows remain visible even where the light source
        // itself is partially visible. This is intentionally weaker than the
        // binary direct occlusion so creases are dark without crushing the image.
        color *= tfContactOcclusion(worldPosition, N);

        // Very small residual sky bounce keeps upward-facing surfaces from
        // looking numerically broken while the direct-light shadow core can
        // still approach black. It is not white lamp spill.
        vec3 skyResidual = tfEnvLookup(Ns) * albedo * 0.010;
        color += skyResidual;

        // Global low-frequency indirect light. It is deliberately separated
        // from direct shadows, so GI fills bounced regions without erasing
        // the shadow core.
        color += albedo * tfEvaluateLowFrequencyGI(Ns);
    }

    float rim = pow(1.0-max(dot(N,V),0.0), 4.0);
    if (!advancedLighting)
        color += envSpecular * rim * 0.22;

    // `light(n)` is emissive: the object's own base color becomes the light
    // energy. This stays in the same single surface pass, so it does not add
    // another mesh draw. The rim term makes the glow visually stronger at
    // grazing angles, similar to a bright emissive material.
    float lightStrength = clamp(tfLightStrength, 0.0, 10.0);
    if (lightStrength > 0.001)
    {
        // Source energy is linear in the declared light strength. The propagation
        // to receivers above remains inverse-square; no artificial energy is
        // added to other objects just to keep the lamp visually obvious.
        const float TF_SOURCE_EMISSION_SCALE = 1.35;
        float emissionEnergy = TF_SOURCE_EMISSION_SCALE * lightStrength;

        // Degrade do emissor trocado: o centro (voltado para a camera) e o
        // mais claro e a borda mais suave, em vez do contrario.
        vec3 emission = albedo * emissionEnergy;
        float centerGlow = pow(max(dot(N,V),0.0), 2.0);
        emission += albedo * emissionEnergy * (0.35 + 0.55 * centerGlow);

        // Small source aura, still proportional to source power. This is a
        // visibility aid, not scene illumination.
        float normalized = clamp(lightStrength / 10.0, 0.0, 1.0);
        float aura = smoothstep(0.0, 1.0, normalized) * (0.08 + 0.28 * normalized);
        emission += albedo * aura;
        color = emission;

        // Perceptual source floor: if exposure/tonemapping would otherwise make
        // the actual light object disappear, lift ONLY this emitter to a tiny
        // positive luminance. Shadows and receiver lighting are untouched.
        const vec3 W = vec3(0.2126, 0.7152, 0.0722);
        float sourceLuma = dot(max(color * max(tfExposure, 0.05), vec3(0.0)), W);
        if (sourceLuma < TF_SOURCE_MIN_LINEAR_LUMA)
        {
            // The floor is evaluated AFTER exposure so a dark-adapted eye can
            // still see a valid emitter. The correction is applied ONLY to the
            // emissive object; receiver illumination remains untouched.
            float deficit = TF_SOURCE_MIN_LINEAR_LUMA - sourceLuma;
            float lift = min(deficit, TF_SOURCE_MAX_LIFT * (0.35 + 0.65 * normalized));
            color += albedo * lift / max(dot(albedo, W) * max(tfExposure, 0.05), 0.05);
        }
    }

    // HDR-safe lighting pipeline: all direct/indirect contributions above can
    // exceed 1.0. Exposure adapts the camera response before the final filmic
    // compression. This preserves bright sources while retaining shadow range.
    color = tfApplyAtmosphere(color, worldPosition, -V);
    color *= clamp(tfExposure, 0.30, 4.00);
    color = tfTonemap(max(color,vec3(0.0)));
    color = pow(color, vec3(1.0/2.2));
    finalColor = vec4(color, colDiffuse.a);
}
)GLSL";
    return glsl.str();
}

static unsigned int tfCompileShaderStage(unsigned int type,
                                         const std::string &source,
                                         std::string &errorText)
{
    unsigned int shader = tfGLCreateShader(type);
    if (!shader)
    {
        errorText = "glCreateShader returned 0";
        return 0;
    }

    const char *sourcePtr = source.c_str();
    tfGLShaderSource(shader, 1, &sourcePtr, nullptr);
    tfGLCompileShader(shader);

    int compiled = 0;
    tfGLGetShaderiv(shader, TF_GL_COMPILE_STATUS, &compiled);
    if (!compiled)
    {
        errorText = tfGetGLInfoLog(true, shader);
        tfGLDeleteShader(shader);
        return 0;
    }
    return shader;
}

static bool tfBuildAdvancedSurfaceShader()
{
    // The host clears tfSurfaceShader/tfSurfaceShaderReady when a graphics
    // scene ends. Keep the bridge installed, but rebuild the GPU program on
    // the next scene instead of reusing a deleted program id.
    if (gTfAdvancedShaderBuilt && gTfSurfaceShaderAddress &&
        gTfSurfaceShaderReadyAddress)
    {
        TFCompatShader current{};
        bool ready = false;
        std::memcpy(&current, gTfSurfaceShaderAddress, sizeof(current));
        std::memcpy(&ready, gTfSurfaceShaderReadyAddress, sizeof(ready));
        if (ready && current.id != 0 && current.locs != nullptr)
            return true;

        gTfAdvancedShaderBuilt = false;
        gTfAdvancedProgram = 0;
        gTfLightStrengthLocation = -1;
        gTfDynamicLightCountLocation = -1;
        gTfDynamicLightsLocation = -1;
        gTfDynamicColorsLocation = -1;
        gTfDynamicLightParamsLocation = -1;
        gTfDynamicOccluderCountLocation = -1;
        gTfDynamicOccludersLocation = -1;
        gTfDynamicOccluderSizesLocation = -1;
        gTfDynamicOccluderRotationsLocation = -1;
        gTfDynamicOccluderSoftALocation = -1;
        gTfDynamicOccluderSoftBLocation = -1;
        gTfDynamicOccluderSoftCLocation = -1;
        gTfDynamicOccluderSoftDLocation = -1;
        gTfDynamicOccluderTintsLocation = -1;
        gTfGroundYLocation = -1;
        gTfGroundValidLocation = -1;
        gTfGroundColorLocation = -1;
        gTfExposureLocation = -1;
        gTfGIIntensityLocation = -1;
        gTfGIDirectionsLocation = -1;
        gTfAtmosphereParamsLocation = -1;
        gTfAtmosphereColorLocation = -1;
        gTfViewPosLocation = -1;
        gTfSwapShadingLocation = -1;
        gTfGlobalLightingUploadedSerial = ~static_cast<uint64_t>(0);
        gTfAdvancedShaderFailed = false;
        gTfAdvancedShaderError.clear();
    }

    if (gTfAdvancedShaderFailed)
        return false;

    if (!tfLoadGLFunctions())
    {
        gTfAdvancedShaderFailed = true;
        gTfAdvancedShaderError = "OpenGL shader entry points unavailable.";
        return false;
    }

    std::string errorText;
    const unsigned int vs = tfCompileShaderStage(
        TF_GL_VERTEX_SHADER, tfBuildAdvancedVertexShader(), errorText);
    if (!vs)
    {
        gTfAdvancedShaderFailed = true;
        gTfAdvancedShaderError = "vertex shader: " + errorText;
        return false;
    }

    const unsigned int fs = tfCompileShaderStage(
        TF_GL_FRAGMENT_SHADER, tfBuildAdvancedFragmentShader(), errorText);
    if (!fs)
    {
        tfGLDeleteShader(vs);
        gTfAdvancedShaderFailed = true;
        gTfAdvancedShaderError = "fragment shader: " + errorText;
        return false;
    }

    const unsigned int program = tfGLCreateProgram();
    if (!program)
    {
        tfGLDeleteShader(vs);
        tfGLDeleteShader(fs);
        gTfAdvancedShaderFailed = true;
        gTfAdvancedShaderError = "glCreateProgram returned 0.";
        return false;
    }

    // raylib's standard mesh attributes: position=0, normal=2.
    tfGLBindAttribLocation(program, 0, "vertexPosition");
    tfGLBindAttribLocation(program, 2, "vertexNormal");
    tfGLAttachShader(program, vs);
    tfGLAttachShader(program, fs);
    tfGLLinkProgram(program);

    int linked = 0;
    tfGLGetProgramiv(program, TF_GL_LINK_STATUS, &linked);
    tfGLDeleteShader(vs);
    tfGLDeleteShader(fs);

    if (!linked)
    {
        gTfAdvancedShaderError = tfGetGLInfoLog(false, program);
        tfGLDeleteProgram(program);
        gTfAdvancedShaderFailed = true;
        return false;
    }

    int *locs = static_cast<int *>(
        std::malloc(TF_RL_MAX_SHADER_LOCATIONS * sizeof(int)));
    if (!locs)
    {
        tfGLDeleteProgram(program);
        gTfAdvancedShaderFailed = true;
        gTfAdvancedShaderError = "shader location allocation failed.";
        return false;
    }

    for (unsigned int i = 0; i < TF_RL_MAX_SHADER_LOCATIONS; ++i)
        locs[i] = -1;

    // raylib ShaderLocationIndex: position 0, normal 3, MVP 6, model 9,
    // normal-matrix 10, view-vector 11, diffuse color 12.
    locs[0] = 0;
    locs[3] = 2;
    locs[6] = tfGLGetUniformLocation(program, "mvp");
    locs[7] = tfGLGetUniformLocation(program, "matView");
    locs[9] = tfGLGetUniformLocation(program, "matModel");
    locs[10] = tfGLGetUniformLocation(program, "matNormal");
    locs[11] = tfGLGetUniformLocation(program, "viewPos");
    locs[12] = tfGLGetUniformLocation(program, "colDiffuse");
    gTfLightStrengthLocation = tfGLGetUniformLocation(program, "tfLightStrength");
    gTfDynamicLightCountLocation = tfGLGetUniformLocation(program, "tfDynamicLightCount");
    gTfDynamicLightsLocation = tfGLGetUniformLocation(program, "tfDynamicLights[0]");
    gTfDynamicColorsLocation = tfGLGetUniformLocation(program, "tfDynamicColors[0]");
    gTfDynamicLightParamsLocation = tfGLGetUniformLocation(program, "tfDynamicLightParams[0]");
    gTfDynamicOccluderCountLocation = tfGLGetUniformLocation(program, "tfDynamicOccluderCount");
    gTfDynamicOccludersLocation = tfGLGetUniformLocation(program, "tfDynamicOccluders[0]");
    gTfDynamicOccluderSizesLocation = tfGLGetUniformLocation(program, "tfDynamicOccluderSizes[0]");
    gTfDynamicOccluderRotationsLocation = tfGLGetUniformLocation(program, "tfDynamicOccluderRotations[0]");
    gTfDynamicOccluderSoftALocation = tfGLGetUniformLocation(program, "tfDynamicOccluderSoftA[0]");
    gTfDynamicOccluderSoftBLocation = tfGLGetUniformLocation(program, "tfDynamicOccluderSoftB[0]");
    gTfDynamicOccluderSoftCLocation = tfGLGetUniformLocation(program, "tfDynamicOccluderSoftC[0]");
    gTfDynamicOccluderSoftDLocation = tfGLGetUniformLocation(program, "tfDynamicOccluderSoftD[0]");
    gTfDynamicOccluderTintsLocation = tfGLGetUniformLocation(program, "tfDynamicOccluderTints[0]");
    gTfGroundYLocation = tfGLGetUniformLocation(program, "tfGroundY");
    gTfGroundValidLocation = tfGLGetUniformLocation(program, "tfGroundValid");
    gTfGroundColorLocation = tfGLGetUniformLocation(program, "tfGroundBounceColor");
    gTfExposureLocation = tfGLGetUniformLocation(program, "tfExposure");
    gTfGIIntensityLocation = tfGLGetUniformLocation(program, "tfGIIntensity");
    gTfGIDirectionsLocation = tfGLGetUniformLocation(program, "tfGIDirections[0]");
    gTfAtmosphereParamsLocation = tfGLGetUniformLocation(program, "tfAtmosphereParams");
    gTfAtmosphereColorLocation = tfGLGetUniformLocation(program, "tfAtmosphereColor");
    gTfViewPosLocation = locs[11];
    gTfSwapShadingLocation = tfGLGetUniformLocation(program, "tfSwapShading");

    if (locs[6] < 0 || locs[9] < 0 || locs[10] < 0 ||
        locs[11] < 0 || locs[12] < 0 || gTfLightStrengthLocation < 0 ||
        gTfDynamicLightCountLocation < 0 || gTfDynamicLightsLocation < 0 ||
        gTfDynamicColorsLocation < 0 || gTfDynamicLightParamsLocation < 0 ||
        gTfDynamicOccluderCountLocation < 0 ||
        gTfDynamicOccludersLocation < 0 || gTfDynamicOccluderSizesLocation < 0 ||
        gTfDynamicOccluderRotationsLocation < 0 || gTfGroundYLocation < 0 ||
        gTfGroundValidLocation < 0 || gTfGroundColorLocation < 0 ||
        !gTfSurfaceShaderAddress || !gTfSurfaceShaderReadyAddress)
    {
        std::free(locs);
        tfGLDeleteProgram(program);
        gTfAdvancedShaderFailed = true;
        gTfAdvancedShaderError = "required renderer locations were unavailable.";
        return false;
    }

    TFCompatShader shader{program, locs};
    std::memcpy(gTfSurfaceShaderAddress, &shader, sizeof(shader));
    const bool ready = true;
    std::memcpy(gTfSurfaceShaderReadyAddress, &ready, sizeof(ready));

    gTfAdvancedProgram = program;
    gTfAdvancedShaderBuilt = true;
    return true;
}

extern "C" void tfShadersAdvancedPrepareSurfaceShader()
{
    tfBuildAdvancedSurfaceShader();
    if (!gTfAdvancedShaderBuilt && gApi.log)
    {
        const std::string message =
            "shaders: advanced lighting fallback: " + gTfAdvancedShaderError;
        gApi.log(message.c_str());
    }
}


static std::string trim(const std::string &text);
static bool isWordBoundary(const std::string &text, size_t pos, size_t length);

static bool tfParseLightPrefix(const std::string &line,
                               float &strengthOut,
                               std::string &rewrittenOut,
                               std::string &objectNameOut)
{
    std::string text = trim(line);
    if (text.rfind("say 3d ", 0) != 0)
        return false;

    std::string rest = trim(text.substr(7));
    // `advanced light` is deliberately NOT part of `say 3d`.
    // It is a standalone global command handled by shadersCommandHook().
    if (rest.rfind("light", 0) != 0)
        return false;

    const size_t lightLen = 5;
    if (!isWordBoundary(rest, 0, lightLen))
        return false;

    size_t cursor = lightLen;
    while (cursor < rest.size() &&
           std::isspace(static_cast<unsigned char>(rest[cursor])))
        ++cursor;

    if (cursor >= rest.size() || rest[cursor] != '(')
        return false;

    const size_t closeParen = rest.find(')', cursor + 1);
    if (closeParen == std::string::npos)
        return false;

    std::string numberText = trim(rest.substr(cursor + 1,
                                              closeParen - cursor - 1));
    if (numberText.empty())
        return false;

    char *end = nullptr;
    const float parsed = std::strtof(numberText.c_str(), &end);
    if (!end || end == numberText.c_str())
        return false;
    while (*end)
    {
        if (!std::isspace(static_cast<unsigned char>(*end)))
            return false;
        ++end;
    }

    strengthOut = std::max(0.0f, std::min(10.0f, parsed));
    rest = trim(rest.substr(closeParen + 1));

    // The normal core parser expects the object format/name in exactly this
    // position, e.g. `"cube" as "player" ...`.
    if (rest.size() < 2 || rest.front() != '"')
        return false;

    const size_t closeObjectQuote = rest.find('"', 1);
    if (closeObjectQuote == std::string::npos)
        return false;

    const size_t asPos = rest.find(" as ", closeObjectQuote + 1);
    if (asPos == std::string::npos)
        return false;

    size_t nameOpen = asPos + 4;
    while (nameOpen < rest.size() &&
           std::isspace(static_cast<unsigned char>(rest[nameOpen])))
        ++nameOpen;
    if (nameOpen >= rest.size() || rest[nameOpen] != '"')
        return false;
    const size_t nameClose = rest.find('"', nameOpen + 1);
    if (nameClose == std::string::npos || nameClose == nameOpen + 1)
        return false;

    objectNameOut = rest.substr(nameOpen + 1, nameClose - nameOpen - 1);
    rewrittenOut = "say 3d " + rest;
    return !objectNameOut.empty();
}


static uintptr_t tfPageStart(uintptr_t address)
{
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0)
        return 0;
    return address & ~static_cast<uintptr_t>(pageSize - 1);
}

static bool tfMakeCodeWritable(void *address, size_t size)
{
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0 || !address || size == 0)
        return false;

    const uintptr_t start = tfPageStart(reinterpret_cast<uintptr_t>(address));
    const uintptr_t end = reinterpret_cast<uintptr_t>(address) + size;
    const uintptr_t endPage =
        (end - 1) & ~static_cast<uintptr_t>(pageSize - 1);
    const size_t total =
        static_cast<size_t>(endPage - start + static_cast<uintptr_t>(pageSize));

    return mprotect(reinterpret_cast<void *>(start), total,
                    PROT_READ | PROT_WRITE | PROT_EXEC) == 0;
}

static size_t tfX86InstructionLength(const unsigned char *code,
                                     bool &safeToRelocate)
{
    safeToRelocate = false;
    if (!code)
        return 0;

    const unsigned char *p = code;
    size_t length = 0;

    // REX / legacy prefixes.
    bool operand16 = false;
    bool address16 = false;
    while (true)
    {
        const unsigned char b = *p;
        if (b == 0x66) { operand16 = true; ++p; ++length; continue; }
        if (b == 0x67) { address16 = true; ++p; ++length; continue; }
        if (b == 0xF0 || b == 0xF2 || b == 0xF3 ||
            (b >= 0x2E && b <= 0x3E && (b & 0x07) == 0x06))
        { ++p; ++length; continue; }
        if (b >= 0x40 && b <= 0x4F)
        { ++p; ++length; continue; }
        break;
    }

    const unsigned char op1 = *p++;
    ++length;

    // CET landing pad.
    if (op1 == 0x0F && p[0] == 0x1E && p[1] == 0xFA)
    {
        safeToRelocate = true;
        return length + 3;
    }

    // Direct relative control-flow in a stolen region cannot simply be copied.
    if (op1 == 0xE8 || op1 == 0xE9 || op1 == 0xEB ||
        (op1 >= 0x70 && op1 <= 0x7F))
        return 0;

    if (op1 >= 0x50 && op1 <= 0x5F)
    { safeToRelocate = true; return length; }
    if (op1 == 0x90 || op1 == 0xC3 || op1 == 0xCB)
    { safeToRelocate = true; return length; }
    if (op1 == 0x68)
    { safeToRelocate = true; return length + 4; }
    if (op1 == 0x6A)
    { safeToRelocate = true; return length + 1; }
    if (op1 >= 0xB8 && op1 <= 0xBF)
    {
        // REX.W is represented in the consumed prefix bytes.
        bool rexW = false;
        for (const unsigned char *q = code; q < p - 1; ++q)
            if (*q >= 0x48 && *q <= 0x4F) rexW = true;
        safeToRelocate = true;
        return length + (rexW ? 8 : (operand16 ? 2 : 4));
    }

    unsigned char op2 = 0;
    if (op1 == 0x0F)
    {
        op2 = *p++; ++length;
        if (op2 >= 0x80 && op2 <= 0x8F)
            return 0; // near Jcc is RIP-relative in the trampoline.
    }

    // Common opcodes with no ModRM.
    if (op1 == 0x04 || op1 == 0x05 || op1 == 0x0C || op1 == 0x0D ||
        op1 == 0x14 || op1 == 0x15 || op1 == 0x1C || op1 == 0x1D ||
        op1 == 0x24 || op1 == 0x25 || op1 == 0x2C || op1 == 0x2D ||
        op1 == 0x34 || op1 == 0x35 || op1 == 0x3C || op1 == 0x3D ||
        op1 == 0xA8 || op1 == 0xA9)
    {
        const size_t imm = (op1 == 0xA8 || (operand16 && op1 != 0xA9)) ? 1 :
                           (operand16 ? 2 : 4);
        safeToRelocate = true;
        return length + imm;
    }

    if (op1 == 0xC2 || op1 == 0xCA)
    { safeToRelocate = true; return length + 2; }

    // Most arithmetic/move/test/LEA/shift forms use ModRM.
    bool hasModRM = false;
    if ((op1 <= 0x03) || (op1 >= 0x08 && op1 <= 0x0B) ||
        (op1 >= 0x10 && op1 <= 0x13) || (op1 >= 0x18 && op1 <= 0x1B) ||
        (op1 >= 0x20 && op1 <= 0x23) || (op1 >= 0x28 && op1 <= 0x2B) ||
        (op1 >= 0x30 && op1 <= 0x33) || (op1 >= 0x38 && op1 <= 0x3B) ||
        op1 == 0x62 || op1 == 0x63 || op1 == 0x69 || op1 == 0x6B ||
        op1 == 0x80 || op1 == 0x81 || op1 == 0x82 || op1 == 0x83 ||
        op1 == 0x84 || op1 == 0x85 || op1 == 0x86 || op1 == 0x87 ||
        op1 == 0x88 || op1 == 0x89 || op1 == 0x8A || op1 == 0x8B ||
        op1 == 0x8C || op1 == 0x8D || op1 == 0x8E ||
        op1 == 0xC0 || op1 == 0xC1 || op1 == 0xC4 || op1 == 0xC5 ||
        op1 == 0xC6 || op1 == 0xC7 || op1 == 0xD0 || op1 == 0xD1 ||
        op1 == 0xD2 || op1 == 0xD3 || op1 == 0xF6 || op1 == 0xF7 ||
        op1 == 0xFE || op1 == 0xFF)
        hasModRM = true;

    if (op1 == 0x0F)
        hasModRM = true;

    if (!hasModRM)
        return 0;

    const unsigned char modrm = *p++;
    ++length;
    const unsigned char mod = modrm >> 6;
    const unsigned char rm = modrm & 0x07;

    if (!address16 && mod == 0 && rm == 4)
    {
        const unsigned char sib = *p++;
        ++length;
        if ((sib & 0x07) == 5)
            return 0; // RIP-relative via SIB base.
    }

    if (!address16 && mod == 0 && rm == 5)
        return 0; // RIP-relative displacement.

    if (mod == 1)
        length += 1;
    else if (mod == 2 || (mod == 0 && address16 && rm == 6))
        length += address16 ? 2 : 4;

    const bool byteImm =
        op1 == 0x6B || op1 == 0x80 || op1 == 0x82 || op1 == 0x83 ||
        op1 == 0xC0 || op1 == 0xC1 || op1 == 0xC6;
    const bool wordImm = operand16 && !byteImm;

    if (op1 == 0x69 || op1 == 0x81 || op1 == 0xC7)
        length += wordImm ? 2 : 4;
    else if (op1 == 0x6B || op1 == 0x83 || op1 == 0x80 || op1 == 0x82 ||
             op1 == 0xC0 || op1 == 0xC1 || op1 == 0xC6)
        length += 1;

    safeToRelocate = true;
    return length;
}

static bool tfInstallTrampolineHook(void *target,
                                     void *replacement,
                                     void **trampolineOut)
{
    if (!target || !replacement || !trampolineOut)
        return false;

    constexpr size_t jumpSize = 12;
    unsigned char *src = static_cast<unsigned char *>(target);
    size_t stolen = 0;

    while (stolen < jumpSize)
    {
        bool safe = false;
        const size_t len = tfX86InstructionLength(src + stolen, safe);
        if (len == 0 || !safe || stolen + len > 64)
            return false;
        stolen += len;
    }

    if (stolen > 48)
        return false;

    const size_t trampolineSize = stolen + jumpSize;
    void *trampoline = mmap(nullptr, trampolineSize,
                            PROT_READ | PROT_WRITE | PROT_EXEC,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (trampoline == MAP_FAILED)
        return false;

    std::memcpy(trampoline, src, stolen);

    unsigned char backJump[jumpSize] =
    {
        0x48, 0xB8, 0,0,0,0,0,0,0,0, 0xFF, 0xE0
    };
    const uintptr_t back = reinterpret_cast<uintptr_t>(target) + stolen;
    std::memcpy(backJump + 2, &back, sizeof(back));
    std::memcpy(static_cast<unsigned char *>(trampoline) + stolen,
                backJump, jumpSize);

    if (!tfMakeCodeWritable(target, stolen))
    {
        munmap(trampoline, trampolineSize);
        return false;
    }

    unsigned char jump[jumpSize] =
    {
        0x48, 0xB8, 0,0,0,0,0,0,0,0, 0xFF, 0xE0
    };
    const uintptr_t dst = reinterpret_cast<uintptr_t>(replacement);
    std::memcpy(jump + 2, &dst, sizeof(dst));
    std::memcpy(target, jump, jumpSize);
    for (size_t i = jumpSize; i < stolen; ++i)
        src[i] = 0x90;

    __builtin___clear_cache(reinterpret_cast<char *>(target),
                            reinterpret_cast<char *>(target) + stolen);
    __builtin___clear_cache(reinterpret_cast<char *>(trampoline),
                            reinterpret_cast<char *>(trampoline) + trampolineSize);

    // Restore RX on all pages covered by the stolen bytes.
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize > 0)
    {
        const uintptr_t start = tfPageStart(reinterpret_cast<uintptr_t>(target));
        const uintptr_t end = reinterpret_cast<uintptr_t>(target) + stolen;
        const uintptr_t endPage = (end - 1) & ~static_cast<uintptr_t>(pageSize - 1);
        const size_t total = static_cast<size_t>(endPage - start + static_cast<uintptr_t>(pageSize));
        mprotect(reinterpret_cast<void *>(start), total, PROT_READ | PROT_EXEC);
    }

    *trampolineOut = trampoline;
    return true;
}

static bool tfSetObjectLightUniform(float strength)
{
    if (!tfGLUniform1f || !tfGLUseProgram || !tfGLGetIntegerv ||
        gTfAdvancedProgram == 0 || gTfLightStrengthLocation < 0)
        return false;

    int previous = 0;
    tfGLGetIntegerv(TF_GL_CURRENT_PROGRAM, &previous);
    tfGLUseProgram(gTfAdvancedProgram);
    tfGLUniform1f(gTfLightStrengthLocation, std::max(0.0f, std::min(10.0f, strength)));
    tfGLUseProgram(static_cast<unsigned int>(std::max(0, previous)));
    return true;
}

struct TFCompatObject3D
{
    std::string name;
    std::string shape;
    std::string color;
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float scaleX = 1.0f, scaleY = 1.0f, scaleZ = 1.0f;
    float rotationX = 0.0f, rotationY = 0.0f, rotationZ = 0.0f;
    float motionBlur = 0.0f;
};

// This matches the host TFObject3D prefix used by the current IDE. Only the
// fields listed here are read; no raylib or interpreter headers are required.
struct TFCompatObject3DView
{
    std::string name;
    std::string shape;
    std::string color;
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float scaleX = 1.0f, scaleY = 1.0f, scaleZ = 1.0f;
    float rotationX = 0.0f, rotationY = 0.0f, rotationZ = 0.0f;
};

// Full mirror of the host TFObject3D layout, including the fields that come
// after the transform. We use it only to read the live objects3D vector from
// the host executable once per frame; no host source changes are required.
struct TFCompatHostObject3D
{
    std::string name;
    std::string shape;
    std::string color;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    float scaleZ = 1.0f;
    float rotationX = 0.0f;
    float rotationY = 0.0f;
    float rotationZ = 0.0f;
    float motionBlur = 0.0f;
    float motionBlurPreviousX = 0.0f;
    float motionBlurPreviousY = 0.0f;
    float motionBlurPreviousZ = 0.0f;
    bool motionBlurPreviousInitialized = false;
};

static void tfDrawRealMotionBlur(const TFCompatHostObject3D &object,
                                 const void *cameraPtr,
                                 int screenWidth,
                                 int screenHeight,
                                 float alpha);

static void *gTfObjects3DAddress = nullptr;
static void *gTfGravityBodiesAddress = nullptr;

// Exact mirror of the host TFGravityBody layout. Keeping the layout identical
// lets the shader bridge read the authoritative deformation state after the
// host physics step without changing the TigerFlash language ABI.
struct TFCompatGravityBody
{
    int type = 0;
    float gravity = 9.80665f;
    struct { float x, y, z; } velocity{};
    struct { float x, y, z; } previousPosition{};
    bool initialized = false;
    float compression = 0.0f;
    float compressionVelocity = 0.0f;
    float stretch = 0.0f;
    float stretchVelocity = 0.0f;
    float bendX = 0.0f;
    float bendXVelocity = 0.0f;
    float bendZ = 0.0f;
    float bendZVelocity = 0.0f;
    float flowX = 0.0f;
    float flowZ = 0.0f;
    float flowXVelocity = 0.0f;
    float flowZVelocity = 0.0f;
    float impactPulse = 0.0f;
    float impactPulseVelocity = 0.0f;
    float liquidSpread = 0.0f;
    float liquidSpreadVelocity = 0.0f;
    std::string contactCollider;
    struct { float x, y, z; } contactNormal{};
    struct { float x, y, z; } contactPoint{};
    float contactStrength = 0.0f;
    float contactStrengthVelocity = 0.0f;
    float contactAge = 100.0f;
    float softness = 10.0f;
    float sweepStartY = 0.0f;
    bool sweepStartInitialized = false;
    float startupHoldRemaining = 0.0f;
    struct { float x, y, z; } startupHoldPosition{};
    bool inputMovementResolvedThisFrame = false;
    bool inputMovementActiveThisFrame = false;
    float pressure = 0.0f;
    float pressureVelocity = 0.0f;
    float volumeError = 0.0f;
    float volumeErrorVelocity = 0.0f;
    float surfaceTension = 0.0f;
    float fluidWaveTime = 0.0f;
    float fluidWaveVelocity = 0.0f;
};

using TFCompatGravityMap = std::unordered_map<std::string, TFCompatGravityBody>;

static std::array<float, 3> tfColorLinearFromName(const std::string &name)
{
    struct NamedColor { const char *name; unsigned char r, g, b; };
    static const NamedColor colors[] =
    {
        {"red",255,0,0}, {"pink",255,105,180}, {"baby_pink",255,182,193},
        {"salmon",250,128,114}, {"wine",114,47,55}, {"burgundy",128,0,32},
        {"carmine",150,0,24}, {"blood",139,0,0}, {"yellow",255,255,0},
        {"cream",255,253,208}, {"ivory",255,255,240}, {"mustard",255,219,88},
        {"amber",255,191,0}, {"blue",0,121,241}, {"sky_blue",135,206,235},
        {"baby_blue",137,207,240}, {"pool_blue",0,191,255}, {"navy",0,0,128},
        {"royal_blue",65,105,225}, {"orange",255,165,0}, {"peach",255,218,185},
        {"terracotta",226,114,91}, {"brick",156,64,55}, {"bronze",205,127,50},
        {"green",0,200,0}, {"mint",152,255,152}, {"lime",50,205,50},
        {"light_green",144,238,144}, {"moss",138,154,91}, {"emerald",80,200,120},
        {"forest_green",34,139,34}, {"purple",128,0,128}, {"violet",148,0,211},
        {"lilac",200,162,200}, {"lavender",230,230,250}, {"mauve",224,176,255},
        {"plum",142,69,133}, {"coral",255,127,80}, {"rust",183,65,14},
        {"mahogany",192,64,0}, {"carrot",237,145,33}, {"olive",128,128,0},
        {"turquoise",64,224,208}, {"cyan",0,255,255}, {"aqua_green",127,255,212},
        {"petroleum_blue",0,78,84}, {"indigo",75,0,130}, {"magenta",255,0,255},
        {"dark_burgundy",70,0,20}, {"hot_pink",255,20,147}, {"cherry",222,49,99},
        {"berry",153,37,84}, {"grape",111,45,168}, {"white",255,255,255},
        {"gray",128,128,128}, {"grey",128,128,128}, {"black",0,0,0},
        {"brown",139,69,19}, {"dark_brown",92,51,23}, {"beige",245,245,220},
        {"gold",255,215,0}, {"silver",192,192,192}, {"goldenrod",218,165,32},
        {"teal",0,128,128}, {"cyan_blue",0,180,220}
    };

    unsigned int r = 255, g = 255, b = 255;
    for (const NamedColor &c : colors)
    {
        if (name == c.name)
        {
            r = c.r; g = c.g; b = c.b;
            break;
        }
    }
    const auto linear = [](float v)
    { return std::pow(std::max(0.0f, v / 255.0f), 2.2f); };
    return {linear(static_cast<float>(r)), linear(static_cast<float>(g)), linear(static_cast<float>(b))};
}

static float tfApproxObjectRadius(const TFCompatObject3DView &obj)
{
    const float sx = std::fabs(obj.scaleX);
    const float sy = std::fabs(obj.scaleY);
    const float sz = std::fabs(obj.scaleZ);
    const float longest = std::max(sx, std::max(sy, sz));

    if (obj.shape == "sphere" || obj.shape == "capsule")
        return std::max(0.08f, 1.20f * longest);
    if (obj.shape == "cone" || obj.shape == "triangle" || obj.shape == "pyramid")
        return std::max(0.08f, 1.55f * longest);
    if (obj.shape == "cylinder" || obj.shape == "rock" || obj.shape == "crystal" || obj.shape == "diamond" || obj.shape == "prism")
        return std::max(0.08f, 1.40f * longest);
    return std::max(0.08f, 1.20f * std::sqrt(sx*sx + sy*sy + sz*sz));
}

// -----------------------------------------------------------------------------
// PHYSICALLY-BASED CRYSTAL / PRISM / DIAMOND OPTICS
// -----------------------------------------------------------------------------
// The IDE already owns the optical render pass and its baked environment
// texture. This bridge replaces only the optical GLSL program at runtime.
//
// The optical model uses:
//   * exact dielectric Fresnel equations (Rs/Rp average),
//   * Snell refraction per RGB wavelength,
//   * wavelength-dependent IOR (Cauchy-style dispersion presets),
//   * Beer-Lambert absorption through an estimated optical thickness,
//   * total internal reflection when sin(theta_t) > 1,
//   * geometry-derived normals for soft-body deformation,
//   * and the IDE's existing environment texture as incoming radiance.
//
// Soft-body deformation does NOT artificially change the material IOR. In real
// physics, deforming the crystal changes its surface normals/path length, not
// its refractive index. The existing `softFluid` field therefore only changes
// the animated geometry, while crystal/diamond/prism keep their material IOR.

struct TFCompatShaderPublic
{
    unsigned int id = 0;
    int *locs = nullptr;
};

static void tfSetOpticalParamsPhysicalProxy(const void *objectPtr,
                                            const void *cameraPtr);

static void logMessage(const char *message);

static bool tfIsCrystalOpticalShape(const std::string &shape)
{
    return shape == "crystal" || shape == "diamond" || shape == "prism";
}

static const char *tfBuildPhysicalOpticalVertexShader()
{
    return R"GLSL(#version 330
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;

uniform mat4 mvp;
uniform mat4 matModel;
uniform mat4 matNormal;
uniform float fluidTime;
uniform float softFluid;

out vec3 fragPosition;
out vec3 fragNormal;

vec3 animateSoftVertex(vec3 p, vec3 n)
{
    float fluid = clamp(softFluid, 0.0, 1.0);
    if (fluid <= 0.001) return p;

    float radial = length(p.xz);
    float w0 = sin(radial * (9.0 + 12.0 * fluid) - fluidTime * (5.5 + 9.5 * fluid));
    float w1 = sin(p.x * 7.0 + p.z * 5.0 + fluidTime * (2.5 + 4.0 * fluid));
    float w2 = sin((p.x - p.z) * 10.0 - fluidTime * (3.0 + 5.0 * fluid));

    float upper = smoothstep(-0.55, 0.85, p.y);
    float rim = smoothstep(0.15, 1.45, radial);
    float mask = (0.34 + 0.66 * upper) * (0.30 + 0.70 * rim);

    float amplitude = (0.012 + 0.040 * fluid) * mask;
    p.y += (w0 * 0.54 + w1 * 0.28 + w2 * 0.18) * amplitude;

    float sideways = (0.004 + 0.012 * fluid) * (0.35 + 0.65 * upper);
    p.x += sin(p.y * 5.0 + fluidTime * (2.0 + 4.0 * fluid)) * sideways;
    p.z += cos(p.y * 4.0 - fluidTime * (2.4 + 3.5 * fluid)) * sideways * 0.82;

    return p;
}

void main()
{
    vec3 localPosition = animateSoftVertex(vertexPosition, vertexNormal);
    vec4 wp = matModel * vec4(localPosition, 1.0);
    fragPosition = wp.xyz;

    // Keep the host mesh normal as a stable base. The fragment stage blends it
    // with the actual geometric derivative normal, so deformation changes the
    // optical surface without changing the material IOR.
    fragNormal = normalize((matNormal * vec4(vertexNormal, 0.0)).xyz);
    gl_Position = mvp * vec4(localPosition, 1.0);
}
)GLSL";
}

static const char *tfBuildPhysicalOpticalFragmentShader()
{
    return R"GLSL(#version 330
in vec3 fragPosition;
in vec3 fragNormal;

uniform vec3 viewPos;
uniform sampler2D texture0;
uniform sampler2D tfScreenTexture;
uniform vec2 tfScreenSize;
uniform vec3 tfCameraRight;
uniform vec3 tfCameraUp;
uniform vec3 tfCameraForward;
uniform float tfCameraTanHalfFov;
uniform float tfCameraAspect;
uniform float tfRefractionScale;
uniform float tfScreenEnabled;

// Water/lens controls. The center/radius are projected from the real 3D object
// by the library so the magnification stays local to the water volume instead
// of zooming the whole screen around (0.5,0.5).
uniform vec2 tfWaterCenterUV;
uniform float tfWaterRadius;
uniform float tfWaterMagnification;
uniform float tfWaterScatter;

uniform vec4 colDiffuse;
uniform vec4 opticalParams;
uniform float opticalAlpha;
uniform float softFluid;
uniform vec4 tfOpticalMaterial;
uniform float tfAdvancedOptical; // 1 = `advanced light`: energy-neutral transmission

uniform float tfOpticalLightCount;
uniform vec4 tfOpticalLights[4];
uniform vec4 tfOpticalLightColors[4];

out vec4 finalColor;
const float PI = 3.14159265358979323846;

vec2 envUV(vec3 d)
{
    d = normalize(d);
    float u = atan(d.z, d.x) / (2.0 * PI) + 0.5;
    float v = asin(clamp(d.y, -1.0, 1.0)) / PI + 0.5;
    return vec2(fract(u), clamp(1.0 - v, 0.001, 0.999));
}

vec3 environment(vec3 d)
{
    return texture(texture0, envUV(d)).rgb;
}

float dielectricFresnel(float cosI, float etaI, float etaT)
{
    cosI = clamp(cosI, 0.0, 1.0);
    float eta = etaI / etaT;
    float sinT2 = eta * eta * max(0.0, 1.0 - cosI * cosI);
    if (sinT2 >= 1.0)
        return 1.0;

    float cosT = sqrt(max(0.0, 1.0 - sinT2));
    float rs = (etaI * cosI - etaT * cosT) /
               max(etaI * cosI + etaT * cosT, 1e-6);
    float rp = (etaT * cosI - etaI * cosT) /
               max(etaT * cosI + etaI * cosT, 1e-6);
    return 0.5 * (rs * rs + rp * rp);
}

vec3 materialIOR(float materialId)
{
    // material 0 = quartz/crystal, 1 = diamond, 2 = prism/BK7,
    // material 3 = liquid water.
    if (materialId > 2.5)
        return vec3(1.3325, 1.3330, 1.3335);
    if (materialId > 1.5)
        return vec3(1.51392, 1.51633, 1.52198);
    if (materialId > 0.5)
        return vec3(2.415, 2.420, 2.430);
    return vec3(1.542, 1.545, 1.548);
}

vec3 safeRefract(vec3 I, vec3 N, float eta)
{
    float c = clamp(dot(-I, N), 0.0, 1.0);
    float sin2T = eta * eta * max(0.0, 1.0 - c * c);
    if (sin2T >= 1.0)
        return reflect(I, N);
    return normalize(refract(I, N, eta));
}

vec3 softGeometryNormal(vec3 baseN, vec3 p, float soft)
{
    vec3 ddx = dFdx(p);
    vec3 ddy = dFdy(p);
    vec3 Ng = normalize(cross(ddx, ddy));
    if (dot(Ng, baseN) < 0.0) Ng = -Ng;
    return normalize(mix(baseN, Ng, clamp(0.92 * soft, 0.0, 0.92)));
}

float thicknessFromView(vec3 N, vec3 V, float baseThickness, float soft)
{
    float nv = max(abs(dot(N, V)), 0.08);
    float grazing = 1.0 / nv;
    float t = baseThickness * clamp(0.78 + 0.55 * (grazing - 1.0), 0.70, 4.4);
    t *= mix(1.0, 1.18, clamp(soft, 0.0, 1.0));
    return t;
}

vec2 projectRayToScreen(vec3 direction)
{
    vec3 c = vec3(dot(direction, tfCameraRight),
                  dot(direction, tfCameraUp),
                  dot(direction, tfCameraForward));
    float z = max(c.z, 0.055);
    float x = c.x / max(z * tfCameraTanHalfFov * tfCameraAspect, 0.0001);
    float y = c.y / max(z * tfCameraTanHalfFov, 0.0001);
    return vec2(x, y);
}

vec2 refractedScreenOffset(vec3 incident,
                           vec3 transmitted,
                           float thickness,
                           float scale)
{
    vec2 pI = projectRayToScreen(incident);
    vec2 pT = projectRayToScreen(transmitted);
    vec2 delta = (pT - pI) * thickness * scale;
    return clamp(delta, vec2(-0.48), vec2(0.48));
}

vec3 dynamicSpecular(vec3 P, vec3 N, vec3 V, vec3 ior, float count)
{
    vec3 result = vec3(0.0);
    vec3 F0 = pow((ior - vec3(1.0)) / (ior + vec3(1.0)), vec3(2.0));

    for (int i = 0; i < 4; ++i)
    {
        if (float(i) >= count) break;
        vec3 Lvec = tfOpticalLights[i].xyz - P;
        float d2 = max(dot(Lvec, Lvec), 0.0001);
        float d = sqrt(d2);
        vec3 L = Lvec / d;
        float NL = max(dot(N, L), 0.0);
        if (NL <= 0.0) continue;

        vec3 H = normalize(V + L);
        float VH = max(dot(V, H), 0.0);
        vec3 F = F0 + (vec3(1.0) - F0) * pow(1.0 - VH, 5.0);
        float rough = 0.08;
        float a = rough * rough;
        float a2 = a*a;
        float NH = max(dot(N,H),0.0);
        float denom = max(PI * pow(NH*NH*(a2-1.0)+1.0, 2.0), 1e-5);
        float D = a2 / denom;
        float k = (rough + 1.0)*(rough + 1.0)/8.0;
        float NV = max(dot(N,V),0.0);
        float Gv = NV / max(NV*(1.0-k)+k, 1e-5);
        float Gl = NL / max(NL*(1.0-k)+k, 1e-5);
        vec3 spec = (D * Gv * Gl * F) /
                    max(4.0 * NV * NL, 1e-5);

        float strength = max(tfOpticalLights[i].w, 0.0);
        vec3 Li = max(tfOpticalLightColors[i].rgb, vec3(0.0)) * strength / d2;
        result += spec * Li * NL;
    }
    return result;
}

float tfLuma(vec3 c)
{
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec2 tfWaterWarp(vec2 uv, vec3 N, vec3 V, float liquid, float curvature)
{
    vec2 slope = vec2(dot(N, tfCameraRight), dot(N, tfCameraUp));
    slope *= vec2(0.90, 0.68);

    // The surface normal controls the primary lens displacement. The animated
    // term is intentionally low-amplitude: it breaks up a flat screen-space
    // offset so water bends different pixels by different amounts.
    vec2 wave = vec2(
        sin(fragPosition.x * 6.7 + fragPosition.z * 3.9 + 4.2 * curvature),
        cos(fragPosition.z * 7.1 - fragPosition.x * 2.8 + 3.3 * curvature)
    );
    wave *= (0.0025 + 0.0095 * liquid);

    vec2 localDelta = uv - tfWaterCenterUV;
    float radius = max(tfWaterRadius, 0.025);
    float inside = 1.0 - smoothstep(radius * 0.52, radius, length(localDelta));

    // Local lens magnification. Dividing the distance to the water center by
    // zoom makes a point/object behind the water appear physically enlarged
    // toward the center of the refracting volume.
    float slopeAmount = clamp(length(slope), 0.0, 1.0);
    float lens = clamp(
        tfWaterMagnification * (0.76 + 1.15 * slopeAmount + 1.35 * curvature),
        1.0, 4.4);
    float zoom = mix(1.0, lens, inside * liquid);

    vec2 magnified = tfWaterCenterUV + localDelta / zoom;
    vec2 displacement = slope * (0.018 + 0.075 * liquid) * (0.48 + 0.92 * inside);
    displacement += wave;
    displacement *= (0.56 + 1.20 * curvature + 0.48 * slopeAmount);

    return magnified + displacement;
}

vec3 tfSampleRGB(vec2 uvR, vec2 uvG, vec2 uvB)
{
    vec3 r = texture(tfScreenTexture, uvR).rgb;
    vec3 g = texture(tfScreenTexture, uvG).rgb;
    vec3 b = texture(tfScreenTexture, uvB).rgb;
    return vec3(r.r, g.g, b.b);
}

vec3 tfWaterScatter(vec2 baseUV,
                    vec2 dirR,
                    vec2 dirG,
                    vec2 dirB,
                    float liquid,
                    float curvature,
                    float bright)
{
    // Eight radial taps emulate the way small bright features are distributed
    // by rippled water. The offsets are symmetric, so the spread is not biased
    // toward one direction.
    const float S2 = 0.70710678118;
    vec2 dirs[8] = vec2[8](
        vec2( 1.0, 0.0), vec2(-1.0, 0.0),
        vec2( 0.0, 1.0), vec2( 0.0,-1.0),
        vec2( S2, S2), vec2(-S2, S2),
        vec2( S2,-S2), vec2(-S2,-S2));

    float radius = (0.0022 + 0.0090 * liquid +
                    0.0100 * curvature + 0.0150 * bright * liquid) *
                   max(0.65, tfWaterScatter);

    vec3 accum = tfSampleRGB(
        clamp(baseUV + dirR, vec2(0.001), vec2(0.999)),
        clamp(baseUV + dirG, vec2(0.001), vec2(0.999)),
        clamp(baseUV + dirB, vec2(0.001), vec2(0.999)));

    for (int i = 0; i < 8; ++i)
    {
        float scale = (0.34 + 0.085 * float(i % 4));
        vec2 d = dirs[i] * radius * scale;
        vec3 tap = tfSampleRGB(
            clamp(baseUV + d + dirR * 0.18, vec2(0.001), vec2(0.999)),
            clamp(baseUV + d + dirG * 0.18, vec2(0.001), vec2(0.999)),
            clamp(baseUV + d + dirB * 0.18, vec2(0.001), vec2(0.999)));
        accum += tap;
    }

    vec3 average = accum / 9.0;

    // Bright sources receive an additional low-frequency halo. This is the
    // part that makes a point light behind rippled water visibly fan out rather
    // than merely move a few pixels.
    float halo = smoothstep(0.35, 1.4, bright) * liquid;
    vec2 h = vec2(radius * 2.1);
    vec3 haloA = tfSampleRGB(
        clamp(baseUV + h + dirR * 0.2, vec2(0.001), vec2(0.999)),
        clamp(baseUV + h + dirG * 0.2, vec2(0.001), vec2(0.999)),
        clamp(baseUV + h + dirB * 0.2, vec2(0.001), vec2(0.999)));
    vec3 haloB = tfSampleRGB(
        clamp(baseUV - h - dirR * 0.2, vec2(0.001), vec2(0.999)),
        clamp(baseUV - h - dirG * 0.2, vec2(0.001), vec2(0.999)),
        clamp(baseUV - h - dirB * 0.2, vec2(0.001), vec2(0.999)));

    average = mix(average, average * 0.78 + (haloA + haloB) * 0.11, 0.34 * halo);
    return average;
}

void main()
{
    float materialId = tfOpticalMaterial.x;
    float baseThickness = max(0.05, tfOpticalMaterial.y);
    float absorptionScale = max(0.0, tfOpticalMaterial.z);
    float soft = clamp(softFluid, 0.0, 1.0);
    bool water = materialId > 2.5;

    // Water is always treated as a liquid optical surface. The host soft-body
    // value still controls how strongly the surface geometry changes.
    float liquid = water ? max(soft, 0.78) : soft;

    vec3 N0 = normalize(fragNormal);
    vec3 V = normalize(viewPos - fragPosition);
    vec3 N = softGeometryNormal(N0, fragPosition, liquid);
    if (dot(N, V) < 0.0) N = -N;

    float cosI = clamp(dot(N, V), 0.0, 1.0);
    vec3 nRGB = materialIOR(materialId);
    vec3 I = -V;

    vec3 T_R = safeRefract(I, N, 1.0 / nRGB.r);
    vec3 T_G = safeRefract(I, N, 1.0 / nRGB.g);
    vec3 T_B = safeRefract(I, N, 1.0 / nRGB.b);

    vec3 F;
    F.r = dielectricFresnel(cosI, 1.0, nRGB.r);
    F.g = dielectricFresnel(cosI, 1.0, nRGB.g);
    F.b = dielectricFresnel(cosI, 1.0, nRGB.b);

    float thickness = thicknessFromView(N, V, baseThickness, liquid);
    vec3 albedo = pow(clamp(colDiffuse.rgb, 0.001, 0.999), vec3(2.2));
    vec3 sigmaA = -log(albedo) * absorptionScale;
    vec3 attenuation = exp(-sigmaA * thickness);

    vec2 screenUV = gl_FragCoord.xy / max(tfScreenSize, vec2(1.0));
    vec2 offR = refractedScreenOffset(I, T_R, thickness,
                                       water ? tfRefractionScale * 1.75 : tfRefractionScale);
    vec2 offG = refractedScreenOffset(I, T_G, thickness,
                                       water ? tfRefractionScale * 1.75 : tfRefractionScale);
    vec2 offB = refractedScreenOffset(I, T_B, thickness,
                                       water ? tfRefractionScale * 1.75 : tfRefractionScale);

    vec2 baseUV = screenUV;
    float curvature = clamp(length(N - N0) * 2.6 +
                            (1.0 - abs(dot(N, V))) * 0.55, 0.0, 1.0);

    if (water)
    {
        baseUV = tfWaterWarp(screenUV + vec2(0.0), N, V, liquid, curvature);
        // Reapply the wavelength-specific angular offsets after the local lens
        // warp so RGB rays are displaced independently.
        offR *= (1.10 + 0.90 * liquid);
        offG *= (1.10 + 0.90 * liquid);
        offB *= (1.10 + 0.90 * liquid);
    }

    vec2 uvR = clamp(baseUV + offR, vec2(0.001), vec2(0.999));
    vec2 uvG = clamp(baseUV + offG, vec2(0.001), vec2(0.999));
    vec2 uvB = clamp(baseUV + offB, vec2(0.001), vec2(0.999));

    vec3 screenTransmitted = tfSampleRGB(uvR, uvG, uvB);
    if (water)
    {
        float localBright = tfLuma(texture(tfScreenTexture, clamp(baseUV, vec2(0.001), vec2(0.999))).rgb);
        screenTransmitted = tfWaterScatter(baseUV,
                                           offR * 0.72,
                                           offG * 0.72,
                                           offB * 0.72,
                                           liquid,
                                           curvature,
                                           localBright);
    }

    vec3 environmentR = environment(T_R);
    vec3 environmentG = environment(T_G);
    vec3 environmentB = environment(T_B);
    vec3 environmentTransmitted = vec3(environmentR.r, environmentG.g, environmentB.b);
    vec3 transmittedSource = mix(environmentTransmitted,
                                  screenTransmitted,
                                  clamp(tfScreenEnabled, 0.0, 1.0));

    // Water should not absorb away a lamp image. Its extinction is weak, while
    // the object is enlarged, bent and spread by the screen-space transmission.
    vec3 transmitted = transmittedSource * attenuation * (vec3(1.0) - F);
    vec3 reflected = environment(reflect(I, N)) * F;

    vec3 bodyTint = mix(vec3(1.0), albedo, water ? 0.035 : 0.12);
    vec3 color = transmitted * bodyTint + reflected;
    float specGain = water ? 0.12 : 0.35;

    if (tfAdvancedOptical > 0.5)
    {
        // Advanced light: the light seen THROUGH the material is distorted
        // (Snell offsets, per-channel dispersion, water lens/scatter above) but
        // not attenuated. Beer-Lambert and body colour keep their hue while their
        // luminance is normalised to 1, and the Fresnel split no longer removes
        // energy from the transmitted image (the reflection is added on top).
        vec3 keepHue = attenuation * bodyTint;
        keepHue /= max(tfLuma(keepHue), 1e-4);
        color = transmittedSource * keepHue + reflected;
        specGain = water ? 0.45 : 0.90;
    }

    color += dynamicSpecular(fragPosition, N, V, nRGB, tfOpticalLightCount) * specGain;

    float edge = 1.0 - cosI;
    color += environment(reflect(I, N)) * pow(edge, 6.0) * (water ? 0.05 : 0.08);

    color = max(color, vec3(0.0));
    float alpha = (tfScreenEnabled > 0.5) ? 1.0 : clamp(opticalAlpha, 0.25, 0.96);
    finalColor = vec4(color, alpha);
}
)GLSL";
}

static bool tfBuildPhysicalOpticalShader()
{
    if (gTfPhysicalOpticalShaderFailed)
        return false;

    if (!tfGLCreateShader || !tfGLShaderSource || !tfGLCompileShader ||
        !tfGLGetShaderiv || !tfGLGetShaderInfoLog || !tfGLDeleteShader ||
        !tfGLCreateProgram || !tfGLAttachShader || !tfGLBindAttribLocation ||
        !tfGLLinkProgram || !tfGLGetProgramiv || !tfGLGetProgramInfoLog ||
        !tfGLDeleteProgram || !tfGLGetUniformLocation)
    {
        gTfPhysicalOpticalShaderFailed = true;
        gTfPhysicalOpticalError = "OpenGL shader entry points unavailable.";
        return false;
    }

    const unsigned int vs = tfCompileShaderStage(
        TF_GL_VERTEX_SHADER, tfBuildPhysicalOpticalVertexShader(), gTfPhysicalOpticalError);
    if (!vs)
    {
        gTfPhysicalOpticalShaderFailed = true;
        return false;
    }

    std::string fsError;
    const unsigned int fs = tfCompileShaderStage(
        TF_GL_FRAGMENT_SHADER, tfBuildPhysicalOpticalFragmentShader(), fsError);
    if (!fs)
    {
        tfGLDeleteShader(vs);
        gTfPhysicalOpticalShaderFailed = true;
        gTfPhysicalOpticalError = "fragment shader: " + fsError;
        return false;
    }

    const unsigned int program = tfGLCreateProgram();
    if (!program)
    {
        tfGLDeleteShader(vs);
        tfGLDeleteShader(fs);
        gTfPhysicalOpticalShaderFailed = true;
        gTfPhysicalOpticalError = "glCreateProgram returned 0.";
        return false;
    }

    tfGLAttachShader(program, vs);
    tfGLAttachShader(program, fs);
    tfGLBindAttribLocation(program, 0, "vertexPosition");
    tfGLBindAttribLocation(program, 2, "vertexNormal");
    tfGLBindAttribLocation(program, 8, "vertexTexCoord");
    tfGLLinkProgram(program);
    tfGLDeleteShader(vs);
    tfGLDeleteShader(fs);

    int linked = 0;
    tfGLGetProgramiv(program, TF_GL_LINK_STATUS, &linked);
    if (!linked)
    {
        gTfPhysicalOpticalError = tfGetGLInfoLog(false, program);
        tfGLDeleteProgram(program);
        gTfPhysicalOpticalShaderFailed = true;
        return false;
    }

    auto loc = [&](const char *name) -> int
    { return tfGLGetUniformLocation(program, name); };

    const int mvp = loc("mvp");
    const int model = loc("matModel");
    const int normal = loc("matNormal");
    const int view = loc("viewPos");
    const int tex = loc("texture0");
    const int color = loc("colDiffuse");
    const int params = loc("opticalParams");
    const int alpha = loc("opticalAlpha");
    const int time = loc("fluidTime");
    const int softFluid = loc("softFluid");
    gTfOpticalMaterialLocation = loc("tfOpticalMaterial");
    gTfOpticalLightCountLocation = loc("tfOpticalLightCount");
    gTfOpticalLightsLocation = loc("tfOpticalLights[0]");
    gTfOpticalLightColorsLocation = loc("tfOpticalLightColors[0]");
    gTfOpticalScreenTextureLocation = loc("tfScreenTexture");
    gTfOpticalScreenSizeLocation = loc("tfScreenSize");
    gTfOpticalCameraRightLocation = loc("tfCameraRight");
    gTfOpticalCameraUpLocation = loc("tfCameraUp");
    gTfOpticalCameraForwardLocation = loc("tfCameraForward");
    gTfOpticalCameraTanHalfFovLocation = loc("tfCameraTanHalfFov");
    gTfOpticalCameraAspectLocation = loc("tfCameraAspect");
    gTfOpticalRefractionScaleLocation = loc("tfRefractionScale");
    gTfOpticalScreenEnabledLocation = loc("tfScreenEnabled");
    gTfOpticalWaterCenterLocation = loc("tfWaterCenterUV");
    gTfOpticalWaterRadiusLocation = loc("tfWaterRadius");
    gTfOpticalWaterMagnificationLocation = loc("tfWaterMagnification");
    gTfOpticalWaterScatterLocation = loc("tfWaterScatter");
    gTfOpticalAdvancedLocation = loc("tfAdvancedOptical");

    if (mvp < 0 || model < 0 || normal < 0 || view < 0 || tex < 0 ||
        color < 0 || params < 0 || alpha < 0 || time < 0 || softFluid < 0 ||
        gTfOpticalMaterialLocation < 0)
    {
        gTfPhysicalOpticalError = "required optical uniforms were unavailable.";
        tfGLDeleteProgram(program);
        gTfPhysicalOpticalShaderFailed = true;
        return false;
    }

    int *locs = static_cast<int *>(std::malloc(TF_RL_MAX_SHADER_LOCATIONS * sizeof(int)));
    if (!locs)
    {
        tfGLDeleteProgram(program);
        gTfPhysicalOpticalShaderFailed = true;
        gTfPhysicalOpticalError = "optical shader location allocation failed.";
        return false;
    }
    for (unsigned int i = 0; i < TF_RL_MAX_SHADER_LOCATIONS; ++i) locs[i] = -1;
    locs[0] = 0;
    locs[3] = 2;
    locs[6] = mvp;
    locs[9] = model;
    locs[10] = normal;
    locs[11] = view;
    locs[12] = color;

    // Replace the host Shader safely. The host later owns/unloads this program
    // through its normal tfUnloadOpticalMaterial() path.
    if (gTfOpticalShaderAddress)
    {
        TFCompatShaderPublic oldShader{};
        std::memcpy(&oldShader, gTfOpticalShaderAddress, sizeof(oldShader));
        if (oldShader.id != 0 && oldShader.id != program)
            tfGLDeleteProgram(oldShader.id);
        if (oldShader.locs)
            std::free(oldShader.locs);
    }

    TFCompatShaderPublic replacement{program, locs};
    std::memcpy(gTfOpticalShaderAddress, &replacement, sizeof(replacement));

    if (gTfOpticalLocViewPosAddress)
        std::memcpy(gTfOpticalLocViewPosAddress, &view, sizeof(view));
    if (gTfOpticalLocEnvironmentAddress)
        std::memcpy(gTfOpticalLocEnvironmentAddress, &tex, sizeof(tex));
    if (gTfOpticalLocParamsAddress)
        std::memcpy(gTfOpticalLocParamsAddress, &params, sizeof(params));
    if (gTfOpticalLocAlphaAddress)
        std::memcpy(gTfOpticalLocAlphaAddress, &alpha, sizeof(alpha));
    if (gTfOpticalLocTimeAddress)
        std::memcpy(gTfOpticalLocTimeAddress, &time, sizeof(time));
    if (gTfOpticalLocSoftFluidAddress)
        std::memcpy(gTfOpticalLocSoftFluidAddress, &softFluid, sizeof(softFluid));

    if (gTfOpticalShaderReadyAddress)
    {
        const bool ready = true;
        std::memcpy(gTfOpticalShaderReadyAddress, &ready, sizeof(ready));
    }

    gTfPhysicalOpticalShaderInstalled = true;
    return true;
}

static void tfEnsurePhysicalOpticalShader()
{
    if (!gTfOpticalShaderAddress || !gTfOpticalShaderReadyAddress)
        return;

    bool ready = false;
    std::memcpy(&ready, gTfOpticalShaderReadyAddress, sizeof(ready));
    TFCompatShaderPublic hostShader{};
    std::memcpy(&hostShader, gTfOpticalShaderAddress, sizeof(hostShader));

    // The host lazily creates its baked optical texture/shader. Let that happen
    // first, then replace only the shader program while keeping the texture.
    if (!ready || hostShader.id == 0)
    {
        // The IDE unloads the optical shader/texture when a 3D scene ends.
        // The old program was therefore destroyed too; force a fresh install
        // for the next scene instead of retaining a stale GL program id.
        gTfPhysicalOpticalShaderInstalled = false;
        if (gTfPrepareOpticalShaderOriginal)
            gTfPrepareOpticalShaderOriginal();
        std::memcpy(&ready, gTfOpticalShaderReadyAddress, sizeof(ready));
        if (!ready)
            return;
    }

    if (gTfPhysicalOpticalShaderInstalled)
        return;

    tfBuildPhysicalOpticalShader();
    if (!gTfPhysicalOpticalShaderInstalled && gTfPhysicalOpticalError.size() > 0)
        logMessage(gTfPhysicalOpticalError.c_str());
}

static void tfUploadOpticalDynamicLights()
{
    if (gTfOpticalLightCountLocation < 0 ||
        gTfOpticalLightsLocation < 0 ||
        !tfGLUniform1f || !tfGLUniform4fv ||
        !tfGLUseProgram || !tfGLGetIntegerv ||
        gTfOpticalShaderAddress == nullptr)
        return;

    TFCompatShaderPublic shader{};
    std::memcpy(&shader, gTfOpticalShaderAddress, sizeof(shader));
    if (shader.id == 0)
        return;

    std::array<float, TF_MAX_DYNAMIC_LIGHTS * 4> lightData{};
    std::array<float, TF_MAX_DYNAMIC_LIGHTS * 4> colorData{};
    int count = 0;
    for (const auto &entry : gTfTrackedObjects)
    {
        const TFTrackedObject &o = entry.second;
        if (!o.isLight || o.lightStrength <= 0.0001f)
            continue;
        if (count >= TF_MAX_DYNAMIC_LIGHTS)
            break;
        lightData[count*4+0] = o.x;
        lightData[count*4+1] = o.y;
        lightData[count*4+2] = o.z;
        lightData[count*4+3] = std::max(0.0f, std::min(10.0f, o.lightStrength));
        const auto c = tfColorLinearFromName(o.color);
        colorData[count*4+0] = c[0];
        colorData[count*4+1] = c[1];
        colorData[count*4+2] = c[2];
        count++;
    }

    int previous = 0;
    tfGLGetIntegerv(TF_GL_CURRENT_PROGRAM, &previous);
    tfGLUseProgram(shader.id);
    tfGLUniform1f(gTfOpticalLightCountLocation, static_cast<float>(count));
    if (gTfOpticalAdvancedLocation >= 0)
        tfGLUniform1f(gTfOpticalAdvancedLocation, gTfAdvancedLightingEnabled ? 1.0f : 0.0f);
    tfGLUniform4fv(gTfOpticalLightsLocation, TF_MAX_DYNAMIC_LIGHTS, lightData.data());
    if (gTfOpticalLightColorsLocation >= 0)
        tfGLUniform4fv(gTfOpticalLightColorsLocation, TF_MAX_DYNAMIC_LIGHTS, colorData.data());
    tfGLUseProgram(static_cast<unsigned int>(std::max(0, previous)));
}


struct TFCamera3DCompat
{
    float px, py, pz;
    float tx, ty, tz;
    float ux, uy, uz;
    float fovy;
    int projection;
};

static bool tfEnsureOpticalScreenTexture(int width, int height)
{
    if (!tfLoadOpticalScreenFunctions() || width <= 0 || height <= 0)
        return false;

    if (gTfOpticalScreenTexture == 0)
        tfGLGenTextures(1, &gTfOpticalScreenTexture);

    if (gTfOpticalScreenTexture == 0)
        return false;

    int active = static_cast<int>(TF_GL_TEXTURE0);
    if (tfGLGetIntegerv)
        tfGLGetIntegerv(TF_GL_ACTIVE_TEXTURE, &active);

    tfGLActiveTexture(TF_GL_TEXTURE1);
    tfGLBindTexture(TF_GL_TEXTURE_2D, gTfOpticalScreenTexture);
    tfGLTexParameteri(TF_GL_TEXTURE_2D, TF_GL_TEXTURE_MIN_FILTER, TF_GL_LINEAR);
    tfGLTexParameteri(TF_GL_TEXTURE_2D, TF_GL_TEXTURE_MAG_FILTER, TF_GL_LINEAR);
    tfGLTexParameteri(TF_GL_TEXTURE_2D, TF_GL_TEXTURE_WRAP_S, TF_GL_CLAMP_TO_EDGE);
    tfGLTexParameteri(TF_GL_TEXTURE_2D, TF_GL_TEXTURE_WRAP_T, TF_GL_CLAMP_TO_EDGE);

    if (gTfOpticalScreenWidth != width || gTfOpticalScreenHeight != height)
    {
        tfGLTexImage2D(TF_GL_TEXTURE_2D, 0, static_cast<int>(TF_GL_RGBA8),
                       width, height, 0, TF_GL_RGBA,
                       TF_GL_UNSIGNED_BYTE, nullptr);
        gTfOpticalScreenWidth = width;
        gTfOpticalScreenHeight = height;
        gTfOpticalScreenCaptured = false;
    }

    tfGLActiveTexture(static_cast<unsigned int>(active));
    return true;
}

static bool tfCaptureOpticalScreen(int width, int height)
{
    if (gTfOpticalScreenCaptured &&
        gTfOpticalScreenWidth == width &&
        gTfOpticalScreenHeight == height)
        return true;

    if (!tfEnsureOpticalScreenTexture(width, height))
        return false;

    int active = static_cast<int>(TF_GL_TEXTURE0);
    if (tfGLGetIntegerv)
        tfGLGetIntegerv(TF_GL_ACTIVE_TEXTURE, &active);

    tfGLActiveTexture(TF_GL_TEXTURE1);
    tfGLBindTexture(TF_GL_TEXTURE_2D, gTfOpticalScreenTexture);
    // glCopyTexSubImage2D reads from the current framebuffer. At the first
    // optical draw that framebuffer contains the complete opaque scene, so the
    // following optical shader can bend the actual scene behind the object.
    tfGLCopyTexSubImage2D(TF_GL_TEXTURE_2D, 0,
                          0, 0, 0, 0, width, height);
    tfGLActiveTexture(static_cast<unsigned int>(active));

    gTfOpticalScreenCaptured = true;
    return true;
}

static void tfSetPhysicalOpticalScreenUniforms(unsigned int shaderId,
                                               const void *objectPtr,
                                               const void *cameraPtr,
                                               int width,
                                               int height,
                                               const std::string &shape,
                                               float thickness)
{
    if (!shaderId || !cameraPtr || !tfLoadOpticalScreenFunctions())
        return;

    const bool captured = tfCaptureOpticalScreen(width, height);
    const TFCamera3DCompat *camera =
        static_cast<const TFCamera3DCompat *>(cameraPtr);

    float forward[3] = {camera->tx-camera->px,
                        camera->ty-camera->py,
                        camera->tz-camera->pz};
    const float flen = std::sqrt(forward[0]*forward[0] +
                                 forward[1]*forward[1] +
                                 forward[2]*forward[2]);
    if (flen > 0.0001f)
    {
        forward[0] /= flen; forward[1] /= flen; forward[2] /= flen;
    }

    float up[3] = {camera->ux, camera->uy, camera->uz};
    // right = forward x up
    float right[3] = {
        forward[1]*up[2] - forward[2]*up[1],
        forward[2]*up[0] - forward[0]*up[2],
        forward[0]*up[1] - forward[1]*up[0]
    };
    const float rlen = std::sqrt(right[0]*right[0] +
                                 right[1]*right[1] +
                                 right[2]*right[2]);
    if (rlen > 0.0001f)
    {
        right[0] /= rlen; right[1] /= rlen; right[2] /= rlen;
    }
    // up = right x forward, making the three axes orthonormal.
    up[0] = right[1]*forward[2] - right[2]*forward[1];
    up[1] = right[2]*forward[0] - right[0]*forward[2];
    up[2] = right[0]*forward[1] - right[1]*forward[0];

    const float pi = 3.14159265358979323846f;
    const float tanHalfFov =
        std::tan(std::max(1.0f, camera->fovy) * (pi / 180.0f) * 0.5f);
    const float aspect = static_cast<float>(std::max(1, width)) /
                        static_cast<float>(std::max(1, height));

    const bool water = !tfIsCrystalOpticalShape(shape);
    float refractionScale = water ? 1.65f : 0.55f;
    if (shape == "diamond") refractionScale = 0.95f;
    else if (shape == "prism") refractionScale = 0.78f;
    else if (shape == "crystal") refractionScale = 0.62f;
    refractionScale *= water
        ? std::max(1.10f, std::min(2.80f, thickness))
        : std::max(0.55f, std::min(1.80f, thickness));

    int previous = 0;
    tfGLGetIntegerv(TF_GL_CURRENT_PROGRAM, &previous);
    tfGLUseProgram(shaderId);

    const int texUnit = 1;
    const float screenSize[2] = {
        static_cast<float>(std::max(1, width)),
        static_cast<float>(std::max(1, height))
    };

    // Project the optical object's center and an approximate projected radius.
    // Water magnification is applied around this local center so the background
    // object grows inside the water footprint instead of zooming the whole view.
    float waterCenter[2] = {0.5f, 0.5f};
    float waterRadius = 0.20f;
    if (objectPtr)
    {
        const TFCompatObject3DView *obj =
            static_cast<const TFCompatObject3DView *>(objectPtr);
        const float rx = obj->x - camera->px;
        const float ry = obj->y - camera->py;
        const float rz = obj->z - camera->pz;
        const float cx = rx * right[0] + ry * right[1] + rz * right[2];
        const float cy = rx * up[0] + ry * up[1] + rz * up[2];
        const float cz = rx * forward[0] + ry * forward[1] + rz * forward[2];
        if (cz > 0.05f)
        {
            waterCenter[0] = 0.5f + 0.5f * cx /
                              std::max(cz * tanHalfFov * aspect, 0.0001f);
            waterCenter[1] = 0.5f + 0.5f * cy /
                              std::max(cz * tanHalfFov, 0.0001f);

            const float scale = std::max(std::fabs(obj->scaleX),
                                std::max(std::fabs(obj->scaleY), std::fabs(obj->scaleZ)));
            const float worldRadius = water ? 1.60f * scale : 1.25f * scale;
            const float ru = 0.5f * worldRadius /
                              std::max(cz * tanHalfFov * aspect, 0.0001f);
            const float rv = 0.5f * worldRadius /
                              std::max(cz * tanHalfFov, 0.0001f);
            waterRadius = std::max(0.025f, std::min(0.72f, std::max(ru, rv)));
        }
    }
    waterCenter[0] = std::max(0.01f, std::min(0.99f, waterCenter[0]));
    waterCenter[1] = std::max(0.01f, std::min(0.99f, waterCenter[1]));

    const float waterMagnification = water ? 2.85f : 1.0f;
    const float waterScatter = water ? 1.25f : 0.0f;
    tfGLUniform1i(gTfOpticalScreenTextureLocation, texUnit);
    tfGLUniform2fv(gTfOpticalScreenSizeLocation, 1, screenSize);
    tfGLUniform3fv(gTfOpticalCameraRightLocation, 1, right);
    tfGLUniform3fv(gTfOpticalCameraUpLocation, 1, up);
    tfGLUniform3fv(gTfOpticalCameraForwardLocation, 1, forward);
    tfGLUniform1f(gTfOpticalCameraTanHalfFovLocation, tanHalfFov);
    tfGLUniform1f(gTfOpticalCameraAspectLocation, aspect);
    tfGLUniform1f(gTfOpticalRefractionScaleLocation, refractionScale);
    tfGLUniform1f(gTfOpticalScreenEnabledLocation, captured ? 1.0f : 0.0f);
    tfGLUniform2fv(gTfOpticalWaterCenterLocation, 1, waterCenter);
    tfGLUniform1f(gTfOpticalWaterRadiusLocation, waterRadius);
    tfGLUniform1f(gTfOpticalWaterMagnificationLocation, waterMagnification);
    tfGLUniform1f(gTfOpticalWaterScatterLocation, waterScatter);

    // Leave the screen texture bound to texture unit 1 for raylib's following
    // DrawMesh call, while restoring whichever texture unit was active before.
    tfGLActiveTexture(TF_GL_TEXTURE1);
    tfGLBindTexture(TF_GL_TEXTURE_2D, gTfOpticalScreenTexture);
    tfGLActiveTexture(TF_GL_TEXTURE0);
    tfGLUseProgram(static_cast<unsigned int>(std::max(0, previous)));
}

static void tfInstallPhysicalOpticalBridge()
{
    if (gTfOpticalShaderAddress && gTfOpticalShaderReadyAddress)
        return;

    gTfOpticalShaderAddress =
        tfFindELFSymbolInMainExecutable("tfOpticalShader");
    gTfOpticalShaderReadyAddress =
        tfFindELFSymbolInMainExecutable("tfOpticalShaderReady");
    gTfOpticalLocViewPosAddress =
        tfFindELFSymbolInMainExecutable("tfOpticalLocViewPos");
    gTfOpticalLocEnvironmentAddress =
        tfFindELFSymbolInMainExecutable("tfOpticalLocEnvironment");
    gTfOpticalLocParamsAddress =
        tfFindELFSymbolInMainExecutable("tfOpticalLocParams");
    gTfOpticalLocAlphaAddress =
        tfFindELFSymbolInMainExecutable("tfOpticalLocAlpha");
    gTfOpticalLocTimeAddress =
        tfFindELFSymbolInMainExecutable("tfOpticalLocTime");
    gTfOpticalLocSoftFluidAddress =
        tfFindELFSymbolInMainExecutable("tfOpticalLocSoftFluid");
    gTfPrepareOpticalShaderAddress =
        tfFindELFSymbolInMainExecutable("_Z22tfPrepareOpticalShaderv");
    gTfSetOpticalParamsAddress =
        tfFindELFSymbolInMainExecutable("_Z18tfSetOpticalParamsRK10TFObject3DRK8Camera3D");

    if (gTfPrepareOpticalShaderAddress)
        gTfPrepareOpticalShaderOriginal =
            reinterpret_cast<TFPrepareOpticalShaderFunction>(gTfPrepareOpticalShaderAddress);

    if (!gTfOpticalShaderAddress || !gTfOpticalShaderReadyAddress ||
        !gTfPrepareOpticalShaderOriginal)
    {
        gTfPhysicalOpticalShaderFailed = true;
        gTfPhysicalOpticalError = "host optical shader symbols were not found.";
        return;
    }

    if (gTfSetOpticalParamsAddress && !gTfSetOpticalParamsHookInstalled)
    {
        void *trampoline = nullptr;
        if (tfInstallTrampolineHook(
                gTfSetOpticalParamsAddress,
                reinterpret_cast<void *>(&tfSetOpticalParamsPhysicalProxy),
                &trampoline))
        {
            gTfSetOpticalParamsTrampoline = trampoline;
            gTfSetOpticalParamsOriginal =
                reinterpret_cast<TFSetOpticalParamsFunction>(trampoline);
            gTfSetOpticalParamsHookInstalled = true;
        }
    }
}

static void tfSetOpticalParamsPhysicalProxy(const void *objectPtr,
                                            const void *cameraPtr)
{
    if (gTfSetOpticalParamsOriginal)
        gTfSetOpticalParamsOriginal(objectPtr, cameraPtr);

    if (!objectPtr || !gTfPhysicalOpticalShaderInstalled ||
        gTfOpticalMaterialLocation < 0)
        return;

    const TFCompatObject3DView *obj =
        static_cast<const TFCompatObject3DView *>(objectPtr);

    // The host's optical pass is also used by soft/liquid bodies. The original
    // library only replaced crystal/diamond/prism, which meant a real soft-body
    // water object never reached the physical screen-space refraction path.
    // Non-crystal optical objects are now treated as water (IOR ~1.333).
    float materialId = 3.0f;
    float thickness = 1.85f;
    float absorption = 0.004f;
    if (obj->shape == "crystal")
    {
        materialId = 0.0f;
        thickness = 1.40f;
        absorption = 0.035f;
    }
    else if (obj->shape == "diamond")
    {
        materialId = 1.0f;
        thickness = 1.05f;
        absorption = 0.010f;
    }
    else if (obj->shape == "prism")
    {
        materialId = 2.0f;
        thickness = 1.65f;
        absorption = 0.025f;
    }

    // Soft bodies reach the optical pass as "water" (materialId 3). Remember them
    // so the advanced-light shadow pass treats them as refractive, not opaque.
    if (materialId > 2.5f && !obj->name.empty())
        gTfSoftRefractiveSeen[obj->name] = tfNowSeconds();

    TFCompatShaderPublic shader{};
    if (!gTfOpticalShaderAddress)
        return;
    std::memcpy(&shader, gTfOpticalShaderAddress, sizeof(shader));
    if (shader.id == 0)
        return;

    int previous = 0;
    tfGLGetIntegerv(TF_GL_CURRENT_PROGRAM, &previous);
    tfGLUseProgram(shader.id);
    const float materialData[4] = {materialId, thickness, absorption, 0.0f};
    tfGLUniform4fv(gTfOpticalMaterialLocation, 1, materialData);
    tfGLUseProgram(static_cast<unsigned int>(std::max(0, previous)));

    tfSetPhysicalOpticalScreenUniforms(
        shader.id,
        objectPtr,
        gTfOpticalCurrentCameraPtr ? gTfOpticalCurrentCameraPtr : cameraPtr,
        gTfOpticalCurrentScreenWidth,
        gTfOpticalCurrentScreenHeight,
        obj->shape,
        thickness);

    tfUploadOpticalDynamicLights();
}

static void tfSyncLiveObjectsFromHost()
{
    if (!gTfObjects3DAddress)
        return;

    auto *hostObjects =
        static_cast<std::vector<TFCompatHostObject3D> *>(gTfObjects3DAddress);
    if (!hostObjects)
        return;

    // The host renderer has already completed input, background-script and
    // physics/collision updates when this hook runs. Reading the vector here
    // therefore gives us the authoritative position for this exact frame.
    std::unordered_map<std::string, bool> liveNames;
    liveNames.reserve(hostObjects->size());

    for (const TFCompatHostObject3D &obj : *hostObjects)
    {
        if (obj.name.empty())
            continue;

        liveNames[obj.name] = true;

        TFTrackedObject &state = gTfTrackedObjects[obj.name];
        state.name = obj.name;
        state.shape = obj.shape;
        state.color = obj.color;
        state.x = obj.x;
        state.y = obj.y;
        state.z = obj.z;
        state.scaleX = obj.scaleX;
        state.scaleY = obj.scaleY;
        state.scaleZ = obj.scaleZ;
        state.rotationX = obj.rotationX;
        state.rotationY = obj.rotationY;
        state.rotationZ = obj.rotationZ;

        auto lightIt = gTfObjectLights.find(obj.name);
        state.isLight = (lightIt != gTfObjectLights.end());
        state.lightStrength = state.isLight
            ? std::max(0.0f, std::min(10.0f, lightIt->second.strength))
            : 0.0f;
        state.advancedLight = state.isLight && lightIt->second.advanced;
    }

    // Drop objects from previous scenes/after deletion so a removed light can
    // never leave a ghost source behind in the dynamic lighting buffers.
    for (auto it = gTfTrackedObjects.begin();
         it != gTfTrackedObjects.end(); )
    {
        if (liveNames.find(it->first) == liveNames.end())
            it = gTfTrackedObjects.erase(it);
        else
            ++it;
    }

    // Also discard the light declaration itself when the host object no
    // longer exists. This prevents a deleted light from ever being revived
    // accidentally by stale library state.
    for (auto it = gTfObjectLights.begin();
         it != gTfObjectLights.end(); )
    {
        if (liveNames.find(it->first) == liveNames.end())
            it = gTfObjectLights.erase(it);
        else
            ++it;
    }

    // Real motion-blur state belongs only to live objects too. This keeps
    // object deletion/recreation from inheriting stale blur history.
    for (auto it = gTfRealMotionBlur.begin();
         it != gTfRealMotionBlur.end(); )
    {
        if (liveNames.find(it->first) == liveNames.end())
            it = gTfRealMotionBlur.erase(it);
        else
            ++it;
    }

    if (!gTfLastCreatedObjectName.empty() &&
        liveNames.find(gTfLastCreatedObjectName) == liveNames.end())
    {
        gTfLastCreatedObjectName.clear();
    }
}

static void tfResolvePhysicalCollisionsLightProxy()
{
    // Every collision pass is one of the host's per-frame synchronization points.
    // Reset the optical framebuffer capture here so the next optical object sees
    // the fresh opaque scene exactly once for this frame.
    gTfOpticalScreenCaptured = false;

    if (gTfResolvePhysicalCollisionsOriginal)
        gTfResolvePhysicalCollisionsOriginal();

    // This is deliberately AFTER the host's collision step. Any light moved
    // by keyboard input, script, gravity or collision resolution is now read
    // from the host's real object position before the first object is drawn.
    tfSyncLiveObjectsFromHost();
    tfUpdateGlobalLightingState();
}

static void tfTrackDrawnObject(const void *objectPtr)
{
    if (!objectPtr)
        return;

    const TFCompatObject3DView *obj =
        static_cast<const TFCompatObject3DView *>(objectPtr);
    if (obj->name.empty())
        return;

    TFTrackedObject &state = gTfTrackedObjects[obj->name];
    state.name = obj->name;
    state.shape = obj->shape;
    state.color = obj->color;
    state.x = obj->x;
    state.y = obj->y;
    state.z = obj->z;
    state.scaleX = obj->scaleX;
    state.scaleY = obj->scaleY;
    state.scaleZ = obj->scaleZ;

    auto it = gTfObjectLights.find(obj->name);
    state.isLight = (it != gTfObjectLights.end());
    state.lightStrength = state.isLight ? std::max(0.0f, std::min(10.0f, it->second.strength)) : 0.0f;
}

static float tfGIObjectRadius(const TFTrackedObject &o)
{
    const float sx = std::fabs(o.scaleX);
    const float sy = std::fabs(o.scaleY);
    const float sz = std::fabs(o.scaleZ);
    if (o.shape == "sphere")
        return std::max(0.10f, 1.50f * std::max(sx, std::max(sy, sz)));
    if (o.shape == "capsule")
        return std::max(0.10f, 1.30f * std::max(sx, std::max(sy, sz)));
    if (o.shape == "crystal" || o.shape == "diamond" || o.shape == "prism")
        return std::max(0.10f, 1.60f * std::max(sx, std::max(sy, sz)));
    return std::max(0.10f, 1.20f * std::sqrt(sx*sx + sy*sy + sz*sz));
}

static void tfResetGIField()
{
    gTfGlobalLighting.giDirections.fill(0.0f);

    // Tiny sky/ground base. These values are deliberately much lower than a
    // direct light so GI can brighten shadows without washing them out.
    gTfGlobalLighting.giDirections[2*4+0] = 0.030f;
    gTfGlobalLighting.giDirections[2*4+1] = 0.045f;
    gTfGlobalLighting.giDirections[2*4+2] = 0.070f;
    gTfGlobalLighting.giDirections[3*4+0] = 0.010f;
    gTfGlobalLighting.giDirections[3*4+1] = 0.008f;
    gTfGlobalLighting.giDirections[3*4+2] = 0.006f;
}

static void tfAccumulateGIFromBounce(const std::array<float, 3> &bounce,
                                     const TFTrackedObject &surface,
                                     const std::array<float, 3> &sceneCenter)
{
    const float nx = surface.x - sceneCenter[0];
    const float ny = surface.y - sceneCenter[1];
    const float nz = surface.z - sceneCenter[2];
    const float len = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (len <= 0.001f)
        return;

    const float ax[3] = {std::fabs(nx/len), std::fabs(ny/len), std::fabs(nz/len)};
    const int baseX = (nx >= 0.0f) ? 0 : 1;
    const int baseY = (ny >= 0.0f) ? 2 : 3;
    const int baseZ = (nz >= 0.0f) ? 4 : 5;

    const float scaleX = ax[0] * 0.55f;
    const float scaleY = ax[1] * 0.75f;
    const float scaleZ = ax[2] * 0.55f;

    gTfGlobalLighting.giDirections[baseX*4+0] += bounce[0] * scaleX;
    gTfGlobalLighting.giDirections[baseX*4+1] += bounce[1] * scaleX;
    gTfGlobalLighting.giDirections[baseX*4+2] += bounce[2] * scaleX;
    gTfGlobalLighting.giDirections[baseY*4+0] += bounce[0] * scaleY;
    gTfGlobalLighting.giDirections[baseY*4+1] += bounce[1] * scaleY;
    gTfGlobalLighting.giDirections[baseY*4+2] += bounce[2] * scaleY;
    gTfGlobalLighting.giDirections[baseZ*4+0] += bounce[0] * scaleZ;
    gTfGlobalLighting.giDirections[baseZ*4+1] += bounce[1] * scaleZ;
    gTfGlobalLighting.giDirections[baseZ*4+2] += bounce[2] * scaleZ;
}

static float tfEstimateSceneLuminance()
{
    // Log-average estimate, similar in spirit to a photographic eye meter.
    // We use the actual light/object state rather than a single fixed scalar,
    // which keeps exposure responsive when the user moves lights or objects.
    double logSum = 0.0;
    int sampleCount = 0;

    std::array<float, 3> sceneCenter = {0.0f, 0.0f, 0.0f};
    int centerCount = 0;
    for (const auto &entry : gTfTrackedObjects)
    {
        const TFTrackedObject &o = entry.second;
        if (o.isLight) continue;
        sceneCenter[0] += o.x;
        sceneCenter[1] += o.y;
        sceneCenter[2] += o.z;
        ++centerCount;
    }
    if (centerCount > 0)
    {
        sceneCenter[0] /= centerCount;
        sceneCenter[1] /= centerCount;
        sceneCenter[2] /= centerCount;
    }

    int sampledObjects = 0;
    for (const auto &entry : gTfTrackedObjects)
    {
        const TFTrackedObject &o = entry.second;
        if (o.isLight) continue;
        if (++sampledObjects > 64) break;

        const auto albedo = tfColorLinearFromName(o.color);
        const float albedoLum = std::max(0.008f,
            0.2126f*albedo[0] + 0.7152f*albedo[1] + 0.0722f*albedo[2]);
        const float radius = tfGIObjectRadius(o);
        const float dx = o.x-sceneCenter[0];
        const float dy = o.y-sceneCenter[1];
        const float dz = o.z-sceneCenter[2];
        const float dCenter = std::sqrt(dx*dx+dy*dy+dz*dz);

        float luminance = 0.055f + 0.018f * std::exp(-dCenter*0.02f);
        for (const auto &lightEntry : gTfTrackedObjects)
        {
            const TFTrackedObject &l = lightEntry.second;
            if (!l.isLight || l.lightStrength <= 0.0001f) continue;
            const float lx = l.x-o.x, ly = l.y-o.y, lz = l.z-o.z;
            const float d2 = std::max(0.25f, lx*lx+ly*ly+lz*lz);
            const float receive = std::min(1.0f, radius / std::sqrt(d2));
            const auto lc = tfColorLinearFromName(l.color);
            const float lightLum = 0.2126f*lc[0] + 0.7152f*lc[1] + 0.0722f*lc[2];
            luminance += 22.0f*l.lightStrength*lightLum*receive / d2 * albedoLum * 0.22f;
        }

        logSum += std::log(std::max(0.00005f, luminance));
        ++sampleCount;
    }

    if (sampleCount == 0)
    {
        float total = 0.055f;
        int lights = 0;
        for (const auto &entry : gTfTrackedObjects)
        {
            const TFTrackedObject &l = entry.second;
            if (!l.isLight || l.lightStrength <= 0.0001f) continue;
            const auto lc = tfColorLinearFromName(l.color);
            const float lightLum = 0.2126f*lc[0] + 0.7152f*lc[1] + 0.0722f*lc[2];
            total += lightLum * l.lightStrength * 0.018f;
            ++lights;
        }
        return std::max(0.00005f, total / std::max(1, lights));
    }

    return static_cast<float>(std::exp(logSum / static_cast<double>(sampleCount)));
}

static void tfRebuildGlobalGI()
{
    tfResetGIField();

    std::array<float, 3> sceneCenter = {0.0f, 0.0f, 0.0f};
    int centerCount = 0;
    for (const auto &entry : gTfTrackedObjects)
    {
        const TFTrackedObject &o = entry.second;
        if (o.isLight) continue;
        sceneCenter[0] += o.x;
        sceneCenter[1] += o.y;
        sceneCenter[2] += o.z;
        ++centerCount;
    }
    if (centerCount > 0)
    {
        sceneCenter[0] /= centerCount;
        sceneCenter[1] /= centerCount;
        sceneCenter[2] /= centerCount;
    }

    // Low-frequency one-bounce color bleeding. Only the nearest 32 surfaces
    // contribute, keeping the CPU work bounded even in larger scenes.
    int surfaceCount = 0;
    for (const auto &surfaceEntry : gTfTrackedObjects)
    {
        const TFTrackedObject &surface = surfaceEntry.second;
        if (surface.isLight) continue;
        if (++surfaceCount > 32) break;

        const auto albedo = tfColorLinearFromName(surface.color);
        const float area = std::min(6.0f, std::max(0.20f, tfGIObjectRadius(surface)*tfGIObjectRadius(surface)*0.16f));
        std::array<float, 3> bounce = {0.0f,0.0f,0.0f};

        for (const auto &lightEntry : gTfTrackedObjects)
        {
            const TFTrackedObject &light = lightEntry.second;
            if (!light.isLight || light.lightStrength <= 0.0001f) continue;

            const float dx = light.x-surface.x;
            const float dy = light.y-surface.y;
            const float dz = light.z-surface.z;
            const float d2 = std::max(0.35f, dx*dx+dy*dy+dz*dz);
            const auto lc = tfColorLinearFromName(light.color);
            const float energy = 22.0f * light.lightStrength / d2;
            const float form = std::min(1.0f, area/(d2+area));

            bounce[0] += lc[0] * energy * form * albedo[0] * 0.018f;
            bounce[1] += lc[1] * energy * form * albedo[1] * 0.018f;
            bounce[2] += lc[2] * energy * form * albedo[2] * 0.018f;
        }

        // Ground-like surfaces are stronger diffuse bouncers because they cover
        // more area than a small object. This supplements, rather than replaces,
        // the existing local ground-bounce calculation.
        const float horizontal = std::max(std::fabs(surface.scaleX), std::fabs(surface.scaleZ));
        const bool groundLike = horizontal >= 1.5f && std::fabs(surface.scaleY) <= horizontal*0.30f;
        if (groundLike)
        {
            bounce[0] *= 1.8f;
            bounce[1] *= 1.8f;
            bounce[2] *= 1.8f;
        }

        tfAccumulateGIFromBounce(bounce, surface, sceneCenter);
    }

    // Prevent accidental runaway from many similarly colored surfaces. This is
    // a physically conservative visual GI approximation, not an energy source.
    for (int i=0; i<TF_GI_DIRECTION_COUNT; ++i)
    {
        gTfGlobalLighting.giDirections[i*4+0] = std::min(0.24f, gTfGlobalLighting.giDirections[i*4+0]);
        gTfGlobalLighting.giDirections[i*4+1] = std::min(0.24f, gTfGlobalLighting.giDirections[i*4+1]);
        gTfGlobalLighting.giDirections[i*4+2] = std::min(0.24f, gTfGlobalLighting.giDirections[i*4+2]);
        gTfGlobalLighting.giDirections[i*4+3] = 1.0f;
    }
}

static void tfUpdateGlobalLightingState()
{
    const double now = tfNowSeconds();
    if (!gTfGlobalLighting.initialized)
    {
        gTfGlobalLighting.initialized = true;
        gTfGlobalLighting.lastUpdateSeconds = now;
        gTfGlobalLighting.exposure = 1.0f;
        gTfGlobalLighting.targetExposure = 1.0f;
        tfRebuildGlobalGI();
        gTfGlobalLighting.serial++;
        return;
    }

    const float dt = static_cast<float>(std::max(0.0, std::min(0.10, now - gTfGlobalLighting.lastUpdateSeconds)));
    gTfGlobalLighting.lastUpdateSeconds = now;
    gTfGlobalLighting.updateAccumulator += dt;

    // GI is rebuilt at a modest fixed cadence. Dynamic direct shadows remain
    // per-frame; GI is deliberately low-frequency to keep CPU overhead small.
    if (gTfGlobalLighting.updateAccumulator >= 0.10f)
    {
        gTfGlobalLighting.updateAccumulator = 0.0f;
        gTfGlobalLighting.measuredLuminance = tfEstimateSceneLuminance();
        gTfGlobalLighting.targetExposure = std::max(0.30f, std::min(4.00f,
            0.18f / std::max(0.0002f, gTfGlobalLighting.measuredLuminance)));
        tfRebuildGlobalGI();
        gTfGlobalLighting.serial++;
    }

    // Photographic-eye style adaptation: brightening is faster, dark adaptation
    // is deliberately slower, so the image does not pump wildly around the eye.
    const float speed = (gTfGlobalLighting.targetExposure < gTfGlobalLighting.exposure)
        ? 3.8f : 1.15f;
    const float alpha = 1.0f - std::exp(-speed*std::max(dt,0.0f));
    const float previousExposure = gTfGlobalLighting.exposure;
    gTfGlobalLighting.exposure +=
        (gTfGlobalLighting.targetExposure - gTfGlobalLighting.exposure) * alpha;
    gTfGlobalLighting.exposure = std::max(0.30f, std::min(4.00f, gTfGlobalLighting.exposure));

    // Exposure is allowed to update every frame while the GI field itself is
    // deliberately rebuilt only every 100 ms. This keeps eye adaptation smooth
    // without paying the CPU cost of rebuilding the bounce field every draw.
    if (std::fabs(gTfGlobalLighting.exposure - previousExposure) > 0.00005f)
        gTfGlobalLighting.serial++;
}

static void tfUploadGlobalLightingUniformsIfNeeded()
{
    if (!gTfAdvancedProgram || !tfGLUseProgram || !tfGLGetIntegerv || !tfGLUniform1f || !tfGLUniform4fv)
        return;

    if (gTfGlobalLightingUploadedSerial == gTfGlobalLighting.serial)
        return;

    int previous = 0;
    tfGLGetIntegerv(TF_GL_CURRENT_PROGRAM, &previous);
    tfGLUseProgram(gTfAdvancedProgram);

    if (gTfExposureLocation >= 0)
        tfGLUniform1f(gTfExposureLocation, gTfGlobalLighting.exposure);
    if (gTfGIIntensityLocation >= 0)
        tfGLUniform1f(gTfGIIntensityLocation, gTfGlobalLighting.giIntensity);
    if (gTfGIDirectionsLocation >= 0)
        tfGLUniform4fv(gTfGIDirectionsLocation, TF_GI_DIRECTION_COUNT, gTfGlobalLighting.giDirections.data());
    if (gTfAtmosphereParamsLocation >= 0)
    {
        // density, height scale, Rayleigh, Mie
        const float params[4] = {0.0028f, 42.0f, 0.72f, 0.34f};
        tfGLUniform4fv(gTfAtmosphereParamsLocation, 1, params);
    }
    if (gTfAtmosphereColorLocation >= 0)
    {
        // Slightly daylight-biased scattering. Alpha stores the sun in-scatter weight.
        const float color[4] = {0.52f, 0.68f, 0.92f, 0.75f};
        tfGLUniform4fv(gTfAtmosphereColorLocation, 1, color);
    }

    tfGLUseProgram(static_cast<unsigned int>(std::max(0, previous)));
    gTfGlobalLightingUploadedSerial = gTfGlobalLighting.serial;
}

static bool tfUploadDynamicLighting(const std::string &receiverName)
{
    tfUpdateGlobalLightingState();
    tfUploadGlobalLightingUniformsIfNeeded();

    if (!tfGLUniform1f || !tfGLUniform4fv || !tfGLUseProgram || !tfGLGetIntegerv ||
        gTfAdvancedProgram == 0 ||
        gTfDynamicLightCountLocation < 0 || gTfDynamicLightsLocation < 0 ||
        gTfDynamicColorsLocation < 0 || gTfDynamicLightParamsLocation < 0 ||
        gTfDynamicOccluderCountLocation < 0 ||
        gTfDynamicOccludersLocation < 0)
        return false;

    struct LightCandidate { float distance2; TFTrackedObject state; };
    std::vector<LightCandidate> lights;
    lights.reserve(gTfTrackedObjects.size());

    float rx = 0.0f, ry = 0.0f, rz = 0.0f;
    auto receiverIt = gTfTrackedObjects.find(receiverName);
    if (receiverIt != gTfTrackedObjects.end())
    {
        rx = receiverIt->second.x;
        ry = receiverIt->second.y;
        rz = receiverIt->second.z;
    }

    for (const auto &entry : gTfTrackedObjects)
    {
        const TFTrackedObject &o = entry.second;
        if (!o.isLight || o.lightStrength <= 0.0001f)
            continue;
        const float dx = o.x-rx, dy = o.y-ry, dz = o.z-rz;
        lights.push_back({dx*dx+dy*dy+dz*dz, o});
    }
    std::sort(lights.begin(), lights.end(),
              [](const LightCandidate &a, const LightCandidate &b)
              { return a.distance2 < b.distance2; });

    std::array<float, TF_MAX_DYNAMIC_LIGHTS * 4> lightData{};
    std::array<float, TF_MAX_DYNAMIC_LIGHTS * 4> colorData{};
    std::array<float, TF_MAX_DYNAMIC_LIGHTS * 4> lightParamsData{};

    auto lightRadiusForObject = [](const TFTrackedObject &o)
    {
        const float sx = std::fabs(o.scaleX);
        const float sy = std::fabs(o.scaleY);
        const float sz = std::fabs(o.scaleZ);
        float radius = 0.10f;
        if (o.shape == "sphere") radius = 1.50f * std::max(sx, std::max(sy, sz));
        else if (o.shape == "cube" || o.shape == "box") radius = 0.95f * std::max(sx, std::max(sy, sz));
        else radius = 1.10f * std::max(sx, std::max(sy, sz));
        return std::max(0.025f, std::min(4.0f, radius));
    };

    const int lightCount = std::min<int>(TF_MAX_DYNAMIC_LIGHTS, static_cast<int>(lights.size()));
    for (int i = 0; i < lightCount; ++i)
    {
        const TFTrackedObject &o = lights[static_cast<size_t>(i)].state;
        lightData[i*4+0] = o.x;
        lightData[i*4+1] = o.y;
        lightData[i*4+2] = o.z;
        lightData[i*4+3] = o.lightStrength;
        const auto c = tfColorLinearFromName(o.color);
        colorData[i*4+0] = c[0];
        colorData[i*4+1] = c[1];
        colorData[i*4+2] = c[2];
        // Alpha is a compact per-light mode flag.
        // 0 = normal light(), 1 = realistic advanced-light shadows.
        // The mode is global: `advanced light` affects every light in the scene.
        colorData[i*4+3] = gTfAdvancedLightingEnabled ? 1.0f : 0.0f;
        lightParamsData[i*4+0] = lightRadiusForObject(o);
        lightParamsData[i*4+1] = 0.0f;
        lightParamsData[i*4+2] = 0.0f;
        lightParamsData[i*4+3] = 0.0f;
    }

    struct OccluderCandidate { float distance2; TFTrackedObject state; };
    std::vector<OccluderCandidate> occluders;
    occluders.reserve(gTfTrackedObjects.size());
    for (const auto &entry : gTfTrackedObjects)
    {
        const TFTrackedObject &o = entry.second;
        const float sx = std::fabs(o.scaleX);
        const float sy = std::fabs(o.scaleY);
        const float sz = std::fabs(o.scaleZ);
        const float horizontal = std::max(sx, sz);
        const bool groundLike =
            horizontal >= 1.5f && sy > 0.0f &&
            ((o.shape == "cube" || o.shape == "box" ||
              o.shape == "ground" || o.shape == "plane")
                ? sy <= horizontal * 0.30f
                : sy <= horizontal * 0.10f);

        // Ground/plane primitives are receivers of light, not vertical
        // occluders. Keeping them out of the shadow-ray list prevents the
        // thickness of a gameplay floor mesh from falsely blocking rays that
        // terminate on the same floor.
        if (o.name == receiverName || o.isLight || groundLike)
            continue;

        const float dx = o.x-rx, dy = o.y-ry, dz = o.z-rz;
        occluders.push_back({dx*dx+dy*dy+dz*dz, o});
    }
    std::sort(occluders.begin(), occluders.end(),
              [](const OccluderCandidate &a, const OccluderCandidate &b)
              { return a.distance2 < b.distance2; });

    // Exact analytic collider packets: center+shape id, size, and rotation.
    std::array<float, TF_MAX_DYNAMIC_OCCLUDERS * 4> occluderData{};
    std::array<float, TF_MAX_DYNAMIC_OCCLUDERS * 4> occluderSizeData{};
    std::array<float, TF_MAX_DYNAMIC_OCCLUDERS * 4> occluderRotationData{};
    std::array<float, TF_MAX_DYNAMIC_OCCLUDERS * 4> occluderSoftAData{};
    std::array<float, TF_MAX_DYNAMIC_OCCLUDERS * 4> occluderSoftBData{};
    std::array<float, TF_MAX_DYNAMIC_OCCLUDERS * 4> occluderSoftCData{};
    std::array<float, TF_MAX_DYNAMIC_OCCLUDERS * 4> occluderSoftDData{};
    std::array<float, TF_MAX_DYNAMIC_OCCLUDERS * 4> occluderTintData{};
    for (int t = 0; t < TF_MAX_DYNAMIC_OCCLUDERS; ++t)
    {
        occluderTintData[t*4+0] = 1.0f;
        occluderTintData[t*4+1] = 1.0f;
        occluderTintData[t*4+2] = 1.0f;
    }
    float groundY = 0.0f;
    bool groundValid = false;
    std::array<float, 3> groundColor = {0.32f, 0.36f, 0.26f};
    float bestGroundDist2 = std::numeric_limits<float>::max();

    auto isGroundLike = [](const TFTrackedObject &o)
    {
        const float sx = std::fabs(o.scaleX);
        const float sy = std::fabs(o.scaleY);
        const float sz = std::fabs(o.scaleZ);
        const float horizontal = std::max(sx, sz);
        if (horizontal < 1.5f || sy <= 0.0f)
            return false;
        if (o.shape == "cube" || o.shape == "box" || o.shape == "ground" || o.shape == "plane")
            return sy <= horizontal * 0.30f;
        return sy <= horizontal * 0.10f;
    };

    const int occluderCount = std::min<int>(TF_MAX_DYNAMIC_OCCLUDERS, static_cast<int>(occluders.size()));
    for (int i = 0; i < occluderCount; ++i)
    {
        const TFTrackedObject &o = occluders[static_cast<size_t>(i)].state;
        const bool groundLike = isGroundLike(o);

        // Match the actual primitive dimensions used by the optimized renderer.
        float hx=1.2f*std::fabs(o.scaleX);
        float hy=1.2f*std::fabs(o.scaleY);
        float hz=1.2f*std::fabs(o.scaleZ);
        int shapeId=5;
        if (o.shape == "cube" || o.shape == "box" || o.shape == "ground" || o.shape == "plane")
        { shapeId=0; hx=1.2f*std::fabs(o.scaleX); hy=1.2f*std::fabs(o.scaleY); hz=1.2f*std::fabs(o.scaleZ); }
        else if (o.shape == "sphere")
        { shapeId=1; hx=1.5f*std::fabs(o.scaleX); hy=1.5f*std::fabs(o.scaleY); hz=1.5f*std::fabs(o.scaleZ); }
        else if (o.shape == "cylinder")
        { shapeId=2; hx=1.5f*std::fabs(o.scaleX); hy=1.5f*std::fabs(o.scaleY); hz=1.5f*std::fabs(o.scaleZ); }
        else if (o.shape == "capsule")
        { shapeId=3; hx=1.15f*std::fabs(o.scaleX); hy=2.0f*std::fabs(o.scaleY); hz=1.15f*std::fabs(o.scaleZ); }
        else if (o.shape == "cone")
        { shapeId=4; hx=1.5f*std::fabs(o.scaleX); hy=1.5f*std::fabs(o.scaleY); hz=1.5f*std::fabs(o.scaleZ); }
        else if (o.shape == "crystal")
        { shapeId=5; hx=1.15f*std::fabs(o.scaleX); hy=1.90f*std::fabs(o.scaleY); hz=1.15f*std::fabs(o.scaleZ); }
        else if (o.shape == "diamond")
        { shapeId=5; hx=1.25f*std::fabs(o.scaleX); hy=1.90f*std::fabs(o.scaleY); hz=1.25f*std::fabs(o.scaleZ); }
        else if (o.shape == "prism")
        { shapeId=5; hx=1.35f*std::fabs(o.scaleX); hy=1.45f*std::fabs(o.scaleY); hz=1.35f*std::fabs(o.scaleZ); }
        else if (o.shape == "rock")
        { shapeId=5; hx=1.25f*std::fabs(o.scaleX); hy=1.15f*std::fabs(o.scaleY); hz=1.10f*std::fabs(o.scaleZ); }
        else
        { shapeId=5; hx=1.30f*std::fabs(o.scaleX); hy=1.80f*std::fabs(o.scaleY); hz=1.30f*std::fabs(o.scaleZ); }

        occluderData[i*4+0] = o.x;
        occluderData[i*4+1] = o.y;
        occluderData[i*4+2] = o.z;
        occluderData[i*4+3] = static_cast<float>(shapeId);
        occluderSizeData[i*4+0] = hx;
        occluderSizeData[i*4+1] = hy;
        occluderSizeData[i*4+2] = hz;
        // O shader le o formato (0 caixa, 1 esfera, 2 cilindro, 3 capsula, 4 cone)
        // em Sizes[i].w; antes ficava 0 e todo oclusor virava caixa.
        occluderSizeData[i*4+3] = static_cast<float>(shapeId);
        occluderRotationData[i*4+0] = o.rotationX * 0.01745329251994329577f;
        occluderRotationData[i*4+1] = o.rotationY * 0.01745329251994329577f;
        occluderRotationData[i*4+2] = o.rotationZ * 0.01745329251994329577f;
        // Refractive material id: light is bent through these bodies, not blocked.
        float refractiveMaterial = 0.0f;
        if (o.shape == "crystal") refractiveMaterial = 1.0f;
        else if (o.shape == "diamond") refractiveMaterial = 2.0f;
        else if (o.shape == "prism") refractiveMaterial = 3.0f;
        else
        {
            auto seenIt = gTfSoftRefractiveSeen.find(o.name);
            if (seenIt != gTfSoftRefractiveSeen.end() &&
                tfNowSeconds() - seenIt->second < 1.0)
                refractiveMaterial = 4.0f;
        }
        occluderRotationData[i*4+3] = refractiveMaterial;

        // Mirror the host's live soft-body solver state into a compact packet.
        // `type == 1` is TFBodyType::Soft in the host enum. If the host map is
        // unavailable, the packets stay zero and rigid shadow logic remains.
        if (gTfGravityBodiesAddress)
        {
            auto *gravityMap = static_cast<TFCompatGravityMap *>(gTfGravityBodiesAddress);
            auto bodyIt = gravityMap->find(o.name);
            if (bodyIt != gravityMap->end() && bodyIt->second.type == 1)
            {
                const TFCompatGravityBody &body = bodyIt->second;
                const float fluid = std::max(0.0f, std::min(1.0f, body.softness / 10.0f));
                occluderSoftAData[i*4+0] = std::max(0.0f, std::min(0.96f, body.compression));
                occluderSoftAData[i*4+1] = std::max(0.0f, std::min(0.80f, body.stretch));
                occluderSoftAData[i*4+2] = std::max(-1.05f, std::min(1.05f, body.bendX));
                occluderSoftAData[i*4+3] = std::max(-1.05f, std::min(1.05f, body.bendZ));
                occluderSoftBData[i*4+0] = std::max(-0.80f, std::min(0.80f, body.flowX));
                occluderSoftBData[i*4+1] = std::max(-0.80f, std::min(0.80f, body.flowZ));
                occluderSoftBData[i*4+2] = std::max(0.0f, std::min(0.90f, body.liquidSpread));
                occluderSoftBData[i*4+3] = std::max(0.0f, std::min(1.0f, body.impactPulse));
                occluderSoftCData[i*4+0] = fluid;
                occluderSoftCData[i*4+1] = 1.0f; // authoritative TFBodyType::Soft
                occluderSoftCData[i*4+2] = std::max(0.0f, std::min(1.0f, body.contactStrength));
                occluderSoftCData[i*4+3] = static_cast<float>(body.fluidWaveTime);

                // Contact point is stored in world space by the host. Convert it
                // to the current object's local shadow space so the deformation
                // follows the exact support location and moves with the body.
                occluderSoftDData[i*4+0] = (body.contactPoint.x - o.x) / std::max(hx, 0.0001f);
                occluderSoftDData[i*4+1] = (body.contactPoint.y - o.y) / std::max(hy, 0.0001f);
                occluderSoftDData[i*4+2] = (body.contactPoint.z - o.z) / std::max(hz, 0.0001f);
                occluderSoftDData[i*4+3] =
                    (std::fabs(body.contactNormal.y) > 0.55f && body.contactStrength > 0.01f)
                        ? 1.0f : 0.0f;
            }
        }

        if (refractiveMaterial > 0.5f)
        {
            // Luminance-normalised tint: the light keeps the crystal's hue but
            // never loses brightness. Water stays almost neutral.
            const auto tc = tfColorLinearFromName(o.color);
            const float lum = std::max(0.0001f, 0.2126f*tc[0] + 0.7152f*tc[1] + 0.0722f*tc[2]);
            const float strength = (refractiveMaterial > 3.5f) ? 0.15f : 0.55f;
            float tr = 1.0f + strength * (tc[0]/lum - 1.0f);
            float tg = 1.0f + strength * (tc[1]/lum - 1.0f);
            float tb = 1.0f + strength * (tc[2]/lum - 1.0f);
            tr = std::max(0.25f, std::min(2.5f, tr));
            tg = std::max(0.25f, std::min(2.5f, tg));
            tb = std::max(0.25f, std::min(2.5f, tb));
            const float tl = std::max(0.0001f, 0.2126f*tr + 0.7152f*tg + 0.0722f*tb);
            occluderTintData[i*4+0] = tr / tl;
            occluderTintData[i*4+1] = tg / tl;
            occluderTintData[i*4+2] = tb / tl;
        }

        if (groundLike)
        {
            const float dx = o.x-rx, dy = o.y-ry, dz = o.z-rz;
            const float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < bestGroundDist2)
            {
                bestGroundDist2 = d2;
                groundValid = true;
                groundY = o.y + std::fabs(o.scaleY) * 0.5f;
                groundColor = tfColorLinearFromName(o.color);
            }
        }
    }

    const float swapShading =
        (receiverIt != gTfTrackedObjects.end() && isGroundLike(receiverIt->second))
            ? 0.0f : 1.0f;

    int previous = 0;
    tfGLGetIntegerv(TF_GL_CURRENT_PROGRAM, &previous);
    tfGLUseProgram(gTfAdvancedProgram);
    if (gTfSwapShadingLocation >= 0)
        tfGLUniform1f(gTfSwapShadingLocation, swapShading);
    tfGLUniform1f(gTfDynamicLightCountLocation, static_cast<float>(lightCount));
    tfGLUniform4fv(gTfDynamicLightsLocation, TF_MAX_DYNAMIC_LIGHTS, lightData.data());
    tfGLUniform4fv(gTfDynamicColorsLocation, TF_MAX_DYNAMIC_LIGHTS, colorData.data());
    tfGLUniform4fv(gTfDynamicLightParamsLocation, TF_MAX_DYNAMIC_LIGHTS, lightParamsData.data());
    tfGLUniform1f(gTfDynamicOccluderCountLocation, static_cast<float>(occluderCount));
    tfGLUniform4fv(gTfDynamicOccludersLocation, TF_MAX_DYNAMIC_OCCLUDERS, occluderData.data());
    tfGLUniform4fv(gTfDynamicOccluderSizesLocation, TF_MAX_DYNAMIC_OCCLUDERS, occluderSizeData.data());
    tfGLUniform4fv(gTfDynamicOccluderRotationsLocation, TF_MAX_DYNAMIC_OCCLUDERS, occluderRotationData.data());
    if (gTfDynamicOccluderSoftALocation >= 0)
        tfGLUniform4fv(gTfDynamicOccluderSoftALocation, TF_MAX_DYNAMIC_OCCLUDERS, occluderSoftAData.data());
    if (gTfDynamicOccluderSoftBLocation >= 0)
        tfGLUniform4fv(gTfDynamicOccluderSoftBLocation, TF_MAX_DYNAMIC_OCCLUDERS, occluderSoftBData.data());
    if (gTfDynamicOccluderSoftCLocation >= 0)
        tfGLUniform4fv(gTfDynamicOccluderSoftCLocation, TF_MAX_DYNAMIC_OCCLUDERS, occluderSoftCData.data());
    if (gTfDynamicOccluderSoftDLocation >= 0)
        tfGLUniform4fv(gTfDynamicOccluderSoftDLocation, TF_MAX_DYNAMIC_OCCLUDERS, occluderSoftDData.data());
    if (gTfDynamicOccluderTintsLocation >= 0)
        tfGLUniform4fv(gTfDynamicOccluderTintsLocation, TF_MAX_DYNAMIC_OCCLUDERS, occluderTintData.data());
    tfGLUniform1f(gTfGroundYLocation, groundY);
    tfGLUniform1f(gTfGroundValidLocation, groundValid ? 1.0f : 0.0f);
    const float groundColor4[4] = {groundColor[0], groundColor[1], groundColor[2], 1.0f};
    tfGLUniform4fv(gTfGroundColorLocation, 1, groundColor4);
    tfGLUseProgram(static_cast<unsigned int>(std::max(0, previous)));
    return true;
}

static void tfUploadViewPosition(const void *cameraPtr)
{
    if (!cameraPtr || !tfGLUniform3fv || !tfGLUseProgram || !tfGLGetIntegerv ||
        gTfAdvancedProgram == 0 || gTfViewPosLocation < 0)
        return;

    const TFCamera3DCompat *camera =
        static_cast<const TFCamera3DCompat *>(cameraPtr);
    const float viewPosition[3] = {camera->px, camera->py, camera->pz};

    int previous = 0;
    tfGLGetIntegerv(TF_GL_CURRENT_PROGRAM, &previous);
    tfGLUseProgram(gTfAdvancedProgram);
    tfGLUniform3fv(gTfViewPosLocation, 1, viewPosition);
    tfGLUseProgram(static_cast<unsigned int>(std::max(0, previous)));
}

static bool tfIsLightObject(const void *objectPtr, float &strength)
{
    strength = 0.0f;
    if (!objectPtr)
        return false;
    const TFCompatObject3DView *view =
        static_cast<const TFCompatObject3DView *>(objectPtr);
    auto it = gTfObjectLights.find(view->name);
    if (it == gTfObjectLights.end())
        return false;
    strength = it->second.strength;
    return strength > 0.0f;
}


static bool tfExecuteLineLightProxy(std::string line)
{
    float strength = 0.0f;
    std::string rewritten;
    std::string objectName;

    if (tfParseLightPrefix(line, strength, rewritten, objectName) &&
        gTfExecuteLineOriginal)
    {
        const bool existedBefore =
            gApi.object3DExists && gApi.object3DExists(objectName.c_str());
        const bool result = gTfExecuteLineOriginal(rewritten);
        const bool existsAfter =
            gApi.object3DExists && gApi.object3DExists(objectName.c_str());

        if (!existedBefore && existsAfter)
        {
            TFObjectLightState &state = gTfObjectLights[objectName];
            state.strength = strength;
            // Advanced shadows are controlled globally by `advanced light`.
            // Keep this field false so the light declaration never opts in by itself.
            state.advanced = false;
        }

        return result;
    }

    bool result = false;
    if (gTfExecuteLineOriginal)
        result = gTfExecuteLineOriginal(line);

    // Key/direct movement actions are executed through executeLine(). Apply
    // the same delta to our light cache immediately, so a light moved by
    // input can change the shadow field on that very frame even when its
    // object appears later in the draw order.
    const std::string text = trim(line);
    if (!text.empty() && text.front() == '"')
    {
        const size_t closeName = text.find('"', 1);
        if (closeName != std::string::npos)
        {
            const std::string target = text.substr(1, closeName - 1);
            const std::string action = trim(text.substr(closeName + 1));
            if (gTfObjectLights.find(target) != gTfObjectLights.end() &&
                action.rfind("add moviment", 0) == 0)
            {
                size_t open = action.find('(', 12);
                size_t close = action.rfind(')');
                if (open != std::string::npos && close != std::string::npos && close > open)
                {
                    std::string body = action.substr(open + 1, close - open - 1);
                    std::stringstream ss(body);
                    std::string ax, ay, az;
                    if (std::getline(ss, ax, ',') && std::getline(ss, ay, ',') && std::getline(ss, az, ','))
                    {
                        char *e0=nullptr, *e1=nullptr, *e2=nullptr;
                        const float dx=std::strtof(trim(ax).c_str(), &e0);
                        const float dy=std::strtof(trim(ay).c_str(), &e1);
                        const float dz=std::strtof(trim(az).c_str(), &e2);
                        if (e0 && e1 && e2 && *e0=='\0' && *e1=='\0' && *e2=='\0')
                        {
                            auto it = gTfTrackedObjects.find(target);
                            if (it != gTfTrackedObjects.end())
                            {
                                it->second.x += dx;
                                it->second.y += dy;
                                it->second.z += dz;
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}

static void tfDrawObjectSingleLightProxy(const void *objectPtr,
                                         const void *cameraPtr,
                                         int screenWidth,
                                         int screenHeight,
                                         float alpha)
{
    if (!gTfDrawObjectSingleOriginal)
        return;

    // The first draw of a scene may be the first point at which the GL context
    // exists. Build the shader before writing the per-object uniform so the
    // `light()` object also lights correctly on its first frame.
    if (!gTfAdvancedShaderBuilt)
        tfBuildAdvancedSurfaceShader();

    // Synchronize all live scene objects before every draw as a fallback too.
    // The once-per-frame collision hook already does this earlier, so this is
    // cheap in the common path and protects hosts with unusual frame order.
    tfSyncLiveObjectsFromHost();

    // Keep the current draw context available to the optical-parameter hook.
    // The host calls tfSetOpticalParams() synchronously immediately before its
    // DrawMesh(), which is the ideal point to capture the opaque framebuffer.
    gTfOpticalCurrentScreenWidth = screenWidth;
    gTfOpticalCurrentScreenHeight = screenHeight;
    gTfOpticalCurrentCameraPtr = cameraPtr;

    // The host copies tfOpticalShader into its Material before it calls
    // tfSetOpticalParams(). Ensure our physical shader has already replaced the
    // host program, otherwise the local Material would still contain the old
    // optical shader for the first water/crystal draw of the frame.
    tfEnsurePhysicalOpticalShader();

    // A normal object resets the emissive uniform to zero; light() objects
    // receive their requested intensity immediately before their draw.
    tfTrackDrawnObject(objectPtr);

    float strength = 0.0f;
    if (!tfIsLightObject(objectPtr, strength))
        strength = 0.0f;

    const TFCompatObject3DView *view =
        static_cast<const TFCompatObject3DView *>(objectPtr);

    // Crystal/prism/diamond use the IDE's optical pass. Replace its GLSL with
    // the physically-based version before the host executes the actual draw.
    // Soft-body motion is preserved through the host's `softFluid` uniform.
    if (tfIsCrystalOpticalShape(view->shape))
    {
        tfEnsurePhysicalOpticalShader();
        if (gTfPhysicalOpticalShaderInstalled)
        {
            float opticalThickness = 1.40f;
            if (view->shape == "diamond") opticalThickness = 1.05f;
            else if (view->shape == "prism") opticalThickness = 1.65f;
            tfSetPhysicalOpticalScreenUniforms(
                gTfOpticalShaderAddress
                    ? [&]() -> unsigned int
                      { TFCompatShaderPublic sh{}; std::memcpy(&sh, gTfOpticalShaderAddress, sizeof(sh)); return sh.id; }()
                    : 0u,
                objectPtr, cameraPtr, screenWidth, screenHeight,
                view->shape, opticalThickness);
        }
    }

    tfUploadViewPosition(cameraPtr);

    if (!tfSetObjectLightUniform(strength) && strength > 0.0f && gApi.log)
        gApi.log("shaders: unable to update light() intensity uniform.");

    if (!tfUploadDynamicLighting(view->name) && gApi.log)
        gApi.log("shaders: dynamic light/shadow uniforms unavailable.");

    // New realistic blur is implemented here, entirely inside shaders.cpp.
    // Legacy `motion trail(...)` is still rendered by the host's original
    // motionBlur path and therefore remains available unchanged.
    const TFCompatHostObject3D *hostObject =
        static_cast<const TFCompatHostObject3D *>(objectPtr);
    if (hostObject)
        tfDrawRealMotionBlur(*hostObject, cameraPtr, screenWidth, screenHeight, alpha);

    gTfDrawObjectSingleOriginal(objectPtr, cameraPtr,
                                screenWidth, screenHeight, alpha);
}

static void tfInstallLightRuntimeHooks()
{
    tfInstallPhysicalOpticalBridge();

    if (gTfExecuteLineHookInstalled &&
        gTfDrawObjectHookInstalled &&
        gTfResolvePhysicalCollisionsHookInstalled)
        return;

    gTfExecuteLineAddress =
        tfFindELFSymbolInMainExecutable(
            "_Z11executeLineNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
    gTfDrawObjectSingleAddress =
        tfFindELFSymbolInMainExecutable(
            "_Z29tfDrawOptimized3DObjectSingleRK10TFObject3DRK8Camera3Diif");
    gTfResolvePhysicalCollisionsAddress =
        tfFindELFSymbolInMainExecutable("_Z25resolvePhysicalCollisionsv");
    gTfObjects3DAddress =
        tfFindELFSymbolInMainExecutable("_Z8objects3D");
    gTfGravityBodiesAddress =
        tfFindELFSymbolInMainExecutable("_Z13gravityBodiesB5cxx11");

    if (!gTfExecuteLineHookInstalled && gTfExecuteLineAddress)
    {
        void *trampoline = nullptr;
        if (tfInstallTrampolineHook(
                gTfExecuteLineAddress,
                reinterpret_cast<void *>(&tfExecuteLineLightProxy),
                &trampoline))
        {
            gTfExecuteLineTrampoline = trampoline;
            gTfExecuteLineOriginal =
                reinterpret_cast<TFExecuteLineFunction>(trampoline);
            gTfExecuteLineHookInstalled = true;
        }
    }

    if (!gTfDrawObjectHookInstalled && gTfDrawObjectSingleAddress)
    {
        void *trampoline = nullptr;
        if (tfInstallTrampolineHook(
                gTfDrawObjectSingleAddress,
                reinterpret_cast<void *>(&tfDrawObjectSingleLightProxy),
                &trampoline))
        {
            gTfDrawObjectSingleTrampoline = trampoline;
            gTfDrawObjectSingleOriginal =
                reinterpret_cast<TFDrawObjectSingleFunction>(trampoline);
            gTfDrawObjectHookInstalled = true;
        }
    }

    if (!gTfResolvePhysicalCollisionsHookInstalled &&
        gTfResolvePhysicalCollisionsAddress &&
        gTfObjects3DAddress)
    {
        void *trampoline = nullptr;
        if (tfInstallTrampolineHook(
                gTfResolvePhysicalCollisionsAddress,
                reinterpret_cast<void *>(&tfResolvePhysicalCollisionsLightProxy),
                &trampoline))
        {
            gTfResolvePhysicalCollisionsTrampoline = trampoline;
            gTfResolvePhysicalCollisionsOriginal =
                reinterpret_cast<TFResolvePhysicalCollisionsFunction>(trampoline);
            gTfResolvePhysicalCollisionsHookInstalled = true;
        }
    }

    if (gApi.log)
    {
        if (gTfExecuteLineHookInstalled)
            gApi.log("shaders: light(n) prefix parser installed.");
        else
            gApi.log("shaders: light(n) prefix parser unavailable on this host.");
        if (gTfDrawObjectHookInstalled)
            gApi.log("shaders: per-object light intensity hook installed.");
        else
            gApi.log("shaders: per-object light intensity hook unavailable on this host.");
        if (gTfResolvePhysicalCollisionsHookInstalled)
            gApi.log("shaders: realtime light position sync installed.");
        else
            gApi.log("shaders: realtime light position sync unavailable on this host.");
    }
}

static void tfInstallRendererLightingBridge()
{
    if (gTfRendererHookInstalled)
        return;

    gTfSurfaceShaderAddress =
        tfFindELFSymbolInMainExecutable("tfSurfaceShader");
    gTfSurfaceShaderReadyAddress =
        tfFindELFSymbolInMainExecutable("tfSurfaceShaderReady");
    gTfPrepareSurfaceShaderAddress =
        tfFindELFSymbolInMainExecutable("_Z22tfPrepareSurfaceShaderv");

    if (!gTfSurfaceShaderAddress || !gTfSurfaceShaderReadyAddress ||
        !gTfPrepareSurfaceShaderAddress)
    {
        if (gApi.log)
        {
            gApi.log(
                "shaders: advanced lighting bridge unavailable; "
                "host renderer symbols were not found."
            );
        }
        tfInstallLightRuntimeHooks();
        return;
    }

    if (tfPatchAbsoluteJump(
            gTfPrepareSurfaceShaderAddress,
            reinterpret_cast<void *>(&tfShadersAdvancedPrepareSurfaceShader)))
    {
        gTfRendererHookInstalled = true;
        if (gApi.log)
            gApi.log("shaders: advanced pre-baked PBR lighting installed.");
    }
    else if (gApi.log)
    {
        gApi.log("shaders: could not install the advanced lighting bridge.");
    }

    // These parser/render hooks are independent of the surface-shader hook.
    // They allow `light(n)` to be attached to individual objects.
    tfInstallLightRuntimeHooks();
}

string trim(const string &text)
{
    size_t first = 0;
    while (first < text.size() &&
           std::isspace(static_cast<unsigned char>(text[first])))
        ++first;

    size_t last = text.size();
    while (last > first &&
           std::isspace(static_cast<unsigned char>(text[last - 1])))
        --last;

    return text.substr(first, last - first);
}

bool isWordBoundary(const string &text, size_t pos, size_t length)
{
    const bool left =
        pos == 0 ||
        !(std::isalnum(static_cast<unsigned char>(text[pos - 1])) ||
          text[pos - 1] == '_');

    const size_t rightPos = pos + length;
    const bool right =
        rightPos >= text.size() ||
        !(std::isalnum(static_cast<unsigned char>(text[rightPos])) ||
          text[rightPos] == '_');

    return left && right;
}

size_t findOutsideQuotes(const string &text,
                         const string &needle,
                         size_t start = 0)
{
    bool inQuotes = false;

    for (size_t i = start; i + needle.size() <= text.size(); ++i)
    {
        if (text[i] == '"')
        {
            inQuotes = !inQuotes;
            continue;
        }

        if (inQuotes)
            continue;

        if (text.compare(i, needle.size(), needle) == 0 &&
            isWordBoundary(text, i, needle.size()))
        {
            return i;
        }
    }

    return string::npos;
}

bool parseNumberInParentheses(const string &text,
                              size_t openParen,
                              float &value,
                              size_t &closeParen)
{
    if (openParen >= text.size() || text[openParen] != '(')
        return false;

    closeParen = text.find(')', openParen + 1);
    if (closeParen == string::npos)
        return false;

    string inside = trim(
        text.substr(openParen + 1, closeParen - openParen - 1));

    if (inside.empty())
        return false;

    char *end = nullptr;
    const float parsed = std::strtof(inside.c_str(), &end);

    if (!end || end == inside.c_str())
        return false;

    while (*end != '\0')
    {
        if (!std::isspace(static_cast<unsigned char>(*end)))
            return false;
        ++end;
    }

    if (!std::isfinite(parsed))
        return false;

    value = std::max(0.0f, std::min(10.0f, parsed));
    return true;
}

enum class TFMotionEffectKind
{
    None,
    RealBlur,
    Trail
};

static bool parseMotionEffectKeyword(const string &text,
                                     size_t pos,
                                     const string &keyword,
                                     float &value,
                                     size_t &end)
{
    size_t cursor = pos + keyword.size();
    while (cursor < text.size() &&
           std::isspace(static_cast<unsigned char>(text[cursor])))
        ++cursor;

    size_t close = string::npos;
    if (!parseNumberInParentheses(text, cursor, value, close))
        return false;

    end = close + 1;
    for (size_t i = end; i < text.size(); ++i)
    {
        if (!std::isspace(static_cast<unsigned char>(text[i])))
            return false;
    }

    return true;
}

bool extractMotionEffect(const string &text,
                         TFMotionEffectKind &kind,
                         float &value,
                         size_t &start,
                         size_t &end)
{
    struct Candidate
    {
        const char *keyword;
        TFMotionEffectKind kind;
        size_t pos;
    };

    const Candidate candidates[] =
    {
        {"motion blur",  TFMotionEffectKind::RealBlur, findOutsideQuotes(text, "motion blur")},
        {"motion trail", TFMotionEffectKind::Trail,    findOutsideQuotes(text, "motion trail")},
        {"trail",        TFMotionEffectKind::Trail,    findOutsideQuotes(text, "trail")}
    };

    size_t bestPos = string::npos;
    const char *bestKeyword = nullptr;
    TFMotionEffectKind bestKind = TFMotionEffectKind::None;

    for (const Candidate &candidate : candidates)
    {
        if (candidate.pos == string::npos)
            continue;

        // `motion trail(...)` also contains `trail(...)`; prefer the longer
        // spelling when both point at the same suffix.
        if (candidate.pos < bestPos ||
            (candidate.pos == bestPos &&
             std::strlen(candidate.keyword) > std::strlen(bestKeyword)))
        {
            bestPos = candidate.pos;
            bestKeyword = candidate.keyword;
            bestKind = candidate.kind;
        }
    }

    if (!bestKeyword)
        return false;

    float parsed = 0.0f;
    size_t parsedEnd = string::npos;
    if (!parseMotionEffectKeyword(text, bestPos, bestKeyword, parsed, parsedEnd))
        return false;

    kind = bestKind;
    value = parsed;
    start = bestPos;
    end = parsedEnd;
    return true;
}

bool extractMotionBlur(const string &text, float &value,
                       size_t &start, size_t &end)
{
    TFMotionEffectKind kind = TFMotionEffectKind::None;
    if (!extractMotionEffect(text, kind, value, start, end))
        return false;
    return kind == TFMotionEffectKind::RealBlur;
}

static bool parseNamedMotionEffectCommand(const string &line,
                                          string &objectName,
                                          TFMotionEffectKind &kind,
                                          float &value)
{
    const string text = trim(line);
    if (text.size() < 2 || text.front() != '"')
        return false;

    const size_t closeQuote = text.find('"', 1);
    if (closeQuote == string::npos)
        return false;

    objectName = text.substr(1, closeQuote - 1);
    if (objectName.empty())
        return false;

    size_t cursor = closeQuote + 1;
    while (cursor < text.size() &&
           std::isspace(static_cast<unsigned char>(text[cursor])))
        ++cursor;

    struct Keyword { const char *text; TFMotionEffectKind kind; };
    const Keyword keywords[] =
    {
        {"motion blur",  TFMotionEffectKind::RealBlur},
        {"motion trail", TFMotionEffectKind::Trail},
        {"trail",        TFMotionEffectKind::Trail}
    };

    const char *matched = nullptr;
    kind = TFMotionEffectKind::None;

    for (const Keyword &keyword : keywords)
    {
        const size_t n = std::strlen(keyword.text);
        if (cursor + n <= text.size() &&
            text.compare(cursor, n, keyword.text) == 0 &&
            isWordBoundary(text, cursor, n))
        {
            matched = keyword.text;
            kind = keyword.kind;
            break;
        }
    }

    if (!matched)
        return false;

    cursor += std::strlen(matched);
    while (cursor < text.size() &&
           std::isspace(static_cast<unsigned char>(text[cursor])))
        ++cursor;

    size_t closeParen = string::npos;
    if (!parseNumberInParentheses(text, cursor, value, closeParen))
        return false;

    for (size_t i = closeParen + 1; i < text.size(); ++i)
    {
        if (!std::isspace(static_cast<unsigned char>(text[i])))
            return false;
    }

    return true;
}

static bool parseSimpleMotionEffectCommand(const string &line,
                                           TFMotionEffectKind &kind,
                                           float &value)
{
    const string text = trim(line);
    struct Keyword { const char *text; TFMotionEffectKind kind; };
    const Keyword keywords[] =
    {
        {"motion blur",  TFMotionEffectKind::RealBlur},
        {"motion trail", TFMotionEffectKind::Trail},
        {"trail",        TFMotionEffectKind::Trail}
    };

    for (const Keyword &keyword : keywords)
    {
        const string prefix(keyword.text);
        if (text.rfind(prefix, 0) != 0)
            continue;
        if (!isWordBoundary(text, 0, prefix.size()))
            continue;

        size_t cursor = prefix.size();
        while (cursor < text.size() &&
               std::isspace(static_cast<unsigned char>(text[cursor])))
            ++cursor;

        size_t closeParen = string::npos;
        if (!parseNumberInParentheses(text, cursor, value, closeParen))
            return false;

        for (size_t i = closeParen + 1; i < text.size(); ++i)
        {
            if (!std::isspace(static_cast<unsigned char>(text[i])))
                return false;
        }

        kind = keyword.kind;
        return true;
    }

    return false;
}

static bool tfSetRealMotionBlur(const string &objectName, float value)
{
    if (!gApi.object3DExists || !gApi.object3DExists(objectName.c_str()))
        return false;

    value = std::max(0.0f, std::min(10.0f, value));
    TFRealMotionBlurState &state = gTfRealMotionBlur[objectName];
    state.strength = value;
    state.previousInitialized = false;

    // Explicitly disable the host's legacy trail for the new realistic blur.
    // The host field remains available for `motion trail(...)`.
    if (gApi.setObjectMotionBlur)
        gApi.setObjectMotionBlur(objectName.c_str(), 0.0f);

    return true;
}

static bool tfSetMotionTrail(const string &objectName, float value)
{
    if (!gApi.setObjectMotionBlur ||
        !gApi.object3DExists ||
        !gApi.object3DExists(objectName.c_str()))
        return false;

    value = std::max(0.0f, std::min(10.0f, value));
    return gApi.setObjectMotionBlur(objectName.c_str(), value);
}

static void tfDrawRealMotionBlur(const TFCompatHostObject3D &object,
                                 const void *cameraPtr,
                                 int screenWidth,
                                 int screenHeight,
                                 float alpha)
{
    if (!gTfDrawObjectSingleOriginal || alpha < 0.999f || object.name.empty())
        return;

    auto stateIt = gTfRealMotionBlur.find(object.name);
    if (stateIt == gTfRealMotionBlur.end())
        return;

    TFRealMotionBlurState &state = stateIt->second;
    const float strength = std::max(0.0f, std::min(10.0f, state.strength));

    if (strength <= 0.001f)
    {
        state.previousX = object.x;
        state.previousY = object.y;
        state.previousZ = object.z;
        state.previousInitialized = true;
        return;
    }

    if (!state.previousInitialized)
    {
        state.previousX = object.x;
        state.previousY = object.y;
        state.previousZ = object.z;
        state.previousInitialized = true;
        return;
    }

    const float dx = object.x - state.previousX;
    const float dy = object.y - state.previousY;
    const float dz = object.z - state.previousZ;
    const float distance = std::sqrt(dx*dx + dy*dy + dz*dz);

    // Teleports/spawn corrections should not turn into a giant smear.
    const float teleportLimit = 6.0f + strength * 3.0f;
    if (distance <= 0.0015f || distance > teleportLimit)
    {
        state.previousX = object.x;
        state.previousY = object.y;
        state.previousZ = object.z;
        state.previousInitialized = true;
        return;
    }

    // Temporal supersampling: many closely spaced, alpha-weighted samples are
    // accumulated between the previous and current transforms. Unlike the old
    // trail mode, the samples are dense, softly weighted and kept close to the
    // motion path so the eye reads one continuous smear rather than distinct
    // transparent copies.
    int samples = 5 + static_cast<int>(std::round(strength * 1.15f));
    samples = std::max(5, std::min(17, samples));

    const float shutter = std::max(0.70f,
                                   std::min(3.15f, 0.70f + strength * 0.245f));
    const float totalGhostAlpha =
        std::max(0.10f, std::min(0.82f, 0.12f + strength * 0.062f));

    float weightSum = 0.0f;
    for (int i = 0; i < samples; ++i)
    {
        const float t = (static_cast<float>(i) + 0.5f) /
                        static_cast<float>(samples);
        const float age = 1.0f - t;
        const float weight = 0.35f + 0.65f * t * t;
        (void)age;
        weightSum += weight;
    }

    const void *camera = cameraPtr;

    for (int i = 0; i < samples; ++i)
    {
        const float t = (static_cast<float>(i) + 0.5f) /
                        static_cast<float>(samples);
        const float weight = 0.35f + 0.65f * t * t;

        // Use the full host object layout here, including rotations and the
        // legacy motion-blur fields. Passing the shorter view struct to the
        // host renderer would make it read garbage past the scale fields.
        TFCompatHostObject3D ghost = object;
        ghost.motionBlur = 0.0f;
        ghost.motionBlurPreviousInitialized = false;
        ghost.x = object.x - dx * (1.0f - t) * shutter;
        ghost.y = object.y - dy * (1.0f - t) * shutter;
        ghost.z = object.z - dz * (1.0f - t) * shutter;

        const float sampleAlpha =
            totalGhostAlpha * (weight / std::max(0.0001f, weightSum));

        if (sampleAlpha <= 0.001f)
            continue;

        gTfDrawObjectSingleOriginal(
            &ghost,
            camera,
            screenWidth,
            screenHeight,
            sampleAlpha);
    }

    state.previousX = object.x;
    state.previousY = object.y;
    state.previousZ = object.z;
    state.previousInitialized = true;
}

bool parseObjectBlurCommand(const string &line,
                            string &objectName,
                            float &value)
{
    string text = trim(line);
    if (text.empty() || text.front() != '"')
        return false;

    const size_t closeQuote = text.find('"', 1);
    if (closeQuote == string::npos)
        return false;

    objectName = text.substr(1, closeQuote - 1);

    size_t cursor = closeQuote + 1;
    while (cursor < text.size() &&
           std::isspace(static_cast<unsigned char>(text[cursor])))
        ++cursor;

    const string keyword = "blur";
    if (cursor + keyword.size() > text.size() ||
        text.compare(cursor, keyword.size(), keyword) != 0)
        return false;

    if (!isWordBoundary(text, cursor, keyword.size()))
        return false;

    cursor += keyword.size();
    while (cursor < text.size() &&
           std::isspace(static_cast<unsigned char>(text[cursor])))
        ++cursor;

    size_t closeParen = string::npos;
    if (!parseNumberInParentheses(text, cursor, value, closeParen))
        return false;

    for (size_t i = closeParen + 1; i < text.size(); ++i)
    {
        if (!std::isspace(static_cast<unsigned char>(text[i])))
            return false;
    }

    return !objectName.empty();
}

void logMessage(const char *message)
{
    if (gApi.log)
        gApi.log(message);
}

bool shadersSay3DOptionsHook(const char *options,
                             char *output,
                             size_t outputCapacity)
{
    if (!options || !output || outputCapacity == 0)
        return false;

    string text(options);
    TFMotionEffectKind kind = TFMotionEffectKind::None;
    float value = 0.0f;
    size_t start = string::npos;
    size_t end = string::npos;

    if (!extractMotionEffect(text, kind, value, start, end))
        return false;

    string cleaned = trim(text.substr(0, start));
    std::strncpy(output, cleaned.c_str(), outputCapacity - 1);
    output[outputCapacity - 1] = '\0';
    return true;
}

bool shadersSay3DHook(const char *objectName,
                      const char *fullLine)
{
    if (!objectName || !fullLine)
        return false;

    // A new say 3d definition supersedes any stale library-only blur state
    // attached to the same object name.
    gTfLastCreatedObjectName = objectName;
    gTfRealMotionBlur.erase(objectName);
    gTfObjectLights.erase(objectName);

    string line(fullLine);
    TFMotionEffectKind kind = TFMotionEffectKind::None;
    float value = 0.0f;
    size_t start = string::npos;
    size_t end = string::npos;

    if (!extractMotionEffect(line, kind, value, start, end))
        return false;

    if (kind == TFMotionEffectKind::RealBlur)
    {
        if (!tfSetRealMotionBlur(objectName, value))
        {
            logMessage("shaders: object for motion blur was not found.");
            return false;
        }

        char message[192];
        std::snprintf(
            message, sizeof(message),
            "shaders: realistic motion blur \"%s\" = %.2f",
            objectName, value);
        logMessage(message);
        return true;
    }

    if (kind == TFMotionEffectKind::Trail)
    {
        if (!tfSetMotionTrail(objectName, value))
        {
            logMessage("shaders: object for motion trail was not found.");
            return false;
        }

        char message[192];
        std::snprintf(
            message, sizeof(message),
            "shaders: motion trail \"%s\" = %.2f",
            objectName, value);
        logMessage(message);
        return true;
    }

    return false;
}

static bool shadersAdvancedLightGlobalCommand(const char *line)
{
    if (!line)
        return false;

    const std::string text = trim(line);
    if (text == "advanced light")
    {
        gTfAdvancedLightingEnabled = true;
        logMessage("shaders: advanced light enabled globally; realistic dynamic shadows ON.");
        return true;
    }

    // Also accept `advanced light off` so scripts can explicitly return to the
    // cheap shadow-free light mode later in the same process. The requested
    // primary syntax remains simply `advanced light`.
    if (text == "advanced light off")
    {
        gTfAdvancedLightingEnabled = false;
        logMessage("shaders: advanced light disabled globally; dynamic shadows OFF.");
        return true;
    }

    return false;
}

bool shadersCommandHook(const char *line)
{
    if (!line)
        return false;

    if (shadersAdvancedLightGlobalCommand(line))
        return true;

    string text = trim(line);

    // `trail(value)` / `motion trail(value)` may be written after the last
    // `say 3d` object, without repeating its name.
    TFMotionEffectKind simpleKind = TFMotionEffectKind::None;
    float simpleValue = 0.0f;
    if (parseSimpleMotionEffectCommand(text, simpleKind, simpleValue))
    {
        if (gTfLastCreatedObjectName.empty())
        {
            logMessage("shaders: no previous 3D object for trail/blur command.");
            return true;
        }

        bool ok = false;
        if (simpleKind == TFMotionEffectKind::RealBlur)
            ok = tfSetRealMotionBlur(gTfLastCreatedObjectName, simpleValue);
        else if (simpleKind == TFMotionEffectKind::Trail)
            ok = tfSetMotionTrail(gTfLastCreatedObjectName, simpleValue);

        if (!ok)
            logMessage("shaders: last 3D object no longer exists.");
        return true;
    }

    // Named form is also supported for convenience: `"player" trail(5)`
    // and `"player" motion blur(5)`. The old `"player" blur(5)` command
    // remains unchanged below.
    if (text.size() >= 2 && text.front() == '"')
    {
        string objectName;
        TFMotionEffectKind namedKind = TFMotionEffectKind::None;
        float namedValue = 0.0f;
        if (parseNamedMotionEffectCommand(text, objectName, namedKind, namedValue))
        {
            bool ok = false;
            if (namedKind == TFMotionEffectKind::RealBlur)
                ok = tfSetRealMotionBlur(objectName, namedValue);
            else if (namedKind == TFMotionEffectKind::Trail)
                ok = tfSetMotionTrail(objectName, namedValue);

            if (!ok)
                logMessage("shaders: object not found for motion effect command.");
            return true;
        }
    }

    // Only claim a command when it really has the `"name" blur(...)` shape.
    if (text.size() < 2 || text.front() != '"')
        return false;

    const size_t blurPos = findOutsideQuotes(text, "blur");
    if (blurPos == string::npos)
        return false;

    string objectName;
    float value = 0.0f;
    if (!parseObjectBlurCommand(text, objectName, value))
    {
        logMessage("shaders: invalid blur syntax. Use: \"object\" blur(value)");
        return true;
    }

    if (!gApi.setObjectMotionBlur(objectName.c_str(), value))
    {
        logMessage("shaders: object not found for blur command.");
        return true;
    }

    return true;
}

bool shadersConditionHook(const char *condition, bool *handled)
{
    if (handled)
        *handled = false;

    if (!condition || !handled)
        return false;

    string text = trim(condition);
    if (text.size() < 2 || text.front() != '"')
        return false;

    if (findOutsideQuotes(text, "blur") == string::npos)
        return false;

    string objectName;
    float requested = 0.0f;

    if (!parseObjectBlurCommand(text, objectName, requested))
    {
        *handled = true;
        logMessage("shaders: invalid blur condition. Use: if \"object\" blur(value)");
        return false;
    }

    if (!gApi.getObjectMotionBlur || !gApi.object3DExists ||
        !gApi.object3DExists(objectName.c_str()))
    {
        *handled = true;
        return false;
    }

    const float current =
        gApi.getObjectMotionBlur(objectName.c_str());

    *handled = true;
    return current >= requested - 0.0001f;
}
} // namespace

extern "C" bool tfLibraryInit(const TFLibraryAPI *api)
{
    if (!api || api->version < 1 ||
        !api->registerSay3DOptionsHook ||
        !api->registerSay3DHook ||
        !api->registerCommandHook ||
        !api->registerConditionHook ||
        !api->setObjectMotionBlur ||
        !api->getObjectMotionBlur ||
        !api->object3DExists)
    {
        return false;
    }

    gApi = *api;
    gTfAdvancedLightingEnabled = false;

    tfInstallRendererLightingBridge();

    if (!gApi.registerSay3DOptionsHook(shadersSay3DOptionsHook))
        return false;

    if (!gApi.registerSay3DHook(shadersSay3DHook))
        return false;

    if (!gApi.registerCommandHook(shadersCommandHook))
        return false;

    if (!gApi.registerConditionHook(shadersConditionHook))
        return false;

    logMessage(
        "shaders loaded: realistic motion blur(value) + motion trail(value) + trail(value) + \"object\" blur(value) + light(value); use standalone advanced light for physically-based inverse-square lighting, soft penumbrae, contact shadows and weak colored one-bounce GI");
    return true;
}
