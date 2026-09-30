#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cctype>
#include <stdexcept>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <atomic>
#include <thread>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <sys/resource.h>
#include <unistd.h>
#include "raylib.h"
#include "rlgl.h"

#if defined(PLATFORM_ANDROID) || defined(__ANDROID__)
    #include <android/asset_manager.h>
    #include <android_native_app_glue.h>
    extern "C" struct android_app *GetAndroidApp(void);
#endif
 
using namespace std;
 
 
unordered_map<string, string> variables;
unordered_map<string, vector<string>> lists;
 
// ============================================================
// INPUT / NAMED OBJECT SYSTEM
// ============================================================
 
struct TFObject2D
{
    string name;
    string shape;
    string color;
    float x = 450.0f;
    float y = 350.0f;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    float rotation = 0.0f;
};
 
struct TFObject3D
{
    string name;
    string shape;
    string color;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    float scaleZ = 1.0f;
    float rotationX = 0.0f;
    float rotationY = 0.0f;
    float rotationZ = 0.0f;

    // Optional visual motion blur supplied by an external TigerFlash
    // library. 0 = disabled, 10 = strongest supported blur.
    float motionBlur = 0.0f;
    Vector3 motionBlurPreviousPosition = {0.0f, 0.0f, 0.0f};
    bool motionBlurPreviousInitialized = false;
};
 
enum class TFInputType
{
    Keyboard,
    Mouse,
    Gamepad,
    Axis,
    Touch,
    VR
};

enum class TFMotionSource
{
    None,
    Mouse,
    Joystick,
    Gamepad,
    Touch,
    VR
};
 
struct TFBinding
{
    TFInputType type = TFInputType::Keyboard;
    string control;
    string objectName;
    string action;
    bool isMovement = true;
    bool isRotation = false;
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    float rx = 0.0f;
    float ry = 0.0f;
    float rz = 0.0f;

    // Optional second input condition.
    // Example: key left_mouse + moviment up mouse "cam" rotate(-1,0,0)
    bool hasMotionCondition = false;
    TFMotionSource motionSource = TFMotionSource::None;
    string motionDirection;
    float motionThreshold = 0.15f;
    int deviceIndex = 0;
};
 
vector<TFObject2D> objects2D;
vector<TFObject3D> objects3D;

// Optional object memory: stores the original object created by say 2d / say 3d.
// delete removes only the active instance; add object restores this saved definition.
unordered_map<string, TFObject2D> objectMemory2D;
unordered_map<string, TFObject3D> objectMemory3D;

// Optional persistent key/value memory. It survives `stop` and also survives
// closing/restarting TigerFlash by using a small local text database.
unordered_map<string, string> tfPersistentMemory;
bool tfPersistentMemoryLoaded = false;
const string TF_MEMORY_FILE = "tigerflash_memory.db";

// ============================================================
// OPTIONAL USER FUNCTIONS / CONTROL FLOW
// ============================================================
struct TFFunction
{
    vector<string> parameters;
    vector<string> body;
};

unordered_map<string, TFFunction> tfFunctions;

unordered_set<string> tfImportedModules;

// ============================================================
// NATIVE BUILD SYSTEM
// `execute block()` builds a native executable for the host PC target.
// `apk block()` builds the same TigerFlash program as a signed Android APK
// for the ARM64 (arm64-v8a) phone target. Both directives are used AFTER
// `stop`.
// ============================================================
vector<string> tfLastExecutedScript;
string tfCurrentScriptPath;
string tfLauncherExecutablePath;
bool tfExecuteBlockArmed = false;
bool tfApkBlockArmed = false;
bool tfPendingSceneWindow = false;

static bool tfBuildExecutableBlock(const vector<string> &sourceScript,
                                   const string &scriptPath);
static bool tfBuildApkBlock(const vector<string> &sourceScript,
                            const string &scriptPath);

// ============================================================
// EXTERNAL TIGERFLASH LIBRARIES
// ============================================================
// `library <name>` loads library/<name>.cpp (compiled to a small .so cache)
// or an already-built library/<name>.so. The C ABI is intentionally small.
// The current implementation supports C++ source libraries; C/Python hooks
// can be added later without changing the TigerFlash language syntax.
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
    uint32_t version = 1;

    bool (*registerSay3DOptionsHook)(TFSay3DOptionsHook hook) = nullptr;
    bool (*registerSay3DHook)(TFSay3DHook hook) = nullptr;
    bool (*registerCommandHook)(TFLibraryCommandHook hook) = nullptr;
    bool (*registerConditionHook)(TFLibraryConditionHook hook) = nullptr;

    bool (*setObjectMotionBlur)(const char *objectName, float value) = nullptr;
    float (*getObjectMotionBlur)(const char *objectName) = nullptr;
    bool (*object3DExists)(const char *objectName) = nullptr;

    void (*log)(const char *message) = nullptr;
};

using TFLibraryInitFunction = bool (*)(const TFLibraryAPI *api);

vector<TFSay3DOptionsHook> tfLibrarySay3DOptionsHooks;
vector<TFSay3DHook> tfLibrarySay3DHooks;
vector<TFLibraryCommandHook> tfLibraryCommandHooks;
vector<TFLibraryConditionHook> tfLibraryConditionHooks;
unordered_set<string> tfLoadedLibraries;
vector<void *> tfLibraryHandles;

// Signals are used internally so `return`, `break` and `continue` can pass
// through nested if blocks without breaking the existing executeLine(bool) API.
enum class TFControlSignal
{
    None,
    Break,
    Continue,
    Return
};

TFControlSignal tfControlSignal = TFControlSignal::None;
string tfReturnValue;
int tfFunctionDepth = 0;
int tfLoopDepth = 0;

vector<TFBinding> inputBindings;
bool sceneHas2D = false;
bool sceneHas3D = false;

// =========================================================
// GUI / HUD (on-screen text and images)
// =========================================================
// gui text(<content>) shape(w,h,d) position(x,y,z) [as "name"]
// gui image "<path>" shape(w,h,d) position(x,y,z) [as "name"]
// Absolute paths may point to ANY existing Linux directory, including spaces.
//
// This is a screen-space overlay, independent of 2D/3D scene objects, so
// it draws on top of both "say 2d" and "say 3d" scenes.
struct TFGuiElement
{
    string name;
    bool isImage = false;
    bool isButton = false;
    string content; // text/image/button content
    float posX = 0.0f, posY = 0.0f, posZ = 0.0f;
    float shapeW = 0.0f, shapeH = 0.0f, shapeD = 0.0f;
    float rotation = 0.0f;
    float scaleX = 1.0f, scaleY = 1.0f;
    float velocityX = 0.0f, velocityY = 0.0f;
    float gravity = 0.0f;
    bool physicsEnabled = false;
    bool grounded = false;
    bool softBody = false;
    float softness = 10.0f;
    float softSquash = 0.0f;
    float softSpread = 0.0f;
    float softVelocity = 0.0f;
    string clickAction;
    bool hovered = false;
    Texture2D texture{};
    bool textureLoadAttempted = false;
    bool textureLoadFailed = false;
};

vector<TFGuiElement> guiElements;
bool sceneHasGui = false;

// GUI animation state. Unlike gui move/rotate/scale, which are immediate
// transforms, these persist and are advanced every frame.
enum class TFGuiAnimationType
{
    Move,
    Rotate,
    Scale
};

struct TFGuiAnimation
{
    string guiName;
    TFGuiAnimationType type = TFGuiAnimationType::Move;
    bool active = true;
    bool loop = false;
    bool forward = true;

    // Move animation.
    float startX = 0.0f;
    float startY = 0.0f;
    float targetX = 0.0f;
    float targetY = 0.0f;

    // Rotation animation.
    float startRotation = 0.0f;
    float targetRotation = 0.0f;

    // Scale animation.
    float startScaleX = 1.0f;
    float startScaleY = 1.0f;
    float targetScaleX = 1.0f;
    float targetScaleY = 1.0f;

    // Units per second: pixels/s for move, degrees/s for rotate,
    // scale units/s for scale.
    float speed = 100.0f;
};

vector<TFGuiAnimation> guiAnimations;

struct TFGuiCollisionPair { string first; string second; };
vector<TFGuiCollisionPair> guiCollisionPairs;

struct TFGuiClickBinding { string guiName; string action; };
vector<TFGuiClickBinding> guiClickBindings;

// Camera can also be controlled by the normal TigerFlash input system.
// "cam" is a reserved target name for the active camera.
Camera3D tfCamera = { 0 };
Vector3 tfCameraRotation = { -24.0f, -45.0f, 0.0f };
bool tfCameraRotationInitialized = false;

// Optional smooth camera follow. Existing camera controls remain unchanged.
bool tfCameraFollowEnabled = false;
string tfCameraFollowTarget;
float tfCameraFollowSmooth = 0.08f;
float tfCameraFollowDistance = 6.5f;
float tfCameraFollowHeight = 1.8f;

// Mouse-driven free camera look (FPS-style: move the mouse, the camera
// turns) is OPT-IN. It starts OFF so the camera only ever moves when the
// script explicitly asks it to - through a normal input binding such as
// "key d \"cam\" rotate(12,0,0)" or a combo like
// "key right_mouse + moviment up mouse \"cam\" rotate(-1,0,0)".
// Scripts that want classic always-on mouse-look can turn it on with the
// "mouse look on" command (see its parser further down). Even then, TigerFlash
// reads the cursor delta without disabling or recentering the native cursor.
float tfMouseLookSensitivity = 0.12f;
bool tfMouseLookEnabled = false;
 
// Forward declaration so key bindings can execute any TigerFlash command.
bool executeLine(string line);
bool evaluateCondition(string condition);
size_t executeBlock(const vector<string> &script, size_t start, int baseIndent);

bool consumeCommentLine(const string &line, bool &insideComment, string &remaining);
int indentationLevel(const string &line);
void tfApplyCameraRotation(Camera3D &camera);
string tfResolveDisplayText(string content);
void tfUpdateBackgroundScript(float dt);
void tfMoveGravityBodyKinematic(TFObject3D &obj, Vector3 delta);
bool tfKinematicInputCollisionMode = false;

// External-library host API forward declarations.
bool tfHostRegisterSay3DOptionsHook(TFSay3DOptionsHook hook);
bool tfHostRegisterSay3DHook(TFSay3DHook hook);
bool tfHostRegisterCommandHook(TFLibraryCommandHook hook);
bool tfHostRegisterConditionHook(TFLibraryConditionHook hook);
bool tfHostSetObjectMotionBlur(const char *objectName, float value);
float tfHostGetObjectMotionBlur(const char *objectName);
bool tfHostObject3DExists(const char *objectName);
void tfHostLibraryLog(const char *message);
bool tfLoadTigerFlashLibrary(const string &requestedName);
bool tfRunLibrarySay3DOptionsHooks(string &options);
void tfRunLibrarySay3DHooks(const string &objectName, const string &fullLine);
bool tfRunLibraryCommandHooks(const string &line);
bool tfRunLibraryConditionHooks(const string &condition, bool &handled);

// Physics helpers used before their full definitions.
bool gravityObjectGrounded(const string &name);
void tfSnapGravityBodyToSupport(const string &name);
struct TFGuiElement;
TFGuiElement *findGuiElement(const string &name);
void addGuiCollision(const string &a, const string &b);
bool guiCollisionSensor(const string &a, const string &b, bool reportErrors);

// ============================================================
// COLLISION / SHAPE PART DEFORMATION
// Added without removing the existing TigerFlash systems.
// ============================================================

// Forward declarations for helpers defined later in the original code.
string trim(string text);
void error(string message);
Color getColor(string colorName);


struct TFDeformation
{
    char side = 'f';
    int segments = 0;
    vector<Vector3> factors;
};

struct TFCollisionPair
{
    string first;
    string second;
};

vector<TFCollisionPair> physicalCollisionPairs;
// Camera uses the same TigerFlash collision syntax as normal 3D objects.
// Camera pairs are kept separate because the camera is not a renderable object.
vector<TFCollisionPair> cameraCollisionPairs;
unordered_map<string, vector<TFDeformation>> shapeDeformations;
string lastCreatedObjectKey;

// ============================================================
// GRAVITY / RIGID BODY / SOFT BODY PHYSICS
// Soft bodies use spring-damper deformation modes instead of a
// simple scale squash. Higher softness values behave more like
// a viscous liquid/gel: more bending, spreading and oscillation.
// ============================================================
enum class TFBodyType
{
    Rigid,
    Soft
};

struct TFGravityBody
{
    TFBodyType type = TFBodyType::Rigid;
    float gravity = 9.80665f;
    Vector3 velocity = {0.0f, 0.0f, 0.0f};
    Vector3 previousPosition = {0.0f, 0.0f, 0.0f};
    bool initialized = false;

    // Dynamic shape modes. These are spring-damper states, not
    // direct scale multipliers.
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

    // Contact-aware molding.  The soft body remembers the strongest
    // collider it is pressing against so the surface can actually take
    // the collider's local shape instead of only scaling globally.
    string contactCollider;
    Vector3 contactNormal = {0.0f, 1.0f, 0.0f};
    Vector3 contactPoint = {0.0f, 0.0f, 0.0f};
    float contactStrength = 0.0f;
    float contactStrengthVelocity = 0.0f;
    float contactAge = 100.0f;

    float softness = 10.0f;

    // Continuous-contact / substep state. These values let the solver
    // recover a body that crossed a surface between two simulation samples
    // instead of allowing it to tunnel through the floor.
    float sweepStartY = 0.0f;
    bool sweepStartInitialized = false;

    // Startup collision lock. The body remains at this exact position until
    // the collision system has had several real solver passes to detect and
    // settle contacts. After the timer expires, normal gravity begins.
    float startupHoldRemaining = 0.0f;
    Vector3 startupHoldPosition = {0.0f, 0.0f, 0.0f};

    // Kinematic input movement is collision-resolved immediately. This flag
    // prevents the main-loop recovery sweep from processing the same movement
    // a second time and trapping the body against an already-touching surface.
    bool inputMovementResolvedThisFrame = false;
    bool inputMovementActiveThisFrame = false;

    // Soft-body material state. These are deliberately internal so the
    // TigerFlash syntax stays unchanged.
    float pressure = 0.0f;
    float pressureVelocity = 0.0f;
    float volumeError = 0.0f;
    float volumeErrorVelocity = 0.0f;
    float surfaceTension = 0.0f;

    float fluidWaveTime = 0.0f;
    float fluidWaveVelocity = 0.0f;
};

unordered_map<string, TFGravityBody> gravityBodies;
unordered_map<string, bool> gravityGrounded;


// =========================================================
// ADAPTIVE PERFORMANCE MONITOR / RESOURCE GOVERNOR
// =========================================================
// The interpreter keeps the requested visual/physics quality, but avoids
// spending the same CPU/GPU work when the current machine is under load.
// Linux process CPU time + current RSS + system RAM + frame time are sampled
// periodically. Physics and soft-body mesh rebuild frequency are adjusted
// only when sustained pressure is detected, preventing oscillation.
struct TFPerformanceState
{
    bool initialized = false;
    double lastCpuSeconds = 0.0;
    chrono::steady_clock::time_point lastWallSample{};
    float sampleTimer = 0.0f;

    float processCPUPercent = 0.0f;     // % of one logical CPU/core.
    float processRSSMB = 0.0f;          // current resident memory.
    float systemRAMPercent = 0.0f;      // used system RAM.
    float frameTimeEMA = 1.0f / 60.0f;

    uint64_t frameIndex = 0;
    uint64_t trianglesDrawn = 0;

    // Runtime quality governors.
    int targetPhysicsHz = 120;
    int collisionPasses = 4;
    int softMeshEveryFrames = 1;
    int organicLOD = 2;
    size_t maxCachedMeshes = 384;
};

TFPerformanceState tfPerf;
bool tfCollisionGeometryBuilding = false;
bool tfBuildingSoftBaseGeometry = false;

static double tfProcessCPUSeconds()
{
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0)
        return 0.0;

    return static_cast<double>(usage.ru_utime.tv_sec) +
           static_cast<double>(usage.ru_utime.tv_usec) * 1.0e-6 +
           static_cast<double>(usage.ru_stime.tv_sec) +
           static_cast<double>(usage.ru_stime.tv_usec) * 1.0e-6;
}

static float tfProcessRSSMB()
{
    FILE *file = fopen("/proc/self/statm", "r");
    if (!file)
        return 0.0f;

    unsigned long totalPages = 0;
    unsigned long residentPages = 0;
    int read = fscanf(file, "%lu %lu", &totalPages, &residentPages);
    fclose(file);

    if (read != 2)
        return 0.0f;

    long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0)
        pageSize = 4096;

    return static_cast<float>(residentPages) *
           static_cast<float>(pageSize) /
           (1024.0f * 1024.0f);
}

static float tfSystemRAMPercent()
{
    FILE *file = fopen("/proc/meminfo", "r");
    if (!file)
        return 0.0f;

    unsigned long long totalKB = 0;
    unsigned long long availableKB = 0;
    char label[64] = {0};
    unsigned long long value = 0;
    char unit[16] = {0};

    while (fscanf(file, "%63s %llu %15s", label, &value, unit) == 3)
    {
        if (strcmp(label, "MemTotal:") == 0)
            totalKB = value;
        else if (strcmp(label, "MemAvailable:") == 0)
            availableKB = value;

        if (totalKB != 0 && availableKB != 0)
            break;
    }

    fclose(file);

    if (totalKB == 0)
        return 0.0f;

    if (availableKB > totalKB)
        availableKB = totalKB;

    return 100.0f *
           static_cast<float>(totalKB - availableKB) /
           static_cast<float>(totalKB);
}

void tfUpdatePerformanceMonitor(float dt)
{
    dt = max(0.0001f, min(0.10f, dt));
    tfPerf.frameIndex++;
    tfPerf.frameTimeEMA +=
        (dt - tfPerf.frameTimeEMA) *
        (1.0f - expf(-4.0f * dt));

    tfPerf.sampleTimer += dt;
    if (tfPerf.initialized && tfPerf.sampleTimer < 0.50f)
        return;

    const double nowCPU = tfProcessCPUSeconds();
    const auto nowWall = chrono::steady_clock::now();

    if (!tfPerf.initialized)
    {
        tfPerf.initialized = true;
        tfPerf.lastCpuSeconds = nowCPU;
        tfPerf.lastWallSample = nowWall;
        tfPerf.sampleTimer = 0.0f;
        return;
    }

    double wallSeconds =
        chrono::duration<double>(nowWall - tfPerf.lastWallSample).count();
    double cpuDelta = nowCPU - tfPerf.lastCpuSeconds;

    if (wallSeconds > 0.001 && cpuDelta >= 0.0)
    {
        // One fully occupied logical CPU ~= 100%. TigerFlash is primarily
        // single-threaded, so this is a useful local pressure signal.
        tfPerf.processCPUPercent =
            static_cast<float>(100.0 * cpuDelta / wallSeconds);
    }

    tfPerf.processRSSMB = tfProcessRSSMB();
    tfPerf.systemRAMPercent = tfSystemRAMPercent();

    const float frameMs = tfPerf.frameTimeEMA * 1000.0f;
    const bool severe =
        tfPerf.processCPUPercent > 95.0f ||
        tfPerf.systemRAMPercent > 94.0f ||
        frameMs > 28.0f ||
        tfPerf.trianglesDrawn > 450000ULL;

    const bool high =
        tfPerf.processCPUPercent > 82.0f ||
        tfPerf.systemRAMPercent > 88.0f ||
        frameMs > 20.0f ||
        tfPerf.trianglesDrawn > 300000ULL;

    const bool moderate =
        tfPerf.processCPUPercent > 70.0f ||
        tfPerf.systemRAMPercent > 82.0f ||
        frameMs > 17.5f ||
        tfPerf.trianglesDrawn > 200000ULL;

    if (severe)
    {
        tfPerf.targetPhysicsHz = 60;
        tfPerf.collisionPasses = 2;
        tfPerf.softMeshEveryFrames = 3;
        tfPerf.organicLOD = 2;
        tfPerf.maxCachedMeshes = 96;
    }
    else if (high)
    {
        tfPerf.targetPhysicsHz = 90;
        tfPerf.collisionPasses = 3;
        tfPerf.softMeshEveryFrames = 2;
        tfPerf.organicLOD = 2;
        tfPerf.maxCachedMeshes = 192;
    }
    else if (moderate)
    {
        tfPerf.targetPhysicsHz = 120;
        tfPerf.collisionPasses = 3;
        // Keep close liquid motion at full temporal resolution. The expensive
        // high-frequency ripple itself is GPU-only; this controls only the
        // slower physical silhouette updates.
        tfPerf.softMeshEveryFrames = 1;
        tfPerf.organicLOD = 1;
        tfPerf.maxCachedMeshes = 256;
    }
    else
    {
        tfPerf.targetPhysicsHz = 120;
        tfPerf.collisionPasses = 4;
        // High-frequency liquid ripples are animated in the GPU shader, so
        // the CPU only needs to rebuild the actual deformed silhouette when
        // its physical state changes. One-frame updates are allowed on a
        // healthy machine for close soft bodies.
        tfPerf.softMeshEveryFrames = 1;
        tfPerf.organicLOD = 0;
        tfPerf.maxCachedMeshes = 384;
    }

    // Never let pathological memory pressure retain a large mesh cache.
    if (tfPerf.systemRAMPercent > 90.0f)
        tfPerf.maxCachedMeshes = min(tfPerf.maxCachedMeshes, static_cast<size_t>(128));

    tfPerf.lastCpuSeconds = nowCPU;
    tfPerf.lastWallSample = nowWall;
    tfPerf.sampleTimer = 0.0f;
}

// =========================================================
// PHYSICS REALISM CONSTANTS
// =========================================================
// These make free-fall and landings behave like real materials instead of
// like an idealized, frictionless, drag-free point mass:
//
// - TF_AIR_DRAG_COEFFICIENT: quadratic air drag (F ~ v^2) opposing the fall.
//   Without this, a falling object accelerates forever, which real objects
//   never do - air resistance eventually balances gravity at a "terminal
//   velocity". The coefficient below is derived from a human-scale terminal
//   velocity of ~53 m/s (a belly-to-earth skydiver): k = g / v_terminal^2.
//   Objects with a custom "gravity" value (via the gravity command) scale
//   their drag proportionally, so the relationship stays physically
//   consistent instead of drag and gravity fighting each other oddly.
//
// - TF_GROUND_FRICTION_RATE: exponential (per-second) decay of horizontal
//   velocity once script/input stops pushing an object. This replaces a
//   flat per-frame multiplier, which made friction feel different at
//   different framerates (e.g. 30 vs 144 FPS) - a real bug for physics.
//   exp(-rate * dt) decays identically no matter the frame time.
//
// - TF_RIGID_RESTITUTION: kept at 0 for stable game-style contact. The
//   collision solver separates the body from the surface and then removes
//   inward vertical velocity, preventing repeated contact bouncing.
//   Soft-body deformation still uses the incoming impact speed before the
//   velocity is stabilized.
const float TF_AIR_DRAG_COEFFICIENT = 9.80665f / (53.0f * 53.0f); // ~0.00349
const float TF_GROUND_FRICTION_RATE = 15.0f;
const float TF_RIGID_RESTITUTION = 0.0f;
const float TF_RESTITUTION_FLOOR = 0.0f;

// During scene startup a gravity object is held exactly at its spawn position
// for a short period while the collision solver acquires/settles all declared
// contacts. This is deliberately a real simulation lock, not a visual delay.
// It prevents the first gravity step from happening while collision geometry
// is still being initialized or while a just-spawned object is being separated.
const float TF_COLLISION_STARTUP_HOLD_SECONDS = 0.40f;

string deformationKey2D(const string &name)
{
    return "2d:" + name;
}

string deformationKey3D(const string &name)
{
    return "3d:" + name;
}

bool parseCollisionSyntax(const string &line,
                          string &first,
                          string &second)
{
    if (line.rfind("colision ", 0) != 0)
        return false;

    string rest = trim(line.substr(9));

    if (rest.empty() || rest.front() != '"')
        return false;

    size_t closeFirst = rest.find('"', 1);
    if (closeFirst == string::npos)
        return false;

    first = rest.substr(1, closeFirst - 1);
    rest = trim(rest.substr(closeFirst + 1));

    if (rest.rfind("to ", 0) != 0)
        return false;

    rest = trim(rest.substr(3));

    if (rest.size() < 2 || rest.front() != '"' || rest.back() != '"')
        return false;

    size_t closeSecond = rest.find('"', 1);
    if (closeSecond == string::npos || closeSecond != rest.size() - 1)
        return false;

    second = rest.substr(1, closeSecond - 1);

    return !first.empty() && !second.empty();
}

int objectDimension(const string &name)
{
    // "cam" is a built-in 3D collision target. It behaves like a small
    // spherical body for collision tests, but it is not stored in objects3D.
    if (name == "cam")
        return 3;

    bool found2D = false;
    bool found3D = false;

    for (const TFObject2D &obj : objects2D)
    {
        if (obj.name == name)
        {
            found2D = true;
            break;
        }
    }

    for (const TFObject3D &obj : objects3D)
    {
        if (obj.name == name)
        {
            found3D = true;
            break;
        }
    }

    if (found2D && found3D)
        return -1;

    if (found2D)
        return 2;

    if (found3D)
        return 3;

    return 0;
}

TFObject2D *findObject2D(const string &name)
{
    for (TFObject2D &obj : objects2D)
        if (obj.name == name)
            return &obj;

    return nullptr;
}

TFObject3D *findObject3D(const string &name)
{
    for (TFObject3D &obj : objects3D)
        if (obj.name == name)
            return &obj;

    return nullptr;
}

bool hasPhysicalCollisionPair(const string &a, const string &b)
{
    for (const TFCollisionPair &pair : physicalCollisionPairs)
    {
        if ((pair.first == a && pair.second == b) ||
            (pair.first == b && pair.second == a))
        {
            return true;
        }
    }

    return false;
}

// ============================================================
// TOUCHING-CLUSTER GROUPING (shared by 2D and 3D)
// Small disjoint-set (union-find) used to group objects that are
// CURRENTLY overlapping into one cluster, one frame at a time. This
// backs the polygon budget system further below: an object glued to
// 49 others is treated completely differently from a lone object.
// ============================================================

string tfClusterFind(unordered_map<string, string> &parent, const string &node)
{
    string root = node;

    while (parent[root] != root)
        root = parent[root];

    string cur = node;

    while (parent[cur] != root)
    {
        string next = parent[cur];
        parent[cur] = root;
        cur = next;
    }

    return root;
}

void tfClusterUnion(unordered_map<string, string> &parent,
                    const string &a,
                    const string &b)
{
    string rootA = tfClusterFind(parent, a);
    string rootB = tfClusterFind(parent, b);

    if (rootA != rootB)
        parent[rootA] = rootB;
}

// -------------------------
// 2D shape geometry
// -------------------------

float shape2DHalfExtentX(const TFObject2D &obj)
{
    if (obj.shape == "sphere")
        return 100.0f * fabsf(obj.scaleX);

    if (obj.shape == "cone")
        return 110.0f * fabsf(obj.scaleX);

    if (obj.shape == "triangle")
        return 100.0f * fabsf(obj.scaleX);

    return 100.0f * fabsf(obj.scaleX);
}

float shape2DHalfExtentY(const TFObject2D &obj)
{
    if (obj.shape == "sphere")
        return 100.0f * fabsf(obj.scaleY);

    if (obj.shape == "cone")
        return 120.0f * fabsf(obj.scaleY);

    if (obj.shape == "triangle")
        return 120.0f * fabsf(obj.scaleY);

    return 100.0f * fabsf(obj.scaleY);
}

float deformationAxisT(char side, const Vector2 &point, 
                       float halfX, float halfY,
                       float minY, float maxY)
{
    (void)halfY;
    float nx = (halfX > 0.00001f) ? point.x / halfX : 0.0f;
    float ny = 0.0f;

    if (maxY - minY > 0.00001f)
        ny = (point.y - minY) / (maxY - minY) * 2.0f - 1.0f;

    nx = max(-1.0f, min(1.0f, nx));
    ny = max(-1.0f, min(1.0f, ny));

    switch (side)
    {
        case 'f': return (ny + 1.0f) * 0.5f;
        case 'b': return 1.0f - (ny + 1.0f) * 0.5f;
        case 'l': return 1.0f - (nx + 1.0f) * 0.5f;
        case 'r': return (nx + 1.0f) * 0.5f;
        case 't': return (ny + 1.0f) * 0.5f;
        case 'd': return 1.0f - (ny + 1.0f) * 0.5f;
        default:  return (nx + 1.0f) * 0.5f;
    }
}

Vector3 interpolateDeformation(const TFDeformation &deformation, float t)
{
    if (deformation.factors.empty())
        return {1.0f, 1.0f, 1.0f};

    if (deformation.factors.size() == 1)
        return deformation.factors.front();

    float position =
        t * (static_cast<float>(deformation.factors.size()) - 1.0f);

    int left = static_cast<int>(floorf(position));
    int right = min(
        static_cast<int>(deformation.factors.size()) - 1,
        left + 1
    );

    left = max(0, left);

    float localT = position - static_cast<float>(left);

    Vector3 a = deformation.factors[left];
    Vector3 b = deformation.factors[right];

    return
    {
        a.x + (b.x - a.x) * localT,
        a.y + (b.y - a.y) * localT,
        a.z + (b.z - a.z) * localT
    };
}

Vector2 apply2DDeformations(const TFObject2D &obj, Vector2 point)
{
    auto it = shapeDeformations.find(deformationKey2D(obj.name));

    if (it == shapeDeformations.end())
        return point;

    float halfX = shape2DHalfExtentX(obj);
    float halfY = shape2DHalfExtentY(obj);

    float minY = -halfY;
    float maxY = halfY;

    if (obj.shape == "cone")
    {
        minY = -120.0f * fabsf(obj.scaleY);
        maxY = 105.0f * fabsf(obj.scaleY);
    }
    else if (obj.shape == "triangle")
    {
        minY = -120.0f * fabsf(obj.scaleY);
        maxY = 100.0f * fabsf(obj.scaleY);
    }

    for (const TFDeformation &deformation : it->second)
    {
        float t = deformationAxisT(
            deformation.side,
            point,
            halfX,
            halfY,
            minY,
            maxY
        );

        Vector3 factor = interpolateDeformation(deformation, t);

        point.x *= factor.x;
        point.y *= factor.y;
    }

    return point;
}

vector<Vector2> make2DShapePolygon(const TFObject2D &obj)
{
    vector<Vector2> points;

    if (obj.shape == "cube")
    {
        points =
        {
            {-100.0f * obj.scaleX, -100.0f * obj.scaleY},
            { 100.0f * obj.scaleX, -100.0f * obj.scaleY},
            { 100.0f * obj.scaleX,  100.0f * obj.scaleY},
            {-100.0f * obj.scaleX,  100.0f * obj.scaleY}
        };
    }
    else if (obj.shape == "sphere")
    {
        const int count = 72;

        for (int i = 0; i < count; i++)
        {
            float angle =
                2.0f * PI * static_cast<float>(i) /
                static_cast<float>(count);

            points.push_back(
            {
                cosf(angle) * 100.0f * obj.scaleX,
                sinf(angle) * 100.0f * obj.scaleY
            });
        }
    }
    else if (obj.shape == "cone")
    {
        // Convex silhouette matching the visible cone footprint.
        points.push_back(
        {
            0.0f,
            -120.0f * obj.scaleY
        });

        const int baseCount = 48;

        for (int i = 0; i <= baseCount; i++)
        {
            float angle =
                PI + PI * static_cast<float>(i) /
                static_cast<float>(baseCount);

            points.push_back(
            {
                cosf(angle) * 110.0f * obj.scaleX,
                80.0f * obj.scaleY +
                    sinf(angle) * 25.0f * obj.scaleY
            });
        }
    }
    else if (obj.shape == "triangle")
    {
        points =
        {
            {0.0f, -120.0f * obj.scaleY},
            {-100.0f * obj.scaleX, 100.0f * obj.scaleY},
            {100.0f * obj.scaleX, 100.0f * obj.scaleY}
        };
    }

    for (Vector2 &point : points)
        point = apply2DDeformations(obj, point);

    for (Vector2 &point : points)
    {
        point.x += obj.x;
        point.y += obj.y;
    }

    return points;
}

float polygonProjectionMin(const vector<Vector2> &polygon,
                           const Vector2 &axis)
{
    float value = polygon[0].x * axis.x + polygon[0].y * axis.y;

    for (size_t i = 1; i < polygon.size(); i++)
    {
        float p = polygon[i].x * axis.x +
                  polygon[i].y * axis.y;

        value = min(value, p);
    }

    return value;
}

float polygonProjectionMax(const vector<Vector2> &polygon,
                           const Vector2 &axis)
{
    float value = polygon[0].x * axis.x + polygon[0].y * axis.y;

    for (size_t i = 1; i < polygon.size(); i++)
    {
        float p = polygon[i].x * axis.x +
                  polygon[i].y * axis.y;

        value = max(value, p);
    }

    return value;
}

bool polygonsOverlap2D(const vector<Vector2> &a,
                        const vector<Vector2> &b)
{
    if (a.empty() || b.empty())
        return false;

    const vector<vector<Vector2> *> polygons =
    {
        const_cast<vector<Vector2> *>(&a),
        const_cast<vector<Vector2> *>(&b)
    };

    for (vector<Vector2> *polygonPtr : polygons)
    {
        const vector<Vector2> &polygon = *polygonPtr;

        for (size_t i = 0; i < polygon.size(); i++)
        {
            Vector2 p1 = polygon[i];
            Vector2 p2 = polygon[(i + 1) % polygon.size()];

            Vector2 edge =
            {
                p2.x - p1.x,
                p2.y - p1.y
            };

            Vector2 axis =
            {
                -edge.y,
                edge.x
            };

            float length = sqrtf(axis.x * axis.x + axis.y * axis.y);

            if (length < 0.00001f)
                continue;

            axis.x /= length;
            axis.y /= length;

            float amin = polygonProjectionMin(a, axis);
            float amax = polygonProjectionMax(a, axis);
            float bmin = polygonProjectionMin(b, axis);
            float bmax = polygonProjectionMax(b, axis);

            if (amax < bmin || bmax < amin)
                return false;
        }
    }

    return true;
}

bool collision2D(const TFObject2D &a, const TFObject2D &b)
{
    return polygonsOverlap2D(
        make2DShapePolygon(a),
        make2DShapePolygon(b)
    );
}

// -------------------------
// 3D vector helpers
// -------------------------

Vector3 tfAdd(Vector3 a, Vector3 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vector3 tfSub(Vector3 a, Vector3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vector3 tfMul(Vector3 a, float s)
{
    return {a.x * s, a.y * s, a.z * s};
}

float tfDot(Vector3 a, Vector3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vector3 tfCross(Vector3 a, Vector3 b)
{
    return
    {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

float tfLength(Vector3 a)
{
    return sqrtf(tfDot(a, a));
}

Vector3 tfNormalize(Vector3 a)
{
    float len = tfLength(a);

    if (len < 0.000001f)
        return {1.0f, 0.0f, 0.0f};

    return tfMul(a, 1.0f / len);
}

float tfClampSoftness(float softness)
{
    return max(0.0f, min(10.0f, softness));
}

float tfSpringStep(float value, float &velocity, float target,
                   float stiffness, float damping, float dt)
{
    float acceleration = (target - value) * stiffness - velocity * damping;
    velocity += acceleration * dt;
    value += velocity * dt;
    return value;
}


TFObject3D *findContactObject3D(const string &name)
{
    if (name.empty())
        return nullptr;
    return findObject3D(name);
}

Vector3 tfClosestPointOnCubeSurface(const TFObject3D &cube,
                                    Vector3 worldPoint,
                                    Vector3 &normalOut)
{
    float ex = 1.2f * fabsf(cube.scaleX);
    float ey = 1.2f * fabsf(cube.scaleY);
    float ez = 1.2f * fabsf(cube.scaleZ);

    Vector3 q =
    {
        worldPoint.x - cube.x,
        worldPoint.y - cube.y,
        worldPoint.z - cube.z
    };

    float ax = fabsf(q.x);
    float ay = fabsf(q.y);
    float az = fabsf(q.z);

    // If the point is outside the box, use the closest face/corner normal.
    // For impact molding we want a face normal whenever possible so a wall
    // produces a genuinely flat contact patch.
    float dx = fabsf(ex - ax);
    float dy = fabsf(ey - ay);
    float dz = fabsf(ez - az);

    if (dx <= dy && dx <= dz)
        normalOut = {q.x >= 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f};
    else if (dy <= dz)
        normalOut = {0.0f, q.y >= 0.0f ? 1.0f : -1.0f, 0.0f};
    else
        normalOut = {0.0f, 0.0f, q.z >= 0.0f ? 1.0f : -1.0f};

    Vector3 surface =
    {
        max(-ex, min(ex, q.x)),
        max(-ey, min(ey, q.y)),
        max(-ez, min(ez, q.z))
    };

    // Force the selected axis exactly onto the supporting face.
    if (fabsf(normalOut.x) > 0.5f)
        surface.x = normalOut.x * ex;
    else if (fabsf(normalOut.y) > 0.5f)
        surface.y = normalOut.y * ey;
    else
        surface.z = normalOut.z * ez;

    return
    {
        cube.x + surface.x,
        cube.y + surface.y,
        cube.z + surface.z
    };
}

Vector3 tfClosestPointOnConeSurface(const TFObject3D &cone,
                                    Vector3 worldPoint,
                                    Vector3 &normalOut)
{
    float sx = max(0.001f, fabsf(cone.scaleX));
    float sy = max(0.001f, fabsf(cone.scaleY));
    float sz = max(0.001f, fabsf(cone.scaleZ));

    Vector3 q =
    {
        (worldPoint.x - cone.x) / sx,
        (worldPoint.y - cone.y) / sy,
        (worldPoint.z - cone.z) / sz
    };

    q.y = max(-1.5f, min(1.5f, q.y));

    float rho = sqrtf(q.x * q.x + q.z * q.z);
    float surfaceRadius = 0.5f * (1.5f - q.y);
    surfaceRadius = max(0.0f, min(1.5f, surfaceRadius));

    Vector3 radial;
    if (rho > 0.0001f)
        radial = {q.x / rho, 0.0f, q.z / rho};
    else
        radial = {1.0f, 0.0f, 0.0f};

    Vector3 surface =
    {
        radial.x * surfaceRadius,
        q.y,
        radial.z * surfaceRadius
    };

    // Gradient of rho - (0.75 - 0.5*y) in the scaled cone coordinates.
    // This is the actual local outward normal of the cone wall.
    Vector3 gradient =
    {
        (rho > 0.0001f ? q.x / (rho * sx) : 1.0f / sx),
        0.5f / sy,
        (rho > 0.0001f ? q.z / (rho * sz) : 0.0f)
    };

    normalOut = tfNormalize(gradient);

    return
    {
        cone.x + surface.x * sx,
        cone.y + surface.y * sy,
        cone.z + surface.z * sz
    };
}

bool tfMoldSoftBodyAgainstCollider(const TFObject3D &obj,
                                   Vector3 &point,
                                   const TFGravityBody &body)
{
    if (body.type != TFBodyType::Soft || body.contactCollider.empty())
        return false;

    TFObject3D *collider = findContactObject3D(body.contactCollider);
    if (!collider)
        return false;

    float fluid = tfClampSoftness(body.softness) / 10.0f;
    float strength = max(0.0f, min(1.0f, body.contactStrength));
    if (strength <= 0.001f)
        return false;

    Vector3 world =
    {
        obj.x + point.x,
        obj.y + point.y,
        obj.z + point.z
    };

    Vector3 surface;
    Vector3 surfaceNormal;

    if (collider->shape == "cube")
    {
        surface = tfClosestPointOnCubeSurface(*collider, world, surfaceNormal);

        // For a box, use a real half-space test against the chosen support
        // plane. Only the contact-facing part of the soft surface is molded.
        Vector3 relative = tfSub(world, surface);
        float signedDistance = tfDot(relative, surfaceNormal);

        // Contact influence fades smoothly away from the remembered impact
        // point. This keeps the opposite side of the blob round.
        Vector3 patchDelta = tfSub(world, body.contactPoint);
        float patchDistance = tfLength(patchDelta);
        float patchRadius = 1.35f + 1.15f * fluid;
        float patchWeight = 1.0f - patchDistance / patchRadius;
        patchWeight = max(0.0f, min(1.0f, patchWeight));
        patchWeight *= patchWeight * (3.0f - 2.0f * patchWeight);

        // A small positive tolerance allows the surface to visibly settle onto
        // the wall while avoiding numerical flicker when the collision solver
        // has already separated the centers.
        float moldDepth = 0.18f + 0.95f * strength;
        if (signedDistance < moldDepth && patchWeight > 0.001f)
        {
            float targetDistance = 0.015f * (1.0f - strength);
            float correction =
                max(0.0f, signedDistance - targetDistance) *
                strength * patchWeight;
            world = tfSub(world, tfMul(surfaceNormal, correction));
            point =
            {
                world.x - obj.x,
                world.y - obj.y,
                world.z - obj.z
            };
            return true;
        }

        return false;
    }

    if (collider->shape == "cone")
    {
        surface = tfClosestPointOnConeSurface(*collider, world, surfaceNormal);
        Vector3 fromSurface = tfSub(world, surface);
        float signedDistance = tfDot(fromSurface, surfaceNormal);

        Vector3 patchDelta = tfSub(world, body.contactPoint);
        float patchDistance = tfLength(patchDelta);
        float patchRadius = 1.55f + 1.25f * fluid;
        float patchWeight = 1.0f - patchDistance / patchRadius;
        patchWeight = max(0.0f, min(1.0f, patchWeight));
        patchWeight = patchWeight * patchWeight * (3.0f - 2.0f * patchWeight);

        // The key difference from the old soft body: vertices that enter the
        // cone are projected toward the CONE SURFACE itself. The contact patch
        // therefore follows the cone's slope instead of becoming a flat dent.
        float moldBand = 0.30f + 1.05f * strength;
        if (signedDistance < moldBand && patchWeight > 0.001f)
        {
            float desiredGap = 0.008f + 0.045f * (1.0f - strength);
            Vector3 target = tfAdd(surface, tfMul(surfaceNormal, desiredGap));
            float blend = strength * patchWeight;

            // Softer materials follow the cone more completely. Lower soft
            // values keep more of the object's original curvature.
            float shapeBlend = 0.20f + 0.80f * fluid;
            blend *= shapeBlend;

            point =
            {
                point.x + (target.x - world.x) * blend,
                point.y + (target.y - world.y) * blend,
                point.z + (target.z - world.z) * blend
            };
            return true;
        }
    }

    return false;
}

Vector3 tfColliderSupportPointWorld(const TFObject3D &collider, Vector3 direction);

void recordSoftBodyCollision(const TFObject3D &soft,
                             const TFObject3D &collider)
{
    auto it = gravityBodies.find(soft.name);
    if (it == gravityBodies.end() || it->second.type != TFBodyType::Soft)
        return;

    TFGravityBody &body = it->second;
    float incoming = tfLength(body.velocity);

    Vector3 center = {soft.x, soft.y, soft.z};
    Vector3 normal = tfNormalize(tfSub(center,
                                      {collider.x, collider.y, collider.z}));
    Vector3 point = {collider.x, collider.y, collider.z};

    point = tfColliderSupportPointWorld(collider, normal);

    float impact = max(0.05f, min(1.0f,
        incoming * (0.018f + 0.055f * (body.softness / 10.0f))));

    if (body.contactCollider != collider.name || impact > body.contactStrength)
    {
        body.contactCollider = collider.name;
        body.contactNormal = normal;
        body.contactPoint = point;
    }

    body.contactAge = 0.0f;
    body.contactStrength = max(body.contactStrength, impact);
    body.contactStrengthVelocity = max(body.contactStrengthVelocity,
                                       impact * 2.5f);
}

void decaySoftBodyContact(TFGravityBody &body, float dt)
{
    body.contactAge += dt;

    // Keep a short-lived imprint after impact, but do not glue the body to the
    // collider forever. High softness retains the molded shape for longer.
    float fluid = tfClampSoftness(body.softness) / 10.0f;
    float hold = 0.10f + 0.45f * fluid;

    if (body.contactAge > hold)
    {
        float rate = 2.8f - 1.7f * fluid;
        body.contactStrengthVelocity -= body.contactStrengthVelocity * rate * dt;
        body.contactStrength -= body.contactStrength * rate * dt;
    }

    body.contactStrength = max(0.0f, min(1.0f, body.contactStrength));
}

Vector3 applySoftBodyDeformation(const TFObject3D &obj, Vector3 point,
                                 const TFGravityBody &body)
{
    if (body.type != TFBodyType::Soft)
        return point;

    const float fluid = tfClampSoftness(body.softness) / 10.0f;
    const float sx = max(0.001f, 1.5f * fabsf(obj.scaleX));
    const float sy = max(0.001f, 1.5f * fabsf(obj.scaleY));
    const float sz = max(0.001f, 1.5f * fabsf(obj.scaleZ));

    float nx = max(-1.0f, min(1.0f, point.x / sx));
    float ny = max(-1.0f, min(1.0f, point.y / sy));
    float nz = max(-1.0f, min(1.0f, point.z / sz));

    // Low softness behaves like a soft solid. At 10 the elastic return is
    // intentionally very weak and the object behaves more like a blob of
    // liquid: it spreads, flows, sags and keeps the deformation for longer.
    // High-softness bodies become liquid-like when supported: their height
    // collapses much more strongly while horizontal area increases to retain
    // most of the volume. Low softness remains a rubbery soft solid.
    // The deformation function does not receive a grounded argument.
    // Read the last contact state produced by the physics solver instead.
    // This avoids calling gravityObjectGrounded() here, which would recurse
    // back into collision3D() while collision geometry is being built.
    auto groundedIt = gravityGrounded.find(obj.name);
    const bool grounded =
        groundedIt != gravityGrounded.end() &&
        groundedIt->second;

    // Do NOT globally crush a liquid vertically. Real water redistributes
    // volume near the contact surface; a global Y scale was the reason the
    // old soft body turned into a pancake whenever it touched the floor.
    const float groundedFluidFlatten = grounded
        ? fluid * (0.065f + 0.180f * fluid)
        : fluid * 0.018f;
    const float persistentFlatten = fluid * fluid * 0.038f + groundedFluidFlatten;
    const float impactFlatten = max(0.0f, min(0.55f,
        body.compression * (0.18f + 0.08f * fluid) +
        body.impactPulse * (0.16f + 0.10f * fluid)));

    float verticalScale = 1.0f - persistentFlatten - impactFlatten *
                          (grounded ? (0.16f + 0.16f * fluid) : 0.10f);
    verticalScale = max(0.54f, min(1.18f, verticalScale));

    // Incompressible-fluid approximation: less height -> more area. High
    // softness intentionally produces a broad puddle/reservoir silhouette
    // instead of leaving the body as a scaled ball.
    float areaScale = 1.0f / sqrtf(max(0.08f, verticalScale));
    areaScale *= 1.0f + fluid *
                 (body.liquidSpread * (1.05f + 0.35f * fluid) +
                  body.volumeError * 0.24f);

    point.y *= verticalScale;
    point.x *= areaScale;
    point.z *= areaScale;

    // ========================================================
    // VERTICAL CONTACT MOLDING
    // ========================================================
    // Lateral molding already follows walls/cones. For a horizontal support,
    // use the remembered contact plane and reshape only the nearby surface
    // band. The top of the fluid remains rounded instead of being globally
    // squashed into a pancake.
    if (!tfCollisionGeometryBuilding &&
        grounded &&
        fabsf(body.contactNormal.y) > 0.55f &&
        body.contactStrength > 0.01f)
    {
        const float side = body.contactNormal.y > 0.0f ? 1.0f : -1.0f;
        const float contactPlaneY = body.contactPoint.y;
        const float worldX = obj.x + point.x;
        const float worldY = obj.y + point.y;
        const float worldZ = obj.z + point.z;

        const float dx = worldX - body.contactPoint.x;
        const float dz = worldZ - body.contactPoint.z;
        const float radialDistance = sqrtf(dx * dx + dz * dz);
        const float patchRadius = sy * (0.85f + 0.75f * fluid) + 0.25f;
        float patch = 1.0f - radialDistance / max(0.05f, patchRadius);
        patch = max(0.0f, min(1.0f, patch));
        patch = patch * patch * (3.0f - 2.0f * patch);

        const float signedDistance = (worldY - contactPlaneY) * side;
        const float band = sy * (0.55f + 0.75f * fluid) + 0.08f;
        float nearSurface = 1.0f - signedDistance / max(0.08f, band);
        nearSurface = max(0.0f, min(1.0f, nearSurface));
        nearSurface = nearSurface * nearSurface * (3.0f - 2.0f * nearSurface);

        if (patch > 0.001f && nearSurface > 0.001f)
        {
            // The lowest layer approaches the real support plane. Layers above
            // it move progressively less, producing a rounded reservoir wall.
            float localDepth = 1.0f - nearSurface * patch;
            float targetDistance =
                0.012f + localDepth * band * (0.34f + 0.30f * (1.0f - fluid));

            float desiredWorldY = contactPlaneY + side * targetDistance;
            float correction = desiredWorldY - worldY;
            float strength = (0.22f + 0.68f * fluid) * patch * nearSurface;

            // Never fold the surface through the support plane.
            if (side > 0.0f)
                correction = min(0.0f, correction);
            else
                correction = max(0.0f, correction);

            point.y += correction * strength;
        }
    }

    // A real impact should make the lower surface collapse first instead of
    // squeezing the entire object uniformly. The contact profile is strongest
    // near the bottom and fades continuously toward the top.
    float bottom = max(0.0f, min(1.0f, (1.0f - ny) * 0.5f));
    float contact = bottom * bottom;
    float impactShift = (body.compression * 0.36f +
                         body.impactPulse * 0.48f) * sy * contact;
    point.y += impactShift;

    // Fluid spreading at the contact surface. This gives a puddle-like rim
    // instead of the classic rubber-ball pancake.
    float rim = contact * contact;
    float rimSpread = 1.0f + (body.compression * 0.62f +
                              body.impactPulse * 0.92f) * (0.35f + 0.65f * fluid) * rim;
    point.x *= rimSpread;
    point.z *= rimSpread;

    // Shear/bending: the upper layers lag behind the lower layers. The sign
    // changes through the center, so the mesh really bends rather than simply
    // translating as a rigid primitive.
    float bendProfile = ny * (0.75f + 0.25f * fabsf(ny));
    point.x += body.bendX * bendProfile * sx * (0.65f + 0.85f * fluid);
    point.z += body.bendZ * bendProfile * sz * (0.65f + 0.85f * fluid);

    // Viscous flow. The middle of the body has more freedom to slosh while
    // the extreme poles stay comparatively coherent.
    float middle = 1.0f - ny * ny;
    float flowWave = sinf((nx * 3.0f + nz * 2.0f) +
                          body.flowX * 4.0f + body.flowZ * 3.0f);
    point.x += (body.flowX * middle * (0.55f + fluid) +
                flowWave * body.impactPulse * 0.035f * fluid) * sx;
    point.z += (body.flowZ * middle * (0.55f + fluid) +
                flowWave * body.impactPulse * 0.035f * fluid) * sz;

    // Stretching before/after impacts. At high fluidity this becomes a long
    // tail/neck instead of a stiff spring-like elongation.
    float stretchAxis = 1.0f + body.stretch * (0.75f + 0.90f * fluid);
    float verticalStretch = max(0.55f, min(1.70f, stretchAxis));
    point.y *= verticalStretch;
    float horizontalCompensation = 1.0f / sqrtf(max(0.20f, verticalStretch));
    point.x *= horizontalCompensation;
    point.z *= horizontalCompensation;

    // High-fluid bodies also lose the perfect spherical silhouette. A small
    // multi-lobed surface field creates rounded tongues and pooled shoulders
    // that move with the viscous flow. It is deliberately continuous so the
    // high-density mesh reads as liquid rather than as a jagged noise shell.
    if (!tfCollisionGeometryBuilding && fluid > 0.45f)
    {
        float angle = atan2f(nz, nx);
        float lobeA = sinf(angle * 4.0f + body.fluidWaveTime * (0.8f + 1.7f * fluid));
        float lobeB = sinf(angle * 7.0f - body.fluidWaveTime * 0.9f + 1.3f);
        float lobeMask = (0.35f + 0.65f * (1.0f - ny * ny));
        float lobeStrength = fluid * (0.018f + 0.055f * body.liquidSpread);
        float lobe = (lobeA * 0.62f + lobeB * 0.38f) * lobeMask;
        point.x *= 1.0f + lobe * lobeStrength;
        point.z *= 1.0f - lobe * lobeStrength * 0.72f;

        // Gravity-defined free surface: flatten the upper cap very slightly
        // and let its height react to the moving wave field.
        float cap = max(0.0f, min(1.0f, (ny - 0.08f) / 0.92f));
        cap = cap * cap * (3.0f - 2.0f * cap);
        point.y -= sy * cap * fluid * (0.016f + 0.038f * body.liquidSpread);
    }

    // Travelling free-surface waves: high-softness liquid visibly sloshes and
    // sends ripples outward after impacts instead of remaining a rubber blob.
    if (!tfCollisionGeometryBuilding && fluid > 0.55f)
    {
        float radial = sqrtf(nx * nx + nz * nz);
        float topMask = max(0.0f, min(1.0f, (ny + 0.04f) * 1.40f));
        topMask = topMask * topMask * (3.0f - 2.0f * topMask);

        // Keep only a very low-amplitude, time-independent micro-relief in
        // the CPU mesh. The actual fast ripple is rendered in the optical
        // vertex shader every GPU frame, so rebuilds cannot make the water
        // visibly jump from one wave phase to another.
        float staticWaveA = sinf(radial * 18.0f + nx * 2.7f);
        float staticWaveB = sinf(nx * 11.0f + nz * 9.0f);
        float staticWaveC = sinf(radial * 31.0f + nz * 3.4f);
        float amplitude = grounded
            ? (0.0028f + 0.010f * fluid) *
              (0.20f + 0.80f * min(1.0f, body.impactPulse + body.liquidSpread))
            : (0.002f + 0.004f * fluid);

        point.y += (staticWaveA * 0.48f + staticWaveB * 0.30f + staticWaveC * 0.22f) *
                   sy * amplitude * topMask;

        float rim = max(0.0f, min(1.0f, (radial - 0.54f) / 0.46f));
        rim = rim * rim * (3.0f - 2.0f * rim);
        point.y += rim * sy * fluid *
                   (0.020f + 0.070f * body.liquidSpread);
    }

    // Keep the procedural mesh closed and numerically stable.
    float maxX = 3.2f * sx;
    float maxY = 3.2f * sy;
    float maxZ = 3.2f * sz;
    point.x = max(-maxX, min(maxX, point.x));
    point.y = max(-maxY, min(maxY, point.y));
    point.z = max(-maxZ, min(maxZ, point.z));

    return point;
}

void updateSoftBodyPhysicsForces(TFGravityBody &body, float dt,
                                  bool grounded, float softness)
{
    softness = tfClampSoftness(softness);
    const float fluid = softness / 10.0f;
    const float solid = 1.0f - fluid;

    // Impact excites the surface-wave oscillator; viscous damping removes
    // energy over time, producing a natural settle rather than a snap.
    body.fluidWaveVelocity += (body.impactPulse * 2.5f -
                               body.fluidWaveVelocity * (1.8f - 0.9f * fluid)) * dt;
    // A very small base phase keeps a liquid surface alive, while impacts add
    // extra energy. High viscosity still damps the actual amplitude below.
    body.fluidWaveTime += dt * (1.4f + 4.2f * fluid);
    body.fluidWaveTime += body.fluidWaveVelocity * dt;

    // ========================================================
    // PHYSICALLY-INSPIRED SOFT BODY MODEL
    // ========================================================
    // The body keeps an approximate volume while allowing controlled
    // compression, stretching, shear and viscous flow.  This is much more
    // stable than simply scaling the whole object on impact.  It is still a
    // lightweight real-time model, not a full FEM/SPH continuum solver.

    const float verticalSpeed = fabsf(body.velocity.y);
    const float horizontalSpeed = sqrtf(
        body.velocity.x * body.velocity.x +
        body.velocity.z * body.velocity.z
    );

    // Impact energy.  Only a descending body can create a hard contact pulse.
    const float incomingSpeed = max(0.0f, -body.velocity.y);
    const float impactKick = grounded
        ? min(1.50f, incomingSpeed * (0.060f + 0.105f * fluid))
        : 0.0f;

    if (impactKick > 0.0f)
    {
        body.impactPulseVelocity +=
            impactKick * (7.0f + 8.0f * fluid);
    }

    // Pressure rises when the body is flattened/compressed.  Higher
    // softness lowers elastic stiffness but increases viscous damping and
    // surface-tension-like recovery.
    float pressureTarget = body.compression +
                           max(0.0f, body.liquidSpread - 0.08f) * (0.55f + 0.90f * fluid);
    pressureTarget = max(0.0f, min(1.35f, pressureTarget));

    float pressureStiffness = 42.0f + 34.0f * solid;
    float pressureDamping = 7.0f + 10.0f * fluid;

    body.pressure = tfSpringStep(
        body.pressure,
        body.pressureVelocity,
        pressureTarget,
        pressureStiffness,
        pressureDamping,
        dt
    );

    body.pressure = max(0.0f, min(1.35f, body.pressure));

    // A normalized volume error: compression tries to reduce height while
    // pressure restores the lost volume by increasing the horizontal area.
    float volumeTarget = (body.compression * 0.68f + body.pressure * 0.32f);
    if (!grounded)
        volumeTarget *= 0.35f;

    body.volumeError = tfSpringStep(
        body.volumeError,
        body.volumeErrorVelocity,
        volumeTarget,
        22.0f - 8.0f * fluid,
        4.0f + 2.0f * fluid,
        dt
    );

    // Impact pulse behaves like a damped oscillation instead of an instant
    // squash-and-reset.
    body.impactPulseVelocity += (
        -34.0f * body.impactPulse -
        (8.0f - 4.5f * fluid) * body.impactPulseVelocity
    ) * dt;
    body.impactPulse += body.impactPulseVelocity * dt;
    body.impactPulse = max(0.0f, min(1.35f, body.impactPulse));

    // Resting contact has a stronger low-frequency deformation than free fall.
    float restingFlatten = grounded
        ? (0.045f + 0.095f * fluid)
        : (0.015f * fluid);

    float impactTarget = min(
        0.82f,
        restingFlatten + body.impactPulse * (0.24f + 0.34f * fluid)
    );

    // Softer material = lower Young-like stiffness, but critically damped
    // enough to avoid numerical wobble when the frame rate changes.
    float elasticStiffness = 135.0f - 112.0f * fluid;
    float elasticDamping = 20.0f - 10.5f * fluid;
    elasticStiffness = max(18.0f, elasticStiffness);
    elasticDamping = max(6.0f, elasticDamping);

    body.compression = tfSpringStep(
        body.compression,
        body.compressionVelocity,
        impactTarget,
        elasticStiffness,
        elasticDamping,
        dt
    );

    // Fast motion stretches the body.  Soft liquids lag behind their center
    // of mass more strongly, creating a viscous tail.
    float stretchTarget = min(
        0.78f,
        verticalSpeed * (0.014f + 0.030f * fluid) +
        horizontalSpeed * 0.007f * fluid
    );

    body.stretch = tfSpringStep(
        body.stretch,
        body.stretchVelocity,
        stretchTarget,
        92.0f - 62.0f * fluid,
        15.0f - 8.0f * fluid,
        dt
    );

    // Horizontal velocity produces shear.  Stiffer soft solids return sooner;
    // fluid-like bodies retain more shear and flow.
    float targetBendX = max(-1.05f, min(1.05f,
        body.velocity.x * (0.014f + 0.042f * fluid)));
    float targetBendZ = max(-1.05f, min(1.05f,
        body.velocity.z * (0.014f + 0.042f * fluid)));

    body.bendX = tfSpringStep(
        body.bendX, body.bendXVelocity,
        targetBendX,
        72.0f - 54.0f * fluid,
        14.0f - 7.0f * fluid,
        dt
    );

    body.bendZ = tfSpringStep(
        body.bendZ, body.bendZVelocity,
        targetBendZ,
        72.0f - 54.0f * fluid,
        14.0f - 7.0f * fluid,
        dt
    );

    // Viscous internal flow.  This is intentionally rate-based so flow slows
    // smoothly rather than snapping to zero when input stops.
    float targetFlowX = max(-0.78f, min(0.78f,
        -body.velocity.x * (0.010f + 0.052f * fluid)));
    float targetFlowZ = max(-0.78f, min(0.78f,
        -body.velocity.z * (0.010f + 0.052f * fluid)));

    // Faster response at high softness makes water follow impacts/input
    // immediately instead of visibly lagging behind the object.
    float flowRate = 8.5f + 18.0f * fluid;
    float flowDamping = 1.9f - 0.7f * fluid;

    body.flowXVelocity += (targetFlowX - body.flowX) * flowRate * dt;
    body.flowZVelocity += (targetFlowZ - body.flowZ) * flowRate * dt;

    body.flowXVelocity *= max(0.0f, 1.0f - flowDamping * dt);
    body.flowZVelocity *= max(0.0f, 1.0f - flowDamping * dt);

    body.flowX += body.flowXVelocity * dt;
    body.flowZ += body.flowZVelocity * dt;

    // Surface tension keeps the free surface coherent, while contact allows
    // the bottom to spread.  More softness => lower tension => more pooling.
    float tensionTarget = grounded
        ? (0.44f - 0.34f * fluid) + body.impactPulse * (0.20f - 0.10f * fluid)
        : (0.58f - 0.30f * fluid);

    body.surfaceTension = tfSpringStep(
        body.surfaceTension,
        body.liquidSpreadVelocity,
        max(0.05f, tensionTarget),
        16.0f + 8.0f * solid,
        5.0f + 2.0f * solid,
        dt
    );

    // Surface spread is driven by deformation and contact pressure, but is
    // damped by surface tension.  High softness produces a wider pool without
    // letting the entire mesh explode sideways.
    float spreadTarget = grounded
        ? min(1.18f,
              fluid * (0.34f + 0.62f * body.impactPulse) +
              body.compression * (0.14f + 0.24f * fluid) +
              body.pressure * 0.08f * fluid)
        : min(0.38f, body.liquidSpread * max(0.0f, 1.0f - 1.9f * dt));

    body.liquidSpread = tfSpringStep(
        body.liquidSpread,
        body.liquidSpreadVelocity,
        spreadTarget,
        14.0f - 8.0f * fluid,
        4.5f - 1.6f * fluid,
        dt
    );

    // Clamp all deformation states to stable physical ranges.
    body.compression = max(0.0f, min(0.96f, body.compression));
    body.stretch = max(0.0f, min(0.80f, body.stretch));
    body.bendX = max(-1.05f, min(1.05f, body.bendX));
    body.bendZ = max(-1.05f, min(1.05f, body.bendZ));
    body.flowX = max(-0.80f, min(0.80f, body.flowX));
    body.flowZ = max(-0.80f, min(0.80f, body.flowZ));
    body.impactPulse = max(0.0f, min(1.35f, body.impactPulse));
    body.liquidSpread = max(0.0f, min(0.90f, body.liquidSpread));
    body.pressure = max(0.0f, min(1.35f, body.pressure));
    body.volumeError = max(0.0f, min(1.20f, body.volumeError));
    body.surfaceTension = max(0.05f, min(1.0f, body.surfaceTension));
}

Vector3 apply3DDeformations(const TFObject3D &obj, Vector3 point)
{
    // Base-mesh capture is render/cache infrastructure only. The original
    // deformation and physics functions remain untouched for normal frames.
    if (tfBuildingSoftBaseGeometry)
        return point;

    auto it = shapeDeformations.find(deformationKey3D(obj.name));

    bool hasStaticDeformation = it != shapeDeformations.end();
    bool hasSoftBody = false;

    auto softCheck = gravityBodies.find(obj.name);
    if (softCheck != gravityBodies.end() &&
        softCheck->second.type == TFBodyType::Soft)
    {
        hasSoftBody = true;
    }

    if (!hasStaticDeformation && !hasSoftBody)
        return point;

    float halfX = 1.2f * fabsf(obj.scaleX);
    float halfY = 1.2f * fabsf(obj.scaleY);
    float halfZ = 1.2f * fabsf(obj.scaleZ);

    float minY = -halfY;
    float maxY = halfY;

    if (obj.shape == "sphere")
    {
        halfX = 1.5f * fabsf(obj.scaleX);
        halfY = 1.5f * fabsf(obj.scaleY);
        halfZ = 1.5f * fabsf(obj.scaleZ);
    }
    else if (obj.shape == "cone")
    {
        halfX = 1.5f * fabsf(obj.scaleX);
        halfY = 1.5f * fabsf(obj.scaleY);
        halfZ = 1.5f * fabsf(obj.scaleZ);
    }
    else if (obj.shape == "triangle")
    {
        halfX = 1.3f * fabsf(obj.scaleX);
        halfY = 1.4f * fabsf(obj.scaleY);
        halfZ = 1.3f * fabsf(obj.scaleZ);
        minY = -1.0f * fabsf(obj.scaleY);
        maxY = 1.8f * fabsf(obj.scaleY);
    }
    else if (obj.shape == "cylinder")
    {
        halfX = 1.5f * fabsf(obj.scaleX);
        halfY = 1.5f * fabsf(obj.scaleY);
        halfZ = 1.5f * fabsf(obj.scaleZ);
    }
    else if (obj.shape == "capsule")
    {
        halfX = 1.15f * fabsf(obj.scaleX);
        halfY = 2.0f * fabsf(obj.scaleY);
        halfZ = 1.15f * fabsf(obj.scaleZ);
        minY = -2.0f * fabsf(obj.scaleY);
        maxY = 2.0f * fabsf(obj.scaleY);
    }
    else if (obj.shape == "crystal")
    {
        halfX = 1.15f * fabsf(obj.scaleX);
        halfY = 1.90f * fabsf(obj.scaleY);
        halfZ = 1.15f * fabsf(obj.scaleZ);
        minY = -1.90f * fabsf(obj.scaleY);
        maxY = 1.90f * fabsf(obj.scaleY);
    }
    else if (obj.shape == "diamond")
    {
        halfX = 1.25f * fabsf(obj.scaleX);
        halfY = 1.90f * fabsf(obj.scaleY);
        halfZ = 1.25f * fabsf(obj.scaleZ);
        minY = -1.90f * fabsf(obj.scaleY);
        maxY = 1.90f * fabsf(obj.scaleY);
    }
    else if (obj.shape == "prism")
    {
        halfX = 1.35f * fabsf(obj.scaleX);
        halfY = 1.45f * fabsf(obj.scaleY);
        halfZ = 1.35f * fabsf(obj.scaleZ);
        minY = -1.45f * fabsf(obj.scaleY);
        maxY = 1.45f * fabsf(obj.scaleY);
    }
    else if (obj.shape == "rock")
    {
        halfX = 1.25f * fabsf(obj.scaleX);
        halfY = 1.15f * fabsf(obj.scaleY);
        halfZ = 1.10f * fabsf(obj.scaleZ);
        minY = -1.05f * fabsf(obj.scaleY);
        maxY = 1.15f * fabsf(obj.scaleY);
    }

    auto axisT3D = [&](char side) -> float
    {
        float nx =
            (halfX > 0.00001f) ? point.x / halfX : 0.0f;

        float ny;

        if (maxY - minY > 0.00001f)
            ny =
                (point.y - minY) /
                (maxY - minY) * 2.0f - 1.0f;
        else
            ny = 0.0f;

        float nz =
            (halfZ > 0.00001f) ? point.z / halfZ : 0.0f;

        nx = max(-1.0f, min(1.0f, nx));
        ny = max(-1.0f, min(1.0f, ny));
        nz = max(-1.0f, min(1.0f, nz));

        switch (side)
        {
            case 'f': return (nz + 1.0f) * 0.5f;
            case 'b': return 1.0f - (nz + 1.0f) * 0.5f;
            case 'l': return 1.0f - (nx + 1.0f) * 0.5f;
            case 'r': return (nx + 1.0f) * 0.5f;
            case 't': return (ny + 1.0f) * 0.5f;
            case 'd': return 1.0f - (ny + 1.0f) * 0.5f;
            default:  return (nz + 1.0f) * 0.5f;
        }
    };

    if (hasStaticDeformation)
    {
        for (const TFDeformation &deformation : it->second)
        {
            Vector3 factor =
                interpolateDeformation(
                    deformation,
                    axisT3D(deformation.side)
                );

            point.x *= factor.x;
            point.y *= factor.y;
            point.z *= factor.z;
        }
    }

    auto softIt = gravityBodies.find(obj.name);
    if (softIt != gravityBodies.end() &&
        softIt->second.type == TFBodyType::Soft)
    {
        point = applySoftBodyDeformation(obj, point, softIt->second);
        if (!tfCollisionGeometryBuilding)
            tfMoldSoftBodyAgainstCollider(obj, point, softIt->second);
    }

    return point;
}

struct TFShape3DData
{
    const TFObject3D *object = nullptr;
    bool analyticSphere = false;
    vector<Vector3> points;
};

// ============================================================
// SHAPE-AWARE 3D COLLISION GEOMETRY
// The renderer and the collision solver use the same mathematical
// silhouettes. This is important for cones/crystals/pyramids:
// collision is no longer resolved from a large generic AABB.
// ============================================================

void tfAddRingPoints(vector<Vector3> &points,
                     float radiusX, float radiusZ,
                     float y,
                     int sides)
{
    sides = max(3, sides);
    for (int i = 0; i < sides; ++i)
    {
        float a = 2.0f * PI * (float)i / (float)sides;
        points.push_back({
            radiusX * cosf(a),
            y,
            radiusZ * sinf(a)
        });
    }
}

void tfAddCapsuleCollisionPoints(vector<Vector3> &points,
                                 float radiusX, float radiusZ,
                                 float halfSegment,
                                 float radiusY,
                                 int sides,
                                 int hemiRings)
{
    sides = max(8, sides);
    hemiRings = max(3, hemiRings);

    // Vertical capsule: cylindrical middle + two hemispherical ends.
    for (int r = 0; r <= hemiRings; ++r)
    {
        float t = (float)r / (float)hemiRings;
        float phi = (PI * 0.5f) * t;
        float cy = cosf(phi);
        float sr = sinf(phi);
        float y = halfSegment + radiusY * cy;
        float rx = radiusX * sr;
        float rz = radiusZ * sr;

        if (r == 0)
            points.push_back({0.0f, y, 0.0f});
        else
            tfAddRingPoints(points, rx, rz, y, sides);
    }

    tfAddRingPoints(points, radiusX, radiusZ, halfSegment, sides);
    tfAddRingPoints(points, radiusX, radiusZ, -halfSegment, sides);

    for (int r = hemiRings; r >= 0; --r)
    {
        float t = (float)r / (float)hemiRings;
        float phi = (PI * 0.5f) * t;
        float cy = cosf(phi);
        float sr = sinf(phi);
        float y = -halfSegment - radiusY * cy;
        float rx = radiusX * sr;
        float rz = radiusZ * sr;

        if (r == 0)
            points.push_back({0.0f, y, 0.0f});
        else
            tfAddRingPoints(points, rx, rz, y, sides);
    }
}

void tfAddCrystalCollisionPoints(vector<Vector3> &points,
                                 float radiusX, float radiusZ,
                                 float bodyHalfHeight,
                                 float tipHeight,
                                 int sides)
{
    sides = max(5, sides);
    tfAddRingPoints(points, radiusX, radiusZ, -bodyHalfHeight, sides);
    tfAddRingPoints(points, radiusX * 0.82f, radiusZ * 0.82f,
                    bodyHalfHeight * 0.72f, sides);
    points.push_back({0.0f, bodyHalfHeight + tipHeight, 0.0f});
    points.push_back({0.0f, -bodyHalfHeight - tipHeight, 0.0f});
}

void tfAddPrismCollisionPoints(vector<Vector3> &points,
                               float radiusX, float radiusZ,
                               float halfHeight,
                               int sides)
{
    sides = max(3, sides);
    tfAddRingPoints(points, radiusX, radiusZ, -halfHeight, sides);
    tfAddRingPoints(points, radiusX, radiusZ, halfHeight, sides);
}

void tfAddRockCollisionPoints(vector<Vector3> &points,
                              float sx, float sy, float sz)
{
    // Fixed deterministic faceted rock. Convex hull remains stable and
    // therefore produces a much tighter collision volume than a sphere/AABB.
    const float rings[3] = { -1.05f, 0.0f, 0.92f };
    const float rx[3] = { 0.80f, 1.22f, 0.72f };
    const float rz[3] = { 0.95f, 1.05f, 0.68f };
    const int sides = 8;

    for (int r = 0; r < 3; ++r)
    {
        for (int i = 0; i < sides; ++i)
        {
            float a = 2.0f * PI * (float)i / (float)sides;
            float wobble = 1.0f + 0.08f * sinf((float)i * 2.31f + (float)r);
            points.push_back({
                sx * rx[r] * wobble * cosf(a),
                sy * rings[r],
                sz * rz[r] * wobble * sinf(a)
            });
        }
    }
}

TFShape3DData make3DShapeData(const TFObject3D &obj)
{
    TFShape3DData data;
    data.object = &obj;

    bool hasDeformation =
        shapeDeformations.count(deformationKey3D(obj.name)) != 0;

    if (obj.shape == "sphere" && !hasDeformation)
    {
        data.analyticSphere = true;
        return data;
    }

    if (obj.shape == "cube")
    {
        const float hx = 1.2f * fabsf(obj.scaleX);
        const float hy = 1.2f * fabsf(obj.scaleY);
        const float hz = 1.2f * fabsf(obj.scaleZ);

        vector<Vector3> base =
        {
            {-hx, -hy, -hz},
            { hx, -hy, -hz},
            { hx,  hy, -hz},
            {-hx,  hy, -hz},
            {-hx, -hy,  hz},
            { hx, -hy,  hz},
            { hx,  hy,  hz},
            {-hx,  hy,  hz}
        };

        for (Vector3 p : base)
        {
            p = apply3DDeformations(obj, p);
            data.points.push_back(
                tfAdd(
                    p,
                    {obj.x, obj.y, obj.z}
                )
            );
        }

        return data;
    }

    if (obj.shape == "sphere")
    {
        // Deformed ellipsoids use a dense but lightweight sphere sample.
        const int rings = tfCollisionGeometryBuilding ? 8 : 14;
        const int slices = tfCollisionGeometryBuilding ? 16 : 28;

        for (int r = 0; r <= rings; r++)
        {
            float v = static_cast<float>(r) /
                      static_cast<float>(rings);

            float phi = PI * v;

            for (int s = 0; s < slices; s++)
            {
                float u = static_cast<float>(s) /
                          static_cast<float>(slices);

                float theta = 2.0f * PI * u;

                Vector3 p =
                {
                    1.5f * obj.scaleX * sinf(phi) * cosf(theta),
                    1.5f * obj.scaleY * cosf(phi),
                    1.5f * obj.scaleZ * sinf(phi) * sinf(theta)
                };

                p = apply3DDeformations(obj, p);

                data.points.push_back(
                    tfAdd(
                        p,
                        {obj.x, obj.y, obj.z}
                    )
                );
            }
        }

        return data;
    }

    if (obj.shape == "cone")
    {
        const int slices = tfCollisionGeometryBuilding ? 20 : 40;

        // Apex.
        Vector3 apex =
        {
            obj.x,
            obj.y + 1.5f * obj.scaleY,
            obj.z
        };

        apex = tfAdd(
            apply3DDeformations(
                obj,
                {0.0f, 1.5f * obj.scaleY, 0.0f}
            ),
            {obj.x, obj.y, obj.z}
        );

        data.points.push_back(apex);

        // Circular base.
        for (int i = 0; i < slices; i++)
        {
            float angle =
                2.0f * PI * static_cast<float>(i) /
                static_cast<float>(slices);

            Vector3 p =
            {
                1.5f * obj.scaleX * cosf(angle),
                -1.5f * obj.scaleY,
                1.5f * obj.scaleZ * sinf(angle)
            };

            p = apply3DDeformations(obj, p);

            data.points.push_back(
                tfAdd(
                    p,
                    {obj.x, obj.y, obj.z}
                )
            );
        }

        return data;
    }

    // TigerFlash's current "triangle" rendering is a square pyramid.
    if (obj.shape == "triangle")
    {
        vector<Vector3> base =
        {
            {
                -1.3f * obj.scaleX,
                -1.0f * obj.scaleY,
                 1.3f * obj.scaleZ
            },
            {
                 1.3f * obj.scaleX,
                -1.0f * obj.scaleY,
                 1.0f * obj.scaleZ
            },
            {
                 0.8f * obj.scaleX,
                -1.0f * obj.scaleY,
                -1.3f * obj.scaleZ
            },
            {
                -1.0f * obj.scaleX,
                -1.0f * obj.scaleY,
                -1.0f * obj.scaleZ
            },
            {
                 0.0f,
                 1.8f * obj.scaleY,
                 0.4f * obj.scaleZ
            }
        };

        for (Vector3 p : base)
        {
            p = apply3DDeformations(obj, p);
            data.points.push_back(
                tfAdd(
                    p,
                    {obj.x, obj.y, obj.z}
                )
            );
        }
    }

    if (obj.shape == "cylinder")
    {
        const int sides = tfCollisionGeometryBuilding ? 12 : 24;
        vector<Vector3> local;
        tfAddPrismCollisionPoints(local,
                                  1.5f * fabsf(obj.scaleX),
                                  1.5f * fabsf(obj.scaleZ),
                                  1.5f * fabsf(obj.scaleY),
                                  sides);
        for (Vector3 p : local)
        {
            p = apply3DDeformations(obj, p);
            data.points.push_back(tfAdd(p, {obj.x, obj.y, obj.z}));
        }
        return data;
    }

    if (obj.shape == "capsule")
    {
        const int sides = tfCollisionGeometryBuilding ? 12 : 20;
        const int rings = tfCollisionGeometryBuilding ? 3 : 6;
        vector<Vector3> local;
        tfAddCapsuleCollisionPoints(local,
                                     1.15f * fabsf(obj.scaleX),
                                     1.15f * fabsf(obj.scaleZ),
                                     0.85f * fabsf(obj.scaleY),
                                     1.15f * fabsf(obj.scaleY),
                                     sides, rings);
        for (Vector3 p : local)
        {
            p = apply3DDeformations(obj, p);
            data.points.push_back(tfAdd(p, {obj.x, obj.y, obj.z}));
        }
        return data;
    }

    if (obj.shape == "crystal")
    {
        vector<Vector3> local;
        tfAddCrystalCollisionPoints(local,
                                    1.15f * fabsf(obj.scaleX),
                                    1.15f * fabsf(obj.scaleZ),
                                    0.95f * fabsf(obj.scaleY),
                                    0.95f * fabsf(obj.scaleY),
                                    6);
        for (Vector3 p : local)
        {
            p = apply3DDeformations(obj, p);
            data.points.push_back(tfAdd(p, {obj.x, obj.y, obj.z}));
        }
        return data;
    }

    if (obj.shape == "diamond")
    {
        vector<Vector3> local =
        {
            { 0.0f,  1.9f * fabsf(obj.scaleY), 0.0f },
            { 0.0f, -1.9f * fabsf(obj.scaleY), 0.0f },
            { 1.25f * obj.scaleX, 0.0f, 0.0f },
            {-1.25f * obj.scaleX, 0.0f, 0.0f },
            { 0.0f, 0.0f, 1.25f * obj.scaleZ },
            { 0.0f, 0.0f,-1.25f * obj.scaleZ }
        };
        for (Vector3 p : local)
        {
            p = apply3DDeformations(obj, p);
            data.points.push_back(tfAdd(p, {obj.x, obj.y, obj.z}));
        }
        return data;
    }

    if (obj.shape == "prism")
    {
        vector<Vector3> local;
        tfAddPrismCollisionPoints(local,
                                  1.35f * fabsf(obj.scaleX),
                                  1.35f * fabsf(obj.scaleZ),
                                  1.45f * fabsf(obj.scaleY),
                                  6);
        for (Vector3 p : local)
        {
            p = apply3DDeformations(obj, p);
            data.points.push_back(tfAdd(p, {obj.x, obj.y, obj.z}));
        }
        return data;
    }

    if (obj.shape == "rock")
    {
        vector<Vector3> local;
        tfAddRockCollisionPoints(local,
                                 fabsf(obj.scaleX),
                                 fabsf(obj.scaleY),
                                 fabsf(obj.scaleZ));
        for (Vector3 p : local)
        {
            p = apply3DDeformations(obj, p);
            data.points.push_back(tfAdd(p, {obj.x, obj.y, obj.z}));
        }
        return data;
    }

    return data;
}

Vector3 support3D(const TFShape3DData &data, Vector3 direction)
{
    if (data.object == nullptr)
        return {0.0f, 0.0f, 0.0f};

    if (data.analyticSphere)
    {
        const TFObject3D &obj = *data.object;

        float rx = 1.5f * fabsf(obj.scaleX);
        float ry = 1.5f * fabsf(obj.scaleY);
        float rz = 1.5f * fabsf(obj.scaleZ);

        Vector3 d = direction;

        float denom =
            sqrtf(
                (rx * d.x) * (rx * d.x) +
                (ry * d.y) * (ry * d.y) +
                (rz * d.z) * (rz * d.z)
            );

        if (denom < 0.000001f)
            return {obj.x, obj.y, obj.z};

        return
        {
            obj.x + (rx * rx * d.x) / denom,
            obj.y + (ry * ry * d.y) / denom,
            obj.z + (rz * rz * d.z) / denom
        };
    }

    if (data.points.empty())
        return data.object ?
            Vector3{data.object->x, data.object->y, data.object->z} :
            Vector3{0.0f, 0.0f, 0.0f};

    Vector3 best = data.points.front();
    float bestDot = tfDot(best, direction);

    for (size_t i = 1; i < data.points.size(); i++)
    {
        float d = tfDot(data.points[i], direction);

        if (d > bestDot)
        {
            bestDot = d;
            best = data.points[i];
        }
    }

    return best;
}

Vector3 tfColliderSupportPointWorld(const TFObject3D &collider,
                                      Vector3 direction)
{
    const bool previousMode = tfCollisionGeometryBuilding;
    tfCollisionGeometryBuilding = true;
    TFShape3DData data = make3DShapeData(collider);
    Vector3 p = support3D(data, direction);
    tfCollisionGeometryBuilding = previousMode;
    return p;
}

Vector3 supportMinkowski3D(const TFShape3DData &a,
                           const TFShape3DData &b,
                           Vector3 direction)
{
    return tfSub(
        support3D(a, direction),
        support3D(b, tfMul(direction, -1.0f))
    );
}

bool sameDirection3D(Vector3 a, Vector3 b)
{
    return tfDot(a, b) > 0.0f;
}

bool handleSimplex3D(vector<Vector3> &simplex,
                     Vector3 &direction)
{
    if (simplex.size() == 2)
    {
        Vector3 a = simplex[1];
        Vector3 b = simplex[0];

        Vector3 ab = tfSub(b, a);
        Vector3 ao = tfMul(a, -1.0f);

        if (tfDot(ab, ao) > 0.0f)
        {
            direction =
                tfCross(
                    tfCross(ab, ao),
                    ab
                );

            if (tfLength(direction) < 0.000001f)
            {
                direction =
                    tfCross(
                        ab,
                        {0.0f, 1.0f, 0.0f}
                    );

                if (tfLength(direction) < 0.000001f)
                {
                    direction =
                        tfCross(
                            ab,
                            {1.0f, 0.0f, 0.0f}
                        );
                }
            }
        }
        else
        {
            simplex = {a};
            direction = ao;
        }

        return false;
    }

    if (simplex.size() == 3)
    {
        Vector3 a = simplex[2];
        Vector3 b = simplex[1];
        Vector3 c = simplex[0];

        Vector3 ab = tfSub(b, a);
        Vector3 ac = tfSub(c, a);
        Vector3 ao = tfMul(a, -1.0f);

        Vector3 abc = tfCross(ab, ac);

        Vector3 abcCrossAc =
            tfCross(abc, ac);

        if (tfDot(abcCrossAc, ao) > 0.0f)
        {
            if (tfDot(ac, ao) > 0.0f)
            {
                simplex = {c, a};

                direction =
                    tfCross(
                        tfCross(ac, ao),
                        ac
                    );
            }
            else
            {
                if (tfDot(ab, ao) > 0.0f)
                {
                    simplex = {b, a};

                    direction =
                        tfCross(
                            tfCross(ab, ao),
                            ab
                        );
                }
                else
                {
                    simplex = {a};
                    direction = ao;
                }
            }

            return false;
        }

        Vector3 abCrossAbc =
            tfCross(ab, abc);

        if (tfDot(abCrossAbc, ao) > 0.0f)
        {
            if (tfDot(ab, ao) > 0.0f)
            {
                simplex = {b, a};

                direction =
                    tfCross(
                        tfCross(ab, ao),
                        ab
                    );
            }
            else
            {
                simplex = {a};
                direction = ao;
            }

            return false;
        }

        if (tfDot(abc, ao) > 0.0f)
        {
            direction = abc;
        }
        else
        {
            simplex = {b, c, a};
            direction = tfMul(abc, -1.0f);
        }

        if (tfLength(direction) < 0.000001f)
            direction = {1.0f, 0.0f, 0.0f};

        return false;
    }

    if (simplex.size() == 4)
    {
        Vector3 a = simplex[3];
        Vector3 b = simplex[2];
        Vector3 c = simplex[1];
        Vector3 d = simplex[0];

        Vector3 ao = tfMul(a, -1.0f);

        Vector3 ab = tfSub(b, a);
        Vector3 ac = tfSub(c, a);
        Vector3 ad = tfSub(d, a);

        Vector3 abc = tfCross(ab, ac);

        if (tfDot(abc, ao) > 0.0f)
        {
            simplex = {c, b, a};
            direction = abc;
            return false;
        }

        Vector3 acd = tfCross(ac, ad);

        if (tfDot(acd, ao) > 0.0f)
        {
            simplex = {d, c, a};
            direction = acd;
            return false;
        }

        Vector3 adb = tfCross(ad, ab);

        if (tfDot(adb, ao) > 0.0f)
        {
            simplex = {b, d, a};
            direction = adb;
            return false;
        }

        return true;
    }

    direction = {1.0f, 0.0f, 0.0f};
    return false;
}

bool gjkCollision3D(const TFShape3DData &a,
                    const TFShape3DData &b,
                    vector<Vector3> *outSimplex = nullptr)
{
    Vector3 centerA =
    {
        a.object ? a.object->x : 0.0f,
        a.object ? a.object->y : 0.0f,
        a.object ? a.object->z : 0.0f
    };

    Vector3 centerB =
    {
        b.object ? b.object->x : 0.0f,
        b.object ? b.object->y : 0.0f,
        b.object ? b.object->z : 0.0f
    };

    Vector3 direction = tfSub(centerB, centerA);

    if (tfLength(direction) < 0.000001f)
        direction = {1.0f, 0.0f, 0.0f};

    vector<Vector3> simplex;

    Vector3 firstPoint =
        supportMinkowski3D(
            a,
            b,
            direction
        );

    simplex.push_back(firstPoint);

    // GJK searches from the first Minkowski point back toward the origin.
    direction = tfMul(firstPoint, -1.0f);

    if (tfLength(direction) < 0.000001f)
        return true;

    for (int iteration = 0; iteration < 48; iteration++)
    {
        Vector3 point =
            supportMinkowski3D(
                a,
                b,
                direction
            );

        if (tfDot(point, direction) < 0.0f)
            return false;

        simplex.push_back(point);

        if (handleSimplex3D(simplex, direction))
        {
            if (outSimplex != nullptr)
                *outSimplex = simplex;
            return true;
        }

        if (tfLength(direction) < 0.000001f)
            return true;
    }

    if (outSimplex != nullptr)
        outSimplex->clear();
    return false;
}

float tfBroadphaseRadius3D(const TFObject3D &obj)
{
    const float sx = fabsf(obj.scaleX);
    const float sy = fabsf(obj.scaleY);
    const float sz = fabsf(obj.scaleZ);

    float r = 1.5f * max(sx, max(sy, sz));
    if (obj.shape == "cube")
        r = 1.2f * sqrtf(sx*sx + sy*sy + sz*sz);
    else if (obj.shape == "cone")
        r = 1.5f * sqrtf(sx*sx + sy*sy + sz*sz);
    else if (obj.shape == "triangle")
        r = 1.85f * sqrtf(sx*sx + sy*sy + sz*sz);
    else if (obj.shape == "cylinder")
        r = sqrtf((1.5f*sx)*(1.5f*sx) + (1.5f*sy)*(1.5f*sy) + (1.5f*sz)*(1.5f*sz));
    else if (obj.shape == "capsule")
        r = sqrtf((1.15f*sx)*(1.15f*sx) + (2.0f*sy)*(2.0f*sy) + (1.15f*sz)*(1.15f*sz));
    else if (obj.shape == "crystal")
        r = sqrtf((1.15f*sx)*(1.15f*sx) + (1.90f*sy)*(1.90f*sy) + (1.15f*sz)*(1.15f*sz));
    else if (obj.shape == "diamond")
        r = sqrtf((1.25f*sx)*(1.25f*sx) + (1.90f*sy)*(1.90f*sy) + (1.25f*sz)*(1.25f*sz));
    else if (obj.shape == "prism")
        r = sqrtf((1.35f*sx)*(1.35f*sx) + (1.45f*sy)*(1.45f*sy) + (1.35f*sz)*(1.35f*sz));
    else if (obj.shape == "rock")
        r = sqrtf((1.25f*sx)*(1.25f*sx) + (1.15f*sy)*(1.15f*sy) + (1.10f*sz)*(1.10f*sz));

    auto it = gravityBodies.find(obj.name);
    if (it != gravityBodies.end() && it->second.type == TFBodyType::Soft)
    {
        const TFGravityBody &body = it->second;
        const float spread = 1.0f +
            min(0.85f, body.liquidSpread * 0.75f + body.volumeError * 0.18f);
        const float stretch = 1.0f + min(0.70f, body.stretch);
        r *= max(spread, stretch);
        r *= 1.10f;
    }

    return max(r, 0.001f);
}

bool tfBroadphaseMayTouch3D(const TFObject3D &a, const TFObject3D &b)
{
    const float ra = tfBroadphaseRadius3D(a);
    const float rb = tfBroadphaseRadius3D(b);
    const float radius = ra + rb;

    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz <= radius * radius;
}

bool collision3D(const TFObject3D &a, const TFObject3D &b)
{
    // Extremely cheap broadphase rejection before constructing any GJK shape
    // samples. Most object pairs are far apart, so this avoids the expensive
    // vector generation in that common case.
    if (!tfBroadphaseMayTouch3D(a, b))
        return false;

    const bool previousMode = tfCollisionGeometryBuilding;
    tfCollisionGeometryBuilding = true;

    TFShape3DData shapeA = make3DShapeData(a);
    TFShape3DData shapeB = make3DShapeData(b);

    const bool result = gjkCollision3D(shapeA, shapeB);
    tfCollisionGeometryBuilding = previousMode;
    return result;
}

float collisionRadius2D(const TFObject2D &obj)
{
    float x = shape2DHalfExtentX(obj);
    float y = shape2DHalfExtentY(obj);
    return sqrtf(x * x + y * y);
}

float collisionRadius3D(const TFObject3D &obj)
{
    if (obj.shape == "sphere")
    {
        float x = 1.5f * fabsf(obj.scaleX);
        float y = 1.5f * fabsf(obj.scaleY);
        float z = 1.5f * fabsf(obj.scaleZ);
        return sqrtf(x*x + y*y + z*z);
    }

    if (obj.shape == "cone")
        return sqrtf(
            (1.5f * fabsf(obj.scaleX)) *
            (1.5f * fabsf(obj.scaleX)) +
            (1.5f * fabsf(obj.scaleY)) *
            (1.5f * fabsf(obj.scaleY)) +
            (1.5f * fabsf(obj.scaleZ)) *
            (1.5f * fabsf(obj.scaleZ))
        );

    if (obj.shape == "triangle")
        return 3.0f * max(
            fabsf(obj.scaleX),
            max(fabsf(obj.scaleY), fabsf(obj.scaleZ))
        );

    if (obj.shape == "cylinder")
        return sqrtf(2.25f*fabsf(obj.scaleX)*fabsf(obj.scaleX) +
                     2.25f*fabsf(obj.scaleY)*fabsf(obj.scaleY) +
                     2.25f*fabsf(obj.scaleZ)*fabsf(obj.scaleZ));

    if (obj.shape == "capsule")
        return sqrtf(1.3225f*fabsf(obj.scaleX)*fabsf(obj.scaleX) +
                     4.0f*fabsf(obj.scaleY)*fabsf(obj.scaleY) +
                     1.3225f*fabsf(obj.scaleZ)*fabsf(obj.scaleZ));

    if (obj.shape == "crystal" || obj.shape == "diamond")
        return sqrtf(1.8f*fabsf(obj.scaleX)*fabsf(obj.scaleX) +
                     4.0f*fabsf(obj.scaleY)*fabsf(obj.scaleY) +
                     1.8f*fabsf(obj.scaleZ)*fabsf(obj.scaleZ));

    return sqrtf(
        (1.2f * fabsf(obj.scaleX)) *
        (1.2f * fabsf(obj.scaleX)) +
        (1.2f * fabsf(obj.scaleY)) *
        (1.2f * fabsf(obj.scaleY)) +
        (1.2f * fabsf(obj.scaleZ)) *
        (1.2f * fabsf(obj.scaleZ))
    );
}

TFObject3D makeCameraCollisionProxy()
{
    TFObject3D cam;
    cam.name = "cam";
    cam.shape = "sphere";
    cam.color = "white";
    cam.x = tfCamera.position.x;
    cam.y = tfCamera.position.y;
    cam.z = tfCamera.position.z;
    // Small collision volume around the camera position.
    cam.scaleX = 0.35f;
    cam.scaleY = 0.35f;
    cam.scaleZ = 0.35f;
    return cam;
}

bool hasCameraPhysicalCollisionPair(const string &a, const string &b)
{
    for (const TFCollisionPair &pair : cameraCollisionPairs)
    {
        if ((pair.first == a && pair.second == b) ||
            (pair.first == b && pair.second == a))
        {
            return true;
        }
    }

    return false;
}

bool sensorCollision(const string &first,
                     const string &second,
                     bool reportErrors)
{
    if (findGuiElement(first) != nullptr && findGuiElement(second) != nullptr)
        return guiCollisionSensor(first, second, reportErrors);

    if (first == "cam" || second == "cam")
    {
        if (first == "cam" && second == "cam")
        {
            if (reportErrors)
                error("Camera cannot collide with itself.");
            return false;
        }

        const string &otherName = (first == "cam") ? second : first;
        TFObject3D *other = findObject3D(otherName);

        if (other == nullptr)
        {
            if (reportErrors)
                error("Camera collision requires a 3D object.");
            return false;
        }

        TFObject3D cam = makeCameraCollisionProxy();
        return (first == "cam") ? collision3D(cam, *other)
                                 : collision3D(*other, cam);
    }

    int dimFirst = objectDimension(first);
    int dimSecond = objectDimension(second);

    if (dimFirst == 0 || dimSecond == 0)
    {
        if (reportErrors)
            error("Collision object not found.");

        return false;
    }

    if (dimFirst == -1 || dimSecond == -1)
    {
        if (reportErrors)
            error("Collision object name is ambiguous between 2D and 3D.");

        return false;
    }

    if (dimFirst != dimSecond)
    {
        if (reportErrors)
            error("Collision requires both objects to be in the same dimension.");

        return false;
    }

    if (dimFirst == 2)
    {
        TFObject2D *a = findObject2D(first);
        TFObject2D *b = findObject2D(second);

        if (a == nullptr || b == nullptr)
            return false;

        return collision2D(*a, *b);
    }

    TFObject3D *a = findObject3D(first);
    TFObject3D *b = findObject3D(second);

    if (a == nullptr || b == nullptr)
        return false;

    return collision3D(*a, *b);
}

void addPhysicalCollision(const string &first,
                          const string &second)
{
    if (findGuiElement(first) != nullptr && findGuiElement(second) != nullptr)
    {
        addGuiCollision(first, second);
        return;
    }

    if (first == "cam" || second == "cam")
    {
        if (first == second)
        {
            error("A collision needs two different objects.");
            return;
        }

        const string &otherName = (first == "cam") ? second : first;
        if (findObject3D(otherName) == nullptr)
        {
            error("Camera collision requires a 3D object.");
            return;
        }

        if (!hasCameraPhysicalCollisionPair(first, second))
            cameraCollisionPairs.push_back({first, second});

        return;
    }

    if (first == second)
    {
        error("A collision needs two different objects.");
        return;
    }

    if (objectDimension(first) == 0 ||
        objectDimension(second) == 0)
    {
        error("Collision object not found.");
        return;
    }

    int firstDim = objectDimension(first);
    int secondDim = objectDimension(second);

    if (firstDim == -1 || secondDim == -1)
    {
        error("Collision object name is ambiguous between 2D and 3D.");
        return;
    }

    if (firstDim != secondDim)
    {
        error("Collision requires both objects to be in the same dimension.");
        return;
    }

    if (hasPhysicalCollisionPair(first, second))
        return;

    physicalCollisionPairs.push_back(
        {first, second}
    );
}

struct TFAABB2D
{
    float minX = 0.0f;
    float maxX = 0.0f;
    float minY = 0.0f;
    float maxY = 0.0f;
};

bool tfGetAABB2D(const TFObject2D &obj, TFAABB2D &box)
{
    vector<Vector2> polygon = make2DShapePolygon(obj);
    if (polygon.empty())
        return false;

    box.minX = box.maxX = polygon.front().x;
    box.minY = box.maxY = polygon.front().y;

    for (const Vector2 &p : polygon)
    {
        box.minX = min(box.minX, p.x);
        box.maxX = max(box.maxX, p.x);
        box.minY = min(box.minY, p.y);
        box.maxY = max(box.maxY, p.y);
    }
    return true;
}

struct TFAABB3D
{
    float minX = 0.0f;
    float maxX = 0.0f;
    float minY = 0.0f;
    float maxY = 0.0f;
    float minZ = 0.0f;
    float maxZ = 0.0f;
};

bool tfGetAABB3D(const TFObject3D &obj, TFAABB3D &box)
{
    if (obj.shape == "sphere" &&
        shapeDeformations.count(deformationKey3D(obj.name)) == 0)
    {
        float hx = 1.5f * fabsf(obj.scaleX);
        float hy = 1.5f * fabsf(obj.scaleY);
        float hz = 1.5f * fabsf(obj.scaleZ);
        box.minX = obj.x - hx; box.maxX = obj.x + hx;
        box.minY = obj.y - hy; box.maxY = obj.y + hy;
        box.minZ = obj.z - hz; box.maxZ = obj.z + hz;
        return true;
    }

    TFShape3DData data = make3DShapeData(obj);
    if (data.points.empty())
    {
        float r = collisionRadius3D(obj);
        box.minX = obj.x - r; box.maxX = obj.x + r;
        box.minY = obj.y - r; box.maxY = obj.y + r;
        box.minZ = obj.z - r; box.maxZ = obj.z + r;
        return true;
    }

    box.minX = box.maxX = data.points.front().x;
    box.minY = box.maxY = data.points.front().y;
    box.minZ = box.maxZ = data.points.front().z;

    for (const Vector3 &p : data.points)
    {
        box.minX = min(box.minX, p.x); box.maxX = max(box.maxX, p.x);
        box.minY = min(box.minY, p.y); box.maxY = max(box.maxY, p.y);
        box.minZ = min(box.minZ, p.z); box.maxZ = max(box.maxZ, p.z);
    }
    return true;
}

void resolve2DPhysicalCollision(TFObject2D &a,
                                TFObject2D &b)
{
    if (!collision2D(a, b))
        return;

    TFAABB2D boxA, boxB;
    if (!tfGetAABB2D(a, boxA) || !tfGetAABB2D(b, boxB))
        return;

    float overlapX = min(boxA.maxX, boxB.maxX) - max(boxA.minX, boxB.minX);
    float overlapY = min(boxA.maxY, boxB.maxY) - max(boxA.minY, boxB.minY);
    if (overlapX <= 0.0f || overlapY <= 0.0f)
        return;

    const float epsilon = 0.01f;
    if (overlapX <= overlapY)
    {
        float centerA = (boxA.minX + boxA.maxX) * 0.5f;
        float centerB = (boxB.minX + boxB.maxX) * 0.5f;
        a.x += (centerA < centerB ? -1.0f : 1.0f) * (overlapX + epsilon);
    }
    else
    {
        float centerA = (boxA.minY + boxA.maxY) * 0.5f;
        float centerB = (boxB.minY + boxB.maxY) * 0.5f;
        a.y += (centerA < centerB ? -1.0f : 1.0f) * (overlapY + epsilon);
    }
}

bool tfHasGravityBody(const string &name)
{
    return gravityBodies.find(name) != gravityBodies.end();
}

// EPA computes the closest separating plane of the actual Minkowski
// volume. Unlike an AABB resolver, this follows cone/pyramid/crystal
// surfaces instead of reacting to empty space inside their bounding box.
struct TFEPAFace
{
    int a = 0;
    int b = 0;
    int c = 0;
    Vector3 normal = {0,0,0};
    float distance = 0.0f;
};

bool tfMakeEPAFace(const vector<Vector3> &vertices,
                   int ia, int ib, int ic,
                   TFEPAFace &face)
{
    if (ia < 0 || ib < 0 || ic < 0 ||
        ia >= (int)vertices.size() ||
        ib >= (int)vertices.size() ||
        ic >= (int)vertices.size())
        return false;

    Vector3 ab = tfSub(vertices[ib], vertices[ia]);
    Vector3 ac = tfSub(vertices[ic], vertices[ia]);
    Vector3 n = tfCross(ab, ac);
    float len = tfLength(n);
    if (len < 0.000001f)
        return false;

    n = tfMul(n, 1.0f / len);
    float d = tfDot(n, vertices[ia]);

    // EPA keeps the origin behind every outward support plane.
    if (d < 0.0f)
    {
        swap(ib, ic);
        ab = tfSub(vertices[ib], vertices[ia]);
        ac = tfSub(vertices[ic], vertices[ia]);
        n = tfNormalize(tfCross(ab, ac));
        d = tfDot(n, vertices[ia]);
    }

    if (d < 0.0f)
        return false;

    face.a = ia;
    face.b = ib;
    face.c = ic;
    face.normal = n;
    face.distance = d;
    return true;
}

void tfEPAAddBoundaryEdge(vector<pair<int,int>> &edges, int a, int b)
{
    for (auto it = edges.begin(); it != edges.end(); ++it)
    {
        if (it->first == b && it->second == a)
        {
            edges.erase(it);
            return;
        }
    }
    edges.push_back({a,b});
}

bool tfEPA3D(const TFShape3DData &shapeA,
             const TFShape3DData &shapeB,
             vector<Vector3> simplex,
             Vector3 &normal,
             float &depth)
{
    if (simplex.size() < 4)
        return false;

    vector<Vector3> vertices = simplex;
    vector<TFEPAFace> faces;
    faces.reserve(96);

    TFEPAFace face;
    if (tfMakeEPAFace(vertices,0,1,2,face)) faces.push_back(face);
    if (tfMakeEPAFace(vertices,0,3,1,face)) faces.push_back(face);
    if (tfMakeEPAFace(vertices,0,2,3,face)) faces.push_back(face);
    if (tfMakeEPAFace(vertices,1,3,2,face)) faces.push_back(face);

    if (faces.size() < 4)
        return false;

    // More iterations only when the machine is currently healthy.
    const int maxIterations =
        (tfPerf.processCPUPercent < 70.0f && tfPerf.frameTimeEMA < 0.018f)
        ? 32 : 18;
    const float tolerance =
        (tfPerf.processCPUPercent < 70.0f && tfPerf.frameTimeEMA < 0.018f)
        ? 0.00001f : 0.00005f;

    for (int iteration = 0; iteration < maxIterations; ++iteration)
    {
        int closest = -1;
        float bestDistance = numeric_limits<float>::infinity();

        for (int i = 0; i < (int)faces.size(); ++i)
        {
            if (faces[i].distance < bestDistance)
            {
                bestDistance = faces[i].distance;
                closest = i;
            }
        }

        if (closest < 0)
            break;

        const TFEPAFace bestFace = faces[closest];
        Vector3 support = supportMinkowski3D(
            shapeA, shapeB, bestFace.normal
        );
        float supportDistance = tfDot(support, bestFace.normal);

        if (supportDistance - bestFace.distance <= tolerance)
        {
            normal = bestFace.normal;
            depth = max(0.0f, supportDistance);
            return depth > 0.000001f;
        }

        bool duplicate = false;
        for (const Vector3 &v : vertices)
        {
            if (tfLength(tfSub(v, support)) < tolerance * 0.5f)
            {
                duplicate = true;
                break;
            }
        }

        if (duplicate)
        {
            normal = bestFace.normal;
            depth = max(0.0f, bestFace.distance);
            return depth > 0.000001f;
        }

        int newIndex = (int)vertices.size();
        vertices.push_back(support);

        vector<pair<int,int>> boundary;
        vector<char> visible(faces.size(), 0);

        for (size_t i = 0; i < faces.size(); ++i)
        {
            const TFEPAFace &f = faces[i];
            Vector3 fromFace = tfSub(support, vertices[f.a]);
            if (tfDot(f.normal, fromFace) > tolerance)
            {
                visible[i] = 1;
                tfEPAAddBoundaryEdge(boundary, f.a, f.b);
                tfEPAAddBoundaryEdge(boundary, f.b, f.c);
                tfEPAAddBoundaryEdge(boundary, f.c, f.a);
            }
        }

        vector<TFEPAFace> kept;
        kept.reserve(faces.size() + boundary.size());
        for (size_t i = 0; i < faces.size(); ++i)
            if (!visible[i])
                kept.push_back(faces[i]);

        faces.swap(kept);

        for (const auto &edge : boundary)
        {
            TFEPAFace nf;
            if (tfMakeEPAFace(vertices,
                              edge.first,
                              edge.second,
                              newIndex,
                              nf))
            {
                faces.push_back(nf);
            }
        }

        if (faces.empty())
            break;
    }

    // Return the best face accumulated so far rather than leaving the body
    // unresolved at the iteration limit.
    int closest = -1;
    float bestDistance = numeric_limits<float>::infinity();
    for (int i = 0; i < (int)faces.size(); ++i)
    {
        if (faces[i].distance < bestDistance)
        {
            bestDistance = faces[i].distance;
            closest = i;
        }
    }

    if (closest >= 0)
    {
        normal = faces[closest].normal;
        depth = max(0.0f, faces[closest].distance);
        return depth > 0.000001f;
    }

    return false;
}

bool tfComputeShapeMTV3D(const TFObject3D &a,
                         const TFObject3D &b,
                         Vector3 &normal,
                         float &depth)
{
    const bool previousMode = tfCollisionGeometryBuilding;
    tfCollisionGeometryBuilding = true;

    TFShape3DData shapeA = make3DShapeData(a);
    TFShape3DData shapeB = make3DShapeData(b);
    vector<Vector3> simplex;

    bool intersects = gjkCollision3D(shapeA, shapeB, &simplex);
    if (!intersects)
    {
        tfCollisionGeometryBuilding = previousMode;
        return false;
    }

    bool solved = tfEPA3D(shapeA, shapeB, simplex, normal, depth);

    tfCollisionGeometryBuilding = previousMode;

    if (!solved)
        return false;

    Vector3 centerDelta =
    {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    };

    // Always orient the contact normal from the collider toward the moving
    // body. This makes the vertical-support test deterministic for cones,
    // pyramids and crystals.
    if (tfLength(centerDelta) > 0.0001f &&
        tfDot(normal, centerDelta) < 0.0f)
    {
        normal = tfMul(normal, -1.0f);
    }

    return true;
}

void resolve3DPhysicalCollision(TFObject3D &a,
                                TFObject3D &b)
{
    if (!collision3D(a, b))
        return;

    bool dynamicA = tfHasGravityBody(a.name);
    bool dynamicB = tfHasGravityBody(b.name);
    if (!dynamicA && !dynamicB)
        return;

    TFObject3D *moving = nullptr;
    TFObject3D *solid = nullptr;
    bool splitCorrection = false;

    if (dynamicA && !dynamicB)
    {
        moving = &a;
        solid = &b;
    }
    else if (!dynamicA && dynamicB)
    {
        moving = &b;
        solid = &a;
    }
    else
    {
        bool groundedA = gravityObjectGrounded(a.name);
        bool groundedB = gravityObjectGrounded(b.name);

        if (groundedA && !groundedB)
        {
            moving = &b;
            solid = &a;
        }
        else if (!groundedA && groundedB)
        {
            moving = &a;
            solid = &b;
        }
        else
        {
            moving = &a;
            solid = &b;
            splitCorrection = true;
        }
    }

    Vector3 normal;
    float depth = 0.0f;
    if (!tfComputeShapeMTV3D(*moving, *solid, normal, depth))
        return;

    // Small positional slop prevents jitter without creating visible gaps.
    const float slop = 0.00001f;
    const float correction = depth + slop;

    moving->x += normal.x * correction;
    moving->y += normal.y * correction;
    moving->z += normal.z * correction;

    if (splitCorrection)
    {
        solid->x -= normal.x * correction;
        solid->y -= normal.y * correction;
        solid->z -= normal.z * correction;
    }

    auto bodyIt = gravityBodies.find(moving->name);
    if (bodyIt != gravityBodies.end())
    {
        TFGravityBody &body = bodyIt->second;
        float vn = body.velocity.x * normal.x +
                   body.velocity.y * normal.y +
                   body.velocity.z * normal.z;
        if (vn < 0.0f)
        {
            body.velocity.x -= vn * normal.x;
            body.velocity.y -= vn * normal.y;
            body.velocity.z -= vn * normal.z;
        }
    }

    if (tfHasGravityBody(moving->name))
        tfSnapGravityBodyToSupport(moving->name);
}
void tfSnapGravityBodyToSupport(const string &name)
{
    TFObject3D *obj = findObject3D(name);
    if (!obj)
        return;

    auto bit = gravityBodies.find(name);
    if (bit == gravityBodies.end())
        return;

    TFGravityBody &body = bit->second;

    const float originalY = obj->y;
    float bestSupportY = -numeric_limits<float>::infinity();
    bool found = false;

    for (const TFCollisionPair &pair : physicalCollisionPairs)
    {
        if (pair.first != name && pair.second != name)
            continue;

        const string &otherName =
            (pair.first == name) ? pair.second : pair.first;
        TFObject3D *other = findObject3D(otherName);
        if (!other)
            continue;

        // A support must actually be below the body's center. This prevents
        // a side wall or a nearby cone from being interpreted as a floor.
        if (other->y > obj->y + 0.0001f)
            continue;

        if (!collision3D(*obj, *other))
            continue;

        Vector3 normal;
        float depth = 0.0f;
        if (!tfComputeShapeMTV3D(*obj, *other, normal, depth))
            continue;

        // Only normals pointing upward represent a floor/support for this body.
        if (normal.y < 0.55f)
            continue;

        // Find the exact contact height by moving only upward until the two
        // shapes stop intersecting, then bisecting the last interval.
        const float search = max(0.25f,
            tfBroadphaseRadius3D(*obj) +
            tfBroadphaseRadius3D(*other) + 0.25f);

        float lowY = obj->y;
        float highY = obj->y + search;

        obj->y = highY;
        bool highCollides = collision3D(*obj, *other);
        if (highCollides)
        {
            obj->y = originalY;
            continue;
        }

        obj->y = lowY;
        for (int i = 0; i < 12; ++i)
        {
            float mid = (lowY + highY) * 0.5f;
            obj->y = mid;
            if (collision3D(*obj, *other))
                lowY = mid;
            else
                highY = mid;
        }

        obj->y = highY - 0.000002f;
        if (obj->y > bestSupportY)
        {
            bestSupportY = obj->y;
            found = true;
        }

        obj->y = originalY;
    }

    if (found)
    {
        obj->y = bestSupportY;
        body.velocity.y = 0.0f;
        gravityGrounded[name] = true;
    }
}
void resolveCameraPhysicalCollision(const string &first, const string &second)
{
    if (first != "cam" && second != "cam")
        return;

    const string &otherName = (first == "cam") ? second : first;
    TFObject3D *other = findObject3D(otherName);
    if (other == nullptr)
        return;

    TFObject3D cam = makeCameraCollisionProxy();
    if (!collision3D(cam, *other))
        return;

    TFAABB3D boxCam, boxOther;
    if (!tfGetAABB3D(cam, boxCam) || !tfGetAABB3D(*other, boxOther))
        return;

    float overlapX = min(boxCam.maxX, boxOther.maxX) - max(boxCam.minX, boxOther.minX);
    float overlapY = min(boxCam.maxY, boxOther.maxY) - max(boxCam.minY, boxOther.minY);
    float overlapZ = min(boxCam.maxZ, boxOther.maxZ) - max(boxCam.minZ, boxOther.minZ);

    if (overlapX <= 0.0f || overlapY <= 0.0f || overlapZ <= 0.0f)
        return;

    const float epsilon = 0.001f;
    if (overlapX <= overlapY && overlapX <= overlapZ)
    {
        float a = (boxCam.minX + boxCam.maxX) * 0.5f;
        float b = (boxOther.minX + boxOther.maxX) * 0.5f;
        tfCamera.position.x += (a < b ? -1.0f : 1.0f) * (overlapX + epsilon);
    }
    else if (overlapY <= overlapZ)
    {
        float a = (boxCam.minY + boxCam.maxY) * 0.5f;
        float b = (boxOther.minY + boxOther.maxY) * 0.5f;
        tfCamera.position.y += (a < b ? -1.0f : 1.0f) * (overlapY + epsilon);
    }
    else
    {
        float a = (boxCam.minZ + boxCam.maxZ) * 0.5f;
        float b = (boxOther.minZ + boxOther.maxZ) * 0.5f;
        tfCamera.position.z += (a < b ? -1.0f : 1.0f) * (overlapZ + epsilon);
    }

    tfApplyCameraRotation(tfCamera);
}

void resolvePhysicalCollisions()
{
    // Several passes let a chain of solid objects settle without
    // requiring velocity, gravity, or momentum.
    for (int pass = 0; pass < max(1, tfPerf.collisionPasses); pass++)
    {
        for (const TFCollisionPair &pair : physicalCollisionPairs)
        {
            int firstDim = objectDimension(pair.first);
            int secondDim = objectDimension(pair.second);

            if (firstDim == 2 && secondDim == 2)
            {
                TFObject2D *a = findObject2D(pair.first);
                TFObject2D *b = findObject2D(pair.second);

                if (a != nullptr && b != nullptr)
                    resolve2DPhysicalCollision(*a, *b);
            }
            else if (firstDim == 3 && secondDim == 3)
            {
                TFObject3D *a = findObject3D(pair.first);
                TFObject3D *b = findObject3D(pair.second);

                if (a != nullptr && b != nullptr)
                {
                    if (collision3D(*a, *b))
                    {
                        // Record the impact for whichever side is the soft body.
                        auto ga = gravityBodies.find(a->name);
                        if (ga != gravityBodies.end() && ga->second.type == TFBodyType::Soft)
                            recordSoftBodyCollision(*a, *b);

                        auto gb = gravityBodies.find(b->name);
                        if (gb != gravityBodies.end() && gb->second.type == TFBodyType::Soft)
                            recordSoftBodyCollision(*b, *a);

                        // Grounding is symmetric with respect to the collision
                        // command order. The gravity object may be either
                        // pair.first or pair.second.
                        auto gravityA = gravityBodies.find(pair.first);
                        auto gravityB = gravityBodies.find(pair.second);

                        if (gravityA != gravityBodies.end() && b->y < a->y)
                            gravityGrounded[pair.first] = true;

                        if (gravityB != gravityBodies.end() && a->y < b->y)
                            gravityGrounded[pair.second] = true;

                        resolve3DPhysicalCollision(*a, *b);
                    }
                }
            }
        }

        for (const TFCollisionPair &pair : cameraCollisionPairs)
            resolveCameraPhysicalCollision(pair.first, pair.second);
    }
}

// ============================================================
// 2D COLLISION CLUSTER POLYGON BUDGET
// Same idea as the 3D version further below: objects that are
// currently touching share ONE fixed segment budget instead of each
// paying full price. See the big comment above the 3D version for
// the full reasoning.
// ============================================================

unordered_map<string, int> tfComputeTouchingClusterSizes2D()
{
    unordered_map<string, string> parent;

    for (const TFObject2D &obj : objects2D)
        parent[obj.name] = obj.name;

    for (const TFCollisionPair &pair : physicalCollisionPairs)
    {
        if (objectDimension(pair.first) != 2 ||
            objectDimension(pair.second) != 2)
        {
            continue;
        }

        TFObject2D *a = findObject2D(pair.first);
        TFObject2D *b = findObject2D(pair.second);

        if (a == nullptr || b == nullptr)
            continue;

        // Only objects touching RIGHT NOW share a budget. A declared
        // "colision" pair that is not currently overlapping costs
        // nothing extra.
        if (!collision2D(*a, *b))
            continue;

        tfClusterUnion(parent, pair.first, pair.second);
    }

    unordered_map<string, int> sizeOfRoot;

    for (const TFObject2D &obj : objects2D)
        sizeOfRoot[tfClusterFind(parent, obj.name)]++;

    unordered_map<string, int> sizeOfObject;

    for (const TFObject2D &obj : objects2D)
        sizeOfObject[obj.name] = sizeOfRoot[tfClusterFind(parent, obj.name)];

    return sizeOfObject;
}

// Eased (smoothed) cluster size per object. An object joining or
// leaving a pile never causes an instant, visible jump in detail -
// the shared budget eases in and out instead of popping.
unordered_map<string, float> tfSmoothedClusterSize2D;
unordered_map<string, int> tfClusterTargetSize2D;
float tfClusterBudgetTimer2D = 1.0f;

void tfUpdateClusterBudgets2D(float dt)
{
    if (objects2D.empty())
        return;

    tfClusterBudgetTimer2D += max(0.0f, dt);
    if (tfClusterBudgetTimer2D >= 0.25f || tfClusterTargetSize2D.empty())
    {
        tfClusterTargetSize2D = tfComputeTouchingClusterSizes2D();
        tfClusterBudgetTimer2D = 0.0f;
    }

    const unordered_map<string, int> &rawSizes = tfClusterTargetSize2D;

    // Higher = reacts to a pile forming/breaking faster; lower = smoother.
    const float smoothingSpeed = 3.5f;
    const float easing = 1.0f - expf(-smoothingSpeed * max(0.0f, dt));

    for (const TFObject2D &obj : objects2D)
    {
        auto rawIt = rawSizes.find(obj.name);
        float target =
            static_cast<float>(rawIt != rawSizes.end() ? rawIt->second : 1);

        float &current = tfSmoothedClusterSize2D[obj.name];

        if (current <= 0.0f)
            current = target; // First sighting: no fake ramp from zero.
        else
            current += (target - current) * easing;
    }

    for (auto it = tfSmoothedClusterSize2D.begin();
         it != tfSmoothedClusterSize2D.end();)
    {
        if (findObject2D(it->first) == nullptr)
            it = tfSmoothedClusterSize2D.erase(it);
        else
            ++it;
    }
}

// 2D outlines are not built from fixed LOD tiers like the 3D meshes,
// so the shared budget scales the segment count of the fan/outline
// directly instead of snapping between steps - continuous, so it is
// even smoother than the 3D tiers.
int tfClusterSegmentCount2D(const string &objectName,
                            int baseSegments,
                            int minSegments)
{
    auto it = tfSmoothedClusterSize2D.find(objectName);
    float clusterSize = (it != tfSmoothedClusterSize2D.end()) ? it->second : 1.0f;
    int n = max(1, static_cast<int>(roundf(clusterSize)));

    if (n <= 1)
        return baseSegments;

    int shared = max(minSegments, baseSegments / n);
    return min(baseSegments, shared);
}

// -------------------------
// 2D rendering with deformation
// -------------------------

void draw2DDeformedShape(const TFObject2D &obj)
{
    Color objectColor = getColor(obj.color);

    if (obj.shape == "cube")
    {
        Vector2 points[4] =
        {
            apply2DDeformations(obj, {-100.0f * obj.scaleX, -100.0f * obj.scaleY}),
            apply2DDeformations(obj, { 100.0f * obj.scaleX, -100.0f * obj.scaleY}),
            apply2DDeformations(obj, { 100.0f * obj.scaleX,  100.0f * obj.scaleY}),
            apply2DDeformations(obj, {-100.0f * obj.scaleX,  100.0f * obj.scaleY})
        };

        for (Vector2 &p : points)
        {
            p.x += obj.x;
            p.y += obj.y;
        }

        DrawTriangle(points[0], points[1], points[2], objectColor);
        DrawTriangle(points[0], points[2], points[3], objectColor);
        return;
    }

    if (obj.shape == "sphere")
    {
        // Shrinks smoothly when this object is part of a touching pile,
        // instead of always paying the full 72-segment price.
        const int count = tfClusterSegmentCount2D(obj.name, 72, 10);
        vector<Vector2> points;

        points.reserve(count);

        for (int i = 0; i < count; i++)
        {
            float angle =
                2.0f * PI * static_cast<float>(i) /
                static_cast<float>(count);

            Vector2 p =
            {
                cosf(angle) * 100.0f * obj.scaleX,
                sinf(angle) * 100.0f * obj.scaleY
            };

            p = apply2DDeformations(obj, p);
            p.x += obj.x;
            p.y += obj.y;

            points.push_back(p);
        }

        DrawTriangleFan(points.data(), count, objectColor);
        return;
    }

    if (obj.shape == "cone")
    {
        Vector2 tip =
            apply2DDeformations(
                obj,
                {0.0f, -120.0f * obj.scaleY}
            );

        Vector2 left =
            apply2DDeformations(
                obj,
                {-110.0f * obj.scaleX, 80.0f * obj.scaleY}
            );

        Vector2 right =
            apply2DDeformations(
                obj,
                {110.0f * obj.scaleX, 80.0f * obj.scaleY}
            );

        tip.x += obj.x;   tip.y += obj.y;
        left.x += obj.x;  left.y += obj.y;
        right.x += obj.x; right.y += obj.y;

        DrawTriangle(tip, left, right, objectColor);

        const int count = tfClusterSegmentCount2D(obj.name, 48, 8);
        vector<Vector2> base;

        base.reserve(count);

        for (int i = 0; i < count; i++)
        {
            float angle =
                2.0f * PI * static_cast<float>(i) /
                static_cast<float>(count);

            Vector2 p =
            {
                cosf(angle) * 110.0f * obj.scaleX,
                80.0f * obj.scaleY +
                    sinf(angle) * 25.0f * obj.scaleY
            };

            p = apply2DDeformations(obj, p);
            p.x += obj.x;
            p.y += obj.y;

            base.push_back(p);
        }

        DrawTriangleFan(base.data(), count, objectColor);
        return;
    }

    if (obj.shape == "triangle")
    {
        Vector2 v1 =
            apply2DDeformations(
                obj,
                {0.0f, -120.0f * obj.scaleY}
            );

        Vector2 v2 =
            apply2DDeformations(
                obj,
                {-100.0f * obj.scaleX, 100.0f * obj.scaleY}
            );

        Vector2 v3 =
            apply2DDeformations(
                obj,
                {100.0f * obj.scaleX, 100.0f * obj.scaleY}
            );

        v1.x += obj.x; v1.y += obj.y;
        v2.x += obj.x; v2.y += obj.y;
        v3.x += obj.x; v3.y += obj.y;

        DrawTriangle(v1, v2, v3, objectColor);
    }
}

// -------------------------
// 3D rendering with deformation
// -------------------------

Vector3 world3D(const TFObject3D &obj, Vector3 local)
{
    local = apply3DDeformations(obj, local);

    return
    {
        obj.x + local.x,
        obj.y + local.y,
        obj.z + local.z
    };
}

void draw3DTriangle(const TFObject3D &obj,
                    Vector3 a,
                    Vector3 b,
                    Vector3 c,
                    Color color)
{
    DrawTriangle3D(
        world3D(obj, a),
        world3D(obj, b),
        world3D(obj, c),
        color
    );
}

void draw3DDeformedCube(const TFObject3D &obj,
                        Color color)
{
    const int grid = 8;

    float hx = 1.2f * fabsf(obj.scaleX);
    float hy = 1.2f * fabsf(obj.scaleY);
    float hz = 1.2f * fabsf(obj.scaleZ);

    auto facePoint =
        [&](int face, float u, float v) -> Vector3
        {
            switch (face)
            {
                case 0: return {-hx, v * hy, u * hz};
                case 1: return { hx, u * hy, v * hz};
                case 2: return { u * hx,  hy, v * hz};
                case 3: return { u * hx, -hy, v * hz};
                case 4: return { u * hx, v * hy,  hz};
                default:return { u * hx, v * hy, -hz};
            }
        };

    for (int face = 0; face < 6; face++)
    {
        for (int y = 0; y < grid; y++)
        {
            float v0 =
                -1.0f + 2.0f *
                static_cast<float>(y) /
                static_cast<float>(grid);

            float v1 =
                -1.0f + 2.0f *
                static_cast<float>(y + 1) /
                static_cast<float>(grid);

            for (int x = 0; x < grid; x++)
            {
                float u0 =
                    -1.0f + 2.0f *
                    static_cast<float>(x) /
                    static_cast<float>(grid);

                float u1 =
                    -1.0f + 2.0f *
                    static_cast<float>(x + 1) /
                    static_cast<float>(grid);

                Vector3 p00 = facePoint(face, u0, v0);
                Vector3 p10 = facePoint(face, u1, v0);
                Vector3 p11 = facePoint(face, u1, v1);
                Vector3 p01 = facePoint(face, u0, v1);

                draw3DTriangle(obj, p00, p10, p11, color);
                draw3DTriangle(obj, p00, p11, p01, color);
            }
        }
    }
}

void draw3DDeformedSphere(const TFObject3D &obj,
                          Color color)
{
    const int rings = 12;
    const int slices = 24;

    for (int r = 0; r < rings; r++)
    {
        float v0 = static_cast<float>(r) /
                   static_cast<float>(rings);

        float v1 = static_cast<float>(r + 1) /
                   static_cast<float>(rings);

        float phi0 = PI * v0;
        float phi1 = PI * v1;

        for (int s = 0; s < slices; s++)
        {
            float u0 = static_cast<float>(s) /
                       static_cast<float>(slices);

            float u1 = static_cast<float>(s + 1) /
                       static_cast<float>(slices);

            float theta0 = 2.0f * PI * u0;
            float theta1 = 2.0f * PI * u1;

            Vector3 p00 =
            {
                1.5f * obj.scaleX * sinf(phi0) * cosf(theta0),
                1.5f * obj.scaleY * cosf(phi0),
                1.5f * obj.scaleZ * sinf(phi0) * sinf(theta0)
            };

            Vector3 p10 =
            {
                1.5f * obj.scaleX * sinf(phi0) * cosf(theta1),
                1.5f * obj.scaleY * cosf(phi0),
                1.5f * obj.scaleZ * sinf(phi0) * sinf(theta1)
            };

            Vector3 p11 =
            {
                1.5f * obj.scaleX * sinf(phi1) * cosf(theta1),
                1.5f * obj.scaleY * cosf(phi1),
                1.5f * obj.scaleZ * sinf(phi1) * sinf(theta1)
            };

            Vector3 p01 =
            {
                1.5f * obj.scaleX * sinf(phi1) * cosf(theta0),
                1.5f * obj.scaleY * cosf(phi1),
                1.5f * obj.scaleZ * sinf(phi1) * sinf(theta0)
            };

            draw3DTriangle(obj, p00, p10, p11, color);
            draw3DTriangle(obj, p00, p11, p01, color);
        }
    }
}

void draw3DDeformedCone(const TFObject3D &obj,
                        Color color)
{
    // Closed cone mesh: the side wall and the base cap use the SAME
    // deformed vertices. This prevents visible seams/open holes when
    // shape p[...] is active.
    const int slices = 40;
    const int rings = 10;

    auto conePoint = [&](float radius, float y, float angle) -> Vector3
    {
        Vector3 p =
        {
            radius * obj.scaleX * cosf(angle),
            y * obj.scaleY,
            radius * obj.scaleZ * sinf(angle)
        };
        return p;
    };

    // Side surface. Every ring shares its vertices conceptually, and the
    // apex is generated as a real deformed vertex rather than a special
    // undeformed replacement.
    for (int r = 0; r < rings; r++)
    {
        float t0 = static_cast<float>(r) /
                   static_cast<float>(rings);
        float t1 = static_cast<float>(r + 1) /
                   static_cast<float>(rings);

        float radius0 = 1.5f * (1.0f - t0);
        float radius1 = 1.5f * (1.0f - t1);

        float y0 = -1.5f + 3.0f * t0;
        float y1 = -1.5f + 3.0f * t1;

        for (int s = 0; s < slices; s++)
        {
            float a0 =
                2.0f * PI * static_cast<float>(s) /
                static_cast<float>(slices);
            float a1 =
                2.0f * PI * static_cast<float>(s + 1) /
                static_cast<float>(slices);

            Vector3 p00 = conePoint(radius0, y0, a0);
            Vector3 p10 = conePoint(radius0, y0, a1);
            Vector3 p11 = conePoint(radius1, y1, a1);
            Vector3 p01 = conePoint(radius1, y1, a0);

            draw3DTriangle(obj, p00, p10, p11, color);
            draw3DTriangle(obj, p00, p11, p01, color);
        }
    }

    // Closed base cap. The center is passed through the exact same
    // deformation pipeline as the surrounding rim.
    Vector3 center = {0.0f, -1.5f * obj.scaleY, 0.0f};

    for (int s = 0; s < slices; s++)
    {
        float a0 =
            2.0f * PI * static_cast<float>(s) /
            static_cast<float>(slices);
        float a1 =
            2.0f * PI * static_cast<float>(s + 1) /
            static_cast<float>(slices);

        Vector3 p0 = conePoint(1.5f, -1.5f, a0);
        Vector3 p1 = conePoint(1.5f, -1.5f, a1);

        draw3DTriangle(obj, center, p1, p0, color);
    }
}

void draw3DDeformedPyramid(const TFObject3D &obj,
                           Color lightColor,
                           Color baseColor,
                           Color shadowColor,
                           Color darkShadow)
{
    Vector3 base0 =
    {
        -1.3f * obj.scaleX,
        -1.0f * obj.scaleY,
         1.3f * obj.scaleZ
    };

    Vector3 base1 =
    {
         1.3f * obj.scaleX,
        -1.0f * obj.scaleY,
         1.0f * obj.scaleZ
    };

    Vector3 base2 =
    {
         0.8f * obj.scaleX,
        -1.0f * obj.scaleY,
        -1.3f * obj.scaleZ
    };

    Vector3 base3 =
    {
        -1.0f * obj.scaleX,
        -1.0f * obj.scaleY,
        -1.0f * obj.scaleZ
    };

    Vector3 top =
    {
        0.0f,
        1.8f * obj.scaleY,
        0.4f * obj.scaleZ
    };

    draw3DTriangle(obj, base0, base1, top, lightColor);
    draw3DTriangle(obj, base1, base2, top, baseColor);
    draw3DTriangle(obj, base2, base3, top, shadowColor);
    draw3DTriangle(obj, base3, base0, top, darkShadow);

    draw3DTriangle(obj, base0, base3, base2, darkShadow);
    draw3DTriangle(obj, base0, base2, base1, darkShadow);
}

void draw3DDeformedObject(const TFObject3D &obj)
{
    Color baseColor = getColor(obj.color);

    Color lightColor =
    {
        (unsigned char)min(255, (int)(baseColor.r * 1.3f)),
        (unsigned char)min(255, (int)(baseColor.g * 1.3f)),
        (unsigned char)min(255, (int)(baseColor.b * 1.3f)),
        255
    };

    Color shadowColor =
    {
        (unsigned char)(baseColor.r * 0.6f),
        (unsigned char)(baseColor.g * 0.6f),
        (unsigned char)(baseColor.b * 0.6f),
        255
    };

    Color darkShadow =
    {
        (unsigned char)(baseColor.r * 0.35f),
        (unsigned char)(baseColor.g * 0.35f),
        (unsigned char)(baseColor.b * 0.35f),
        255
    };

    if (obj.shape == "cube")
    {
        draw3DDeformedCube(obj, baseColor);
    }
    else if (obj.shape == "sphere")
    {
        draw3DDeformedSphere(obj, baseColor);
    }
    else if (obj.shape == "cone")
    {
        draw3DDeformedCone(obj, baseColor);
    }
    else if (obj.shape == "triangle")
    {
        draw3DDeformedPyramid(
            obj,
            lightColor,
            baseColor,
            shadowColor,
            darkShadow
        );
    }
    else if (obj.shape == "cylinder")
    {
        rlPushMatrix();
        rlTranslatef(obj.x, obj.y, obj.z);
        rlScalef(obj.scaleX, obj.scaleY, obj.scaleZ);
        DrawCylinder({0,0,0},1.5f,1.5f,3.0f,32,baseColor);
        rlPopMatrix();
    }
    else if (obj.shape == "capsule")
    {
        rlPushMatrix();
        rlTranslatef(obj.x, obj.y, obj.z);
        rlScalef(obj.scaleX, obj.scaleY, obj.scaleZ);
        DrawCylinder({0,0,0},1.15f,1.15f,1.7f,24,baseColor);
        DrawSphere({0,0.85f,0},1.15f,baseColor);
        DrawSphere({0,-0.85f,0},1.15f,baseColor);
        rlPopMatrix();
    }
}

 
 
// ============================================================
// OPTIMIZED 3D MESH / LOD / FRUSTUM CULLING
// Procedural closed meshes replace the primitive draw path.
// The logical TigerFlash object always remains; GPU mesh data is
// created only when visible and discarded when culled.
// ============================================================

struct TFOptimized3DMeshCache
{
    Mesh mesh = { 0 };
    bool valid = false;
    int lod = -1;
    size_t signature = 0;
    size_t observedSignature = 0;
    uint64_t lastSignatureCheckFrame = 0;
    uint64_t lastVisibleFrame = 0;
    uint64_t lastBuildFrame = 0;

    // Soft bodies keep one undeformed CPU template and update their existing
    // GPU mesh in place. This is the SAME old molding function, just sampled
    // every render frame instead of rebuilding the whole mesh.
    vector<float> softBaseVertices;
    vector<int> softNormalGroup;
    vector<Vector3> softNormalSums;
};

unordered_map<string, TFOptimized3DMeshCache> optimized3DMeshes;
Material optimized3DMaterial = { 0 };
bool optimized3DMaterialReady = false;

// ============================================================
// BAKED OPTICAL MATERIALS: CRYSTAL / DIAMOND / PRISM
// ============================================================
// These three materials use one small environment texture that is generated
// ONCE when the 3D window starts. It contains precomputed light gradients,
// caustic bands, spectral highlights and several broad light sources. The
// shader then samples that baked field for reflection/refraction instead of
// running ray tracing every frame.
//
// The optical objects are intentionally only slightly transparent. Their
// strong Fresnel term keeps them visually dense while the refraction channel
// lets a controlled amount of the scene/light field show through.
Shader tfSurfaceShader = { 0 };
bool tfSurfaceShaderReady = false;
int tfSurfaceLocColor = -1;

Shader tfOpticalShader = { 0 };
bool tfOpticalShaderReady = false;
Texture2D tfBakedOpticalEnvironment = { 0 };
bool tfBakedOpticalEnvironmentReady = false;
int tfOpticalLocViewPos = -1;
int tfOpticalLocEnvironment = -1;
int tfOpticalLocParams = -1;   // reflectivity, transmission, IOR, dispersion
int tfOpticalLocAlpha = -1;
int tfOpticalLocTime = -1;
int tfOpticalLocSoftFluid = -1;

bool tfIsOpticalShape(const TFObject3D &obj)
{
    if (obj.shape == "crystal" ||
        obj.shape == "diamond" ||
        obj.shape == "prism")
    {
        return true;
    }

    auto it = gravityBodies.find(obj.name);
    return it != gravityBodies.end() &&
           it->second.type == TFBodyType::Soft &&
           it->second.softness > 0.01f;
}

static Color tfOpticalBakeColor(float r, float g, float b, float exposure)
{
    r = 255.0f * (1.0f - expf(-max(0.0f, r) * exposure));
    g = 255.0f * (1.0f - expf(-max(0.0f, g) * exposure));
    b = 255.0f * (1.0f - expf(-max(0.0f, b) * exposure));

    return {
        (unsigned char)max(0.0f, min(255.0f, r)),
        (unsigned char)max(0.0f, min(255.0f, g)),
        (unsigned char)max(0.0f, min(255.0f, b)),
        255
    };
}

void tfBuildBakedOpticalEnvironment()
{
    if (tfBakedOpticalEnvironmentReady)
        return;

    // Small enough to keep memory tiny, large enough to avoid blocky
    // reflections on the faceted shapes.
    const int width = 512;
    const int height = 256;

    Image image = GenImageColor(width, height, BLACK);
    if (image.data == nullptr)
        return;

    Color *pixels = static_cast<Color *>(image.data);

    auto gauss = [](float x, float y, float cx, float cy,
                    float sx, float sy) -> float
    {
        float dx = (x - cx) / max(0.0001f, sx);
        float dy = (y - cy) / max(0.0001f, sy);
        return expf(-0.5f * (dx * dx + dy * dy));
    };

    for (int y = 0; y < height; ++y)
    {
        float v = (float)y / (float)(height - 1);
        float sky = 1.0f - v;
        float horizon = expf(-powf((v - 0.58f) / 0.16f, 2.0f));

        for (int x = 0; x < width; ++x)
        {
            float u = (float)x / (float)(width - 1);

            // Base physically-inspired sky/environment radiance.
            float rr = 0.025f + 0.075f * sky + 0.035f * horizon;
            float gg = 0.040f + 0.105f * sky + 0.060f * horizon;
            float bb = 0.080f + 0.180f * sky + 0.105f * horizon;

            // Broad pre-baked lights.
            float sun = gauss(u, v, 0.77f, 0.22f, 0.055f, 0.075f);
            float warm = gauss(u, v, 0.18f, 0.38f, 0.10f, 0.14f);
            float cool = gauss(u, v, 0.55f, 0.68f, 0.16f, 0.08f);

            rr += 5.2f * sun + 1.5f * warm + 0.35f * cool;
            gg += 4.5f * sun + 0.75f * warm + 0.85f * cool;
            bb += 2.9f * sun + 0.25f * warm + 1.85f * cool;

            // Pre-rendered caustic/ring energy. It is fixed in the baked
            // texture, so runtime cost is just a few texture samples.
            float wave1 = 0.5f + 0.5f * sinf(u * 72.0f + sinf(v * 31.0f) * 4.0f);
            float wave2 = 0.5f + 0.5f * sinf(u * 127.0f - v * 41.0f);
            float caustic = powf(max(0.0f, wave1 * wave2 - 0.38f), 3.0f);
            float streak = expf(-powf((v - 0.48f - 0.09f*sinf(u*9.0f))/0.028f, 2.0f));

            rr += caustic * 1.8f + streak * 0.7f;
            gg += caustic * 2.4f + streak * 0.95f;
            bb += caustic * 3.6f + streak * 1.4f;

            pixels[y * width + x] = tfOpticalBakeColor(rr, gg, bb, 0.65f);
        }
    }

    tfBakedOpticalEnvironment = LoadTextureFromImage(image);
    UnloadImage(image);

    tfBakedOpticalEnvironmentReady =
        tfBakedOpticalEnvironment.id != 0;
}

void tfPrepareSurfaceShader()
{
    if (tfSurfaceShaderReady)
        return;

    static const char *vertexShader = R"GLSL(
#version 330

in vec3 vertexPosition;
in vec3 vertexNormal;

uniform mat4 mvp;

uniform mat4 matModel;

out vec3 fragNormal;

void main()
{
    fragNormal = normalize((matModel * vec4(vertexNormal, 0.0)).xyz);
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
)GLSL";

    static const char *fragmentShader = R"GLSL(
#version 330

in vec3 fragNormal;
uniform vec4 colDiffuse;
out vec4 finalColor;

void main()
{
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(vec3(-0.42, 0.84, 0.34));
    float diffuse = max(dot(N, L), 0.0);
    float fill = 0.22 * max(dot(N, normalize(vec3(0.20, 0.35, -0.92))), 0.0);
    float lighting = 0.70 + diffuse * 0.30 + fill;
    lighting = clamp(lighting, 0.62, 1.14);
    finalColor = vec4(clamp(colDiffuse.rgb * lighting, 0.0, 1.0), colDiffuse.a);
}
)GLSL";

    tfSurfaceShader = LoadShaderFromMemory(vertexShader, fragmentShader);
    if (tfSurfaceShader.id == 0)
        return;

    tfSurfaceShader.locs[SHADER_LOC_MATRIX_MVP] =
        GetShaderLocation(tfSurfaceShader, "mvp");
    tfSurfaceShader.locs[SHADER_LOC_MATRIX_MODEL] =
        GetShaderLocation(tfSurfaceShader, "matModel");
    tfSurfaceLocColor =
        GetShaderLocation(tfSurfaceShader, "colDiffuse");

    tfSurfaceShaderReady = true;
}

void tfPrepareOpticalShader()
{
    if (tfOpticalShaderReady)
        return;

    tfBuildBakedOpticalEnvironment();

    static const char *vertexShader = R"GLSL(
#version 330

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

vec3 animateLiquidVertex(vec3 p, vec3 n)
{
    float fluid = clamp(softFluid, 0.0, 1.0);
    if (fluid <= 0.001) return p;

    // This is deliberately a vertex-stage deformation: the existing high
    // polygon mesh is preserved, but the water surface can animate every GPU
    // frame without rebuilding/uploading the mesh on the CPU.
    float radial = length(p.xz);
    float waveA = sin(radial * (9.0 + 12.0 * fluid) - fluidTime * (5.5 + 9.5 * fluid));
    float waveB = sin(p.x * 7.0 + p.z * 5.0 + fluidTime * (2.5 + 4.0 * fluid));
    float waveC = sin((p.x - p.z) * 10.0 - fluidTime * (3.0 + 5.0 * fluid));

    float upper = smoothstep(-0.55, 0.85, p.y);
    float rim = smoothstep(0.15, 1.45, radial);
    float mask = (0.34 + 0.66 * upper) * (0.30 + 0.70 * rim);

    float amplitude = (0.012 + 0.040 * fluid) * mask;
    p.y += (waveA * 0.54 + waveB * 0.28 + waveC * 0.18) * amplitude;

    float sideways = (0.004 + 0.012 * fluid) * (0.35 + 0.65 * upper);
    p.x += sin(p.y * 5.0 + fluidTime * (2.0 + 4.0 * fluid)) * sideways;
    p.z += cos(p.y * 4.0 - fluidTime * (2.4 + 3.5 * fluid)) * sideways * 0.82;

    return p;
}

void main()
{
    vec3 localPosition = animateLiquidVertex(vertexPosition, vertexNormal);
    vec4 worldPosition = matModel * vec4(localPosition, 1.0);
    fragPosition = worldPosition.xyz;

    // Re-normalize a slightly perturbed normal for smooth specular response.
    vec3 animatedNormal = normalize(vertexNormal + vec3(0.10, 0.18, 0.10) *
                                    softFluid *
                                    vec3(
                                        sin(fluidTime + vertexPosition.z * 4.0),
                                        0.0,
                                        cos(fluidTime + vertexPosition.x * 4.0)
                                    ));
    fragNormal = normalize((matNormal * vec4(animatedNormal, 0.0)).xyz);
    gl_Position = mvp * vec4(localPosition, 1.0);
}
)GLSL";

    static const char *fragmentShader = R"GLSL(
#version 330

in vec3 fragPosition;
in vec3 fragNormal;

uniform vec3 viewPos;
uniform sampler2D texture0;
uniform vec4 colDiffuse;
uniform vec4 opticalParams;
uniform float opticalAlpha;

out vec4 finalColor;

const float PI = 3.14159265359;

vec2 envUV(vec3 d)
{
    d = normalize(d);
    float u = atan(d.z, d.x) / (2.0 * PI) + 0.5;
    float v = asin(clamp(d.y, -1.0, 1.0)) / PI + 0.5;
    return vec2(u, 1.0 - v);
}

vec3 sampleEnvironment(vec3 direction, vec2 distortion)
{
    vec2 uv = envUV(direction) + distortion;
    // Repeat in X because the baked environment is spherical.
    uv.x = fract(uv.x);
    uv.y = clamp(uv.y, 0.001, 0.999);
    return texture(texture0, uv).rgb;
}

void main()
{
    float reflectivity = clamp(opticalParams.x, 0.0, 1.0);
    float transmission = clamp(opticalParams.y, 0.0, 1.0);
    float ior = max(1.01, opticalParams.z);
    float dispersion = max(0.0, opticalParams.w);

    vec3 N = normalize(fragNormal);
    vec3 V = normalize(viewPos - fragPosition);

    // Keep the normal oriented toward the camera for stable Fresnel on both
    // sides of the closed mesh.
    if (dot(N, V) < 0.0)
        N = -N;

    float ndotv = clamp(dot(N, V), 0.0, 1.0);
    float fresnel = pow(1.0 - ndotv, 5.0);
    fresnel = mix(0.04, 1.0, fresnel);
    fresnel = clamp(fresnel * (0.55 + 0.65 * reflectivity), 0.0, 1.0);

    vec3 R = reflect(-V, N);
    vec3 T = refract(-V, N, 1.0 / ior);
    if (length(T) < 0.001)
        T = R;

    // Facet-dependent optical distortion. Because the environment itself is
    // baked, these offsets produce stable crystalline light bends without
    // a costly dynamic ray-traced path.
    vec2 facetWarp = N.xy * (0.010 + 0.035 * transmission);
    vec2 chromatic = normalize(N.xy + vec2(0.0001)) * dispersion * (0.35 + 1.75 * fresnel);

    vec3 reflected = sampleEnvironment(R, facetWarp * 0.70);
    vec3 refractedR = sampleEnvironment(T, facetWarp + chromatic * 1.65);
    vec3 refractedG = sampleEnvironment(T, facetWarp);
    vec3 refractedB = sampleEnvironment(T, facetWarp - chromatic * 1.65);

    // Mild spectral dispersion, strongest at glancing angles.
    vec3 spectral = vec3(refractedR.r, refractedG.g, refractedB.b);

    // Central body stays denser than the edges; this prevents the crystal
    // from looking like a transparent ghost.
    float bodyMix = transmission * (0.34 + 0.38 * (1.0 - fresnel));
    vec3 optical = mix(spectral, reflected, clamp(reflectivity * fresnel + 0.18, 0.0, 1.0));
    optical = mix(optical, reflected, fresnel * 0.48);
    optical += reflected * pow(fresnel, 3.0) * 0.75;

    // Strongly preserve the TigerFlash color argument.
    // The baked environment supplies light/reflection, while the requested
    // object color tints the transmitted and reflected spectrum rather than
    // replacing it with one shared cyan/white material.
    vec3 tint = max(colDiffuse.rgb, vec3(0.002));
    vec3 coloredSpectral = spectral * (0.62 + 0.78 * tint);
    vec3 coloredReflected = reflected * (0.72 + 0.42 * tint);
    optical = mix(coloredSpectral, coloredReflected,
                  clamp(reflectivity * fresnel + 0.18, 0.0, 1.0));
    optical = mix(optical, coloredReflected, fresnel * 0.48);
    optical += coloredReflected * pow(fresnel, 3.0) * 0.75;

    // Preserve saturated crystal color without removing the white specular
    // component that makes a transparent material read as glass/crystal.
    optical = mix(optical, optical * (0.78 + 0.42 * tint), 0.72);
    optical = max(optical, tint * 0.06);

    // Slightly transparent, but still dense.
    float alpha = clamp(opticalAlpha + bodyMix * 0.05, 0.38, 0.97);

    finalColor = vec4(optical, alpha);
}
)GLSL";

    tfOpticalShader = LoadShaderFromMemory(vertexShader, fragmentShader);
    if (tfOpticalShader.id == 0)
        return;

    tfOpticalShader.locs[SHADER_LOC_MATRIX_MVP] =
        GetShaderLocation(tfOpticalShader, "mvp");
    tfOpticalShader.locs[SHADER_LOC_MATRIX_MODEL] =
        GetShaderLocation(tfOpticalShader, "matModel");
    tfOpticalShader.locs[SHADER_LOC_MATRIX_NORMAL] =
        GetShaderLocation(tfOpticalShader, "matNormal");
    tfOpticalShader.locs[SHADER_LOC_VECTOR_VIEW] =
        GetShaderLocation(tfOpticalShader, "viewPos");
    tfOpticalShader.locs[SHADER_LOC_MAP_DIFFUSE] =
        GetShaderLocation(tfOpticalShader, "texture0");
    tfOpticalShader.locs[SHADER_LOC_COLOR_DIFFUSE] =
        GetShaderLocation(tfOpticalShader, "colDiffuse");

    tfOpticalLocViewPos =
        GetShaderLocation(tfOpticalShader, "viewPos");
    tfOpticalLocEnvironment =
        GetShaderLocation(tfOpticalShader, "texture0");
    tfOpticalLocParams =
        GetShaderLocation(tfOpticalShader, "opticalParams");
    tfOpticalLocAlpha =
        GetShaderLocation(tfOpticalShader, "opticalAlpha");
    tfOpticalLocTime =
        GetShaderLocation(tfOpticalShader, "fluidTime");
    tfOpticalLocSoftFluid =
        GetShaderLocation(tfOpticalShader, "softFluid");

    tfOpticalShaderReady = true;
}

void tfUnloadOpticalMaterial()
{
    if (tfOpticalShaderReady)
    {
        UnloadShader(tfOpticalShader);
        tfOpticalShader = { 0 };
        tfOpticalShaderReady = false;
        tfOpticalLocTime = -1;
        tfOpticalLocSoftFluid = -1;
    }

    if (tfBakedOpticalEnvironmentReady)
    {
        UnloadTexture(tfBakedOpticalEnvironment);
        tfBakedOpticalEnvironment = { 0 };
        tfBakedOpticalEnvironmentReady = false;
    }
}

void tfSetOpticalParams(const TFObject3D &obj,
                        const Camera3D &camera)
{
    if (!tfOpticalShaderReady)
        return;

    Vector3 viewPosition = camera.position;
    SetShaderValue(tfOpticalShader,
                   tfOpticalLocViewPos,
                   &viewPosition,
                   SHADER_UNIFORM_VEC3);

    int textureUnit = 0;
    SetShaderValue(tfOpticalShader,
                   tfOpticalLocEnvironment,
                   &textureUnit,
                   SHADER_UNIFORM_INT);

    Vector4 params;
    float alpha = 0.86f;

    // Crystal-like primitives retain their existing material response.
    if (obj.shape == "diamond")
    {
        params = {0.90f, 0.18f, 2.35f, 0.050f};
        alpha = 0.91f;
    }
    else if (obj.shape == "crystal")
    {
        params = {0.72f, 0.27f, 1.54f, 0.026f};
        alpha = 0.86f;
    }
    else if (obj.shape == "prism")
    {
        params = {0.76f, 0.31f, 1.56f, 0.042f};
        alpha = 0.84f;
    }

    // Softness is a material parameter now: the more `soft body (N)` is used
    // in gravity, the closer the object gets to a water/gel-like optical
    // response. The TigerFlash color is preserved through colDiffuse in the
    // shader, so blue remains blue, red remains red, etc.
    auto it = gravityBodies.find(obj.name);
    if (it != gravityBodies.end() && it->second.type == TFBodyType::Soft)
    {
        float fluid = tfClampSoftness(it->second.softness) / 10.0f;

        // At 0: nearly normal translucent material.
        // At 10: strong reflection + transmission, water-like IOR.
        float reflectivity = 0.16f + 0.84f * fluid;
        float transmission = 0.10f + 0.86f * fluid;
        float ior = 1.50f - 0.18f * fluid;
        float dispersion = 0.008f + 0.024f * fluid;

        params = {
            reflectivity,
            transmission,
            ior,
            dispersion
        };

        // Higher softness = more transparent, but never completely invisible.
        alpha = 0.96f - 0.42f * fluid;
    }

    SetShaderValue(tfOpticalShader,
                   tfOpticalLocParams,
                   &params,
                   SHADER_UNIFORM_VEC4);
    SetShaderValue(tfOpticalShader,
                   tfOpticalLocAlpha,
                   &alpha,
                   SHADER_UNIFORM_FLOAT);

    float fluidAmount = 0.0f;
    auto fluidIt = gravityBodies.find(obj.name);
    if (fluidIt != gravityBodies.end() &&
        fluidIt->second.type == TFBodyType::Soft)
    {
        fluidAmount = tfClampSoftness(fluidIt->second.softness) / 10.0f;
    }

    float timeSeconds = static_cast<float>(GetTime());
    SetShaderValue(tfOpticalShader,
                   tfOpticalLocTime,
                   &timeSeconds,
                   SHADER_UNIFORM_FLOAT);
    SetShaderValue(tfOpticalShader,
                   tfOpticalLocSoftFluid,
                   &fluidAmount,
                   SHADER_UNIFORM_FLOAT);
}
struct TFMeshBuilder
{
    vector<Vector3> vertices;
    vector<Vector3> normals;

    void reserveTriangles(size_t triangleCount)
    {
        const size_t vertexCount = triangleCount * 3;
        vertices.reserve(vertexCount);
        normals.reserve(vertexCount);
    }

    void addTriangle(Vector3 a, Vector3 b, Vector3 c, Vector3 outwardHint)
    {
        Vector3 ab = tfSub(b, a);
        Vector3 ac = tfSub(c, a);
        Vector3 normal = tfCross(ab, ac);
        float normalLenSq =
            normal.x * normal.x +
            normal.y * normal.y +
            normal.z * normal.z;

        if (normalLenSq < 0.000000000001f)
            return;

        // Closed convex meshes need consistent winding. Several older
        // shape builders passed {0,0,0} as the hint, which left the winding
        // undefined and could make whole faces disappear when backface
        // culling was active. For those faces, derive the outward direction
        // from the triangle centroid instead of guessing.
        float hintLenSq =
            outwardHint.x * outwardHint.x +
            outwardHint.y * outwardHint.y +
            outwardHint.z * outwardHint.z;

        if (hintLenSq < 0.000000000001f)
        {
            outwardHint =
            {
                (a.x + b.x + c.x) / 3.0f,
                (a.y + b.y + c.y) / 3.0f,
                (a.z + b.z + c.z) / 3.0f
            };

            hintLenSq =
                outwardHint.x * outwardHint.x +
                outwardHint.y * outwardHint.y +
                outwardHint.z * outwardHint.z;
        }

        if (hintLenSq > 0.000000000001f &&
            tfDot(normal, outwardHint) < 0.0f)
        {
            swap(b, c);
            ab = tfSub(b, a);
            ac = tfSub(c, a);
            normal = tfCross(ab, ac);
            normalLenSq =
                normal.x * normal.x +
                normal.y * normal.y +
                normal.z * normal.z;
        }

        const float invLength = 1.0f / sqrtf(normalLenSq);
        normal.x *= invLength;
        normal.y *= invLength;
        normal.z *= invLength;

        vertices.push_back(a);
        vertices.push_back(b);
        vertices.push_back(c);

        normals.push_back(normal);
        normals.push_back(normal);
        normals.push_back(normal);
    }
};

float tfDeformationMultiplier3D(const TFObject3D &obj)
{
    auto it = shapeDeformations.find(deformationKey3D(obj.name));
    if (it == shapeDeformations.end())
        return 1.0f;

    float largest = 1.0f;
    for (const TFDeformation &d : it->second)
    {
        for (const Vector3 &f : d.factors)
        {
            largest = max(largest, fabsf(f.x));
            largest = max(largest, fabsf(f.y));
            largest = max(largest, fabsf(f.z));
        }
    }

    return max(1.0f, largest);
}

float tfObjectBoundsRadius3D(const TFObject3D &obj)
{
    float sx = fabsf(obj.scaleX);
    float sy = fabsf(obj.scaleY);
    float sz = fabsf(obj.scaleZ);
    float radius = 1.5f;

    if (obj.shape == "cube")
        radius = 1.2f * sqrtf(sx*sx + sy*sy + sz*sz);
    else if (obj.shape == "sphere")
        radius = 1.5f * max(sx, max(sy, sz));
    else if (obj.shape == "cone")
        radius = 1.5f * sqrtf(sx*sx + sy*sy + sz*sz);
    else if (obj.shape == "triangle")
    {
        float ex = 1.3f * sx;
        float ey = 1.8f * sy;
        float ez = 1.3f * sz;
        radius = sqrtf(ex*ex + ey*ey + ez*ez);
    }
    else if (obj.shape == "cylinder")
        radius = sqrtf((1.5f*sx)*(1.5f*sx) + (1.5f*sy)*(1.5f*sy) + (1.5f*sz)*(1.5f*sz));
    else if (obj.shape == "capsule")
        radius = sqrtf((1.15f*sx)*(1.15f*sx) + (2.0f*sy)*(2.0f*sy) + (1.15f*sz)*(1.15f*sz));
    else if (obj.shape == "crystal")
        radius = sqrtf((1.15f*sx)*(1.15f*sx) + (1.90f*sy)*(1.90f*sy) + (1.15f*sz)*(1.15f*sz));
    else if (obj.shape == "diamond")
        radius = sqrtf((1.25f*sx)*(1.25f*sx) + (1.90f*sy)*(1.90f*sy) + (1.25f*sz)*(1.25f*sz));
    else if (obj.shape == "prism")
        radius = sqrtf((1.35f*sx)*(1.35f*sx) + (1.45f*sy)*(1.45f*sy) + (1.35f*sz)*(1.35f*sz));
    else if (obj.shape == "rock")
        radius = sqrtf((1.25f*sx)*(1.25f*sx) + (1.15f*sy)*(1.15f*sy) + (1.10f*sz)*(1.10f*sz));

    return radius * tfDeformationMultiplier3D(obj) * 1.08f;
}

bool tfObjectInsideFrustum3D(const TFObject3D &obj,
                              const Camera3D &camera,
                              float aspect,
                              float farClip)
{
    Vector3 forward = tfNormalize(tfSub(camera.target, camera.position));
    Vector3 right = tfNormalize(tfCross(forward, camera.up));
    Vector3 up = tfNormalize(tfCross(right, forward));
    Vector3 rel = tfSub({obj.x, obj.y, obj.z}, camera.position);

    float radius = tfObjectBoundsRadius3D(obj);
    float depth = tfDot(rel, forward);

    if (depth + radius < 0.05f)
        return false;

    if (tfLength(rel) - radius > farClip)
        return false;

    float verticalHalf = camera.fovy * DEG2RAD * 0.5f;
    float horizontalHalf = atanf(tanf(verticalHalf) * aspect);

    float horizontalDistance = fabsf(tfDot(rel, right));
    float verticalDistance = fabsf(tfDot(rel, up));

    // Safety margin for frustum culling. The object may remain partially
    // outside the visible screen without being discarded too early. This
    // prevents moving/soft objects such as slimes from disappearing at the
    // screen border because of small camera/deformation/camera-step changes.
    float cullingMargin = radius * 1.8f;

    if (depth > 0.0f)
    {
        if (horizontalDistance - cullingMargin >
            depth * tanf(horizontalHalf))
            return false;

        if (verticalDistance - cullingMargin >
            depth * tanf(verticalHalf))
            return false;
    }

    return true;
}

// ============================================================
// ADAPTIVE ORGANIC / WATER DETAIL
// The renderer measures the real frame time of the current PC and
// automatically chooses how much mesh detail soft/liquid objects
// need. Organic objects never fall to the old 1x1 cube grid because
// that is where deformations can visually open up / look perforated.
// The TigerFlash syntax is unchanged.
// ============================================================

float tfOrganicFrameTimeEMA = 1.0f / 60.0f;
int tfAdaptiveOrganicLOD = 0; // Start at full liquid quality; adaptive governor may reduce it under sustained load.
float tfAdaptiveOrganicTimer = 0.0f;

bool tfIsOrganic3DObject(const TFObject3D &obj)
{
    auto it = gravityBodies.find(obj.name);
    if (it != gravityBodies.end() && it->second.type == TFBodyType::Soft)
        return true;

    return false;
}

void tfUpdateAdaptiveOrganicLOD(float dt)
{
    dt = max(0.0001f, min(0.10f, dt));

    // Exponential moving average: short spikes do not immediately destroy
    // water quality, but a genuinely slow machine is detected quickly.
    float alpha = 1.0f - expf(-2.5f * dt);
    tfOrganicFrameTimeEMA +=
        (dt - tfOrganicFrameTimeEMA) * alpha;

    float fps = 1.0f / max(0.0001f, tfOrganicFrameTimeEMA);

    int desiredLOD;

    // LOD 0 = maximum organic detail, LOD 1 = high detail,
    // LOD 2 = balanced liquid detail, LOD 3 = distant/pressure fallback.
    if (fps >= 58.0f)
        desiredLOD = 0;
    else if (fps >= 48.0f)
        desiredLOD = 1;
    else if (fps >= 36.0f)
        desiredLOD = 2;
    else
        desiredLOD = 3;

    // Hold each decision briefly so the mesh does not constantly rebuild
    // when FPS hovers around a threshold.
    tfAdaptiveOrganicTimer += dt;

    if (desiredLOD != tfAdaptiveOrganicLOD &&
        tfAdaptiveOrganicTimer >= 0.60f)
    {
        // Moving toward more detail is deliberately conservative.
        // The PC must sustain the higher FPS for the full hold period.
        tfAdaptiveOrganicLOD = desiredLOD;
        tfAdaptiveOrganicTimer = 0.0f;
    }
}

uint64_t tfSoftMeshRebuildIntervalFrames(const TFObject3D &obj,
                                          const Camera3D &camera,
                                          int lod)
{
    float distance = tfLength(tfSub(
        {obj.x, obj.y, obj.z},
        camera.position
    ));

    // Fast ripples no longer depend on this interval: they run in the GPU.
    // These intervals now govern only the slower physical silhouette changes.
    if (lod <= 0 && distance < 16.0f) return 1;
    if (lod <= 1 && distance < 40.0f) return 2;
    if (lod <= 2 && distance < 90.0f) return 3;
    return 5;
}

int tfChooseLOD3D(const TFObject3D &obj,
                  const Camera3D &camera,
                  int screenHeight,
                  int previousLOD)
{
    Vector3 position = {obj.x, obj.y, obj.z};
    float distance = tfLength(tfSub(position, camera.position));
    float radius = max(0.001f, tfObjectBoundsRadius3D(obj));

    float focalPixels =
        screenHeight /
        (2.0f * tanf(camera.fovy * DEG2RAD * 0.5f));

    float projectedRadius =
        (focalPixels * radius) / max(0.001f, distance);

    // 0 = very close/high detail, 3 = very far/minimal detail.
    // Wide hysteresis keeps changes quiet instead of visibly popping.
    int lod = 3;

    if (projectedRadius >= 150.0f)
        lod = 0;
    else if (projectedRadius >= 55.0f)
        lod = 1;
    else if (projectedRadius >= 18.0f)
        lod = 2;
    else
        lod = 3;

    if (previousLOD >= 0 && previousLOD <= 3)
    {
        if (previousLOD == 0 && projectedRadius > 125.0f) lod = 0;
        if (previousLOD == 1 && projectedRadius > 45.0f && projectedRadius < 165.0f) lod = 1;
        if (previousLOD == 2 && projectedRadius > 14.0f && projectedRadius < 65.0f) lod = 2;
        if (previousLOD == 3 && projectedRadius < 23.0f) lod = 3;
    }

    return lod;
}

// ============================================================
// 3D COLLISION CLUSTER POLYGON BUDGET
// When several bodies are physically touching (a soft-body pile,
// parts stuck to one another, a chain of glued objects) the total
// polygon cost must NOT simply add up as more objects join the
// pile - 50 objects glued together should not cost 50x the
// triangles of a single object. Instead, every object that belongs
// to the same touching group shares ONE fixed triangle budget
// (the cost of a single full-detail object). That budget is split
// evenly between the group's current members, so the bigger the
// pile gets, the smaller each individual member's own polygon
// allowance becomes - total scene complexity stays bounded and the
// frame rate stays stable on any machine, no matter how large a
// pile of glued objects grows.
//
// This works ALONGSIDE the camera-distance LOD above, never
// instead of it: whichever of the two asks for less detail wins.
// A lone object up close still gets full quality; a big glued pile
// stays cheap even at point-blank range.
// ============================================================

// Triangle count produced by each LOD tier, per shape. Mirrors the
// rings/slices/grid tables used by tfBuildOptimized*() above - keep
// both in sync if those tables ever change.
int tfTriangleCountForLOD3D(const string &shape, int lod, bool organic = false)
{
    lod = max(0, min(3, lod));

    if (shape == "sphere")
    {
        int rings = organic
            ? (lod == 0 ? 40 : (lod == 1 ? 28 : (lod == 2 ? 18 : 12)))
            : (lod == 0 ? 18 : (lod == 1 ? 10 : (lod == 2 ? 6 : 4)));
        int slices = organic
            ? (lod == 0 ? 64 : (lod == 1 ? 48 : (lod == 2 ? 32 : 20)))
            : (lod == 0 ? 32 : (lod == 1 ? 18 : (lod == 2 ? 12 : 8)));
        return rings * slices * 2;
    }

    if (shape == "cone")
    {
        int slices = organic
            ? (lod == 0 ? 64 : (lod == 1 ? 40 : (lod == 2 ? 24 : 14)))
            : (lod == 0 ? 40 : (lod == 1 ? 20 : (lod == 2 ? 12 : 8)));
        int rings = organic
            ? (lod == 0 ? 16 : (lod == 1 ? 10 : (lod == 2 ? 6 : 3)))
            : (lod == 0 ? 10 : (lod == 1 ? 5  : (lod == 2 ? 3  : 1)));
        return rings * slices * 2 + slices; // side quads (2 tris each) + base fan
    }

    if (shape == "cube")
    {
        int grid = organic
            ? (lod == 0 ? 20 : (lod == 1 ? 14 : (lod == 2 ? 8 : 4)))
            : (lod == 0 ? 8 : (lod == 1 ? 4 : (lod == 2 ? 2 : 1)));
        return 6 * grid * grid * 2;
    }

    if (shape == "cylinder")
    {
        int sides = (lod == 0 ? 32 : (lod == 1 ? 20 : (lod == 2 ? 12 : 8)));
        return sides * 4;
    }

    if (shape == "capsule")
    {
        int sides = organic
            ? (lod == 0 ? 48 : (lod == 1 ? 32 : (lod == 2 ? 20 : 12)))
            : (lod == 0 ? 28 : (lod == 1 ? 18 : (lod == 2 ? 12 : 8)));
        int rings = (lod == 0 ? 10 : (lod == 1 ? 7 : (lod == 2 ? 5 : 3)));
        return sides * rings * 2;
    }

    if (shape == "crystal")
    {
        int sides = (lod == 0 ? 8 : 6);
        return sides * 4;
    }

    if (shape == "diamond")
        return 8;

    if (shape == "prism")
        return 24;

    if (shape == "rock")
    {
        int sides = (lod == 0 ? 8 : (lod == 1 ? 6 : 4));
        return sides * 6;
    }

    // Pyramid has a small, fixed triangle count at every LOD tier.
    return 6;
}

// Groups every 3D object that is CURRENTLY overlapping another one
// (real geometric contact, not just a declared "colision" pair) into
// clusters, and returns how many objects share each object's group.
unordered_map<string, int> tfComputeTouchingClusterSizes3D()
{
    unordered_map<string, string> parent;

    for (const TFObject3D &obj : objects3D)
        parent[obj.name] = obj.name;

    for (const TFCollisionPair &pair : physicalCollisionPairs)
    {
        if (objectDimension(pair.first) != 3 ||
            objectDimension(pair.second) != 3)
        {
            continue;
        }

        TFObject3D *a = findObject3D(pair.first);
        TFObject3D *b = findObject3D(pair.second);

        if (a == nullptr || b == nullptr)
            continue;

        // Only objects touching RIGHT NOW share a budget. A declared
        // "colision" pair that is not currently overlapping costs
        // nothing extra.
        if (!collision3D(*a, *b))
            continue;

        tfClusterUnion(parent, pair.first, pair.second);
    }

    unordered_map<string, int> sizeOfRoot;

    for (const TFObject3D &obj : objects3D)
        sizeOfRoot[tfClusterFind(parent, obj.name)]++;

    unordered_map<string, int> sizeOfObject;

    for (const TFObject3D &obj : objects3D)
        sizeOfObject[obj.name] = sizeOfRoot[tfClusterFind(parent, obj.name)];

    return sizeOfObject;
}

// Eased (smoothed) cluster size per object. An object joining or
// leaving a pile never causes an instant, visible drop or jump in
// detail - the shared budget eases in and out over a fraction of a
// second instead of popping.
unordered_map<string, float> tfSmoothedClusterSize3D;
unordered_map<string, int> tfClusterTargetSize3D;
float tfClusterBudgetTimer3D = 1.0f;

void tfUpdateClusterBudgets3D(float dt)
{
    if (objects3D.empty())
        return;

    tfClusterBudgetTimer3D += max(0.0f, dt);
    if (tfClusterBudgetTimer3D >= 0.20f || tfClusterTargetSize3D.empty())
    {
        tfClusterTargetSize3D = tfComputeTouchingClusterSizes3D();
        tfClusterBudgetTimer3D = 0.0f;
    }

    const unordered_map<string, int> &rawSizes = tfClusterTargetSize3D;

    // Higher = reacts to a pile forming/breaking faster; lower = smoother.
    const float smoothingSpeed = 3.5f;
    const float easing = 1.0f - expf(-smoothingSpeed * max(0.0f, dt));

    for (const TFObject3D &obj : objects3D)
    {
        auto rawIt = rawSizes.find(obj.name);
        float target =
            static_cast<float>(rawIt != rawSizes.end() ? rawIt->second : 1);

        float &current = tfSmoothedClusterSize3D[obj.name];

        if (current <= 0.0f)
            current = target; // First sighting: no fake ramp from zero.
        else
            current += (target - current) * easing;
    }

    for (auto it = tfSmoothedClusterSize3D.begin();
         it != tfSmoothedClusterSize3D.end();)
    {
        if (findObject3D(it->first) == nullptr)
            it = tfSmoothedClusterSize3D.erase(it);
        else
            ++it;
    }
}

// Given how many objects are (smoothly) sharing the same touching
// group, returns the lowest LOD tier ("most detailed") whose own
// triangle count still fits inside this object's fair share of the
// group's ONE shared budget. A lone object (group size 1) always
// returns 0, meaning "no cluster penalty - decide purely by camera
// distance", exactly like before this system existed.
int tfClusterLODFloor3D(const TFObject3D &obj, float smoothedClusterSize)
{
    int n = max(1, static_cast<int>(roundf(smoothedClusterSize)));

    if (n <= 1)
        return 0;

    const bool organic = tfIsOrganic3DObject(obj);
    const int oneObjectBudget =
        tfTriangleCountForLOD3D(obj.shape, 0, organic);
    const int perMemberBudget = max(1, oneObjectBudget / n);

    for (int lod = 0; lod <= 3; lod++)
    {
        if (tfTriangleCountForLOD3D(obj.shape, lod, organic) <= perMemberBudget)
            return lod;
    }

    return 3;
}

size_t tfMeshSignature3D(const TFObject3D &obj)
{
    // Hash numeric state directly. The old string concatenation/to_string path
    // allocated and formatted many temporary strings every frame.
    size_t h = hash<string>{}(obj.shape);

    auto mix = [&](size_t value)
    {
        h ^= value +
             static_cast<size_t>(0x9e3779b97f4a7c15ULL) +
             (h << 6) +
             (h >> 2);
    };

    auto q = [](float v) -> long long
    {
        return llround(v * 10000.0f);
    };

    mix(static_cast<size_t>(q(obj.scaleX)));
    mix(static_cast<size_t>(q(obj.scaleY)));
    mix(static_cast<size_t>(q(obj.scaleZ)));

    auto it = shapeDeformations.find(deformationKey3D(obj.name));
    if (it != shapeDeformations.end())
    {
        for (const TFDeformation &d : it->second)
        {
            mix(static_cast<size_t>(d.side));
            mix(static_cast<size_t>(d.segments));
            for (const Vector3 &f : d.factors)
            {
                mix(static_cast<size_t>(q(f.x)));
                mix(static_cast<size_t>(q(f.y)));
                mix(static_cast<size_t>(q(f.z)));
            }
        }
    }

    // Soft-body deformation state is deliberately excluded from the mesh
    // signature. Its topology does not change; the legacy mold is streamed
    // into the existing dynamic mesh every visible frame instead.

    return h;
}

Mesh tfUploadMeshBuilder(TFMeshBuilder &builder, bool smoothNormals = false, bool dynamic = false)
{
    Mesh mesh = { 0 };

    if (builder.vertices.empty())
        return mesh;

    if (smoothNormals)
    {
        struct SmoothKey
        {
            int x, y, z;
            bool operator==(const SmoothKey &other) const
            {
                return x == other.x && y == other.y && z == other.z;
            }
        };
        struct SmoothKeyHash
        {
            size_t operator()(const SmoothKey &k) const noexcept
            {
                size_t h = static_cast<size_t>(k.x) * 73856093u;
                h ^= static_cast<size_t>(k.y) * 19349663u;
                h ^= static_cast<size_t>(k.z) * 83492791u;
                return h;
            }
        };

        unordered_map<SmoothKey, Vector3, SmoothKeyHash> sums;
        sums.reserve(builder.vertices.size());

        auto keyOf = [](const Vector3 &p) -> SmoothKey
        {
            return {
                (int)llround(p.x * 10000.0f),
                (int)llround(p.y * 10000.0f),
                (int)llround(p.z * 10000.0f)
            };
        };

        for (size_t i = 0; i < builder.vertices.size(); ++i)
        {
            const SmoothKey key = keyOf(builder.vertices[i]);
            Vector3 &sum = sums[key];
            sum.x += builder.normals[i].x;
            sum.y += builder.normals[i].y;
            sum.z += builder.normals[i].z;
        }

        for (size_t i = 0; i < builder.vertices.size(); ++i)
        {
            const SmoothKey key = keyOf(builder.vertices[i]);
            Vector3 n = sums[key];
            float len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
            if (len > 0.000001f)
            {
                n.x /= len;
                n.y /= len;
                n.z /= len;
                builder.normals[i] = n;
            }
        }
    }

    mesh.vertexCount = static_cast<int>(builder.vertices.size());
    mesh.triangleCount = mesh.vertexCount / 3;

    mesh.vertices =
        static_cast<float *>(MemAlloc(sizeof(float) * mesh.vertexCount * 3));
    mesh.normals =
        static_cast<float *>(MemAlloc(sizeof(float) * mesh.vertexCount * 3));

    for (int i = 0; i < mesh.vertexCount; i++)
    {
        mesh.vertices[i * 3 + 0] = builder.vertices[i].x;
        mesh.vertices[i * 3 + 1] = builder.vertices[i].y;
        mesh.vertices[i * 3 + 2] = builder.vertices[i].z;

        mesh.normals[i * 3 + 0] = builder.normals[i].x;
        mesh.normals[i * 3 + 1] = builder.normals[i].y;
        mesh.normals[i * 3 + 2] = builder.normals[i].z;
    }

    UploadMesh(&mesh, dynamic);
    return mesh;
}

Mesh tfBuildOptimizedCube(const TFObject3D &obj, int lod)
{
    TFMeshBuilder builder;
    bool organic = tfIsOrganic3DObject(obj);

    // Normal objects keep the original lightweight 8/4/2/1 grids.
    // Soft/liquid objects get 12/8/4/2, with LOD 2 as the minimum safe
    // quality selected by the adaptive PC-performance system.
    int grid = organic
        ? (lod == 0 ? 20 : (lod == 1 ? 14 : (lod == 2 ? 8 : 4)))
        : (lod == 0 ? 8  : (lod == 1 ? 4 : (lod == 2 ? 2 : 1)));

    builder.reserveTriangles(static_cast<size_t>(6 * grid * grid * 2));

    float hx = 1.2f * fabsf(obj.scaleX);
    float hy = 1.2f * fabsf(obj.scaleY);
    float hz = 1.2f * fabsf(obj.scaleZ);

    auto facePoint = [&](int face, float u, float v) -> Vector3
    {
        switch (face)
        {
            case 0: return {-hx, v * hy, u * hz};
            case 1: return { hx, u * hy, v * hz};
            case 2: return { u * hx,  hy, v * hz};
            case 3: return { u * hx, -hy, v * hz};
            case 4: return { u * hx, v * hy,  hz};
            default:return { u * hx, v * hy, -hz};
        }
    };

    for (int face = 0; face < 6; face++)
    {
        for (int y = 0; y < grid; y++)
        {
            float v0 = -1.0f + 2.0f * y / grid;
            float v1 = -1.0f + 2.0f * (y + 1) / grid;

            for (int x = 0; x < grid; x++)
            {
                float u0 = -1.0f + 2.0f * x / grid;
                float u1 = -1.0f + 2.0f * (x + 1) / grid;

                Vector3 a = apply3DDeformations(obj, facePoint(face, u0, v0));
                Vector3 b = apply3DDeformations(obj, facePoint(face, u1, v0));
                Vector3 c = apply3DDeformations(obj, facePoint(face, u1, v1));
                Vector3 d = apply3DDeformations(obj, facePoint(face, u0, v1));

                Vector3 hint =
                {
                    (a.x + b.x + c.x + d.x) * 0.25f,
                    (a.y + b.y + c.y + d.y) * 0.25f,
                    (a.z + b.z + c.z + d.z) * 0.25f
                };

                builder.addTriangle(a, b, c, hint);
                builder.addTriangle(a, c, d, hint);
            }
        }
    }

    return tfUploadMeshBuilder(builder, tfIsOrganic3DObject(obj), tfIsOrganic3DObject(obj));
}

Mesh tfBuildOptimizedSphere(const TFObject3D &obj, int lod)
{
    TFMeshBuilder builder;
    const bool organic = tfIsOrganic3DObject(obj);
    int rings = organic
        ? (lod == 0 ? 40 : (lod == 1 ? 28 : (lod == 2 ? 18 : 12)))
        : (lod == 0 ? 18 : (lod == 1 ? 10 : (lod == 2 ? 6 : 4)));
    int slices = organic
        ? (lod == 0 ? 64 : (lod == 1 ? 48 : (lod == 2 ? 32 : 20)))
        : (lod == 0 ? 32 : (lod == 1 ? 18 : (lod == 2 ? 12 : 8)));
    builder.reserveTriangles(static_cast<size_t>(rings * slices * 2));

    vector<float> sinPhi(rings + 1);
    vector<float> cosPhi(rings + 1);
    for (int r = 0; r <= rings; ++r)
    {
        float phi = PI * static_cast<float>(r) / static_cast<float>(rings);
        sinPhi[r] = sinf(phi);
        cosPhi[r] = cosf(phi);
    }

    vector<float> sinTheta(slices + 1);
    vector<float> cosTheta(slices + 1);
    for (int s = 0; s <= slices; ++s)
    {
        float theta = 2.0f * PI * static_cast<float>(s) / static_cast<float>(slices);
        sinTheta[s] = sinf(theta);
        cosTheta[s] = cosf(theta);
    }

    for (int r = 0; r < rings; r++)
    {
        for (int s = 0; s < slices; s++)
        {
            auto spherePoint = [&](int ring, int slice) -> Vector3
            {
                Vector3 p =
                {
                    1.5f * obj.scaleX * sinPhi[ring] * cosTheta[slice],
                    1.5f * obj.scaleY * cosPhi[ring],
                    1.5f * obj.scaleZ * sinPhi[ring] * sinTheta[slice]
                };
                return apply3DDeformations(obj, p);
            };

            Vector3 a = spherePoint(r, s);
            Vector3 b = spherePoint(r, s + 1);
            Vector3 c = spherePoint(r + 1, s + 1);
            Vector3 d = spherePoint(r + 1, s);

            Vector3 hint1 = tfAdd(tfAdd(a, b), c);
            Vector3 hint2 = tfAdd(tfAdd(a, c), d);
            builder.addTriangle(a, b, c, hint1);
            builder.addTriangle(a, c, d, hint2);
        }
    }

    return tfUploadMeshBuilder(builder, tfIsOrganic3DObject(obj), tfIsOrganic3DObject(obj));
}

Mesh tfBuildOptimizedCone(const TFObject3D &obj, int lod)
{
    TFMeshBuilder builder;
    bool organic = tfIsOrganic3DObject(obj);
    int slices = organic
        ? (lod == 0 ? 64 : (lod == 1 ? 40 : (lod == 2 ? 24 : 14)))
        : (lod == 0 ? 40 : (lod == 1 ? 20 : (lod == 2 ? 12 : 8)));
    int rings = organic
        ? (lod == 0 ? 16 : (lod == 1 ? 10 : (lod == 2 ? 6 : 3)))
        : (lod == 0 ? 10 : (lod == 1 ? 5 : (lod == 2 ? 3 : 1)));
    builder.reserveTriangles(static_cast<size_t>(rings * slices * 2 + slices));

    auto conePoint = [&](float radius, float y, float angle) -> Vector3
    {
        Vector3 p =
        {
            radius * obj.scaleX * cosf(angle),
            y * obj.scaleY,
            radius * obj.scaleZ * sinf(angle)
        };
        return apply3DDeformations(obj, p);
    };

    for (int r = 0; r < rings; r++)
    {
        float t0 = static_cast<float>(r) / rings;
        float t1 = static_cast<float>(r + 1) / rings;
        float radius0 = 1.5f * (1.0f - t0);
        float radius1 = 1.5f * (1.0f - t1);
        float y0 = -1.5f + 3.0f * t0;
        float y1 = -1.5f + 3.0f * t1;

        for (int s = 0; s < slices; s++)
        {
            float a0 = 2.0f * PI * s / slices;
            float a1 = 2.0f * PI * (s + 1) / slices;

            Vector3 a = conePoint(radius0, y0, a0);
            Vector3 b = conePoint(radius0, y0, a1);
            Vector3 c = conePoint(radius1, y1, a1);
            Vector3 d = conePoint(radius1, y1, a0);

            builder.addTriangle(a, b, c, tfAdd(tfAdd(a, b), c));
            builder.addTriangle(a, c, d, tfAdd(tfAdd(a, c), d));
        }
    }

    Vector3 center = apply3DDeformations(obj, {0.0f, -1.5f * obj.scaleY, 0.0f});
    for (int s = 0; s < slices; s++)
    {
        float a0 = 2.0f * PI * s / slices;
        float a1 = 2.0f * PI * (s + 1) / slices;
        Vector3 a = conePoint(1.5f, -1.5f, a0);
        Vector3 b = conePoint(1.5f, -1.5f, a1);
        builder.addTriangle(center, b, a, {0.0f, -1.0f, 0.0f});
    }

    return tfUploadMeshBuilder(builder, tfIsOrganic3DObject(obj), tfIsOrganic3DObject(obj));
}

Mesh tfBuildOptimizedCylinder(const TFObject3D &obj, int lod)
{
    TFMeshBuilder builder;
    int sides = (lod == 0 ? 32 : (lod == 1 ? 20 : (lod == 2 ? 12 : 8)));
    float rx = 1.5f * obj.scaleX;
    float rz = 1.5f * obj.scaleZ;
    float hy = 1.5f * obj.scaleY;
    builder.reserveTriangles((size_t)sides * 4);

    for (int i=0;i<sides;++i)
    {
        float a0=2.0f*PI*i/sides, a1=2.0f*PI*(i+1)/sides;
        Vector3 b0=apply3DDeformations(obj,{rx*cosf(a0),-hy,rz*sinf(a0)});
        Vector3 b1=apply3DDeformations(obj,{rx*cosf(a1),-hy,rz*sinf(a1)});
        Vector3 t1=apply3DDeformations(obj,{rx*cosf(a1),hy,rz*sinf(a1)});
        Vector3 t0=apply3DDeformations(obj,{rx*cosf(a0),hy,rz*sinf(a0)});
        builder.addTriangle(b0,b1,t1,{0,0,0});
        builder.addTriangle(b0,t1,t0,{0,0,0});
        Vector3 top=apply3DDeformations(obj,{0,hy,0});
        Vector3 bot=apply3DDeformations(obj,{0,-hy,0});
        builder.addTriangle(top,t1,t0,{0,1,0});
        builder.addTriangle(bot,b0,b1,{0,-1,0});
    }
    return tfUploadMeshBuilder(builder, tfIsOrganic3DObject(obj), tfIsOrganic3DObject(obj));
}

Mesh tfBuildOptimizedCapsule(const TFObject3D &obj, int lod)
{
    TFMeshBuilder builder;
    bool organic = tfIsOrganic3DObject(obj);
    int slices = organic
        ? (lod == 0 ? 48 : (lod == 1 ? 32 : (lod == 2 ? 20 : 12)))
        : (lod == 0 ? 28 : (lod == 1 ? 18 : (lod == 2 ? 12 : 8)));
    int hemi = organic
        ? (lod == 0 ? 10 : (lod == 1 ? 7 : (lod == 2 ? 5 : 3)))
        : (lod == 0 ? 6 : (lod == 1 ? 5 : (lod == 2 ? 4 : 3)));
    float rx=1.15f*obj.scaleX, rz=1.15f*obj.scaleZ;
    float r=1.15f*fabsf(obj.scaleY), half=0.85f*fabsf(obj.scaleY);
    vector<vector<Vector3>> rings;
    vector<float> ys; vector<pair<float,float>> rad;
    ys.push_back(half+r); rad.push_back({0,0});
    for(int i=1;i<=hemi;++i){float t=(float)i/hemi;float ph=(PI*0.5f)*t;ys.push_back(half+r*cosf(ph));float rr=r*sinf(ph);rad.push_back({rr*rx/r,rr*rz/r});}
    ys.push_back(-half); rad.push_back({rx,rz});
    for(int i=0;i<hemi;++i){float t=(float)i/hemi;float ph=(PI*0.5f)*t;ys.push_back(-half-r*sinf(ph));float rr=r*cosf(ph);rad.push_back({rr*rx/r,rr*rz/r});}
    ys.push_back(-half-r); rad.push_back({0,0});
    // Replace zero-radius poles with degenerate single-point rings by building triangles explicitly.
    for(size_t j=0;j+1<ys.size();++j){
        bool topPole=(rad[j].first<1e-5f), botPole=(rad[j+1].first<1e-5f);
        if(topPole){Vector3 p=apply3DDeformations(obj,{0,ys[j],0}); for(int i=0;i<slices;++i){float a0=2*PI*i/slices,a1=2*PI*(i+1)/slices;Vector3 q0=apply3DDeformations(obj,{rad[j+1].first*cosf(a0),ys[j+1],rad[j+1].second*sinf(a0)});Vector3 q1=apply3DDeformations(obj,{rad[j+1].first*cosf(a1),ys[j+1],rad[j+1].second*sinf(a1)});builder.addTriangle(p,q0,q1,{0,1,0});}}
        else if(botPole){Vector3 p=apply3DDeformations(obj,{0,ys[j+1],0}); for(int i=0;i<slices;++i){float a0=2*PI*i/slices,a1=2*PI*(i+1)/slices;Vector3 q0=apply3DDeformations(obj,{rad[j].first*cosf(a0),ys[j],rad[j].second*sinf(a0)});Vector3 q1=apply3DDeformations(obj,{rad[j].first*cosf(a1),ys[j],rad[j].second*sinf(a1)});builder.addTriangle(p,q1,q0,{0,-1,0});}}
        else {for(int i=0;i<slices;++i){float a0=2*PI*i/slices,a1=2*PI*(i+1)/slices;Vector3 a=apply3DDeformations(obj,{rad[j].first*cosf(a0),ys[j],rad[j].second*sinf(a0)});Vector3 b=apply3DDeformations(obj,{rad[j].first*cosf(a1),ys[j],rad[j].second*sinf(a1)});Vector3 c=apply3DDeformations(obj,{rad[j+1].first*cosf(a1),ys[j+1],rad[j+1].second*sinf(a1)});Vector3 d=apply3DDeformations(obj,{rad[j+1].first*cosf(a0),ys[j+1],rad[j+1].second*sinf(a0)});builder.addTriangle(a,b,c,{0,ys[j],0});builder.addTriangle(a,c,d,{0,ys[j],0});}}
    }
    return tfUploadMeshBuilder(builder, tfIsOrganic3DObject(obj), tfIsOrganic3DObject(obj));
}

Mesh tfBuildOptimizedCrystal(const TFObject3D &obj, int lod)
{
    TFMeshBuilder builder;
    const int sides = (lod == 0 ? 8 : (lod == 1 ? 6 : 6));

    const float rx = 1.15f * fabsf(obj.scaleX);
    const float rz = 1.15f * fabsf(obj.scaleZ);
    const float halfBody = 0.72f * fabsf(obj.scaleY);
    const float tip = 0.82f * fabsf(obj.scaleY);

    vector<Vector3> lower(sides);
    vector<Vector3> upper(sides);
    vector<Vector3> lowerShoulder(sides);
    vector<Vector3> upperShoulder(sides);

    // A closed crystal has four rings/regions: lower shoulder, main body,
    // upper shoulder, then two real pointed terminations. Every neighboring
    // region shares the same mathematical silhouette, so there are no holes.
    for (int i = 0; i < sides; ++i)
    {
        float a = 2.0f * PI * (float)i / (float)sides;
        float ca = cosf(a);
        float sa = sinf(a);

        lower[i] = apply3DDeformations(obj,
            {rx * 0.72f * ca, -halfBody * 0.88f, rz * 0.72f * sa});
        lowerShoulder[i] = apply3DDeformations(obj,
            {rx * ca, -halfBody * 0.46f, rz * sa});
        upperShoulder[i] = apply3DDeformations(obj,
            {rx * ca,  halfBody * 0.46f, rz * sa});
        upper[i] = apply3DDeformations(obj,
            {rx * 0.72f * ca, halfBody * 0.88f, rz * 0.72f * sa});
    }

    Vector3 bottomTip = apply3DDeformations(obj,
        {0.0f, -halfBody * 0.88f - tip, 0.0f});
    Vector3 topTip = apply3DDeformations(obj,
        {0.0f, halfBody * 0.88f + tip, 0.0f});

    for (int i = 0; i < sides; ++i)
    {
        int j = (i + 1) % sides;

        builder.addTriangle(bottomTip, lower[j], lower[i],
                            {0.0f, -1.0f, 0.0f});

        builder.addTriangle(lower[i], lower[j], lowerShoulder[j],
                            {0.0f, -0.5f, 0.0f});
        builder.addTriangle(lower[i], lowerShoulder[j], lowerShoulder[i],
                            {0.0f, -0.2f, 0.0f});

        builder.addTriangle(lowerShoulder[i], lowerShoulder[j], upperShoulder[j],
                            {0.0f, 0.0f, 0.0f});
        builder.addTriangle(lowerShoulder[i], upperShoulder[j], upperShoulder[i],
                            {0.0f, 0.0f, 0.0f});

        builder.addTriangle(upperShoulder[i], upperShoulder[j], upper[j],
                            {0.0f, 0.5f, 0.0f});
        builder.addTriangle(upperShoulder[i], upper[j], upper[i],
                            {0.0f, 0.7f, 0.0f});

        builder.addTriangle(topTip, upper[i], upper[j],
                            {0.0f, 1.0f, 0.0f});
    }

    return tfUploadMeshBuilder(builder, tfIsOrganic3DObject(obj), tfIsOrganic3DObject(obj));
}

Mesh tfBuildOptimizedDiamond(const TFObject3D &obj)
{
    TFMeshBuilder builder; builder.reserveTriangles(8);
    Vector3 top=apply3DDeformations(obj,{0,1.9f*obj.scaleY,0}); Vector3 bot=apply3DDeformations(obj,{0,-1.9f*obj.scaleY,0});
    Vector3 xp=apply3DDeformations(obj,{1.25f*obj.scaleX,0,0}); Vector3 xn=apply3DDeformations(obj,{-1.25f*obj.scaleX,0,0});
    Vector3 zp=apply3DDeformations(obj,{0,0,1.25f*obj.scaleZ}); Vector3 zn=apply3DDeformations(obj,{0,0,-1.25f*obj.scaleZ});
    builder.addTriangle(top,xp,zp,{0,1,0}); builder.addTriangle(top,zp,xn,{0,1,0}); builder.addTriangle(top,xn,zn,{0,1,0}); builder.addTriangle(top,zn,xp,{0,1,0});
    builder.addTriangle(bot,zp,xp,{0,-1,0}); builder.addTriangle(bot,xn,zp,{0,-1,0}); builder.addTriangle(bot,zn,xn,{0,-1,0}); builder.addTriangle(bot,xp,zn,{0,-1,0});
    return tfUploadMeshBuilder(builder, tfIsOrganic3DObject(obj), tfIsOrganic3DObject(obj));
}

Mesh tfBuildOptimizedPrism(const TFObject3D &obj, int lod)
{
    TFMeshBuilder builder;
    const int sides = 6;
    const float rx = 1.35f * fabsf(obj.scaleX);
    const float rz = 1.35f * fabsf(obj.scaleZ);
    const float hy = 1.45f * fabsf(obj.scaleY);

    vector<Vector3> top(sides);
    vector<Vector3> bottom(sides);

    // Regular closed hexagonal prism. Both end caps are explicit triangle
    // fans and the side quads are split into two triangles, so every boundary
    // edge is paired and the volume is watertight.
    Vector3 topCenter = apply3DDeformations(obj, {0.0f, hy, 0.0f});
    Vector3 bottomCenter = apply3DDeformations(obj, {0.0f, -hy, 0.0f});

    for (int i = 0; i < sides; ++i)
    {
        float a = 2.0f * PI * (float)i / (float)sides;
        top[i] = apply3DDeformations(obj,
            {rx * cosf(a), hy, rz * sinf(a)});
        bottom[i] = apply3DDeformations(obj,
            {rx * cosf(a), -hy, rz * sinf(a)});
    }

    for (int i = 0; i < sides; ++i)
    {
        int j = (i + 1) % sides;

        builder.addTriangle(bottom[i], bottom[j], top[j],
                            {0.0f, 0.0f, 0.0f});
        builder.addTriangle(bottom[i], top[j], top[i],
                            {0.0f, 0.0f, 0.0f});

        builder.addTriangle(topCenter, top[i], top[j],
                            {0.0f, 1.0f, 0.0f});
        builder.addTriangle(bottomCenter, bottom[j], bottom[i],
                            {0.0f, -1.0f, 0.0f});
    }

    return tfUploadMeshBuilder(builder, tfIsOrganic3DObject(obj), tfIsOrganic3DObject(obj));
}

Mesh tfBuildOptimizedRock(const TFObject3D &obj, int lod)
{
    TFMeshBuilder builder;
    const int sides = (lod == 0 ? 8 : (lod == 1 ? 8 : 6));

    // Deterministic irregular convex rock with a complete top and bottom.
    // The middle ring is wider, giving a natural faceted silhouette without
    // relying on open strips or a missing cap.
    const float yBottom = -1.02f;
    const float yLower = -0.38f;
    const float yUpper = 0.58f;
    const float yTop = 1.02f;

    const float radiusLowerX = 0.82f;
    const float radiusLowerZ = 0.90f;
    const float radiusMidX = 1.20f;
    const float radiusMidZ = 1.08f;
    const float radiusUpperX = 0.88f;
    const float radiusUpperZ = 0.80f;

    vector<Vector3> lower(sides);
    vector<Vector3> upper(sides);

    for (int i = 0; i < sides; ++i)
    {
        float a = 2.0f * PI * (float)i / (float)sides;
        float wobble1 = 1.0f + 0.10f * sinf((float)i * 2.31f);
        float wobble2 = 1.0f + 0.07f * cosf((float)i * 1.73f + 0.8f);
        lower[i] = apply3DDeformations(obj,
            {obj.scaleX * radiusLowerX * wobble1 * cosf(a),
             obj.scaleY * yLower,
             obj.scaleZ * radiusLowerZ * wobble2 * sinf(a)});
        upper[i] = apply3DDeformations(obj,
            {obj.scaleX * radiusUpperX * wobble2 * cosf(a + 0.08f),
             obj.scaleY * yUpper,
             obj.scaleZ * radiusUpperZ * wobble1 * sinf(a + 0.08f)});
    }

    vector<Vector3> mid(sides);
    for (int i = 0; i < sides; ++i)
    {
        float a = 2.0f * PI * (float)i / (float)sides;
        float wobble = 1.0f + 0.09f * sinf((float)i * 2.17f + 1.1f);
        mid[i] = apply3DDeformations(obj,
            {obj.scaleX * radiusMidX * wobble * cosf(a),
             obj.scaleY * 0.06f,
             obj.scaleZ * radiusMidZ * wobble * sinf(a)});
    }

    Vector3 bottom = apply3DDeformations(obj,
        {0.0f, obj.scaleY * yBottom, 0.0f});
    Vector3 top = apply3DDeformations(obj,
        {0.0f, obj.scaleY * yTop, 0.0f});

    for (int i = 0; i < sides; ++i)
    {
        int j = (i + 1) % sides;

        builder.addTriangle(bottom, lower[j], lower[i],
                            {0.0f, -1.0f, 0.0f});
        builder.addTriangle(lower[i], lower[j], mid[j],
                            {0.0f, 0.0f, 0.0f});
        builder.addTriangle(lower[i], mid[j], mid[i],
                            {0.0f, 0.0f, 0.0f});
        builder.addTriangle(mid[i], mid[j], upper[j],
                            {0.0f, 0.0f, 0.0f});
        builder.addTriangle(mid[i], upper[j], upper[i],
                            {0.0f, 0.0f, 0.0f});
        builder.addTriangle(top, upper[i], upper[j],
                            {0.0f, 1.0f, 0.0f});
    }

    return tfUploadMeshBuilder(builder, tfIsOrganic3DObject(obj), tfIsOrganic3DObject(obj));
}

Mesh tfBuildOptimizedPyramid(const TFObject3D &obj)
{
    TFMeshBuilder builder;
    builder.reserveTriangles(6);

    Vector3 a = apply3DDeformations(obj,
    {-1.3f * obj.scaleX, -1.0f * obj.scaleY, 1.3f * obj.scaleZ});
    Vector3 b = apply3DDeformations(obj,
    {1.3f * obj.scaleX, -1.0f * obj.scaleY, 1.0f * obj.scaleZ});
    Vector3 c = apply3DDeformations(obj,
    {0.8f * obj.scaleX, -1.0f * obj.scaleY, -1.3f * obj.scaleZ});
    Vector3 d = apply3DDeformations(obj,
    {-1.0f * obj.scaleX, -1.0f * obj.scaleY, -1.0f * obj.scaleZ});
    Vector3 top = apply3DDeformations(obj,
    {0.0f, 1.8f * obj.scaleY, 0.4f * obj.scaleZ});

    builder.addTriangle(a, b, top, tfAdd(tfAdd(a, b), top));
    builder.addTriangle(b, c, top, tfAdd(tfAdd(b, c), top));
    builder.addTriangle(c, d, top, tfAdd(tfAdd(c, d), top));
    builder.addTriangle(d, a, top, tfAdd(tfAdd(d, a), top));
    builder.addTriangle(a, d, c, {0.0f, -1.0f, 0.0f});
    builder.addTriangle(a, c, b, {0.0f, -1.0f, 0.0f});

    return tfUploadMeshBuilder(builder, tfIsOrganic3DObject(obj), tfIsOrganic3DObject(obj));
}

Mesh tfBuildOptimizedMesh3D(const TFObject3D &obj, int lod)
{
    if (obj.shape == "cube") return tfBuildOptimizedCube(obj, lod);
    if (obj.shape == "sphere") return tfBuildOptimizedSphere(obj, lod);
    if (obj.shape == "cone") return tfBuildOptimizedCone(obj, lod);
    if (obj.shape == "cylinder") return tfBuildOptimizedCylinder(obj, lod);
    if (obj.shape == "capsule") return tfBuildOptimizedCapsule(obj, lod);
    if (obj.shape == "crystal") return tfBuildOptimizedCrystal(obj, lod);
    if (obj.shape == "diamond") return tfBuildOptimizedDiamond(obj);
    if (obj.shape == "prism") return tfBuildOptimizedPrism(obj, lod);
    if (obj.shape == "rock") return tfBuildOptimizedRock(obj, lod);
    return tfBuildOptimizedPyramid(obj);
}

void tfDiscardOptimizedMesh3D(const string &name)
{
    auto it = optimized3DMeshes.find(name);
    if (it == optimized3DMeshes.end())
        return;

    if (it->second.valid)
        UnloadMesh(it->second.mesh);

    it->second.softBaseVertices.clear();
    it->second.softNormalGroup.clear();
    it->second.softNormalSums.clear();
    optimized3DMeshes.erase(it);
}

void tfPrepareOptimizedMaterial3D()
{
    if (optimized3DMaterialReady)
        return;

    optimized3DMaterial = LoadMaterialDefault();
    optimized3DMaterialReady = true;
    tfPrepareSurfaceShader();
    tfPrepareOpticalShader();
}

Matrix tfMatrixRotateXYZ(float x, float y, float z)
{
    float cx = cosf(x), sx = sinf(x);
    float cy = cosf(y), sy = sinf(y);
    float cz = cosf(z), sz = sinf(z);

    // Same XYZ convention used by raylib/raymath, written directly
    // into raylib's Matrix field layout.
    float r00 = cy * cz;
    float r01 = cy * sz;
    float r02 = -sy;

    float r10 = sx * sy * cz - cx * sz;
    float r11 = sx * sy * sz + cx * cz;
    float r12 = sx * cy;

    float r20 = cx * sy * cz + sx * sz;
    float r21 = cx * sy * sz - sx * cz;
    float r22 = cx * cy;

    Matrix m =
    {
        r00, r10, r20, 0.0f,
        r01, r11, r21, 0.0f,
        r02, r12, r22, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    return m;
}

// ============================================================
// FAST SOFT-BODY MOLD STREAM
// ============================================================
// The molding math below is intentionally the ORIGINAL TigerFlash mold:
// apply3DDeformations() -> applySoftBodyDeformation() ->
// tfMoldSoftBodyAgainstCollider().
//
// The only change is where the result goes. Instead of destroying and
// rebuilding the whole 5k+ triangle mesh whenever a soft-body state changes,
// keep the same mesh/topology and stream its molded vertex positions into the
// existing dynamic GPU buffers every visible frame.
// ============================================================

vector<float> tfBuildSoftBaseVertices(const TFObject3D &obj, int lod)
{
    bool previous = tfBuildingSoftBaseGeometry;
    tfBuildingSoftBaseGeometry = true;

    Mesh baseMesh = tfBuildOptimizedMesh3D(obj, lod);
    vector<float> base;

    if (baseMesh.vertexCount > 0 && baseMesh.vertices != nullptr)
    {
        const size_t count = static_cast<size_t>(baseMesh.vertexCount) * 3u;
        base.assign(baseMesh.vertices, baseMesh.vertices + count);
    }

    if (baseMesh.vertexCount > 0)
        UnloadMesh(baseMesh);

    tfBuildingSoftBaseGeometry = previous;
    return base;
}

void tfPrepareSoftNormalGroups(TFOptimized3DMeshCache &cache)
{
    cache.softNormalGroup.clear();
    cache.softNormalSums.clear();

    if (cache.softBaseVertices.empty())
        return;

    struct Key
    {
        int x;
        int y;
        int z;

        bool operator==(const Key &other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct KeyHash
    {
        size_t operator()(const Key &k) const noexcept
        {
            size_t h = static_cast<size_t>(k.x) * 73856093u;
            h ^= static_cast<size_t>(k.y) * 19349663u;
            h ^= static_cast<size_t>(k.z) * 83492791u;
            return h;
        }
    };

    const size_t vertexCount = cache.softBaseVertices.size() / 3u;
    cache.softNormalGroup.resize(vertexCount, -1);

    unordered_map<Key, int, KeyHash> groups;
    groups.reserve(vertexCount);

    auto keyOf = [](float x, float y, float z) -> Key
    {
        return {
            static_cast<int>(llround(x * 10000.0f)),
            static_cast<int>(llround(y * 10000.0f)),
            static_cast<int>(llround(z * 10000.0f))
        };
    };

    int nextGroup = 0;
    for (size_t i = 0; i < vertexCount; ++i)
    {
        const size_t k = i * 3u;
        Key key = keyOf(
            cache.softBaseVertices[k + 0],
            cache.softBaseVertices[k + 1],
            cache.softBaseVertices[k + 2]
        );

        auto it = groups.find(key);
        if (it == groups.end())
            it = groups.emplace(key, nextGroup++).first;

        cache.softNormalGroup[i] = it->second;
    }

    cache.softNormalSums.assign(static_cast<size_t>(nextGroup), {0.0f, 0.0f, 0.0f});
}

void tfUpdateSoftBodyMoldFast(const TFObject3D &obj,
                              TFOptimized3DMeshCache &cache)
{
    if (!cache.valid || cache.mesh.vertices == nullptr)
        return;

    if (cache.softBaseVertices.size() !=
        static_cast<size_t>(cache.mesh.vertexCount) * 3u)
        return;

    if (cache.softNormalGroup.size() !=
        static_cast<size_t>(cache.mesh.vertexCount))
        tfPrepareSoftNormalGroups(cache);

    const int vertexCount = cache.mesh.vertexCount;
    if (vertexCount <= 0)
        return;

    // EXACT old molding path. No new physical equations are introduced here.
    for (int i = 0; i < vertexCount; ++i)
    {
        const size_t k = static_cast<size_t>(i) * 3u;
        Vector3 base =
        {
            cache.softBaseVertices[k + 0],
            cache.softBaseVertices[k + 1],
            cache.softBaseVertices[k + 2]
        };

        Vector3 molded = apply3DDeformations(obj, base);

        cache.mesh.vertices[k + 0] = molded.x;
        cache.mesh.vertices[k + 1] = molded.y;
        cache.mesh.vertices[k + 2] = molded.z;
    }

    // Preserve the same smooth-normal behavior as the old mesh builder, but
    // reuse precomputed vertex-group indices so there is no unordered_map
    // construction in every frame.
    if (cache.mesh.normals != nullptr &&
        cache.softNormalSums.size() > 0)
    {
        fill(cache.softNormalSums.begin(), cache.softNormalSums.end(),
             Vector3{0.0f, 0.0f, 0.0f});

        const int triangleCount = cache.mesh.vertexCount / 3;
        for (int t = 0; t < triangleCount; ++t)
        {
            const int i0 = t * 3;
            const int i1 = i0 + 1;
            const int i2 = i0 + 2;

            Vector3 p0 =
            {
                cache.mesh.vertices[i0 * 3 + 0],
                cache.mesh.vertices[i0 * 3 + 1],
                cache.mesh.vertices[i0 * 3 + 2]
            };
            Vector3 p1 =
            {
                cache.mesh.vertices[i1 * 3 + 0],
                cache.mesh.vertices[i1 * 3 + 1],
                cache.mesh.vertices[i1 * 3 + 2]
            };
            Vector3 p2 =
            {
                cache.mesh.vertices[i2 * 3 + 0],
                cache.mesh.vertices[i2 * 3 + 1],
                cache.mesh.vertices[i2 * 3 + 2]
            };

            Vector3 e1 = tfSub(p1, p0);
            Vector3 e2 = tfSub(p2, p0);
            Vector3 n = tfCross(e1, e2);

            int g0 = cache.softNormalGroup[i0];
            int g1 = cache.softNormalGroup[i1];
            int g2 = cache.softNormalGroup[i2];

            if (g0 >= 0) cache.softNormalSums[g0] = tfAdd(cache.softNormalSums[g0], n);
            if (g1 >= 0) cache.softNormalSums[g1] = tfAdd(cache.softNormalSums[g1], n);
            if (g2 >= 0) cache.softNormalSums[g2] = tfAdd(cache.softNormalSums[g2], n);
        }

        for (int i = 0; i < vertexCount; ++i)
        {
            int group = cache.softNormalGroup[i];
            if (group < 0 || group >= static_cast<int>(cache.softNormalSums.size()))
                continue;

            Vector3 n = tfNormalize(cache.softNormalSums[group]);
            cache.mesh.normals[i * 3 + 0] = n.x;
            cache.mesh.normals[i * 3 + 1] = n.y;
            cache.mesh.normals[i * 3 + 2] = n.z;
        }
    }

    UpdateMeshBuffer(
        cache.mesh,
        0,
        cache.mesh.vertices,
        cache.mesh.vertexCount * 3 * static_cast<int>(sizeof(float)),
        0
    );

    if (cache.mesh.normals != nullptr)
    {
        UpdateMeshBuffer(
            cache.mesh,
            2,
            cache.mesh.normals,
            cache.mesh.vertexCount * 3 * static_cast<int>(sizeof(float)),
            0
        );
    }
}

void tfDrawOptimized3DObjectSingle(const TFObject3D &obj,
                                    const Camera3D &camera,
                                    int screenWidth,
                                    int screenHeight,
                                    float alpha = 1.0f)
{
    const float aspect =
        static_cast<float>(screenWidth) /
        static_cast<float>(max(1, screenHeight));
    const float farClip = 220.0f;

    if (!tfObjectInsideFrustum3D(obj, camera, aspect, farClip))
    {
        // Keep the GPU mesh cached while the object is temporarily outside
        // the camera. Rebuilding geometry every time an object crosses the
        // frustum is a large CPU/GPU spike. A separate memory governor evicts
        // genuinely cold meshes when the cache exceeds its RAM budget.
        return;
    }

    tfPrepareOptimizedMaterial3D();

    TFOptimized3DMeshCache &cache = optimized3DMeshes[obj.name];
    cache.lastVisibleFrame = tfPerf.frameIndex;
    const bool softForCache = tfIsOrganic3DObject(obj);
    size_t signature = cache.signature;
    int lod = tfChooseLOD3D(
        obj,
        camera,
        screenHeight,
        cache.valid ? cache.lod : -1
    );

    // Shared cluster polygon budget: never render finer than this
    // object's fair share of its touching-group's budget allows,
    // even when the camera is right on top of it.
    auto clusterIt = tfSmoothedClusterSize3D.find(obj.name);
    float clusterSize =
        (clusterIt != tfSmoothedClusterSize3D.end()) ? clusterIt->second : 1.0f;
    lod = max(lod, tfClusterLODFloor3D(obj, clusterSize));

    // Organic/soft meshes use the measured machine performance as the final
    // quality guard. This intentionally wins over the old collision-cluster
    // budget when that budget would force a liquid/deformed surface below its
    // safe mesh density. Ordinary rigid objects keep the original cluster
    // optimization unchanged.
    if (tfIsOrganic3DObject(obj))
    {
        // Combine the existing FPS-based governor with the CPU/RAM-aware
        // resource governor. The higher (more conservative) tier wins.
        lod = max(lod, tfAdaptiveOrganicLOD);
        lod = max(lod, tfPerf.organicLOD);
        lod = min(lod, 3);
    }

    const bool softObject = softForCache;

    if (!softObject ||
        !cache.valid ||
        cache.lod != lod ||
        tfPerf.frameIndex - cache.lastSignatureCheckFrame >=
            tfSoftMeshRebuildIntervalFrames(obj, camera, lod))
    {
        signature = tfMeshSignature3D(obj);
        cache.observedSignature = signature;
        cache.lastSignatureCheckFrame = tfPerf.frameIndex;
    }
    else
    {
        signature = cache.observedSignature;
    }

    const bool meshChanged =
        !cache.valid ||
        cache.signature != signature ||
        cache.lod != lod;

    bool mayRebuild = true;
    if (softObject && cache.valid && meshChanged)
    {
        const uint64_t elapsedFrames =
            tfPerf.frameIndex - cache.lastBuildFrame;
        const uint64_t distanceInterval =
            tfSoftMeshRebuildIntervalFrames(obj, camera, lod);
        const uint64_t governorInterval =
            static_cast<uint64_t>(max(1, tfPerf.softMeshEveryFrames));
        mayRebuild = elapsedFrames >= max(distanceInterval, governorInterval);
    }

    if (meshChanged && mayRebuild)
    {
        if (cache.valid)
            UnloadMesh(cache.mesh);

        cache.mesh = tfBuildOptimizedMesh3D(obj, lod);
        cache.valid = cache.mesh.vertexCount > 0;
        cache.lod = lod;
        cache.signature = signature;
        cache.observedSignature = signature;
        cache.lastBuildFrame = tfPerf.frameIndex;

        if (softObject && cache.valid)
        {
            cache.softBaseVertices = tfBuildSoftBaseVertices(obj, lod);
            tfPrepareSoftNormalGroups(cache);
        }
        else
        {
            cache.softBaseVertices.clear();
            cache.softNormalGroup.clear();
            cache.softNormalSums.clear();
        }
    }

    if (!cache.valid)
        return;

    if (softObject)
    {
        if (cache.softBaseVertices.empty())
        {
            cache.softBaseVertices = tfBuildSoftBaseVertices(obj, lod);
            tfPrepareSoftNormalGroups(cache);
        }
        tfUpdateSoftBodyMoldFast(obj, cache);
    }

    Material material = optimized3DMaterial;
    Color drawColor = getColor(obj.color);
    if (alpha < 0.999f)
        drawColor.a = (unsigned char)max(0.0f, min(255.0f, alpha * 255.0f));
    material.maps[MATERIAL_MAP_DIFFUSE].color = drawColor;
    const bool translucentPass = alpha < 0.999f;

    Matrix transform = tfMatrixRotateXYZ(
        obj.rotationX * DEG2RAD,
        obj.rotationY * DEG2RAD,
        obj.rotationZ * DEG2RAD
    );
    transform.m12 = obj.x;
    transform.m13 = obj.y;
    transform.m14 = obj.z;

    if (tfIsOpticalShape(obj) &&
        tfOpticalShaderReady &&
        tfBakedOpticalEnvironmentReady)
    {
        material.shader = tfOpticalShader;
        material.maps[MATERIAL_MAP_DIFFUSE].texture =
            tfBakedOpticalEnvironment;
        // Keep the exact color requested by say 3d so each crystal can be
        // ruby, blue, green, purple, etc. The shader adds the optical tint.
        material.maps[MATERIAL_MAP_DIFFUSE].color = drawColor;

        tfSetOpticalParams(obj, camera);

        // Draw optical objects after the opaque scene. Depth testing remains
        // active, but depth writes are disabled so overlapping glass-like
        // objects can blend instead of punching opaque holes into each other.
        rlDisableDepthMask();
        BeginBlendMode(BLEND_ALPHA);
        DrawMesh(cache.mesh, material, transform);
        EndBlendMode();
        rlEnableDepthMask();
    }
    else
    {
        if (tfSurfaceShaderReady)
        {
            material.shader = tfSurfaceShader;
            material.maps[MATERIAL_MAP_DIFFUSE].texture = { 0 };
            material.maps[MATERIAL_MAP_DIFFUSE].color = drawColor;
        }

        if (translucentPass)
        {
            rlDisableDepthMask();
            BeginBlendMode(BLEND_ALPHA);
            DrawMesh(cache.mesh, material, transform);
            EndBlendMode();
            rlEnableDepthMask();
        }
        else
        {
            DrawMesh(cache.mesh, material, transform);
        }
    }

    tfPerf.trianglesDrawn += static_cast<uint64_t>(max(0, cache.mesh.triangleCount));
}

// External-library motion blur wrapper. The language library controls the
// property; the core renderer supplies a low-cost temporal trail. A stopped
// object produces no extra draws.
void tfDrawOptimized3DObject(const TFObject3D &obj,
                             const Camera3D &camera,
                             int screenWidth,
                             int screenHeight)
{
    const float blur = max(0.0f, min(10.0f, obj.motionBlur));

    if (blur > 0.001f &&
        obj.motionBlurPreviousInitialized &&
        !tfIsOpticalShape(obj))
    {
        Vector3 delta =
        {
            obj.x - obj.motionBlurPreviousPosition.x,
            obj.y - obj.motionBlurPreviousPosition.y,
            obj.z - obj.motionBlurPreviousPosition.z
        };

        if (tfLength(delta) > 0.0015f)
        {
            const float strength = blur / 10.0f;
            int samples = 2 + static_cast<int>(ceilf(blur * 0.8f));
            samples = max(2, min(10, samples));

            // The original one-frame distance can be very small (for example
            // key movement of 0.10). Amplify the visual trail so motion blur
            // is actually visible without changing the object's real position.
            const float trailScale =
                max(0.75f, min(3.40f, 0.75f + strength * 2.60f));

            for (int i = samples; i >= 1; --i)
            {
                const float t =
                    static_cast<float>(i) /
                    static_cast<float>(samples + 1);

                TFObject3D ghost = obj;
                ghost.motionBlur = 0.0f;
                ghost.motionBlurPreviousInitialized = false;
                ghost.x = obj.x - delta.x * t * trailScale;
                ghost.y = obj.y - delta.y * t * trailScale;
                ghost.z = obj.z - delta.z * t * trailScale;

                // Make the trail visible enough at ordinary TigerFlash
                // movement speeds while keeping the real object fully opaque.
                const float trailAlpha =
                    (0.045f + 0.20f * strength) * (1.0f - t);

                if (trailAlpha > 0.001f)
                    tfDrawOptimized3DObjectSingle(
                        ghost,
                        camera,
                        screenWidth,
                        screenHeight,
                        trailAlpha);
            }
        }
    }

    tfDrawOptimized3DObjectSingle(
        obj,
        camera,
        screenWidth,
        screenHeight,
        1.0f);
}

void tfMaintainOptimized3DMeshCache()
{
    if (optimized3DMeshes.size() <= tfPerf.maxCachedMeshes)
        return;

    // Prefer eviction of meshes that have not been visible for a while.
    // Soft meshes are retained longer because rebuilding them is more costly.
    vector<string> eviction;
    eviction.reserve(optimized3DMeshes.size());

    for (const auto &entry : optimized3DMeshes)
    {
        const TFOptimized3DMeshCache &cache = entry.second;
        if (tfPerf.frameIndex > cache.lastVisibleFrame + 180)
            eviction.push_back(entry.first);
    }

    sort(eviction.begin(), eviction.end(),
         [&](const string &a, const string &b)
    {
        return optimized3DMeshes[a].lastVisibleFrame <
               optimized3DMeshes[b].lastVisibleFrame;
    });

    size_t target = tfPerf.maxCachedMeshes;
    if (tfPerf.systemRAMPercent > 90.0f)
        target = min(target, static_cast<size_t>(128));

    for (const string &name : eviction)
    {
        if (optimized3DMeshes.size() <= target)
            break;
        tfDiscardOptimizedMesh3D(name);
    }

    // Hard cap: if nearly every mesh is active, remove the least-recently
    // visible entries until memory use returns to the configured envelope.
    while (optimized3DMeshes.size() > target)
    {
        auto oldest = optimized3DMeshes.end();
        for (auto it = optimized3DMeshes.begin(); it != optimized3DMeshes.end(); ++it)
        {
            if (oldest == optimized3DMeshes.end() ||
                it->second.lastVisibleFrame < oldest->second.lastVisibleFrame)
                oldest = it;
        }
        if (oldest == optimized3DMeshes.end())
            break;
        tfDiscardOptimizedMesh3D(oldest->first);
    }
}

void tfClearOptimized3DMeshes()
{
    for (auto &entry : optimized3DMeshes)
    {
        if (entry.second.valid)
            UnloadMesh(entry.second.mesh);
    }

    optimized3DMeshes.clear();
}


// =========================
// ERROR
// =========================
 
void error(string message)
{
    cout << "\033[1;31m";
    cout << "ERROR: " << message << endl;
    cout << "\033[0m";
}
 
// =========================
// VALID NAME
// =========================
 
bool validName(string name)
{
    if (name.empty())
        return false;
 
    unsigned char first = static_cast<unsigned char>(name[0]);
 
    if (!isalpha(first) && name[0] != '_')
        return false;
 
    for (char c : name)
    {
        unsigned char uc = static_cast<unsigned char>(c);
 
        if (!isalnum(uc) && c != '_')
            return false;
    }
 
    return true;
}
 
// =========================
// NUMBER
// =========================
 
// Forward declaration used by the numeric helpers below.
string trim(string text);
 
bool isNumber(string value)
{
    value = trim(value);
 
    if (value.empty())
        return false;
 
    size_t used = 0;
 
    try
    {
        stod(value, &used);
    }
    catch (...)
    {
        return false;
    }
 
    while (used < value.size() &&
           isspace(static_cast<unsigned char>(value[used])))
    {
        used++;
    }
 
    return used == value.size();
}
 
bool isIntegerNumber(string value)
{
    value = trim(value);
 
    if (value.empty())
        return false;
 
    size_t start = 0;
 
    if (value[0] == '+' || value[0] == '-')
        start = 1;
 
    if (start >= value.size())
        return false;
 
    for (size_t i = start; i < value.size(); i++)
    {
        if (!isdigit(static_cast<unsigned char>(value[i])))
            return false;
    }
 
    return true;
}
 
// =========================
// TRIM
// =========================
 
string trim(string text)
{
    while (!text.empty() &&
           isspace(static_cast<unsigned char>(text.front())))
    {
        text.erase(0, 1);
    }
 
    while (!text.empty() &&
           isspace(static_cast<unsigned char>(text.back())))
    {
        text.pop_back();
    }
 
    return text;
}
 
// =========================
// CASE INSENSITIVITY
// =========================
//
// TigerFlash commands, keywords, and identifiers (key, rotate, cube, true,
// variable/object names typed bare, etc.) can be written in any mix of
// upper/lower case: "KEY", "Key" and "key" all work the same way.
//
// Text the player actually wants to keep exactly as written - anything
// between double quotes, such as printed messages ("growl(\"Hello World\")")
// or names given to objects/variables - is left untouched. Only the
// characters OUTSIDE quotes are folded to lowercase before the rest of the
// interpreter ever looks at the line.
//
string toLowerOutsideQuotes(const string &text)
{
    string result = text;
    bool insideQuotes = false;
 
    for (size_t i = 0; i < result.size(); i++)
    {
        char c = result[i];
 
        if (c == '"')
        {
            insideQuotes = !insideQuotes;
            continue;
        }
 
        if (!insideQuotes && c >= 'A' && c <= 'Z')
            result[i] = static_cast<char>(c - 'A' + 'a');
    }
 
    return result;
}
 
// =========================
// REPLACE VARIABLES
// =========================
 
bool isIdentifierByte(unsigned char c)
{
    return isalnum(c) || c == '_' || c >= 128;
}
 
string replaceVariables(string text)
{
    string result;
 
    for (size_t i = 0; i < text.size();)
    {
        if (isalpha((unsigned char)text[i]) || text[i] == '_')
        {
            size_t startName = i;
            string name;
 
            while (i < text.size() &&
                   isIdentifierByte((unsigned char)text[i]))
            {
                name += text[i];
                i++;
            }
 
            bool canReplace = variables.count(name) != 0;
 
            if (!canReplace)
            {
                result.append(text, startName, i - startName);
            }
            else
            {
                result += variables[name];
            }
        }
        else
        {
            result += text[i];
            i++;
        }
    }
 
    return result;
}
 
// ============================================================
// SPLIT GROWL PARTS
// ============================================================
 
vector<string> splitMovement(string text)
{
    vector<string> parts;
    string current;
 
    for (char c : text)
    {
        if (c == ',')
        {
            parts.push_back(trim(current));
            current.clear();
        }
        else
        {
            current += c;
        }
    }
 
    parts.push_back(trim(current));
    return parts;
}
 
vector<string> splitParts(string text)
{
    vector<string> parts;
    string current;
    char quote = 0;
    int parentheses = 0;
 
    for (char c : text)
    {
        if (quote != 0)
        {
            current += c;
            if (c == quote)
                quote = 0;
            continue;
        }
 
        if (c == '"' || c == '\'')
        {
            quote = c;
            current += c;
            continue;
        }
 
        if (c == '(')
        {
            parentheses++;
            current += c;
            continue;
        }
 
        if (c == ')')
        {
            if (parentheses > 0)
                parentheses--;
            current += c;
            continue;
        }
 
        if (c == ';' && parentheses == 0)
        {
            parts.push_back(trim(current));
            current.clear();
            continue;
        }
 
        current += c;
    }
 
    parts.push_back(trim(current));
    return parts;
}
 
// ============================================================
// COUNT PARSER
// ============================================================
 
class CountParser
{
private:
    string expression;
    size_t position;
 
public:
 
    CountParser(string exp)
    {
        expression = exp;
        position = 0;
    }
 
    void skipSpaces()
    {
        while (position < expression.size() &&
               isspace(static_cast<unsigned char>(expression[position])))
        {
            position++;
        }
    }
 
    double parseNumber()
    {
        skipSpaces();
 
        string number;
        bool hasDigit = false;
        bool hasDot = false;
 
        if (position < expression.size() &&
            (expression[position] == '+' || expression[position] == '-'))
        {
            number += expression[position];
            position++;
        }
 
        while (position < expression.size())
        {
            unsigned char c = static_cast<unsigned char>(expression[position]);
 
            if (isdigit(c))
            {
                hasDigit = true;
                number += expression[position];
                position++;
                continue;
            }
 
            if (expression[position] == '.')
            {
                if (hasDot)
                    throw runtime_error("Invalid numeric expression.");
 
                hasDot = true;
                number += expression[position];
                position++;
                continue;
            }
 
            break;
        }
 
        if (!hasDigit)
            throw runtime_error("Invalid numeric expression.");
 
        try
        {
            return stod(number);
        }
        catch (...)
        {
            throw runtime_error("Invalid numeric expression.");
        }
    }
 
    double parseFactor()
    {
        skipSpaces();
 
        if (position < expression.size() &&
            expression[position] == '(')
        {
            position++;
 
            double value = parseExpression();
 
            skipSpaces();
 
            if (position >= expression.size() ||
                expression[position] != ')')
            {
                throw runtime_error("Missing closing parenthesis.");
            }
 
            position++;
 
            return value;
        }
 
        // Sinal unário, mantendo a mesma aritmética da linguagem.
        if (position < expression.size() &&
            (expression[position] == '+' || expression[position] == '-'))
        {
            char sign = expression[position];
            position++;
 
            double value = parseFactor();
 
            if (sign == '-')
                return -value;
 
            return value;
        }
 
        if (position < expression.size() &&
            (isdigit(static_cast<unsigned char>(expression[position])) ||
             expression[position] == '.'))
        {
            return parseNumber();
        }
 
        if (position < expression.size() &&
            (isalpha(static_cast<unsigned char>(expression[position])) ||
             expression[position] == '_'))
        {
            string name;
 
            while (position < expression.size() &&
                   (isalnum(static_cast<unsigned char>(expression[position])) ||
                    expression[position] == '_'))
            {
                name += expression[position];
                position++;
            }
 
            if (!variables.count(name))
            {
                throw runtime_error("Variable not found: " + name);
            }
 
            string value = variables[name];
 
            if (!isNumber(value))
            {
                throw runtime_error(
                    "You cannot mix a string with a numeric value, or strings are not mathematical values TigerFlash can count on."
                );
            }
 
            return stod(value);
        }
 
        throw runtime_error("Invalid numeric expression.");
    }
 
    double parseTerm()
    {
        double value = parseFactor();
 
        while (true)
        {
            skipSpaces();
 
            if (position >= expression.size())
                break;
 
            char op = expression[position];
 
            if (op != '+' && op != '-' && op != '*' && op != '/')
                break;
 
            position++;
 
            double next = parseFactor();
 
            if (op == '+')
            {
                value += next;
            }
            else if (op == '-')
            {
                value -= next;
            }
            else if (op == '*')
            {
                value *= next;
            }
            else
            {
                if (next == 0)
                    throw runtime_error("Division by zero.");
 
                value /= next;
            }
        }
 
        return value;
    }
 
    double parseExpression()
    {
        return parseTerm();
    }
 
    double parse()
    {
        double result = parseExpression();
 
        skipSpaces();
 
        if (position != expression.size())
        {
            throw runtime_error("Invalid numeric expression.");
        }
 
        return result;
    }
};
 
// =========================
// FORMAT NUMBER
// =========================
 
string formatNumber(double value)
{
    if (!std::isfinite(value))
        return to_string(value);
 
    const double minLongLong =
        static_cast<double>(numeric_limits<long long>::lowest());
 
    const double maxLongLong =
        static_cast<double>(numeric_limits<long long>::max());
 
    if (value >= minLongLong &&
        value <= maxLongLong &&
        value == static_cast<long long>(value))
    {
        return to_string(static_cast<long long>(value));
    }
 
    string result = to_string(value);
 
    while (!result.empty() && result.back() == '0')
        result.pop_back();
 
    if (!result.empty() && result.back() == '.')
        result.pop_back();
 
    return result;
}
 
// ============================================================
// RANDOM
// ============================================================
 
mt19937 &randomGenerator()
{
    static mt19937 generator(
        static_cast<unsigned int>(
            chrono::high_resolution_clock::now().time_since_epoch().count()
        )
    );
    return generator;
}
 
string randomNumber()
{
    uniform_int_distribution<long long> lengthDist(1, 9);
    uniform_int_distribution<int> digitDist(0, 9);
 
    int length = lengthDist(randomGenerator());
    string result;
 
    for (int i = 0; i < length; i++)
    {
        int digit = digitDist(randomGenerator());
 
        if (i == 0 && length > 1)
            digit = uniform_int_distribution<int>(1, 9)(randomGenerator());
 
        result += char('0' + digit);
    }
 
    return result;
}
 
string randomText()
{
    const string chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789";
 
    uniform_int_distribution<int> lengthDist(1, 30);
    int length = lengthDist(randomGenerator());
 
    uniform_int_distribution<int> charDist(
        0, static_cast<int>(chars.size()) - 1
    );
 
    string result;
 
    for (int i = 0; i < length; i++)
        result += chars[charDist(randomGenerator())];
 
    return result;
}
 
string randomShape2D()
{
    const vector<string> shapes = {"cube", "sphere", "cone", "triangle"};
    uniform_int_distribution<int> dist(0, static_cast<int>(shapes.size()) - 1);
    return shapes[dist(randomGenerator())];
}

string randomShape()
{
    const vector<string> shapes =
    {
        "cube", "sphere", "cone", "triangle",
        "cylinder", "capsule", "crystal", "diamond",
        "prism", "rock"
    };
 
    uniform_int_distribution<int> dist(
        0, static_cast<int>(shapes.size()) - 1
    );
 
    return shapes[dist(randomGenerator())];
}
 
string randomColor()
{
    const vector<string> colors =
    {
        "white",
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "purple",
        "pink",
        "gray",
        "black"
    };
 
    uniform_int_distribution<int> dist(
        0, static_cast<int>(colors.size()) - 1
    );
 
    return colors[dist(randomGenerator())];
}
 
bool isRandomText(const string &value)
{
    return value.size() >= 2 &&
           ((value.front() == '"' && value.back() == '"') ||
            (value.front() == '\'' && value.back() == '\'')) &&
           trim(value.substr(1, value.size() - 2)) == "random";
}
 
// ============================================================
// LISTS (list nome {a; b; c})
// ============================================================
 
bool tryNumber(string value, double &number); // forward declaration
 
// Resolve o valor de um único item de lista: string entre aspas, número,
// "random", texto aleatório ou variável — mesma lógica usada no growl.
string resolveListElementValue(string part)
{
    part = trim(part);
 
    if (part.empty())
        return "";
 
    if (part == "random")
        return randomNumber();
 
    if (isRandomText(part))
        return randomText();
 
    string expression = part;
 
    if (expression.size() >= 2 &&
        ((expression.front() == '"' && expression.back() == '"') ||
         (expression.front() == '\'' && expression.back() == '\'')))
    {
        return expression.substr(1, expression.size() - 2);
    }
 
    try
    {
        CountParser parser(expression);
        double result = parser.parse();
        return formatNumber(result);
    }
    catch (runtime_error &)
    {
    }
 
    return replaceVariables(part);
}
 
// Procura por padrões "list nome[posicao]" dentro de um texto e troca cada
// ocorrência pelo valor guardado naquela posição (posição começa em 1).
string resolveListAccess(string text)
{
    string result;
    size_t i = 0;
 
    while (i < text.size())
    {
        unsigned char c = static_cast<unsigned char>(text[i]);
 
        if (isalpha(c) || text[i] == '_')
        {
            string word;
 
            while (i < text.size() &&
                   isIdentifierByte((unsigned char)text[i]))
            {
                word += text[i];
                i++;
            }
 
            if (word != "list")
            {
                result += word;
                continue;
            }
 
            size_t j = i;
 
            while (j < text.size() && isspace((unsigned char)text[j]))
                j++;
 
            string listName;
 
            while (j < text.size() &&
                   isIdentifierByte((unsigned char)text[j]))
            {
                listName += text[j];
                j++;
            }
 
            size_t k = j;
 
            while (k < text.size() && isspace((unsigned char)text[k]))
                k++;
 
            if (!listName.empty() &&
                k < text.size() &&
                text[k] == '[')
            {
                size_t closeBracket = text.find(']', k);
 
                if (closeBracket != string::npos)
                {
                    string indexText =
                        trim(text.substr(k + 1, closeBracket - k - 1));
 
                    double indexNumber = 0;
 
                    if (tryNumber(replaceVariables(indexText), indexNumber))
                    {
                        long long index = (long long)indexNumber;
 
                        if (!lists.count(listName))
                        {
                            error("List not found: \"" + listName + "\". Make sure it was created earlier with: list " + listName + " = {...}");
                        }
                        else if (index < 1 ||
                                 index > (long long)lists[listName].size())
                        {
                            error("List position out of range for \"" + listName + "\". Positions start at 1 and must be within the list's current size.");
                        }
                        else
                        {
                            result += lists[listName][index - 1];
                        }
 
                        i = closeBracket + 1;
                        continue;
                    }
                }
            }
 
            // Não é um acesso de lista válido: mantém o texto original.
            result += word;
            continue;
        }
 
        result += text[i];
        i++;
    }
 
    return result;
}
 
// =========================
// EXECUTE GROWL
// =========================
 
void executegrowl(string content, bool count, bool alert)
{
    content = trim(content);
    content = resolveListAccess(content);
 
    if (!count && content == "random")
    {
        if (alert)
            cout << "\033[1;31m";
 
        cout << randomNumber() << endl;
        cout << "\033[0m";
        return;
    }
 
    if (!count && isRandomText(content))
    {
        if (alert)
            cout << "\033[1;31m";
 
        cout << randomText() << endl;
        cout << "\033[0m";
        return;
    }
 
    if (alert)
        cout << "\033[1;31m";
 
    if (count)
    {
        vector<string> parts = splitParts(content);
        bool first = true;
 
        for (string part : parts)
        {
            part = trim(part);
 
            if (part.empty())
                continue;
 
            string value = part;
            string expression = part;
 
            if (expression.size() >= 2 &&
                ((expression.front() == '"' && expression.back() == '"') ||
                 (expression.front() == '\'' && expression.back() == '\'')))
            {
                expression = expression.substr(1, expression.size() - 2);
            }
 
            bool isNumeric = false;
            bool isRandomValue = false;
 
            if (part == "random")
            {
                value = randomNumber();
                isRandomValue = true;
            }
            else if (isRandomText(part))
            {
                value = randomText();
                isRandomValue = true;
            }
            else
            {
                try
                {
                    CountParser parser(expression);
                    double result = parser.parse();
                    value = formatNumber(result);
                    isNumeric = true;
                }
                catch (runtime_error &)
                {
                }
            }
 
            if (!isNumeric && !isRandomValue)
            {
                value = replaceVariables(part);
            }
 
            if (!first)
                cout << ' ';
 
            cout << value;
            first = false;
        }
 
        cout << endl;
        cout << "\033[0m";
        return;
    }
 
    string output = replaceVariables(content);
 
    cout << output << endl;
    cout << "\033[0m";
}
 
// =========================
// COLOR SYSTEM (2D / 3D)
// =========================
 
Color rgb(unsigned char r, unsigned char g, unsigned char b)
{
    return Color{r, g, b, 255};
}
 
Color getColor(string colorName)
{
    static const unordered_map<string, Color> colors =
    {
        // Primarias
        {"red", rgb(255, 0, 0)},
        {"pink", rgb(255, 105, 180)},
        {"baby_pink", rgb(255, 182, 193)},
        {"salmon", rgb(250, 128, 114)},
        {"wine", rgb(114, 47, 55)},
        {"burgundy", rgb(128, 0, 32)},
        {"carmine", rgb(150, 0, 24)},
        {"blood", rgb(139, 0, 0)},
 
        {"yellow", rgb(255, 255, 0)},
        {"pastel_yellow", rgb(253, 253, 150)},
        {"cream", rgb(255, 253, 208)},
        {"ivory", rgb(255, 255, 240)},
        {"mustard", rgb(255, 219, 88)},
        {"old_gold", rgb(207, 181, 59)},
        {"amber", rgb(255, 191, 0)},
 
        {"blue", rgb(0, 121, 241)},
        {"sky_blue", rgb(135, 206, 235)},
        {"baby_blue", rgb(137, 207, 240)},
        {"pool_blue", rgb(0, 191, 255)},
        {"navy", rgb(0, 0, 128)},
        {"royal_blue", rgb(65, 105, 225)},
        {"night_blue", rgb(15, 20, 50)},
 
        // Secundarias
        {"orange", rgb(255, 165, 0)},
        {"peach", rgb(255, 218, 185)},
        {"melon", rgb(253, 188, 180)},
        {"light_coral", rgb(240, 128, 128)},
        {"terracotta", rgb(226, 114, 91)},
        {"brick", rgb(156, 64, 55)},
        {"bronze", rgb(205, 127, 50)},
 
        {"green", rgb(0, 200, 0)},
        {"mint", rgb(152, 255, 152)},
        {"lime", rgb(50, 205, 50)},
        {"light_green", rgb(144, 238, 144)},
        {"moss", rgb(138, 154, 91)},
        {"military_green", rgb(75, 83, 32)},
        {"emerald", rgb(80, 200, 120)},
        {"forest_green", rgb(34, 139, 34)},
 
        {"purple", rgb(128, 0, 128)},
        {"violet", rgb(148, 0, 211)},
        {"lilac", rgb(200, 162, 200)},
        {"lavender", rgb(230, 230, 250)},
        {"mauve", rgb(224, 176, 255)},
        {"eggplant", rgb(97, 64, 81)},
        {"plum", rgb(142, 69, 133)},
        {"dark_purple", rgb(74, 20, 100)},
 
        // Terciarias
        {"coral", rgb(255, 127, 80)},
        {"light_salmon", rgb(255, 160, 122)},
        {"pastel_coral", rgb(255, 179, 171)},
        {"rust", rgb(183, 65, 14)},
        {"mahogany", rgb(192, 64, 0)},
 
        {"carrot", rgb(237, 145, 33)},
        {"apricot", rgb(251, 206, 177)},
        {"naples_yellow", rgb(250, 218, 94)},
        {"dark_bronze", rgb(128, 74, 0)},
        {"caramel", rgb(198, 135, 72)},
 
        {"cane_green", rgb(171, 205, 65)},
        {"pistachio", rgb(147, 197, 114)},
        {"tea_green", rgb(208, 240, 192)},
        {"light_avocado", rgb(190, 220, 120)},
        {"olive", rgb(128, 128, 0)},
        {"dark_moss", rgb(72, 88, 45)},
 
        {"turquoise", rgb(64, 224, 208)},
        {"cyan", rgb(0, 255, 255)},
        {"aqua_green", rgb(127, 255, 212)},
        {"light_turquoise", rgb(175, 238, 238)},
        {"petroleum_blue", rgb(0, 78, 84)},
        {"petroleum_green", rgb(0, 92, 75)},
 
        {"indigo", rgb(75, 0, 130)},
        {"anil", rgb(44, 62, 142)},
        {"lavender_blue", rgb(181, 184, 255)},
        {"periwinkle", rgb(204, 204, 255)},
        {"dark_navy", rgb(0, 0, 64)},
        {"midnight_blue", rgb(25, 25, 112)},
 
        {"magenta", rgb(255, 0, 255)},
        {"dark_burgundy", rgb(70, 0, 20)},
        {"hot_pink", rgb(255, 20, 147)},
        {"light_fuchsia", rgb(255, 119, 255)},
        {"cherry", rgb(222, 49, 99)},
        {"berry", rgb(153, 37, 84)},
        {"grape", rgb(111, 45, 168)},
 
        // Neutras e aliases úteis
        {"white", WHITE},
        {"gray", GRAY},
        {"grey", GRAY},
        {"black", BLACK},
        {"brown", rgb(139, 69, 19)},
        {"dark_brown", rgb(92, 51, 23)},
        {"beige", rgb(245, 245, 220)},
        {"gold", rgb(255, 215, 0)},
        {"silver", rgb(192, 192, 192)},
        {"goldenrod", rgb(218, 165, 32)},
        {"teal", rgb(0, 128, 128)},
        {"cyan_blue", rgb(0, 180, 220)}
    };
 
    auto it = colors.find(colorName);
    if (it != colors.end())
        return it->second;
 
    return WHITE;
}
 
bool validColor(string colorName)
{
    static const unordered_map<string, bool> names =
    {
        {"red", true}, {"pink", true}, {"baby_pink", true}, {"salmon", true},
        {"wine", true}, {"burgundy", true}, {"carmine", true}, {"blood", true},
        {"yellow", true}, {"pastel_yellow", true}, {"cream", true}, {"ivory", true},
        {"mustard", true}, {"old_gold", true}, {"amber", true},
        {"blue", true}, {"sky_blue", true}, {"baby_blue", true}, {"pool_blue", true},
        {"navy", true}, {"royal_blue", true}, {"night_blue", true},
        {"orange", true}, {"peach", true}, {"melon", true}, {"light_coral", true},
        {"terracotta", true}, {"brick", true}, {"bronze", true},
        {"green", true}, {"mint", true}, {"lime", true}, {"light_green", true},
        {"moss", true}, {"military_green", true}, {"emerald", true}, {"forest_green", true},
        {"purple", true}, {"violet", true}, {"lilac", true}, {"lavender", true},
        {"mauve", true}, {"eggplant", true}, {"plum", true}, {"dark_purple", true},
        {"coral", true}, {"light_salmon", true}, {"pastel_coral", true}, {"rust", true},
        {"mahogany", true}, {"carrot", true}, {"apricot", true}, {"naples_yellow", true},
        {"dark_bronze", true}, {"caramel", true}, {"cane_green", true}, {"pistachio", true},
        {"tea_green", true}, {"light_avocado", true}, {"olive", true}, {"dark_moss", true},
        {"turquoise", true}, {"cyan", true}, {"aqua_green", true}, {"light_turquoise", true},
        {"petroleum_blue", true}, {"petroleum_green", true}, {"indigo", true}, {"anil", true},
        {"lavender_blue", true}, {"periwinkle", true}, {"dark_navy", true}, {"midnight_blue", true},
        {"magenta", true}, {"dark_burgundy", true}, {"hot_pink", true}, {"light_fuchsia", true},
        {"cherry", true}, {"berry", true}, {"grape", true}, {"white", true}, {"gray", true},
        {"grey", true}, {"black", true}, {"brown", true}, {"dark_brown", true}, {"beige", true},
        {"gold", true}, {"silver", true}, {"goldenrod", true}, {"teal", true}, {"cyan_blue", true}
    };
 
    return names.count(colorName) != 0;
}
 
// =========================
// SAY 2D
// =========================
 
 
void draw2DObject(const TFObject2D &obj)
{
    Color objectColor = getColor(obj.color);
 
    float sx = obj.scaleX;
    float sy = obj.scaleY;
 
    if (obj.shape == "cube")
    {
        float width = 200.0f * fabsf(sx);
        float height = 200.0f * fabsf(sy);
        Rectangle rect = {obj.x - width / 2.0f, obj.y - height / 2.0f, width, height};
        Vector2 origin = {width / 2.0f, height / 2.0f};
        DrawRectanglePro(rect, origin, obj.rotation, objectColor);
    }
    else if (obj.shape == "sphere")
    {
        DrawEllipse((int)obj.x, (int)obj.y,
                    100.0f * fabsf(sx),
                    100.0f * fabsf(sy),
                    objectColor);
    }
    else if (obj.shape == "cone")
    {
        Vector2 tip = { obj.x, obj.y - 120.0f * sy };
        Vector2 left = { obj.x - 110.0f * sx, obj.y + 80.0f * sy };
        Vector2 right = { obj.x + 110.0f * sx, obj.y + 80.0f * sy };

        auto rotatePoint = [&](Vector2 p) -> Vector2
        {
            float a = -obj.rotation * DEG2RAD;
            float c = cosf(a);
            float sn = sinf(a);
            float dx = p.x - obj.x;
            float dy = p.y - obj.y;
            return {obj.x + dx * c - dy * sn, obj.y + dx * sn + dy * c};
        };

        tip = rotatePoint(tip);
        left = rotatePoint(left);
        right = rotatePoint(right);

        DrawTriangle(tip, left, right, objectColor);
        DrawEllipse((int)obj.x, (int)(obj.y + 80.0f * sy),
                    110.0f * fabsf(sx),
                    25.0f * fabsf(sy),
                    objectColor);
    }
    else if (obj.shape == "triangle")
    {
        Vector2 v1 = { obj.x, obj.y - 120.0f * sy };
        Vector2 v2 = { obj.x - 100.0f * sx, obj.y + 100.0f * sy };
        Vector2 v3 = { obj.x + 100.0f * sx, obj.y + 100.0f * sy };

        auto rotatePoint = [&](Vector2 p) -> Vector2
        {
            float a = -obj.rotation * DEG2RAD;
            float c = cosf(a);
            float sn = sinf(a);
            float dx = p.x - obj.x;
            float dy = p.y - obj.y;
            return {obj.x + dx * c - dy * sn, obj.y + dx * sn + dy * c};
        };

        v1 = rotatePoint(v1);
        v2 = rotatePoint(v2);
        v3 = rotatePoint(v3);
        DrawTriangle(v1, v2, v3, objectColor);
    }
}
 
void draw3DObject(const TFObject3D &obj)
{
    Color baseColor = getColor(obj.color);
 
    Color lightColor =
    {
        (unsigned char)min(255, (int)(baseColor.r * 1.3f)),
        (unsigned char)min(255, (int)(baseColor.g * 1.3f)),
        (unsigned char)min(255, (int)(baseColor.b * 1.3f)),
        255
    };
 
    Color shadowColor =
    {
        (unsigned char)(baseColor.r * 0.6f),
        (unsigned char)(baseColor.g * 0.6f),
        (unsigned char)(baseColor.b * 0.6f),
        255
    };
 
    Color darkShadow =
    {
        (unsigned char)(baseColor.r * 0.35f),
        (unsigned char)(baseColor.g * 0.35f),
        (unsigned char)(baseColor.b * 0.35f),
        255
    };
 
    Vector3 p = { obj.x, obj.y, obj.z };
    float sx = obj.scaleX;
    float sy = obj.scaleY;
    float sz = obj.scaleZ;
 
    if (obj.shape == "cube")
    {
        DrawCube(p, 2.4f * fabsf(sx), 2.4f * fabsf(sy), 2.4f * fabsf(sz), baseColor);
    }
    else if (obj.shape == "sphere")
    {
        rlPushMatrix();
        rlTranslatef(p.x, p.y, p.z);
        rlScalef(sx, sy, sz);
        DrawSphere({ 0.0f, 0.0f, 0.0f }, 1.5f, baseColor);
        rlPopMatrix();
    }
    else if (obj.shape == "cone")
    {
        rlPushMatrix();
        rlTranslatef(p.x, p.y, p.z);
        rlScalef(sx, sy, sz);
        DrawCylinder({ 0.0f, 0.0f, 0.0f }, 1.5f, 0.0f, 3.0f, 32, baseColor);
        rlPopMatrix();
    }
    else if (obj.shape == "triangle")
    {
        Vector3 top = { p.x, p.y + 1.8f * sy, p.z + 0.4f * sz };
        Vector3 v1 = { p.x - 1.3f * sx, p.y - 1.0f * sy, p.z + 1.3f * sz };
        Vector3 v2 = { p.x + 1.3f * sx, p.y - 1.0f * sy, p.z + 1.0f * sz };
        Vector3 v3 = { p.x + 0.8f * sx, p.y - 1.0f * sy, p.z - 1.3f * sz };
        Vector3 v4 = { p.x - 1.0f * sx, p.y - 1.0f * sy, p.z - 1.0f * sz };
 
        DrawTriangle3D(v1, v2, top, lightColor);
        DrawTriangle3D(v2, v3, top, baseColor);
        DrawTriangle3D(v3, v4, top, shadowColor);
        DrawTriangle3D(v4, v1, top, darkShadow);
        DrawTriangle3D(v1, v4, v3, darkShadow);
        DrawTriangle3D(v1, v3, v2, darkShadow);
    }
}
 
string normalizeInputControl(string value)
{
    value = trim(value);
 
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\'')))
    {
        value = value.substr(1, value.size() - 2);
    }
 
    for (char &c : value)
        if (c >= 'A' && c <= 'Z')
            c = (char)(c - 'A' + 'a');
 
    return value;
}
 
int keyboardKeyFromName(const string &name)
{
    static const unordered_map<string, int> keys =
    {
        // Main keyboard symbols
        {"apostrophe", KEY_APOSTROPHE}, {"'", KEY_APOSTROPHE},
        {"comma", KEY_COMMA}, {",", KEY_COMMA},
        {"minus", KEY_MINUS}, {"-", KEY_MINUS},
        {"period", KEY_PERIOD}, {".", KEY_PERIOD},
        {"slash", KEY_SLASH}, {"/", KEY_SLASH},
        {"semicolon", KEY_SEMICOLON}, {";", KEY_SEMICOLON},
        {"equal", KEY_EQUAL}, {"=", KEY_EQUAL},
        {"left_bracket", KEY_LEFT_BRACKET}, {"[", KEY_LEFT_BRACKET},
        {"right_bracket", KEY_RIGHT_BRACKET}, {"]", KEY_RIGHT_BRACKET},
        {"backslash", KEY_BACKSLASH}, {"\\", KEY_BACKSLASH},
        {"grave", KEY_GRAVE}, {"`", KEY_GRAVE},

        // Control/navigation
        {"space", KEY_SPACE}, {"enter", KEY_ENTER}, {"return", KEY_ENTER},
        {"escape", KEY_ESCAPE}, {"esc", KEY_ESCAPE},
        {"tab", KEY_TAB}, {"backspace", KEY_BACKSPACE},
        {"insert", KEY_INSERT}, {"ins", KEY_INSERT},
        {"delete", KEY_DELETE}, {"del", KEY_DELETE},
        {"home", KEY_HOME}, {"end", KEY_END},
        {"pageup", KEY_PAGE_UP}, {"page_up", KEY_PAGE_UP},
        {"pagedown", KEY_PAGE_DOWN}, {"page_down", KEY_PAGE_DOWN},
        {"up", KEY_UP}, {"down", KEY_DOWN}, {"left", KEY_LEFT}, {"right", KEY_RIGHT},
        {"capslock", KEY_CAPS_LOCK}, {"caps_lock", KEY_CAPS_LOCK},
        {"scrolllock", KEY_SCROLL_LOCK}, {"scroll_lock", KEY_SCROLL_LOCK},
        {"numlock", KEY_NUM_LOCK}, {"num_lock", KEY_NUM_LOCK},
        {"printscreen", KEY_PRINT_SCREEN}, {"print_screen", KEY_PRINT_SCREEN},
        {"pause", KEY_PAUSE},

        // Modifiers
        {"shift", KEY_LEFT_SHIFT}, {"left_shift", KEY_LEFT_SHIFT}, {"right_shift", KEY_RIGHT_SHIFT},
        {"ctrl", KEY_LEFT_CONTROL}, {"control", KEY_LEFT_CONTROL},
        {"left_ctrl", KEY_LEFT_CONTROL}, {"left_control", KEY_LEFT_CONTROL},
        {"right_ctrl", KEY_RIGHT_CONTROL}, {"right_control", KEY_RIGHT_CONTROL},
        {"alt", KEY_LEFT_ALT}, {"left_alt", KEY_LEFT_ALT}, {"right_alt", KEY_RIGHT_ALT},
        {"super", KEY_LEFT_SUPER}, {"left_super", KEY_LEFT_SUPER},
        {"right_super", KEY_RIGHT_SUPER}, {"win", KEY_LEFT_SUPER},
        {"menu", KEY_KB_MENU}, {"kb_menu", KEY_KB_MENU},

        // Function keys
        {"f1", KEY_F1}, {"f2", KEY_F2}, {"f3", KEY_F3}, {"f4", KEY_F4},
        {"f5", KEY_F5}, {"f6", KEY_F6}, {"f7", KEY_F7}, {"f8", KEY_F8},
        {"f9", KEY_F9}, {"f10", KEY_F10}, {"f11", KEY_F11}, {"f12", KEY_F12},

        // Keypad
        {"kp0", KEY_KP_0}, {"kp1", KEY_KP_1}, {"kp2", KEY_KP_2}, {"kp3", KEY_KP_3},
        {"kp4", KEY_KP_4}, {"kp5", KEY_KP_5}, {"kp6", KEY_KP_6}, {"kp7", KEY_KP_7},
        {"kp8", KEY_KP_8}, {"kp9", KEY_KP_9}, {"kp_decimal", KEY_KP_DECIMAL},
        {"kp_divide", KEY_KP_DIVIDE}, {"kp_multiply", KEY_KP_MULTIPLY},
        {"kp_subtract", KEY_KP_SUBTRACT}, {"kp_add", KEY_KP_ADD},
        {"kp_enter", KEY_KP_ENTER}, {"kp_equal", KEY_KP_EQUAL},

        // Android/system keys exposed by raylib
        {"android_back", KEY_BACK}, {"android_menu", KEY_MENU},
        {"volume_up", KEY_VOLUME_UP}, {"volume_down", KEY_VOLUME_DOWN}
    };

    auto it = keys.find(name);
    if (it != keys.end())
        return it->second;

    if (name.size() == 1)
    {
        char c = name[0];
        if (c >= 'a' && c <= 'z') return KEY_A + (c - 'a');
        if (c >= '0' && c <= '9') return KEY_ZERO + (c - '0');
    }
    return -1;
}

int mouseButtonFromName(const string &name)
{
    if (name == "mouse_left" || name == "left_mouse" || name == "left_click" || name == "mouse1") return MOUSE_BUTTON_LEFT;
    if (name == "mouse_right" || name == "right_mouse" || name == "right_click" || name == "mouse2") return MOUSE_BUTTON_RIGHT;
    if (name == "mouse_middle" || name == "middle_mouse" || name == "middle_click" || name == "mouse3") return MOUSE_BUTTON_MIDDLE;
    if (name == "mouse_side" || name == "side_mouse" || name == "mouse4") return MOUSE_BUTTON_SIDE;
    if (name == "mouse_extra" || name == "extra_mouse" || name == "mouse5") return MOUSE_BUTTON_EXTRA;
    if (name == "mouse_forward" || name == "forward_mouse" || name == "mouse6") return MOUSE_BUTTON_FORWARD;
    if (name == "mouse_back" || name == "back_mouse" || name == "mouse7") return MOUSE_BUTTON_BACK;
    return -1;
}

int gamepadButtonFromName(const string &name)
{
    if (name == "gamepad_dpad_up" || name == "gamepad_up" || name == "dpad_up") return GAMEPAD_BUTTON_LEFT_FACE_UP;
    if (name == "gamepad_dpad_right" || name == "gamepad_right" || name == "dpad_right") return GAMEPAD_BUTTON_LEFT_FACE_RIGHT;
    if (name == "gamepad_dpad_down" || name == "gamepad_down" || name == "dpad_down") return GAMEPAD_BUTTON_LEFT_FACE_DOWN;
    if (name == "gamepad_dpad_left" || name == "gamepad_left" || name == "dpad_left") return GAMEPAD_BUTTON_LEFT_FACE_LEFT;
    if (name == "gamepad_y" || name == "pad_y") return GAMEPAD_BUTTON_RIGHT_FACE_UP;
    if (name == "gamepad_b" || name == "pad_b") return GAMEPAD_BUTTON_RIGHT_FACE_RIGHT;
    if (name == "gamepad_a" || name == "pad_a") return GAMEPAD_BUTTON_RIGHT_FACE_DOWN;
    if (name == "gamepad_x" || name == "pad_x") return GAMEPAD_BUTTON_RIGHT_FACE_LEFT;
    if (name == "gamepad_l1" || name == "gamepad_lb") return GAMEPAD_BUTTON_LEFT_TRIGGER_1;
    if (name == "gamepad_l2" || name == "gamepad_lt") return GAMEPAD_BUTTON_LEFT_TRIGGER_2;
    if (name == "gamepad_r1" || name == "gamepad_rb") return GAMEPAD_BUTTON_RIGHT_TRIGGER_1;
    if (name == "gamepad_r2" || name == "gamepad_rt") return GAMEPAD_BUTTON_RIGHT_TRIGGER_2;
    if (name == "gamepad_select" || name == "gamepad_back") return GAMEPAD_BUTTON_MIDDLE_LEFT;
    if (name == "gamepad_start" || name == "gamepad_menu") return GAMEPAD_BUTTON_MIDDLE_RIGHT;
    if (name == "gamepad_middle" || name == "gamepad_home") return GAMEPAD_BUTTON_MIDDLE;
    if (name == "gamepad_l3" || name == "left_thumb" || name == "left_stick_click") return GAMEPAD_BUTTON_LEFT_THUMB;
    if (name == "gamepad_r3" || name == "right_thumb" || name == "right_stick_click") return GAMEPAD_BUTTON_RIGHT_THUMB;
    return -1;
}

int gamepadAxisFromName(const string &name)
{
    if (name == "left_x" || name == "left_stick_x" || name == "joystick_left_x" || name == "gamepad_left_x") return GAMEPAD_AXIS_LEFT_X;
    if (name == "left_y" || name == "left_stick_y" || name == "joystick_left_y" || name == "gamepad_left_y") return GAMEPAD_AXIS_LEFT_Y;
    if (name == "right_x" || name == "right_stick_x" || name == "joystick_right_x" || name == "gamepad_right_x") return GAMEPAD_AXIS_RIGHT_X;
    if (name == "right_y" || name == "right_stick_y" || name == "joystick_right_y" || name == "gamepad_right_y") return GAMEPAD_AXIS_RIGHT_Y;
    if (name == "left_trigger" || name == "left_trigger_axis" || name == "gamepad_left_trigger") return GAMEPAD_AXIS_LEFT_TRIGGER;
    if (name == "right_trigger" || name == "right_trigger_axis" || name == "gamepad_right_trigger") return GAMEPAD_AXIS_RIGHT_TRIGGER;
    return -1;
}

bool tfIsMotionControlName(const string &control)
{
    string c = normalizeInputControl(control);
    static const vector<string> names =
    {
        "mouse_up", "mouse_down", "mouse_left", "mouse_right",
        "mouse_move_up", "mouse_move_down", "mouse_move_left", "mouse_move_right",
        "mouse_front", "mouse_back", "mouse_forward", "mouse_backward",
        "wheel_up", "wheel_down",
        "joystick_up", "joystick_down", "joystick_left", "joystick_right",
        "left_joystick_up", "left_joystick_down", "left_joystick_left", "left_joystick_right",
        "right_joystick_up", "right_joystick_down", "right_joystick_left", "right_joystick_right",
        "vr_up", "vr_down", "vr_left", "vr_right", "vr_front", "vr_back",
        "vr_forward", "vr_backward"
    };
    return find(names.begin(), names.end(), c) != names.end();
}

enum class TFAxisDirection
{
    None,
    Up,
    Down,
    Left,
    Right,
    Front,
    Back,
    Any
};

TFAxisDirection tfAxisDirectionFromName(string direction)
{
    direction = normalizeInputControl(direction);
    if (direction == "up" || direction == "north") return TFAxisDirection::Up;
    if (direction == "down" || direction == "south") return TFAxisDirection::Down;
    if (direction == "left" || direction == "west") return TFAxisDirection::Left;
    if (direction == "right" || direction == "east") return TFAxisDirection::Right;
    if (direction == "front" || direction == "forward") return TFAxisDirection::Front;
    if (direction == "back" || direction == "backward") return TFAxisDirection::Back;
    if (direction == "any" || direction == "move" || direction == "movement") return TFAxisDirection::Any;
    return TFAxisDirection::None;
}

// Cursor policy: mouse input never captures or hides the native cursor.


bool tfDirectionActive(float x, float y, float threshold, TFAxisDirection direction)
{
    threshold = max(0.001f, fabsf(threshold));
    switch (direction)
    {
        case TFAxisDirection::Up:    return y < -threshold;
        case TFAxisDirection::Down:  return y > threshold;
        case TFAxisDirection::Left:  return x < -threshold;
        case TFAxisDirection::Right: return x > threshold;
        case TFAxisDirection::Front: return y < -threshold;
        case TFAxisDirection::Back:  return y > threshold;
        case TFAxisDirection::Any:   return fabsf(x) > threshold || fabsf(y) > threshold;
        default: return false;
    }
}

bool tfMouseMotionActive(const string &direction, float threshold)
{
    TFAxisDirection d = tfAxisDirectionFromName(direction);
    threshold = max(0.001f, fabsf(threshold));

    if (d == TFAxisDirection::Front || d == TFAxisDirection::Back)
    {
        float wheel = GetMouseWheelMove();
        return d == TFAxisDirection::Front ? wheel > threshold : wheel < -threshold;
    }

    Vector2 delta = GetMouseDelta();
    return tfDirectionActive(delta.x, delta.y, threshold, d);
}

bool tfGamepadMotionActive(int gamepad, const string &direction, float threshold, bool rightStick)
{
    if (gamepad < 0) gamepad = 0;
    if (!IsGamepadAvailable(gamepad)) return false;

    Vector2 axis =
    {
        GetGamepadAxisMovement(gamepad, rightStick ? GAMEPAD_AXIS_RIGHT_X : GAMEPAD_AXIS_LEFT_X),
        GetGamepadAxisMovement(gamepad, rightStick ? GAMEPAD_AXIS_RIGHT_Y : GAMEPAD_AXIS_LEFT_Y)
    };
    return tfDirectionActive(axis.x, axis.y, threshold, tfAxisDirectionFromName(direction));
}

bool tfTouchMotionActive(const string &direction, float threshold)
{
    if (GetTouchPointCount() <= 0) return false;

    static Vector2 previous = {0.0f, 0.0f};
    static bool initialized = false;
    Vector2 current = GetTouchPosition(0);
    if (!initialized)
    {
        previous = current;
        initialized = true;
        return false;
    }
    Vector2 delta = {current.x - previous.x, current.y - previous.y};
    previous = current;
    return tfDirectionActive(delta.x, delta.y, threshold, tfAxisDirectionFromName(direction));
}

// Stable VR input abstraction. It is ready for a headset/runtime to provide
// head translation and orientation without changing the TigerFlash syntax.
struct TFVRInputState
{
    float headX = 0.0f;
    float headY = 0.0f;
    float headZ = 0.0f;
    float yaw = 0.0f;
    float pitch = 0.0f;
    float roll = 0.0f;
    bool available = false;
};

TFVRInputState tfVRInput;

bool tfVRMotionActive(const string &direction, float threshold)
{
    if (!tfVRInput.available) return false;
    TFAxisDirection d = tfAxisDirectionFromName(direction);
    if (d == TFAxisDirection::Left || d == TFAxisDirection::Right)
        return tfDirectionActive(tfVRInput.yaw, 0.0f, threshold, d);
    if (d == TFAxisDirection::Up || d == TFAxisDirection::Down)
        return tfDirectionActive(0.0f, tfVRInput.pitch, threshold, d);
    if (d == TFAxisDirection::Front)
        return tfVRInput.headZ < -fabsf(threshold);
    if (d == TFAxisDirection::Back)
        return tfVRInput.headZ > fabsf(threshold);
    return tfDirectionActive(tfVRInput.headX, tfVRInput.headY, threshold, d);
}

bool inputAxisDirectionDown(const string &control)
{
    string c = normalizeInputControl(control);

    if (c == "mouse_up" || c == "mouse_move_up" || c == "mouse_movement_up") return tfMouseMotionActive("up", 0.15f);
    if (c == "mouse_down" || c == "mouse_move_down" || c == "mouse_movement_down") return tfMouseMotionActive("down", 0.15f);
    if (c == "mouse_left" || c == "mouse_move_left" || c == "mouse_movement_left") return tfMouseMotionActive("left", 0.15f);
    if (c == "mouse_right" || c == "mouse_move_right" || c == "mouse_movement_right") return tfMouseMotionActive("right", 0.15f);
    if (c == "mouse_front" || c == "mouse_forward" || c == "wheel_up") return tfMouseMotionActive("front", 0.01f);
    if (c == "mouse_back" || c == "mouse_backward" || c == "wheel_down") return tfMouseMotionActive("back", 0.01f);

    if (c == "joystick_up" || c == "left_joystick_up") return tfGamepadMotionActive(0, "up", 0.15f, false);
    if (c == "joystick_down" || c == "left_joystick_down") return tfGamepadMotionActive(0, "down", 0.15f, false);
    if (c == "joystick_left" || c == "left_joystick_left") return tfGamepadMotionActive(0, "left", 0.15f, false);
    if (c == "joystick_right" || c == "left_joystick_right") return tfGamepadMotionActive(0, "right", 0.15f, false);
    if (c == "right_joystick_up") return tfGamepadMotionActive(0, "up", 0.15f, true);
    if (c == "right_joystick_down") return tfGamepadMotionActive(0, "down", 0.15f, true);
    if (c == "right_joystick_left") return tfGamepadMotionActive(0, "left", 0.15f, true);
    if (c == "right_joystick_right") return tfGamepadMotionActive(0, "right", 0.15f, true);

    if (c == "vr_up") return tfVRMotionActive("up", 0.15f);
    if (c == "vr_down") return tfVRMotionActive("down", 0.15f);
    if (c == "vr_left") return tfVRMotionActive("left", 0.15f);
    if (c == "vr_right") return tfVRMotionActive("right", 0.15f);
    if (c == "vr_front" || c == "vr_forward") return tfVRMotionActive("front", 0.15f);
    if (c == "vr_back" || c == "vr_backward") return tfVRMotionActive("back", 0.15f);
    return false;
}

bool inputDown(const TFBinding &binding)
{
    string control = normalizeInputControl(binding.control);
    int key = keyboardKeyFromName(control);
    if (key >= 0) return IsKeyDown(key);

    int mouse = mouseButtonFromName(control);
    if (mouse >= 0) return IsMouseButtonDown(mouse);

    int gamepad = gamepadButtonFromName(control);
    if (gamepad >= 0) return IsGamepadAvailable(binding.deviceIndex) && IsGamepadButtonDown(binding.deviceIndex, gamepad);

    if (inputAxisDirectionDown(control)) return true;
    return false;
}

bool inputPressed(const TFBinding &binding)
{
    string control = normalizeInputControl(binding.control);
    int key = keyboardKeyFromName(control);
    if (key >= 0) return IsKeyPressed(key);

    int mouse = mouseButtonFromName(control);
    if (mouse >= 0) return IsMouseButtonPressed(mouse);

    int gamepad = gamepadButtonFromName(control);
    if (gamepad >= 0) return IsGamepadAvailable(binding.deviceIndex) && IsGamepadButtonPressed(binding.deviceIndex, gamepad);

    return inputAxisDirectionDown(control);
}

bool tfMotionConditionActive(const TFBinding &binding)
{
    if (!binding.hasMotionCondition) return true;

    switch (binding.motionSource)
    {
        case TFMotionSource::Mouse:
            return tfMouseMotionActive(binding.motionDirection, binding.motionThreshold);
        case TFMotionSource::Joystick:
            return tfGamepadMotionActive(binding.deviceIndex, binding.motionDirection, binding.motionThreshold, false);
        case TFMotionSource::Gamepad:
            return tfGamepadMotionActive(binding.deviceIndex, binding.motionDirection, binding.motionThreshold, true);
        case TFMotionSource::Touch:
            return tfTouchMotionActive(binding.motionDirection, binding.motionThreshold);
        case TFMotionSource::VR:
            return tfVRMotionActive(binding.motionDirection, binding.motionThreshold);
        default:
            return false;
    }
}

// ============================================================
// ROTATION / CAMERA INPUT
// key d "obj" rotate(0,1,0)
// key d "cam" rotate(0,1,0)
// key d "cam" rotate()
// ============================================================

Vector3 tfRotateVectorAroundAxis(Vector3 value, Vector3 axis, float radians)
{
    axis = tfNormalize(axis);
    float c = cosf(radians);
    float sn = sinf(radians);

    return tfAdd(
        tfAdd(tfMul(value, c), tfMul(tfCross(axis, value), sn)),
        tfMul(axis, tfDot(axis, value) * (1.0f - c))
    );
}

void tfInitializeCameraRotation(Camera3D &camera)
{
    if (tfCameraRotationInitialized)
        return;

    Vector3 forward = tfNormalize(tfSub(camera.target, camera.position));
    float horizontal = sqrtf(forward.x * forward.x + forward.z * forward.z);

    if (horizontal > 0.000001f)
    {
        tfCameraRotation.x = atan2f(forward.y, horizontal) / DEG2RAD;
        tfCameraRotation.y = atan2f(forward.x, -forward.z) / DEG2RAD;
    }
    else
    {
        tfCameraRotation.x = forward.y >= 0.0f ? 89.0f : -89.0f;
        tfCameraRotation.y = 0.0f;
    }

    tfCameraRotationInitialized = true;
}

void tfApplyCameraRotation(Camera3D &camera)
{
    float pitch = tfCameraRotation.x * DEG2RAD;
    float yaw = tfCameraRotation.y * DEG2RAD;
    float roll = tfCameraRotation.z * DEG2RAD;

    pitch = max(-89.0f * DEG2RAD, min(89.0f * DEG2RAD, pitch));

    float cp = cosf(pitch);
    float sp = sinf(pitch);
    float cy = cosf(yaw);
    float sy = sinf(yaw);

    Vector3 forward =
    {
        sy * cp,
        sp,
        -cy * cp
    };

    forward = tfNormalize(forward);

    Vector3 worldUp = {0.0f, 1.0f, 0.0f};
    Vector3 right = tfNormalize(tfCross(forward, worldUp));

    if (tfLength(right) < 0.000001f)
        right = {1.0f, 0.0f, 0.0f};

    Vector3 up = tfNormalize(tfCross(right, forward));

    if (fabsf(roll) > 0.000001f)
    {
        right = tfRotateVectorAroundAxis(right, forward, roll);
        up = tfRotateVectorAroundAxis(up, forward, roll);
    }

    camera.target = tfAdd(camera.position, forward);
    camera.up = up;
}

// Release the native mouse cursor once after the window is created.
// Mouse input is intentionally independent from cursor capture: reading
// buttons, movement, wheel, GUI clicks, or key bindings must never lock the
// cursor to the TigerFlash window.
static inline void tfReleaseMouseCursor()
{
    if (!IsWindowReady())
        return;

    ShowCursor();
    EnableCursor();
    SetMouseCursor(MOUSE_CURSOR_DEFAULT);
}

void tfUpdateCameraMouseLook(Camera3D &camera)
{
    if (!tfMouseLookEnabled || !IsWindowReady())
        return;

    Vector2 delta = GetMouseDelta();

    if (fabsf(delta.x) > 0.0001f || fabsf(delta.y) > 0.0001f)
    {
        // Mouse X rotates left/right. Mouse Y rotates up/down.
        tfCameraRotation.y += delta.x * tfMouseLookSensitivity;
        tfCameraRotation.x -= delta.y * tfMouseLookSensitivity;

        // Prevent the camera from flipping over itself.
        tfCameraRotation.x = max(-89.0f, min(89.0f, tfCameraRotation.x));

        tfApplyCameraRotation(camera);
    }
}

void tfUpdateCameraFollow(Camera3D &camera, float dt)
{
    if (!tfCameraFollowEnabled || tfCameraFollowTarget.empty())
        return;

    TFObject3D *targetObject = findObject3D(tfCameraFollowTarget);
    if (!targetObject)
        return;

    // The camera stays behind the target according to the current yaw.
    // Mouse pitch only controls where the camera looks, not its orbit height,
    // so the camera remains physically behind the player instead of tilting
    // its follow position into the ground or sky.
    float yaw = tfCameraRotation.y * DEG2RAD;
    float cy = cosf(yaw);
    float sy = sinf(yaw);

    Vector3 desiredPosition =
    {
        targetObject->x - sy * tfCameraFollowDistance,
        targetObject->y + tfCameraFollowHeight,
        targetObject->z + cy * tfCameraFollowDistance
    };

    // 0.66 means roughly two thirds of the target's movement is caught up
    // each frame at 60 FPS, while still retaining a visible smooth lag.
    float response = max(0.001f, min(1.0f,
        tfCameraFollowSmooth * dt * 60.0f));

    camera.position.x += (desiredPosition.x - camera.position.x) * response;
    camera.position.y += (desiredPosition.y - camera.position.y) * response;
    camera.position.z += (desiredPosition.z - camera.position.z) * response;

    // Keep the mouse-controlled pitch/yaw in charge of the view direction.
    tfApplyCameraRotation(camera);
}

TFGuiElement *findGuiElement(const string &name)
{
    for (TFGuiElement &el : guiElements)
        if (el.name == name)
            return &el;
    return nullptr;
}

bool guiContainsPoint(const TFGuiElement &el, Vector2 p)
{
    float w = fabsf(el.shapeW * el.scaleX);
    float h = fabsf(el.shapeH * el.scaleY);
    if (w <= 0.0f || h <= 0.0f)
        return false;
    return p.x >= el.posX && p.x <= el.posX + w &&
           p.y >= el.posY && p.y <= el.posY + h;
}

void tfUpdateGuiHover()
{
    Vector2 mouse = GetMousePosition();
    for (TFGuiElement &el : guiElements)
        el.hovered = el.isButton && guiContainsPoint(el, mouse);
}

void tfProcessGuiClickBindings()
{
    if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
        return;

    Vector2 mouse = GetMousePosition();
    for (TFGuiClickBinding &binding : guiClickBindings)
    {
        TFGuiElement *el = findGuiElement(binding.guiName);
        if (el != nullptr && el->isButton && guiContainsPoint(*el, mouse))
        {
            if (!binding.action.empty())
                executeLine(binding.action);
        }
    }
}

bool guiCollisionAABB(const TFGuiElement &a, const TFGuiElement &b)
{
    float aw = fabsf(a.shapeW * a.scaleX);
    float ah = fabsf(a.shapeH * a.scaleY);
    float bw = fabsf(b.shapeW * b.scaleX);
    float bh = fabsf(b.shapeH * b.scaleY);

    return a.posX < b.posX + bw &&
           a.posX + aw > b.posX &&
           a.posY < b.posY + bh &&
           a.posY + ah > b.posY;
}

void resolveGuiPhysicalCollisions()
{
    for (int pass = 0; pass < 3; pass++)
    {
        for (const TFGuiCollisionPair &pair : guiCollisionPairs)
        {
            TFGuiElement *a = findGuiElement(pair.first);
            TFGuiElement *b = findGuiElement(pair.second);
            if (!a || !b || !guiCollisionAABB(*a, *b))
                continue;

            float aw = fabsf(a->shapeW * a->scaleX);
            float ah = fabsf(a->shapeH * a->scaleY);
            float bw = fabsf(b->shapeW * b->scaleX);
            float bh = fabsf(b->shapeH * b->scaleY);

            float overlapX = min(a->posX + aw, b->posX + bw) - max(a->posX, b->posX);
            float overlapY = min(a->posY + ah, b->posY + bh) - max(a->posY, b->posY);
            if (overlapX <= 0.0f || overlapY <= 0.0f)
                continue;

            bool aDynamic = a->physicsEnabled;
            bool bDynamic = b->physicsEnabled;
            if (!aDynamic && !bDynamic)
                continue;

            if (overlapX < overlapY)
            {
                if (aDynamic && !bDynamic)
                    a->posX += (a->posX < b->posX ? -overlapX : overlapX);
                else if (!aDynamic && bDynamic)
                    b->posX += (b->posX < a->posX ? -overlapX : overlapX);
                else
                {
                    float half = overlapX * 0.5f;
                    a->posX += (a->posX < b->posX ? -half : half);
                    b->posX += (b->posX < a->posX ? half : -half);
                }
            }
            else
            {
                if (aDynamic && !bDynamic)
                    a->posY += (a->posY < b->posY ? -overlapY : overlapY);
                else if (!aDynamic && bDynamic)
                    b->posY += (b->posY < a->posY ? -overlapY : overlapY);
                else
                {
                    float half = overlapY * 0.5f;
                    a->posY += (a->posY < b->posY ? -half : half);
                    b->posY += (b->posY < a->posY ? half : -half);
                }
            }
        }
    }
}

void updateGuiAnimations(float dt)
{
    if (!sceneHasGui || guiAnimations.empty())
        return;

    dt = max(0.0f, min(0.10f, dt));

    for (size_t i = 0; i < guiAnimations.size(); )
    {
        TFGuiAnimation &anim = guiAnimations[i];
        TFGuiElement *gui = findGuiElement(anim.guiName);

        if (!gui || !anim.active)
        {
            guiAnimations.erase(guiAnimations.begin() + static_cast<long>(i));
            continue;
        }

        bool finished = false;

        if (anim.type == TFGuiAnimationType::Move)
        {
            float dx = anim.targetX - gui->posX;
            float dy = anim.targetY - gui->posY;
            float distance = sqrtf(dx * dx + dy * dy);
            float step = max(0.0f, anim.speed) * dt;

            if (distance <= max(0.5f, step))
            {
                gui->posX = anim.targetX;
                gui->posY = anim.targetY;
                finished = true;
            }
            else if (distance > 0.00001f && step > 0.0f)
            {
                float factor = step / distance;
                gui->posX += dx * factor;
                gui->posY += dy * factor;
            }
        }
        else if (anim.type == TFGuiAnimationType::Rotate)
        {
            float delta = anim.targetRotation - gui->rotation;
            float step = max(0.0f, anim.speed) * dt;

            if (fabsf(delta) <= max(0.25f, step))
            {
                gui->rotation = anim.targetRotation;
                finished = true;
            }
            else if (step > 0.0f)
            {
                gui->rotation += (delta > 0.0f ? step : -step);
            }
        }
        else if (anim.type == TFGuiAnimationType::Scale)
        {
            float dx = anim.targetScaleX - gui->scaleX;
            float dy = anim.targetScaleY - gui->scaleY;
            float distance = sqrtf(dx * dx + dy * dy);
            float step = max(0.0f, anim.speed) * dt;

            if (distance <= max(0.0025f, step))
            {
                gui->scaleX = max(0.01f, anim.targetScaleX);
                gui->scaleY = max(0.01f, anim.targetScaleY);
                finished = true;
            }
            else if (distance > 0.000001f && step > 0.0f)
            {
                float factor = step / distance;
                gui->scaleX = max(0.01f, gui->scaleX + dx * factor);
                gui->scaleY = max(0.01f, gui->scaleY + dy * factor);
            }
        }

        if (finished)
        {
            if (!anim.loop)
            {
                guiAnimations.erase(guiAnimations.begin() + static_cast<long>(i));
                continue;
            }

            // Ping-pong loop: reverse between the two endpoints instead of
            // teleporting back to the beginning. This produces a continuous
            // animation with no visible jump.
            if (anim.type == TFGuiAnimationType::Move)
            {
                float oldTargetX = anim.targetX;
                float oldTargetY = anim.targetY;
                anim.targetX = anim.startX;
                anim.targetY = anim.startY;
                anim.startX = oldTargetX;
                anim.startY = oldTargetY;
            }
            else if (anim.type == TFGuiAnimationType::Rotate)
            {
                float oldTarget = anim.targetRotation;
                anim.targetRotation = anim.startRotation;
                anim.startRotation = oldTarget;
            }
            else
            {
                float oldTargetX = anim.targetScaleX;
                float oldTargetY = anim.targetScaleY;
                anim.targetScaleX = anim.startScaleX;
                anim.targetScaleY = anim.startScaleY;
                anim.startScaleX = oldTargetX;
                anim.startScaleY = oldTargetY;
            }
        }

        ++i;
    }
}

void updateGuiPhysics(float dt)
{
    if (!sceneHasGui)
        return;

    for (TFGuiElement &el : guiElements)
    {
        if (!el.physicsEnabled)
            continue;

        el.grounded = false;
        if (el.gravity != 0.0f)
        {
            el.velocityY += el.gravity * dt;
            el.posY += el.velocityY * dt;
        }

        if (el.softBody)
        {
            float softness = max(0.0f, min(10.0f, el.softness));
            float fluid = softness / 10.0f;
            float speed = fabsf(el.velocityY);
            float targetSquash = min(0.55f, speed * (0.004f + 0.012f * fluid));
            el.softSquash += (targetSquash - el.softSquash) * min(1.0f, dt * (8.0f - 4.0f * fluid));
            float targetSpread = el.softSquash * (0.25f + 0.75f * fluid);
            el.softSpread += (targetSpread - el.softSpread) * min(1.0f, dt * (10.0f - 5.0f * fluid));
        }
    }

    resolveGuiPhysicalCollisions();

    for (TFGuiElement &el : guiElements)
    {
        if (!el.physicsEnabled)
            continue;

        float h = fabsf(el.shapeH * el.scaleY);
        if (el.posY + h >= GetScreenHeight())
        {
            el.posY = max(0.0f, (float)GetScreenHeight() - h);
            if (el.velocityY > 0.0f)
                el.velocityY = 0.0f;
            el.grounded = true;
        }
        if (el.posY < 0.0f)
        {
            el.posY = 0.0f;
            if (el.velocityY < 0.0f)
                el.velocityY = 0.0f;
        }
    }
}

bool hasGuiCollisionPair(const string &a, const string &b)
{
    for (const TFGuiCollisionPair &p : guiCollisionPairs)
        if ((p.first == a && p.second == b) || (p.first == b && p.second == a))
            return true;
    return false;
}

void addGuiCollision(const string &a, const string &b)
{
    if (!findGuiElement(a) || !findGuiElement(b))
    {
        error("GUI collision requires two existing GUI names.");
        return;
    }
    if (a == b)
    {
        error("A GUI collision needs two different names.");
        return;
    }
    if (!hasGuiCollisionPair(a, b))
        guiCollisionPairs.push_back({a, b});
}

bool guiCollisionSensor(const string &a, const string &b, bool reportErrors)
{
    TFGuiElement *ga = findGuiElement(a);
    TFGuiElement *gb = findGuiElement(b);
    if (!ga || !gb)
    {
        if (reportErrors) error("GUI collision requires two existing GUI names.");
        return false;
    }
    return guiCollisionAABB(*ga, *gb);
}

void applyInputBindings(float scale, Camera3D &camera)
{
    tfInitializeCameraRotation(camera);

    for (const TFBinding &binding : inputBindings)
    {
        bool active = inputDown(binding);
        if (active && binding.hasMotionCondition)
            active = tfMotionConditionActive(binding);

        if (!active)
        {
            // Ordinary commands remain one-shot below, exactly as before.
            if (binding.isMovement || binding.isRotation || binding.hasMotionCondition)
                continue;
        }

        if (binding.isMovement)
        {
            if (binding.objectName == "cam")
            {
                camera.position.x += binding.dx * scale;
                camera.position.y += binding.dy * scale;
                camera.position.z += binding.dz * scale;
                tfApplyCameraRotation(camera);
                continue;
            }

            if (sceneHas2D)
            {
                for (TFObject2D &obj : objects2D)
                {
                    if (obj.name == binding.objectName)
                    {
                        obj.x += binding.dx * scale;
                        obj.y += binding.dy * scale;
                        break;
                    }
                }
            }

            if (sceneHas3D)
            {
                for (TFObject3D &obj : objects3D)
                {
                    if (obj.name == binding.objectName)
                    {
                        auto gravityIt = gravityBodies.find(obj.name);
                        if (gravityIt != gravityBodies.end())
                        {
                            tfMoveGravityBodyKinematic(
                                obj,
                                {
                                    binding.dx * scale,
                                    binding.dy * scale,
                                    binding.dz * scale
                                }
                            );
                        }
                        else
                        {
                            obj.x += binding.dx * scale;
                            obj.y += binding.dy * scale;
                            obj.z += binding.dz * scale;
                        }
                        break;
                    }
                }
            }

            if (sceneHasGui)
            {
                TFGuiElement *gui = findGuiElement(binding.objectName);
                if (gui != nullptr)
                {
                    gui->posX += binding.dx * scale;
                    gui->posY += binding.dy * scale;
                }
            }
        }
        else if (binding.isRotation)
        {
            if (binding.objectName == "cam")
            {
                tfCameraRotation.x += binding.rx * scale;
                tfCameraRotation.y += binding.ry * scale;
                tfCameraRotation.z += binding.rz * scale;
                tfApplyCameraRotation(camera);
                continue;
            }

            if (sceneHas2D)
            {
                for (TFObject2D &obj : objects2D)
                {
                    if (obj.name == binding.objectName)
                    {
                        obj.rotation += binding.ry * scale;
                        break;
                    }
                }
            }

            if (sceneHas3D)
            {
                for (TFObject3D &obj : objects3D)
                {
                    if (obj.name == binding.objectName)
                    {
                        obj.rotationX += binding.rx * scale;
                        obj.rotationY += binding.ry * scale;
                        obj.rotationZ += binding.rz * scale;
                        break;
                    }
                }
            }

            if (sceneHasGui)
            {
                TFGuiElement *gui = findGuiElement(binding.objectName);
                if (gui != nullptr)
                    gui->rotation += binding.rz * scale;
            }
        }
        else
        {
            // Comandos continuam sendo disparados uma vez por pressionamento.
            if (inputPressed(binding) && !binding.action.empty())
                executeLine(binding.action);
        }
    }
}

TFObject3D *findGravityObject3D(const string &name)
{
    return findObject3D(name);
}

bool gravityObjectGrounded(const string &name)
{
    auto recorded = gravityGrounded.find(name);
    if (recorded != gravityGrounded.end() && recorded->second)
        return true;

    TFObject3D *obj = findObject3D(name);
    if (!obj)
        return false;

    TFAABB3D objBox;
    if (!tfGetAABB3D(*obj, objBox))
        return false;

    float centerObjY = (objBox.minY + objBox.maxY) * 0.5f;

    for (const TFCollisionPair &pair : physicalCollisionPairs)
    {
        if (pair.first != name && pair.second != name)
            continue;

        const string &otherName = (pair.first == name) ? pair.second : pair.first;
        TFObject3D *other = findObject3D(otherName);
        if (!other)
            continue;

        TFAABB3D otherBox;
        if (!tfGetAABB3D(*other, otherBox))
            continue;

        float overlapX = min(objBox.maxX, otherBox.maxX) - max(objBox.minX, otherBox.minX);
        float overlapZ = min(objBox.maxZ, otherBox.maxZ) - max(objBox.minZ, otherBox.minZ);
        if (overlapX <= 0.0f || overlapZ <= 0.0f)
            continue;

        float centerOtherY = (otherBox.minY + otherBox.maxY) * 0.5f;
        if (centerOtherY >= centerObjY)
            continue;

        float verticalGap = objBox.minY - otherBox.maxY;
        // A small contact envelope prevents numerical/frame-rate jitter from
        // losing grounded state when the body is only a few millimetres above
        // a support. Penetration is also accepted because the snap solver
        // immediately repairs it.
        if (verticalGap <= 0.08f && verticalGap >= -0.08f)
            return true;
    }

    return false;
}

bool tfPointSegmentBroadphaseMayTouch(const TFObject3D &moving,
                                      const TFObject3D &other,
                                      Vector3 start,
                                      Vector3 end)
{
    Vector3 segment = tfSub(end, start);
    float lengthSq = tfDot(segment, segment);
    float t = 0.0f;
    if (lengthSq > 0.0000001f)
    {
        Vector3 toOther = tfSub(
            {other.x, other.y, other.z},
            start
        );
        t = tfDot(toOther, segment) / lengthSq;
        t = max(0.0f, min(1.0f, t));
    }

    Vector3 closest = tfAdd(start, tfMul(segment, t));
    Vector3 delta = tfSub(
        closest,
        {other.x, other.y, other.z}
    );

    float radius = tfBroadphaseRadius3D(moving) +
                   tfBroadphaseRadius3D(other) + 0.08f;
    return tfDot(delta, delta) <= radius * radius;
}

bool tfRecoverGravityBodySweptCollision(TFObject3D &obj,
                                        TFGravityBody &body,
                                        Vector3 start,
                                        Vector3 end)
{
    // Collision movement must slide along surfaces. A wall should block the
    // component that enters the wall, but it must NOT cancel the whole player
    // movement. The same rule applies to floors, ceilings, slopes and corners.
    Vector3 current = start;
    Vector3 remaining = tfSub(end, start);
    bool hadContact = false;

    // ------------------------------------------------------------
    // 1) Repair any existing overlap at the starting point.
    // ------------------------------------------------------------
    // This is important when the previous frame ended exactly on a surface.
    // We use the real GJK + EPA shape solver, not an AABB, to get the normal.
    for (int pass = 0; pass < 4; ++pass)
    {
        bool repaired = false;

        for (const TFCollisionPair &pair : physicalCollisionPairs)
        {
            if (pair.first != obj.name && pair.second != obj.name)
                continue;

            const string &otherName =
                pair.first == obj.name ? pair.second : pair.first;
            TFObject3D *other = findObject3D(otherName);
            if (!other || other->name == obj.name)
                continue;

            obj.x = current.x;
            obj.y = current.y;
            obj.z = current.z;

            if (!collision3D(obj, *other))
                continue;

            Vector3 normal;
            float depth = 0.0f;
            if (!tfComputeShapeMTV3D(obj, *other, normal, depth))
                continue;

            if (tfKinematicInputCollisionMode)
            {
                if (normal.y > 0.55f)
                    normal = {0.0f, 1.0f, 0.0f};
                else if (normal.y < -0.55f)
                    normal = {0.0f, -1.0f, 0.0f};
                else
                {
                    normal.y = 0.0f;
                    normal = tfNormalize(normal);
                }
            }

            const float epsilon = 0.00005f;
            current = tfAdd(
                current,
                tfMul(normal, depth + epsilon)
            );

            // Keep tangent motion and discard only motion INTO the surface.
            float inwardTravel = tfDot(remaining, normal);
            if (inwardTravel < 0.0f)
            {
                remaining = tfSub(
                    remaining,
                    tfMul(normal, inwardTravel)
                );
            }

            float inwardVelocity = tfDot(body.velocity, normal);
            if (inwardVelocity < 0.0f)
            {
                body.velocity = tfSub(
                    body.velocity,
                    tfMul(normal, inwardVelocity)
                );
            }

            body.contactCollider = other->name;
            body.contactNormal = normal;
            body.contactPoint = current;
            body.contactAge = 0.0f;
            body.contactStrength = max(body.contactStrength, 0.35f);
            gravityGrounded[obj.name] = normal.y > 0.55f;

            hadContact = true;
            repaired = true;
        }

        if (!repaired)
            break;
    }

    // ------------------------------------------------------------
    // 2) Sweep the remaining movement and slide on the first hit.
    // ------------------------------------------------------------
    for (int slideIteration = 0; slideIteration < 4; ++slideIteration)
    {
        if (tfLength(remaining) < 0.00001f)
            break;

        Vector3 segmentStart = current;
        Vector3 segmentEnd = tfAdd(current, remaining);
        float distance = tfLength(remaining);

        float earliestHit = 2.0f;
        TFObject3D *hitObject = nullptr;

        for (const TFCollisionPair &pair : physicalCollisionPairs)
        {
            if (pair.first != obj.name && pair.second != obj.name)
                continue;

            const string &otherName =
                pair.first == obj.name ? pair.second : pair.first;
            TFObject3D *other = findObject3D(otherName);
            if (!other || other->name == obj.name)
                continue;

            if (!tfPointSegmentBroadphaseMayTouch(obj, *other,
                                                  segmentStart,
                                                  segmentEnd))
                continue;

            int samples = static_cast<int>(ceilf(distance / 0.08f));
            samples = max(4, min(96, samples));

            float previousT = 0.0f;

            for (int sample = 1; sample <= samples; ++sample)
            {
                float t = static_cast<float>(sample) /
                          static_cast<float>(samples);
                Vector3 p = tfAdd(segmentStart,
                                  tfMul(remaining, t));
                obj.x = p.x;
                obj.y = p.y;
                obj.z = p.z;

                if (collision3D(obj, *other))
                {
                    float low = previousT;
                    float high = t;

                    for (int iteration = 0; iteration < 14; ++iteration)
                    {
                        float mid = (low + high) * 0.5f;
                        Vector3 m = tfAdd(segmentStart,
                                          tfMul(remaining, mid));
                        obj.x = m.x;
                        obj.y = m.y;
                        obj.z = m.z;

                        if (collision3D(obj, *other))
                            high = mid;
                        else
                            low = mid;
                    }

                    if (high < earliestHit)
                    {
                        earliestHit = high;
                        hitObject = other;
                    }
                    break;
                }

                previousT = t;
            }
        }

        if (!hitObject || earliestHit > 1.0f)
        {
            current = segmentEnd;
            break;
        }

        // Move to the first safe point before penetration.
        float safeT = max(0.0f, earliestHit - 0.0005f);
        Vector3 safe = tfAdd(
            segmentStart,
            tfMul(remaining, safeT)
        );

        // Probe just beyond the contact so EPA can determine the surface
        // normal. The visible object is restored to the safe position below.
        Vector3 probe = tfAdd(
            segmentStart,
            tfMul(remaining,
                  min(1.0f, earliestHit + 0.0015f))
        );
        obj.x = probe.x;
        obj.y = probe.y;
        obj.z = probe.z;

        Vector3 normal = tfNormalize(tfSub(
            {obj.x, obj.y, obj.z},
            {hitObject->x, hitObject->y, hitObject->z}
        ));

        Vector3 mtvNormal;
        float mtvDepth = 0.0f;
        if (collision3D(obj, *hitObject) &&
            tfComputeShapeMTV3D(obj, *hitObject,
                                mtvNormal, mtvDepth))
        {
            normal = mtvNormal;
        }

        // Kinematic player movement needs a stable floor/wall normal. EPA can
        // return a tiny diagonal component at a perfectly flat contact, which
        // would incorrectly eat part of the horizontal input every frame.
        if (tfKinematicInputCollisionMode)
        {
            if (normal.y > 0.55f)
                normal = {0.0f, 1.0f, 0.0f};
            else if (normal.y < -0.55f)
                normal = {0.0f, -1.0f, 0.0f};
            else
            {
                normal.y = 0.0f;
                normal = tfNormalize(normal);
            }
        }

        current = safe;
        obj.x = current.x;
        obj.y = current.y;
        obj.z = current.z;
        hadContact = true;

        body.contactCollider = hitObject->name;
        body.contactNormal = normal;
        body.contactPoint = current;
        body.contactAge = 0.0f;
        body.contactStrength = max(body.contactStrength, 0.35f);
        gravityGrounded[obj.name] = normal.y > 0.55f;

        // Preserve all tangent motion. Only the inward component is removed.
        // Example: pressing right while touching a floor keeps X movement;
        // pressing right into a wall removes X but keeps Z/Y movement.
        float inwardTravel = tfDot(remaining, normal);
        if (inwardTravel < 0.0f)
        {
            remaining = tfSub(
                remaining,
                tfMul(normal, inwardTravel)
            );
        }

        float inwardVelocity = tfDot(body.velocity, normal);
        if (inwardVelocity < 0.0f)
        {
            body.velocity = tfSub(
                body.velocity,
                tfMul(normal, inwardVelocity)
            );
        }

        // The next iteration continues exactly from the safe contact point.
        // This lets a diagonal movement resolve against a corner in several
        // small slide steps instead of freezing at the first surface.
    }

    obj.x = current.x;
    obj.y = current.y;
    obj.z = current.z;
    return hadContact;
}

// ============================================================
// COLLISION-SAFE KINEMATIC INPUT MOVEMENT
// ============================================================
// Input movement is deliberately solved independently from inertial gravity.
// The old swept path could detect the surface that the body was already
// touching at t=0 and then reduce the whole remaining vector repeatedly. That
// created the visible "slows down until it stops" bug while a key was held.
//
// This solver first repairs a genuine penetration, then tests the requested
// endpoint. If an axis is blocked, only that axis is reduced by binary search.
// X/Z movement along a floor therefore remains constant-speed, while movement
// into a wall is clipped only on the blocked axis. It is also considerably
// cheaper than doing dozens of samples plus a binary search for every input
// frame.
bool tfKinematicPositionCollides(TFObject3D &obj, Vector3 position)
{
    const float oldX = obj.x;
    const float oldY = obj.y;
    const float oldZ = obj.z;

    obj.x = position.x;
    obj.y = position.y;
    obj.z = position.z;

    bool hit = false;
    for (const TFCollisionPair &pair : physicalCollisionPairs)
    {
        if (pair.first != obj.name && pair.second != obj.name)
            continue;

        const string &otherName =
            pair.first == obj.name ? pair.second : pair.first;
        TFObject3D *other = findObject3D(otherName);
        if (!other || other->name == obj.name)
            continue;

        if (collision3D(obj, *other))
        {
            // Touching a platform is valid contact. Only true penetration
            // blocks kinematic movement, so the player can walk while standing.
            Vector3 normal;
            float depth = 0.0f;
            if (tfComputeShapeMTV3D(obj, *other, normal, depth) &&
                depth > 0.00020f)
            {
                hit = true;
                break;
            }
        }
    }

    obj.x = oldX;
    obj.y = oldY;
    obj.z = oldZ;
    return hit;
}

void tfRepairKinematicStartingContact(TFObject3D &obj)
{
    for (int pass = 0; pass < 3; ++pass)
    {
        bool repaired = false;

        for (const TFCollisionPair &pair : physicalCollisionPairs)
        {
            if (pair.first != obj.name && pair.second != obj.name)
                continue;

            const string &otherName =
                pair.first == obj.name ? pair.second : pair.first;
            TFObject3D *other = findObject3D(otherName);
            if (!other || other->name == obj.name)
                continue;

            if (!collision3D(obj, *other))
                continue;

            Vector3 normal;
            float depth = 0.0f;
            if (!tfComputeShapeMTV3D(obj, *other, normal, depth))
                continue;

            // Tangent contact is already stable. Repair only actual penetration;
            // otherwise pressing X/Z while standing on a platform can get stuck.
            if (depth <= 0.00020f)
                continue;

            // Do not let a tiny numerical overlap remain active. A stable
            // epsilon is enough; it is far below normal TigerFlash movement.
            const float epsilon = 0.00008f;
            obj.x += normal.x * (depth + epsilon);
            obj.y += normal.y * (depth + epsilon);
            obj.z += normal.z * (depth + epsilon);
            repaired = true;

            if (normal.y > 0.55f)
                gravityGrounded[obj.name] = true;
        }

        if (!repaired)
            break;
    }
}

float tfKinematicMoveAxis(TFObject3D &obj, Vector3 axis, float amount)
{
    if (fabsf(amount) < 0.000001f)
        return 0.0f;

    Vector3 start = {obj.x, obj.y, obj.z};
    Vector3 target = tfAdd(start, tfMul(axis, amount));

    // The body may be exactly touching a surface from the previous frame.
    // Repair only true penetration, never tangent contact.
    tfRepairKinematicStartingContact(obj);
    start = {obj.x, obj.y, obj.z};
    target = tfAdd(start, tfMul(axis, amount));

    // Most input frames take this extremely cheap path.
    if (!tfKinematicPositionCollides(obj, target))
    {
        obj.x = target.x;
        obj.y = target.y;
        obj.z = target.z;
        return amount;
    }

    // A real barrier exists. Find the largest safe fraction of THIS axis.
    float low = 0.0f;
    float high = 1.0f;
    const int iterations = 14;

    for (int i = 0; i < iterations; ++i)
    {
        float mid = (low + high) * 0.5f;
        Vector3 candidate = tfAdd(start, tfMul(axis, amount * mid));

        if (tfKinematicPositionCollides(obj, candidate))
            high = mid;
        else
            low = mid;
    }

    const float safeAmount = amount * max(0.0f, low - 0.00005f);
    Vector3 safe = tfAdd(start, tfMul(axis, safeAmount));
    obj.x = safe.x;
    obj.y = safe.y;
    obj.z = safe.z;
    return safeAmount;
}

void tfMoveGravityBodyKinematic(TFObject3D &obj, Vector3 delta)
{
    auto it = gravityBodies.find(obj.name);
    if (it == gravityBodies.end())
        return;

    TFGravityBody &body = it->second;
    body.inputMovementActiveThisFrame = true;

    // Startup collision hold has absolute priority. Input is ignored until the
    // collision startup window has completed, exactly like gravity is.
    if (body.startupHoldRemaining > 0.0f)
        return;

    if (tfLength(delta) < 0.000001f)
        return;

    tfKinematicInputCollisionMode = true;

    // Solve horizontal axes independently. This prevents a floor contact from
    // interfering with X/Z movement and naturally produces wall sliding.
    const float movedX = tfKinematicMoveAxis(
        obj, {1.0f, 0.0f, 0.0f}, delta.x
    );
    const float movedZ = tfKinematicMoveAxis(
        obj, {0.0f, 0.0f, 1.0f}, delta.z
    );

    // Vertical input is solved separately, so jumping/falling input can still
    // work while horizontal motion remains independent of the floor contact.
    const float movedY = tfKinematicMoveAxis(
        obj, {0.0f, 1.0f, 0.0f}, delta.y
    );

    tfKinematicInputCollisionMode = false;

    // The physical position is authoritative. These velocity components are
    // only used by the soft-body deformation model; gravity itself will sample
    // the measured frame displacement again on the next stage.
    body.velocity.x = movedX * 60.0f;
    body.velocity.z = movedZ * 60.0f;
    if (fabsf(movedY) > 0.000001f)
        body.velocity.y = movedY * 60.0f;

    body.inputMovementResolvedThisFrame = true;
}

// ============================================================
// CONTACT-FIRST GRAVITY
// ============================================================
// The previous solver waited for a true GJK intersection before treating a
// body as supported. That is too late for resting contact: two shapes may be
// visually touching while their support surfaces are still separated by a
// tiny floating-point gap. This helper performs a short, shape-accurate
// downward look-ahead and finds the first real intersection before gravity is
// integrated. The body is then placed at the contact height and its downward
// velocity is blocked for that substep.
//
// AABB data is used only as a cheap horizontal candidate filter. The final
// contact decision is still made by the real 3D shape collision + MTV solver.
void tfPrepareGravitySupportContacts(float step)
{
    for (auto &entry : gravityBodies)
    {
        const string &name = entry.first;
        TFGravityBody &body = entry.second;
        TFObject3D *obj = findGravityObject3D(name);
        if (!obj)
            continue;

        // Only a body moving downward (or already resting) needs a floor
        // look-ahead. Upward motion must remain free so jumps can leave the
        // surface normally.
        if (body.velocity.y > 0.0f)
            continue;

        bool supported = false;

        for (const TFCollisionPair &pair : physicalCollisionPairs)
        {
            if (pair.first != name && pair.second != name)
                continue;

            const string &otherName =
                (pair.first == name) ? pair.second : pair.first;
            TFObject3D *other = findObject3D(otherName);
            if (!other || other->name == name)
                continue;

            // The candidate collider must be below the body's center. This
            // prevents nearby walls/ceilings from becoming a floor contact.
            if (other->y >= obj->y + 0.0001f)
                continue;

            TFAABB3D objBox;
            TFAABB3D otherBox;
            if (!tfGetAABB3D(*obj, objBox) || !tfGetAABB3D(*other, otherBox))
                continue;

            const float overlapX =
                min(objBox.maxX, otherBox.maxX) -
                max(objBox.minX, otherBox.minX);
            const float overlapZ =
                min(objBox.maxZ, otherBox.maxZ) -
                max(objBox.minZ, otherBox.minZ);

            if (overlapX <= 0.0f || overlapZ <= 0.0f)
                continue;

            const float currentY = obj->y;

            // First handle a genuine overlap immediately. This is important
            // when the body was placed directly on, or a microscopic amount
            // inside, a support by the script or another solver.
            if (collision3D(*obj, *other))
            {
                Vector3 normal;
                float depth = 0.0f;

                if (tfComputeShapeMTV3D(*obj, *other, normal, depth) &&
                    normal.y > 0.55f)
                {
                    obj->x += normal.x * (depth + 0.00001f);
                    obj->y += normal.y * (depth + 0.00001f);
                    obj->z += normal.z * (depth + 0.00001f);

                    body.velocity.y = 0.0f;
                    body.contactCollider = other->name;
                    body.contactNormal = normal;
                    body.contactPoint = {obj->x, obj->y, obj->z};
                    body.contactAge = 0.0f;
                    body.contactStrength = max(body.contactStrength, 0.35f);
                    gravityGrounded[name] = true;
                    supported = true;
                }
                else
                {
                    // It is already intersecting this collider, but the
                    // contact normal is not a floor. Do not reinterpret a wall
                    // or an underside as a support just because a downward
                    // probe would remain inside the same collider.
                    obj->y = currentY;
                    continue;
                }
            }

            if (supported)
                break;

            // Look only a short distance ahead. The distance follows the
            // body's current speed, so a fast downward-moving object gets a
            // larger early-contact envelope without changing normal physics.
            const float lookAhead = max(
                0.06f,
                min(0.35f, fabsf(body.velocity.y) * step + 0.06f)
            );

            obj->y = currentY - lookAhead;
            bool probeCollides = collision3D(*obj, *other);

            if (!probeCollides)
            {
                obj->y = currentY;
                continue;
            }

            // We now know there is a real shape collision below the current
            // position. Bisect the interval to locate the first contact with
            // sub-pixel precision, without requiring penetration as the final
            // resting position.
            float high = currentY;          // known non-penetrating side
            float low = currentY - lookAhead; // known colliding side

            for (int iteration = 0; iteration < 16; ++iteration)
            {
                const float mid = (high + low) * 0.5f;
                obj->y = mid;

                if (collision3D(*obj, *other))
                    low = mid;
                else
                    high = mid;
            }

            obj->y = high;

            Vector3 normal;
            float depth = 0.0f;
            bool validSupport = false;

            // Probe a hair into the support to obtain an exact MTV normal.
            const float probeDepth = 0.00002f;
            obj->y = high - probeDepth;

            if (collision3D(*obj, *other) &&
                tfComputeShapeMTV3D(*obj, *other, normal, depth) &&
                normal.y > 0.55f)
            {
                validSupport = true;
            }

            // Restore the safe side before committing the contact.
            obj->y = high;

            if (validSupport)
            {
                body.velocity.y = 0.0f;
                body.contactCollider = other->name;
                body.contactNormal = normal;
                body.contactPoint = {obj->x, obj->y, obj->z};
                body.contactAge = 0.0f;
                body.contactStrength = max(body.contactStrength, 0.35f);
                gravityGrounded[name] = true;
                supported = true;
            }

            if (supported)
                break;

            obj->y = currentY;
        }
    }
}

// ============================================================
// STARTUP COLLISION LOCK
// ============================================================
// A gravity body must not start moving until the collision system has had
// time to inspect its declared colliders. The body is held at the exact spawn
// position for a few hundred milliseconds while real collision tests and the
// normal physical resolver run. Input/script movement cannot accidentally
// bypass this lock because the position is restored before gravity integrates.
// Once the timer reaches zero, the body is released and normal gravity starts
// from the stabilized position.
bool tfUpdateGravityStartupCollisionLock(float dt)
{
    bool anyLocked = false;

    for (auto &entry : gravityBodies)
    {
        TFGravityBody &body = entry.second;
        TFObject3D *obj = findGravityObject3D(entry.first);
        if (!obj || body.startupHoldRemaining <= 0.0f)
            continue;

        anyLocked = true;

        // Restore the exact held pose. This prevents keyboard movement,
        // follow systems, or another pre-physics command from moving a body
        // before the startup collision window has finished.
        obj->x = body.startupHoldPosition.x;
        obj->y = body.startupHoldPosition.y;
        obj->z = body.startupHoldPosition.z;
        body.velocity = {0.0f, 0.0f, 0.0f};

        body.startupHoldRemaining =
            max(0.0f, body.startupHoldRemaining - dt);
    }

    if (!anyLocked)
        return false;

    // Several complete passes make the startup state deterministic. These
    // are the same physical collision routines used during normal gameplay.
    const int passes = max(8, tfPerf.collisionPasses + 4);
    for (int pass = 0; pass < passes; ++pass)
        resolvePhysicalCollisions();

    // If the resolver had to separate an initially overlapping body, keep the
    // corrected position as the held pose so the body does not snap back into
    // the collider on the next startup frame.
    for (auto &entry : gravityBodies)
    {
        TFGravityBody &body = entry.second;
        if (body.startupHoldRemaining <= 0.0f)
            continue;

        TFObject3D *obj = findGravityObject3D(entry.first);
        if (!obj)
            continue;

        body.startupHoldPosition = {obj->x, obj->y, obj->z};
        body.previousPosition = body.startupHoldPosition;
        body.sweepStartY = obj->y;
        body.sweepStartInitialized = true;
        body.initialized = true;
    }

    return true;
}

void updateGravityBodies(float dt)
{
    if (gravityBodies.empty()) return;

    // Startup lock: no gravity integration is allowed until collision has been
    // given the dedicated startup window to detect and settle the scene.
    if (tfUpdateGravityStartupCollisionLock(dt))
        return;

    // 240 Hz fixed-ish substeps provide continuous-feeling contact without
    // changing the public TigerFlash gravity syntax.
    const float maxStep = 1.0f / static_cast<float>(max(60, tfPerf.targetPhysicsHz));
    int steps = max(1, (int)ceilf(dt / maxStep));
    steps = min(steps, 4);
    float step = dt / (float)steps;

    for (auto &entry : gravityBodies)
    {
        TFGravityBody &body = entry.second;
        TFObject3D *obj = findGravityObject3D(entry.first);
        if (!obj) continue;

        if (!body.initialized)
        {
            body.previousPosition = {obj->x, obj->y, obj->z};
            body.sweepStartY = obj->y;
            body.sweepStartInitialized = true;
            body.initialized = true;
            body.velocity.x = 0.0f;
            body.velocity.z = 0.0f;
        }

        float invDt = 1.0f / max(0.0001f, dt);
        float measuredX = (obj->x - body.previousPosition.x) * invDt;
        float measuredZ = (obj->z - body.previousPosition.z) * invDt;
        float friction = expf(-TF_GROUND_FRICTION_RATE * dt);

        if (body.inputMovementActiveThisFrame)
        {
            // Keyboard/gamepad movement is kinematic input, not an inertial
            // force. Never let ground friction make a held movement command
            // decay toward zero.
            if (fabsf(measuredX) > 0.0001f) body.velocity.x = measuredX;
            if (fabsf(measuredZ) > 0.0001f) body.velocity.z = measuredZ;
        }
        else
        {
            if (fabsf(measuredX) > 0.0001f) body.velocity.x = measuredX;
            else body.velocity.x *= friction;
            if (fabsf(measuredZ) > 0.0001f) body.velocity.z = measuredZ;
            else body.velocity.z *= friction;
        }
    }

    for (int sub = 0; sub < steps; sub++)
    {
        gravityGrounded.clear();

        // Detect support BEFORE integrating gravity. This is the critical
        // ordering fix: resting contact no longer has to wait for penetration.
        tfPrepareGravitySupportContacts(step);

        for (auto &entry : gravityBodies)
        {
            TFGravityBody &body = entry.second;
            TFObject3D *obj = findGravityObject3D(entry.first);
            if (!obj) continue;
            body.sweepStartY = obj->y;
            body.sweepStartInitialized = true;
        }

        // Semi-implicit Euler with quadratic drag.
        for (auto &entry : gravityBodies)
        {
            const string &name = entry.first;
            TFGravityBody &body = entry.second;
            TFObject3D *obj = findGravityObject3D(name);
            if (!obj) continue;

            if (gravityGrounded.count(name) != 0 &&
                gravityGrounded[name] &&
                body.velocity.y <= 0.0f)
            {
                // The contact-first stage already positioned this body at the
                // support surface. Do not apply one more downward gravity step.
                body.velocity.y = 0.0f;
                continue;
            }

            float speed = fabsf(body.velocity.y);
            float dragScale = body.gravity / 9.80665f;
            float drag = TF_AIR_DRAG_COEFFICIENT * dragScale * speed * speed;
            if (body.velocity.y > 0.0f) drag = -drag;
            body.velocity.y += (-body.gravity + drag) * step;
            obj->y += body.velocity.y * step;
        }

        // Full 3D continuous sweep after the gravity integration. This is the
        // primary anti-tunneling barrier for rigid and soft gravity bodies.
        for (auto &entry : gravityBodies)
        {
            const string &name = entry.first;
            TFGravityBody &body = entry.second;
            TFObject3D *obj = findGravityObject3D(name);
            if (!obj) continue;

            Vector3 sweepStart = {obj->x, body.sweepStartY, obj->z};
            Vector3 sweepEnd = {obj->x, obj->y, obj->z};
            tfRecoverGravityBodySweptCollision(*obj, body, sweepStart, sweepEnd);
        }

        // Continuous downward sweep against declared collision supports.
        // This catches a surface crossing even if the final position has
        // already moved beyond it during a high-speed frame.
        for (auto &entry : gravityBodies)
        {
            const string &name = entry.first;
            TFGravityBody &body = entry.second;
            TFObject3D *obj = findGravityObject3D(name);
            if (!obj || body.velocity.y > 0.0f) continue;

            TFAABB3D endBox;
            if (!tfGetAABB3D(*obj, endBox)) continue;

            float h = endBox.maxY - endBox.minY;
            float startBottom = body.sweepStartY - h * 0.5f;
            float endBottom = endBox.minY;
            float bestTop = -numeric_limits<float>::infinity();
            bool hit = false;

            for (const TFCollisionPair &pair : physicalCollisionPairs)
            {
                if (pair.first != name && pair.second != name) continue;
                const string &otherName = (pair.first == name) ? pair.second : pair.first;
                TFObject3D *other = findObject3D(otherName);
                if (!other) continue;

                TFAABB3D otherBox;
                if (!tfGetAABB3D(*other, otherBox)) continue;

                float ox = min(endBox.maxX, otherBox.maxX) - max(endBox.minX, otherBox.minX);
                float oz = min(endBox.maxZ, otherBox.maxZ) - max(endBox.minZ, otherBox.minZ);
                if (ox <= 0.0f || oz <= 0.0f) continue;

                bool crossed = startBottom >= otherBox.maxY - 0.001f &&
                               endBottom <= otherBox.maxY + 0.003f;
                bool resting = fabsf(endBottom - otherBox.maxY) <= 0.045f;

                if ((crossed || resting) && otherBox.maxY > bestTop)
                {
                    bestTop = otherBox.maxY;
                    hit = true;
                }
            }

            if (hit)
            {
                obj->y += bestTop - endBox.minY;
                body.velocity.y = 0.0f;
                gravityGrounded[name] = true;
            }
        }

        resolvePhysicalCollisions();

        // Restore every support after all pair resolutions. This is the final
        // barrier against a chain of contacts moving a body below the floor.
        for (auto &entry : gravityBodies)
        {
            const string &name = entry.first;
            TFGravityBody &body = entry.second;
            TFObject3D *obj = findGravityObject3D(name);
            if (!obj) continue;

            bool grounded = gravityObjectGrounded(name);
            if (grounded) body.velocity.y = 0.0f;

            if (body.type == TFBodyType::Soft)
            {
                updateSoftBodyPhysicsForces(body, step, grounded, body.softness);
                if (grounded) body.contactAge = min(body.contactAge, 0.05f);
                decaySoftBodyContact(body, step);
            }

            tfSnapGravityBodyToSupport(name);
            grounded = gravityObjectGrounded(name);
            if (grounded) body.velocity.y = 0.0f;
        }
    }

    for (auto &entry : gravityBodies)
    {
        TFGravityBody &body = entry.second;
        TFObject3D *obj = findGravityObject3D(entry.first);
        if (!obj) continue;
        body.previousPosition = {obj->x, obj->y, obj->z};
    }
}
// Resolves a GUI image path without changing the TigerFlash syntax.
// We deliberately try several common filesystem locations because TigerFlash
// receives its script through stdin and therefore does not automatically know
// the directory of the .tf source file.
string tfResolveGuiImagePath(const string &inputPath)
{
    if (inputPath.empty())
        return inputPath;

    string path = inputPath;

    // Support the usual Linux home shorthand: ~/Pictures/logo.png
    if (path.rfind("~/", 0) == 0)
    {
        const char *home = getenv("HOME");
        if (home != nullptr)
            path = string(home) + "/" + path.substr(2);
    }

    // Absolute Linux paths are used exactly as written.  This means the
    // TigerFlash script can load an image from any existing directory,
    // including paths containing spaces, for example:
    // gui image "/home/ierrie/Imagens/Papel de parede.png" ...
    if (!path.empty() && path.front() == '/')
        return path;

    vector<string> candidates;
    candidates.push_back(path);

    // The executable's directory is important when Geany starts TigerFlash
    // with a different working directory.
    char exeBuffer[4096] = {0};
    ssize_t exeLength = readlink("/proc/self/exe", exeBuffer,
                                 sizeof(exeBuffer) - 1);

    if (exeLength > 0)
    {
        exeBuffer[exeLength] = '\0';
        string executablePath(exeBuffer);
        size_t slash = executablePath.find_last_of('/');

        if (slash != string::npos)
            candidates.push_back(executablePath.substr(0, slash) + "/" + path);
    }

    // Also try the current working directory explicitly.
    char cwdBuffer[4096] = {0};
    if (getcwd(cwdBuffer, sizeof(cwdBuffer)) != nullptr)
        candidates.push_back(string(cwdBuffer) + "/" + path);

    // Remove duplicate candidate paths before testing them.
    for (const string &candidate : candidates)
    {
        if (FileExists(candidate.c_str()))
            return candidate;
    }

    // Keep the original value for the final diagnostic.
    return path;
}

// Loads a GUI image in two stages. LoadTexture() is the normal fast path.
// If it fails, LoadImage()/LoadTextureFromImage() gives raylib another decode
// path and also makes it possible to verify whether the file itself decoded.
bool tfLoadGuiImageTexture(const string &resolvedPath, Texture2D &texture)
{
    texture = {0};

    if (!FileExists(resolvedPath.c_str()))
        return false;

    texture = LoadTexture(resolvedPath.c_str());
    if (texture.id != 0)
        return true;

    Image image = LoadImage(resolvedPath.c_str());
    if (image.data == nullptr || image.width <= 0 || image.height <= 0)
    {
        if (image.data != nullptr)
            UnloadImage(image);
        return false;
    }

    texture = LoadTextureFromImage(image);
    UnloadImage(image);

    return texture.id != 0;
}

// Draws every "gui text(...)"/"gui image ..." element as a screen-space
// overlay, sorted so a higher position(...) z draws on top of a lower one.
// Image textures are loaded lazily (only once InitWindow has run) and
// cached on the element; text is re-resolved every call so it can show a
// live value, the same way growl() prints the current value each time.
void tfDrawGuiElements()
{
    if (guiElements.empty())
        return;

    vector<size_t> order(guiElements.size());
    for (size_t i = 0; i < order.size(); i++)
        order[i] = i;

    sort(order.begin(), order.end(), [](size_t a, size_t b)
    {
        return guiElements[a].posZ < guiElements[b].posZ;
    });

    tfUpdateGuiHover();

    for (size_t idx : order)
    {
        TFGuiElement &el = guiElements[idx];

        if (el.isButton)
        {
            float w = el.shapeW > 0.0f ? fabsf(el.shapeW * el.scaleX) : 160.0f;
            float h = el.shapeH > 0.0f ? fabsf(el.shapeH * el.scaleY) : 50.0f;
            Rectangle rect = {el.posX, el.posY, w, h};
            Color background = el.hovered ? Color{80, 150, 220, 255} : Color{55, 105, 165, 255};
            DrawRectangleRec(rect, background);
            DrawRectangleLines((int)rect.x, (int)rect.y, (int)rect.width, (int)rect.height, WHITE);

            string text = tfResolveDisplayText(el.content);
            int fontSize = (int)(el.shapeH > 8.0f ? el.shapeH * 0.42f : 20.0f);
            if (fontSize < 8) fontSize = 8;
            int tw = MeasureText(text.c_str(), fontSize);
            int tx = (int)(el.posX + (w - tw) * 0.5f);
            int ty = (int)(el.posY + (h - fontSize) * 0.5f);
            DrawText(text.c_str(), tx, ty, fontSize, WHITE);
            continue;
        }

        if (el.isImage)
        {
            if (!el.textureLoadAttempted)
            {
                el.textureLoadAttempted = true;
                string resolvedPath = tfResolveGuiImagePath(el.content);
                el.textureLoadFailed = !tfLoadGuiImageTexture(resolvedPath, el.texture);
                if (el.textureLoadFailed)
                    error("gui image: could not open image file. Path requested: \"" + el.content + "\"");
            }

            if (!el.textureLoadFailed && el.texture.id != 0)
            {
                float w = el.shapeW > 0.0f ? fabsf(el.shapeW * el.scaleX) : (float)el.texture.width;
                float h = el.shapeH > 0.0f ? fabsf(el.shapeH * el.scaleY) : (float)el.texture.height;
                Rectangle src = {0.0f, 0.0f, (float)el.texture.width, (float)el.texture.height};
                Rectangle dst = {el.posX + w * 0.5f, el.posY + h * 0.5f, w, h};
                SetTextureFilter(el.texture, TEXTURE_FILTER_BILINEAR);
                DrawTexturePro(el.texture, src, dst, {w * 0.5f, h * 0.5f}, el.rotation, WHITE);
            }
        }
        else
        {
            string text = tfResolveDisplayText(el.content);
            int fontSize = (int)(el.shapeH > 0.0f ? el.shapeH : 20.0f);
            if (fontSize < 4) fontSize = 4;

            if (el.shapeW > 0.0f)
            {
                int textWidth = MeasureText(text.c_str(), fontSize);
                if (textWidth > (int)fabsf(el.shapeW * el.scaleX) && textWidth > 0)
                {
                    float scale = fabsf(el.shapeW * el.scaleX) / (float)textWidth;
                    int scaledSize = (int)(fontSize * scale);
                    fontSize = scaledSize < 4 ? 4 : scaledSize;
                }
            }

            float w = max(1.0f, fabsf((float)MeasureText(text.c_str(), fontSize) * el.scaleX));
            float h = max(1.0f, (float)fontSize * el.scaleY);
            DrawTextPro(GetFontDefault(), text.c_str(),
                        {el.posX, el.posY}, {0.0f, 0.0f},
                        el.rotation, fontSize, 1.0f, WHITE);
            (void)w;
            (void)h;
        }
    }
}


// GPU textures must be freed while the GL context (the window) still
// exists, so this runs right before CloseWindow(), not in the main()
// scene reset (which only clears CPU-side state).
void tfUnloadGuiTextures()
{
    for (TFGuiElement &el : guiElements)
    {
        if (el.isImage && el.textureLoadAttempted && !el.textureLoadFailed && el.texture.id != 0)
            UnloadTexture(el.texture);
    }
}


// ============================================================
// COLLISION PRE-WARM / PHYSICS STARTUP LOCK
// ============================================================
// Collision geometry is generated lazily by collision3D(). If gravity is
// allowed to run on the very first frame, the first GJK/shape build can take
// long enough for a falling object to visibly move before its first collision
// test completes. This function forces every declared collision pair through
// the real collision path BEFORE the simulation loop starts, then resolves
// any initial overlap and snapshots the exact starting positions for gravity.
// No fake AABBs are used as the final collision test.
void tfPrewarmCollisionSystem()
{
    // Warm all declared 3D collision pairs several times. Repeating the pass
    // makes lazy shape/material/deformation caches settle before the first
    // visible frame and avoids a one-time spike being charged to gameplay.
    for (int pass = 0; pass < 2; ++pass)
    {
        for (const TFCollisionPair &pair : physicalCollisionPairs)
        {
            if (objectDimension(pair.first) != 3 || objectDimension(pair.second) != 3)
                continue;

            TFObject3D *a = findObject3D(pair.first);
            TFObject3D *b = findObject3D(pair.second);
            if (!a || !b)
                continue;

            (void)collision3D(*a, *b);
        }
    }

    // Warm 2D polygon generation and overlap tests too, so a 2D scene does not
    // have the same first-frame hitch.
    for (const TFCollisionPair &pair : physicalCollisionPairs)
    {
        if (objectDimension(pair.first) != 2 || objectDimension(pair.second) != 2)
            continue;

        TFObject2D *a = findObject2D(pair.first);
        TFObject2D *b = findObject2D(pair.second);
        if (!a || !b)
            continue;

        (void)collision2D(*a, *b);
    }

    // Resolve any overlap that already exists at script startup BEFORE the
    // first gravity integration. This is particularly important for a player
    // that intentionally spawns inside/against a platform or floor.
    const int settlePasses = max(6, tfPerf.collisionPasses + 2);
    for (int pass = 0; pass < settlePasses; ++pass)
        resolvePhysicalCollisions();

    // Gravity starts from the now-stable, collision-resolved position. No
    // initial falling step can happen before its support geometry is ready.
    for (auto &entry : gravityBodies)
    {
        TFGravityBody &body = entry.second;
        TFObject3D *obj = findGravityObject3D(entry.first);
        if (!obj)
            continue;

        body.previousPosition = {obj->x, obj->y, obj->z};
        body.sweepStartY = obj->y;
        body.sweepStartInitialized = true;
        body.initialized = true;
        body.velocity = {0.0f, 0.0f, 0.0f};
        body.startupHoldPosition = {obj->x, obj->y, obj->z};
        body.startupHoldRemaining = TF_COLLISION_STARTUP_HOLD_SECONDS;
    }

    gravityGrounded.clear();
    for (auto &entry : gravityBodies)
    {
        const string &name = entry.first;
        TFGravityBody &body = entry.second;
        body.contactCollider.clear();
        body.contactAge = 100.0f;

        // Seed grounded state from the exact collision solver/support query.
        gravityGrounded[name] = gravityObjectGrounded(name);
        if (gravityGrounded[name])
            body.velocity.y = 0.0f;
    }
}

void runGraphicsWindow()
{
    if (!sceneHas2D && !sceneHas3D && !sceneHasGui)
        return;
 
    const int screenWidth = 900;
    const int screenHeight = 700;
 
    // raylib logs an INFO line to the console for routine internal
    // work (loading/unloading a mesh, textures, shaders, etc). Since
    // every bend/move that changes a soft body's shape can rebuild
    // its mesh, that INFO channel alone would spam a message per
    // change. LOG_WARNING keeps real warnings/errors but drops the
    // routine INFO chatter. Raylib must be told this BEFORE InitWindow.
    SetTraceLogLevel(LOG_WARNING);

    string windowTitle = "TigerFlash";
    if (!tfCurrentScriptPath.empty())
    {
        filesystem::path scriptPath(tfCurrentScriptPath);
        if (!scriptPath.filename().empty())
            windowTitle = scriptPath.filename().string();
    }
    else
    {
        windowTitle = sceneHas3D ? "TigerFlash 3D" : "TigerFlash 2D";
    }

    InitWindow(screenWidth, screenHeight, windowTitle.c_str());
    SetTargetFPS(60);

    // TigerFlash never captures or hides the mouse cursor.
    // Release the native cursor once after window creation. From this point
    // onward, mouse input is read normally without repeatedly changing the
    // OS cursor mode. Keyboard bindings remain completely independent.
    tfReleaseMouseCursor();
 
    tfCamera = { 0 };
    tfCamera.position = { 0.0f, 1.5f, 6.5f };
    tfCamera.target = { 0.0f, 0.0f, 0.0f };
    tfCamera.up = { 0.0f, 1.0f, 0.0f };
    tfCamera.fovy = 45.0f;
    tfCamera.projection = CAMERA_PERSPECTIVE;
    if (!tfCameraRotationInitialized)
        tfCameraRotation = {-12.0f, 0.0f, 0.0f};
    tfCameraRotationInitialized = true;
    tfApplyCameraRotation(tfCamera);

    // Collision must be completely ready before the first simulation frame.
    // This eliminates startup tunneling/falling while lazy collision geometry
    // is being built.
    tfPrewarmCollisionSystem();
 
    while (!WindowShouldClose())
    {
        float dt = min(GetFrameTime(), 0.05f);
        tfUpdatePerformanceMonitor(dt);
        tfPerf.trianglesDrawn = 0;

        for (auto &entry : gravityBodies)
        {
            entry.second.inputMovementResolvedThisFrame = false;
            entry.second.inputMovementActiveThisFrame = false;
        }

        tfUpdateCameraMouseLook(tfCamera);
        applyInputBindings(dt * 60.0f, tfCamera);

        // Script/animation movement that did not pass through the kinematic
        // input path is still protected by the frame-to-frame swept solver.
        for (auto &entry : gravityBodies)
        {
            TFObject3D *obj = findGravityObject3D(entry.first);
            if (!obj || !entry.second.initialized ||
                entry.second.inputMovementResolvedThisFrame)
                continue;

            Vector3 start = entry.second.previousPosition;
            Vector3 end = {obj->x, obj->y, obj->z};
            tfRecoverGravityBodySweptCollision(*obj, entry.second, start, end);
        }

        updateGuiAnimations(dt);
        updateGravityBodies(dt);
        updateGuiPhysics(dt);
        tfProcessGuiClickBindings();
        tfUpdateCameraFollow(tfCamera, dt);
        resolvePhysicalCollisions();

        // `if`/`repeat` run cooperatively here, after the physics/collision
        // step, so collision sensors see the actual current frame state.
        tfUpdateBackgroundScript(dt);

        // Re-evaluate which objects are currently glued into a pile
        // and ease their shared polygon budget toward it.
        tfUpdateClusterBudgets3D(dt);
        tfUpdateClusterBudgets2D(dt);
        tfUpdateAdaptiveOrganicLOD(dt);

        BeginDrawing();
        ClearBackground(LIGHTGRAY);
 
        if (sceneHas3D)
        {
            BeginMode3D(tfCamera);

            // Opaque geometry first so the optical pass can use the finished
            // depth buffer and blend crystals/prisms/diamonds over the scene.
            for (const TFObject3D &obj : objects3D)
            {
                if (!tfIsOpticalShape(obj))
                    tfDrawOptimized3DObject(obj, tfCamera, screenWidth, screenHeight);
            }

            for (const TFObject3D &obj : objects3D)
            {
                if (tfIsOpticalShape(obj))
                    tfDrawOptimized3DObject(obj, tfCamera, screenWidth, screenHeight);
            }

            EndMode3D();
        }
        else
        {
            for (const TFObject2D &obj : objects2D)
            {
                if (shapeDeformations.count(deformationKey2D(obj.name)) != 0)
                    draw2DDeformedShape(obj);
                else
                    draw2DObject(obj);
            }
        }
 
        tfDrawGuiElements();

        EndDrawing();

        // Save the completed frame position. The next frame compares against
        // this transform, so a stationary object produces no blur trail.
        for (TFObject3D &obj : objects3D)
        {
            obj.motionBlurPreviousPosition = {obj.x, obj.y, obj.z};
            obj.motionBlurPreviousInitialized = true;
        }

        // Cache maintenance happens AFTER the frame so a mesh that was just
        // rendered always has the newest visibility timestamp.
        tfMaintainOptimized3DMeshCache();
    }
 
    tfClearOptimized3DMeshes();
    tfUnloadGuiTextures();

    if (optimized3DMaterialReady)
    {
        UnloadMaterial(optimized3DMaterial);
        optimized3DMaterial = { 0 };
        optimized3DMaterialReady = false;
    }

    if (tfSurfaceShaderReady)
    {
        UnloadShader(tfSurfaceShader);
        tfSurfaceShader = { 0 };
        tfSurfaceShaderReady = false;
        tfSurfaceLocColor = -1;
    }

    tfUnloadOpticalMaterial();

    CloseWindow();
}
 
// ============================================================
// CONDITIONS
 
// ============================================================
 
string stripQuotes(string value)
{
    value = trim(value);
 
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\'')))
    {
        return value.substr(1, value.size() - 2);
    }
 
    return value;
}
 
int indentationLevel(const string &line)
{
    int level = 0;
 
    for (char c : line)
    {
        if (c == ' ')
            level++;
        else if (c == '\t')
            level += 4;
        else
            break;
    }
 
    return level;
}
 
bool isNullValue(string value)
{
    value = trim(value);
    return value == "null";
}
 
bool valueExists(string name)
{
    return variables.count(name) != 0;
}
 
string conditionValue(string value)
{
    value = trim(value);
 
    if (variables.count(value))
        return variables[value];
 
    if (value == "null")
        return "null";
 
    return stripQuotes(value);
}
 
// Fast non-throwing numeric conversion used by conditions and repeat.
// The old path relied on std::stod + exceptions for every failed conversion.
// Conditions often compare text, so that made a normal string test unnecessarily
// expensive. This implementation accepts the same ordinary decimal/scientific
// numeric forms handled by std::stod closely enough for TigerFlash values.
bool tryNumber(string value, double &number)
{
    value = trim(value);

    if (value.empty())
        return false;

    const char *begin = value.c_str();
    const char *p = begin;
    const char *end = begin + value.size();

    bool negative = false;
    if (p < end && (*p == '+' || *p == '-'))
    {
        negative = (*p == '-');
        ++p;
    }

    bool hasDigits = false;
    double result = 0.0;

    while (p < end && isdigit(static_cast<unsigned char>(*p)))
    {
        hasDigits = true;
        result = result * 10.0 + static_cast<double>(*p - '0');
        ++p;
    }

    if (p < end && *p == '.')
    {
        ++p;
        double place = 0.1;
        while (p < end && isdigit(static_cast<unsigned char>(*p)))
        {
            hasDigits = true;
            result += static_cast<double>(*p - '0') * place;
            place *= 0.1;
            ++p;
        }
    }

    if (!hasDigits)
        return false;

    if (p < end && (*p == 'e' || *p == 'E'))
    {
        ++p;
        bool expNegative = false;
        if (p < end && (*p == '+' || *p == '-'))
        {
            expNegative = (*p == '-');
            ++p;
        }

        if (p >= end || !isdigit(static_cast<unsigned char>(*p)))
            return false;

        int exponent = 0;
        while (p < end && isdigit(static_cast<unsigned char>(*p)))
        {
            int digit = *p - '0';
            if (exponent < 10000)
                exponent = min(10000, exponent * 10 + digit);
            ++p;
        }

        const double scale = pow(10.0, expNegative ? -exponent : exponent);
        result *= scale;
    }

    if (p != end)
        return false;

    number = negative ? -result : result;
    return true;
}

// Find the first comparison operator that is OUTSIDE quoted strings.
// This prevents expressions such as `if "a=b" = "a=b"` from being split at
// the `=` that belongs to the text itself. Longer operators are checked first
// so `>=`, `<=`, `=/` and `!=` stay unambiguous.
size_t findConditionOperator(const string &condition, string &op)
{
    static const char *operators[] =
    {
        ">=", "<=", "=/", "!=", "=", ">", "<"
    };

    char quote = 0;

    for (size_t i = 0; i < condition.size(); ++i)
    {
        char c = condition[i];

        if (quote != 0)
        {
            if (c == quote)
                quote = 0;
            continue;
        }

        if (c == '"' || c == '\'')
        {
            quote = c;
            continue;
        }

        for (const char *candidate : operators)
        {
            size_t length = strlen(candidate);
            if (i + length <= condition.size() &&
                condition.compare(i, length, candidate) == 0)
            {
                op = candidate;
                return i;
            }
        }
    }

    op.clear();
    return string::npos;
}

bool evaluateCondition(string condition)
{
    condition = trim(condition);
    condition = resolveListAccess(condition);

    // gravity inside if = grounded/contact sensor. This is evaluated each
    // frame by the secondary-flow scheduler, so `if gravity "player"` means
    // the body is currently supported rather than checking only at startup.
    if (condition.rfind("gravity ", 0) == 0)
    {
        string rest = trim(condition.substr(8));
        bool guiTarget = false;
        if (rest.rfind("gui ", 0) == 0)
        {
            guiTarget = true;
            rest = trim(rest.substr(4));
        }

        if (rest.size() < 2 || rest.front() != '"' || rest.back() != '"')
            return false;

        string name = rest.substr(1, rest.size() - 2);
        if (guiTarget)
        {
            TFGuiElement *gui = findGuiElement(name);
            return gui != nullptr && gui->physicsEnabled && gui->grounded;
        }

        auto grounded = gravityGrounded.find(name);
        return grounded != gravityGrounded.end() && grounded->second;
    }

    // colision inside if = sensor mode (true/false only).
    if (condition.rfind("colision ", 0) == 0)
    {
        string first;
        string second;

        if (!parseCollisionSyntax(condition, first, second))
        {
            error("Invalid collision sensor syntax. Use: colision \"name1\" to \"name2\"");
            return false;
        }

        // Collision sensors are polled continuously. A deleted/temporarily inactive
        // object is a normal "not colliding" state, not a runtime error.
        return sensorCollision(first, second, false);
    }

    {
        bool handled = false;
        bool result = tfRunLibraryConditionHooks(condition, handled);
        if (handled)
            return result;
    }

    if (condition == "true")
        return true;

    if (condition == "false")
        return false;

    string op;
    size_t pos = findConditionOperator(condition, op);
    if (pos != string::npos)
    {
        string left = trim(condition.substr(0, pos));
        string right = trim(condition.substr(pos + op.size()));

        if (left.empty() || right.empty())
            return false;

        string leftValue = conditionValue(left);
        string rightValue = conditionValue(right);

        if (op == "=")
        {
            if (leftValue == rightValue)
                return true;

            double a, b;
            if (tryNumber(leftValue, a) && tryNumber(rightValue, b))
                return a == b;

            return false;
        }

        if (op == "=/" || op == "!=")
        {
            if (leftValue == rightValue)
                return false;

            double a, b;
            if (tryNumber(leftValue, a) && tryNumber(rightValue, b))
                return a != b;

            return true;
        }

        double a, b;

        if (!tryNumber(leftValue, a) || !tryNumber(rightValue, b))
            return false;

        if (op == ">")
            return a > b;

        if (op == "<")
            return a < b;

        if (op == ">=")
            return a >= b;

        if (op == "<=")
            return a <= b;
    }

    if (variables.count(condition))
    {
        const string &value = variables.find(condition)->second;
        return !value.empty() && value != "null";
    }

    return !isNullValue(condition);
}
 
// ============================================================
// REPEAT / VARIABLE ADJUSTMENT
// ============================================================
 
bool parseRepeatAdjustment(const string &line, double &amount, string &name)
{
    if (line.rfind("repeat ", 0) != 0)
        return false;
 
    string content = trim(line.substr(7));
    size_t equal = content.find('=');
 
    if (equal == string::npos)
        return false;
 
    string change = trim(content.substr(0, equal));
    name = trim(content.substr(equal + 1));
 
    if (!validName(name))
        return false;
 
    if (change.size() < 2 ||
        (change[0] != '+' && change[0] != '-'))
        return false;
 
    string number = trim(change.substr(1));
    if (!isIntegerNumber(number))
        return false;
 
    amount = stod(number);
 
    if (change[0] == '-')
        amount = -amount;
 
    return true;
}
 
bool executeRepeatAdjustment(const string &line)
{
    double amount = 0;
    string name;
 
    if (!parseRepeatAdjustment(line, amount, name))
        return false;
 
    if (!variables.count(name))
    {
        error("Variable not found: " + name);
        return true;
    }
 
    double current = 0;
    if (!tryNumber(variables[name], current))
    {
        error("Repeat adjustment only works with numeric variables.");
        return true;
    }
 
    variables[name] = formatNumber(current + amount);
    return true;
}
 
// ============================================================
// TIGERFLASH SYNTAX COLOR
// ============================================================
 
string colorTigerFlashCommand(const string &line)
{
    string text = line;
 
    const vector<string> commands =
    {
        "growl", "count", "alert", "see", "if", "else",
        "repeat", "say", "3d", "2d", "random",
        "list", "add", "remove", "shape", "colision", "rotate",
        "key", "gravity", "camera", "moviment", "mouse", "click", "joystick",
        "gamepad", "touch", "vr", "axis", "threshold",
        "function", "return", "break", "continue", "memory", "import"
    };
 
    string result;
    size_t i = 0;
 
    while (i < text.size())
    {
        if (isspace((unsigned char)text[i]))
        {
            result += text[i];
            i++;
            continue;
        }
 
        if (isalpha((unsigned char)text[i]) || text[i] == '_')
        {
            string word;
 
            while (i < text.size() &&
                   (isalnum((unsigned char)text[i]) || text[i] == '_'))
            {
                word += text[i];
                i++;
            }
 
            bool isCommand = false;
 
            for (const string &command : commands)
            {
                if (word == command)
                {
                    isCommand = true;
                    break;
                }
            }
 
            if (isCommand)
                result += "\033[1;34m" + word + "\033[0m";
            else
                result += word;
 
            continue;
        }
 
        result += text[i];
        i++;
    }
 
    return result;
}
 
void printTigerFlashCommand(const string &line)
{
    cout << colorTigerFlashCommand(line) << endl;
}
 
// ============================================================
// WORLD POSITION
// ============================================================
// `at(x,y)` / `at(x,y,z)` is the normal TigerFlash object-position syntax.
// `position(...)` remains accepted as an alias for compatibility.
// ============================================================

// ============================================================
// SHAPE / SCALE
// ============================================================
 
bool parseShapeScale(string text, vector<double> &values)
{
    text = trim(text);
 
    if (text.empty() || text.front() != '(' || text.back() != ')')
        return false;
 
    string inside = text.substr(1, text.size() - 2);
    vector<string> parts = splitMovement(inside);
 
    if (parts.size() != 2 && parts.size() != 3)
        return false;
 
    values.clear();
 
    for (string part : parts)
    {
        try
        {
            CountParser parser(replaceVariables(trim(part)));
            values.push_back(parser.parse());
        }
        catch (...)
        {
            return false;
        }
    }
 
    return true;
}
 
bool extractInlineShape(string &text, vector<double> &values)
{
    size_t pos = text.find("shape(");
 
    if (pos == string::npos)
    {
        pos = text.find("shape (");
    }
 
    if (pos == string::npos)
        return false;
 
    size_t open = text.find('(', pos + 5);
    if (open == string::npos)
        return false;
 
    size_t close = text.find(')', open);
    if (close == string::npos)
        return false;
 
    if (!parseShapeScale(text.substr(open, close - open + 1), values))
        return false;
 
    text = trim(text.substr(0, pos) + " " + text.substr(close + 1));
    return true;
}
 
bool containsInlinePositionToken(const string &text)
{
    const vector<string> keywords = {"position", "at"};

    for (const string &keyword : keywords)
    {
        size_t p = text.find(keyword + "(");
        if (p == string::npos)
            p = text.find(keyword + " (");

        if (p == string::npos)
            continue;

        bool leftOk = (p == 0 || isspace((unsigned char)text[p - 1]));
        if (leftOk)
            return true;
    }

    return false;
}

bool extractInlinePosition(string &text, vector<double> &values)
{
    // `at(...)` is the short TigerFlash world-position syntax.
    // `position(...)` remains supported as a backwards-compatible alias.
    const vector<string> keywords = {"position", "at"};

    size_t pos = string::npos;
    size_t keywordLen = 0;

    for (const string &keyword : keywords)
    {
        size_t p = text.find(keyword + "(");
        if (p == string::npos)
            p = text.find(keyword + " (");

        if (p == string::npos)
            continue;

        bool leftOk = (p == 0 || isspace((unsigned char)text[p - 1]));
        if (!leftOk)
            continue;

        if (pos == string::npos || p < pos)
        {
            pos = p;
            keywordLen = keyword.size();
        }
    }

    if (pos == string::npos)
        return false;

    size_t open = text.find('(', pos + keywordLen);
    if (open == string::npos)
        return false;

    size_t close = text.find(')', open);
    if (close == string::npos)
        return false;

    string inside = text.substr(open + 1, close - open - 1);
    vector<string> parts = splitMovement(inside);

    if (parts.size() != 2 && parts.size() != 3)
        return false;

    values.clear();
    for (const string &part : parts)
    {
        try
        {
            CountParser parser(replaceVariables(trim(part)));
            values.push_back(parser.parse());
        }
        catch (...)
        {
            return false;
        }
    }

    text = trim(text.substr(0, pos) + " " + text.substr(close + 1));
    return true;
}

bool extractObjectName(string &text, string &objectName)
{
    size_t pos = text.find("as");
 
    while (pos != string::npos)
    {
        bool leftOk = (pos == 0 || isspace((unsigned char)text[pos - 1]));
        bool rightOk = (pos + 2 >= text.size() || isspace((unsigned char)text[pos + 2]));
 
        if (leftOk && rightOk)
        {
            string rest = trim(text.substr(pos + 2));
            if (!rest.empty() && rest.front() == '"')
            {
                size_t close = rest.find('"', 1);
                if (close == string::npos)
                    return false;
 
                objectName = rest.substr(1, close - 1);
                text = trim(text.substr(0, pos) + " " + rest.substr(close + 1));
                return true;
            }
        }
 
        pos = text.find("as", pos + 2);
    }
 
    return true;
}
 
bool extractInlineColor(string &text, string &color)
{
    text = trim(text);
    if (text.empty())
        return false;
 
    vector<string> words;
    string current;
    for (char c : text)
    {
        if (isspace((unsigned char)c))
        {
            if (!current.empty())
            {
                words.push_back(current);
                current.clear();
            }
        }
        else
        {
            current += c;
        }
    }
    if (!current.empty())
        words.push_back(current);
 
    for (const string &word : words)
    {
        if (word == "random" || validColor(word))
        {
            color = word;
            return true;
        }
    }
 
    return false;
}
 
bool applyShapeToObject(const string &objectName, const vector<double> &values)
{
    for (TFObject2D &obj : objects2D)
    {
        if (obj.name != objectName)
            continue;
 
        if (values.size() != 2)
        {
            error("2D shape requires (x,y).");
            return true;
        }
 
        obj.scaleX = (float)values[0];
        obj.scaleY = (float)values[1];
        return true;
    }
 
    for (TFObject3D &obj : objects3D)
    {
        if (obj.name != objectName)
            continue;
 
        if (values.size() != 3)
        {
            error("3D shape requires (x,y,z).");
            return true;
        }
 
        obj.scaleX = (float)values[0];
        obj.scaleY = (float)values[1];
        obj.scaleZ = (float)values[2];
        return true;
    }
 
    error("Object not found: " + objectName);
    return true;
}
 

// ============================================================
// SHAPE PART DEFORMATION PARSER
// Syntax: shape p[f/b/l/r/t/d]N (x,y,z)(x,y,z)...(x,y,z)
// It applies to the most recently created 2D or 3D object.
// ============================================================

bool parseShapePartDeformation(const string &line,
                                TFDeformation &deformation)
{
    string text = trim(line);

    if (text.rfind("shape p[", 0) != 0)
        return false;

    size_t openBracket = text.find('[', 7);
    size_t closeBracket = text.find(']', openBracket + 1);

    if (openBracket == string::npos ||
        closeBracket == string::npos ||
        closeBracket <= openBracket + 1)
    {
        error("Invalid shape deformation. Use: shape p[f]3 (x,y,z)(x,y,z)(x,y,z)");
        return true;
    }

    string sideText =
        trim(
            text.substr(
                openBracket + 1,
                closeBracket - openBracket - 1
            )
        );

    if (sideText.size() != 1 ||
        string("fblrtd").find(sideText[0]) == string::npos)
    {
        error("Invalid shape deformation direction. Use f, b, l, r, t or d.");
        return true;
    }

    size_t numberStart = closeBracket + 1;

    while (numberStart < text.size() &&
           isspace((unsigned char)text[numberStart]))
    {
        numberStart++;
    }

    size_t numberEnd = numberStart;

    while (numberEnd < text.size() &&
           isdigit((unsigned char)text[numberEnd]))
    {
        numberEnd++;
    }

    if (numberEnd == numberStart)
    {
        error("Shape deformation requires the number of segments.");
        return true;
    }

    int segments = 0;

    try
    {
        segments = stoi(
            text.substr(
                numberStart,
                numberEnd - numberStart
            )
        );
    }
    catch (...)
    {
        error("Invalid deformation segment count.");
        return true;
    }

    if (segments < 1 || segments > 512)
    {
        error("Shape deformation segments must be from 1 to 512.");
        return true;
    }

    size_t pos = numberEnd;
    vector<Vector3> factors;

    for (int i = 0; i < segments; i++)
    {
        while (pos < text.size() &&
               isspace((unsigned char)text[pos]))
        {
            pos++;
        }

        if (pos >= text.size() || text[pos] != '(')
        {
            error("Shape deformation requires one (x,y,z) factor per segment.");
            return true;
        }

        size_t close = text.find(')', pos + 1);

        if (close == string::npos)
        {
            error("Missing ')' in shape deformation.");
            return true;
        }

        string inside =
            text.substr(
                pos + 1,
                close - pos - 1
            );

        vector<string> parts =
            splitMovement(inside);

        if (parts.size() != 3)
        {
            error("Each shape deformation factor must be (x,y,z).");
            return true;
        }

        double x = 0;
        double y = 0;
        double z = 0;

        if (!tryNumber(replaceVariables(parts[0]), x) ||
            !tryNumber(replaceVariables(parts[1]), y) ||
            !tryNumber(replaceVariables(parts[2]), z))
        {
            try
            {
                CountParser px(replaceVariables(trim(parts[0])));
                CountParser py(replaceVariables(trim(parts[1])));
                CountParser pz(replaceVariables(trim(parts[2])));

                x = px.parse();
                y = py.parse();
                z = pz.parse();
            }
            catch (...)
            {
                error("Shape deformation factors must be numeric.");
                return true;
            }
        }

        factors.push_back(
        {
            (float)x,
            (float)y,
            (float)z
        });

        pos = close + 1;
    }

    while (pos < text.size() &&
           isspace((unsigned char)text[pos]))
    {
        pos++;
    }

    if (pos != text.size())
    {
        error("Unexpected text after shape deformation factors.");
        return true;
    }

    if (lastCreatedObjectKey.empty())
    {
        error("Shape deformation requires an object created before it.");
        return true;
    }

    deformation.side = sideText[0];
    deformation.segments = segments;
    deformation.factors = factors;

    shapeDeformations[lastCreatedObjectKey].push_back(deformation);

    return true;
}

// ============================================================
// GUI HELPERS
// ============================================================

// Extracts the text inside a parenthesis group that starts at
// text[openParenIndex] (which must be '('), matching nested parentheses.
// endIndexOut is set to the index right after the matching ')', or
// string::npos if the parentheses never close.
string extractParenGroup(const string &text, size_t openParenIndex, size_t &endIndexOut)
{
    if (openParenIndex >= text.size() || text[openParenIndex] != '(')
    {
        endIndexOut = string::npos;
        return "";
    }

    int depth = 0;

    for (size_t i = openParenIndex; i < text.size(); i++)
    {
        if (text[i] == '(')
        {
            depth++;
        }
        else if (text[i] == ')')
        {
            depth--;

            if (depth == 0)
            {
                endIndexOut = i + 1;
                return text.substr(openParenIndex + 1, i - openParenIndex - 1);
            }
        }
    }

    endIndexOut = string::npos;
    return "";
}

// Looks for "keyword(x,y,z)" anywhere in text. Returns false if the
// keyword isn't present at all. Returns true if it IS present; errorOut
// is left empty on success (x/y/z filled in) or set to a description of
// what's wrong (missing ')', wrong number of values, non-numeric values).
bool tfFindNamedVector(const string &text, const string &keyword,
                        float &x, float &y, float &z, string &errorOut)
{
    errorOut.clear();
    size_t pos = text.find(keyword + "(");

    if (pos == string::npos)
        return false;

    size_t endIndex;
    string inner = extractParenGroup(text, pos + keyword.size(), endIndex);

    if (endIndex == string::npos)
    {
        errorOut = "Missing closing ')' for " + keyword + "(...).";
        return true;
    }

    vector<string> parts = splitMovement(inner);

    if (parts.size() != 3)
    {
        errorOut = keyword + "(...) requires exactly 3 numbers: (x,y,z). Example: " + keyword + "(10,10,0)";
        return true;
    }

    double dx = 0.0, dy = 0.0, dz = 0.0;

    if (!tryNumber(parts[0], dx) || !tryNumber(parts[1], dy) || !tryNumber(parts[2], dz))
    {
        errorOut = keyword + "(...) values must be numeric. Example: " + keyword + "(10,10,0)";
        return true;
    }

    x = (float)dx;
    y = (float)dy;
    z = (float)dz;
    return true;
}

// Resolves growl-style content (variables, "list name[index]", "random",
// random text) WITHOUT growl's "count"/CSV-splitting behaviour. Used by
// "gui text(...)" so the on-screen text understands exactly the same
// content syntax growl() does, and can show a live value every frame.
string tfResolveDisplayText(string content)
{
    content = trim(content);
    content = resolveListAccess(content);

    if (content == "random")
        return randomNumber();

    if (isRandomText(content))
        return randomText();

    return replaceVariables(content);
}

// ============================================================
// OPTIONAL WAIT / REPEAT INTERVAL
// ============================================================
// `wait(seconds)` is intentionally additive. A value of 1 means one second;
// decimals such as 0.30 are also accepted.
// The standalone line is used as an optional interval for the next repeat:
//
//     wait(1)
//     repeat
//         growl("hello")
//
// Existing `repeat` without a preceding wait keeps its original behavior.

bool parseWaitCommand(const string &line, float &seconds)
{
    string text = trim(line);
    if (text.rfind("wait(", 0) != 0 || text.empty() || text.back() != ')')
        return false;

    string inside = trim(text.substr(5, text.size() - 6));
    if (inside.empty())
        return false;

    double value = 0.0;
    if (!tryNumber(replaceVariables(inside), value) || !isfinite(value) || value < 0.0)
        return false;

    seconds = static_cast<float>(value);
    return true;
}

bool isStandaloneWaitCommand(const string &line)
{
    float seconds = 0.0f;
    return parseWaitCommand(line, seconds);
}

bool parseWaitRepeatCommand(const string &line,
                            float &seconds,
                            string &repeatLine)
{
    string text = trim(line);
    if (text.rfind("wait(", 0) != 0)
        return false;

    size_t open = text.find('(', 4);
    if (open == string::npos)
        return false;

    size_t close = text.find(')', open + 1);
    if (close == string::npos)
        return false;

    string waitText = trim(text.substr(open + 1, close - open - 1));
    if (waitText.empty())
        return false;

    double waitValue = 0.0;
    if (!tryNumber(replaceVariables(waitText), waitValue) ||
        !isfinite(waitValue) || waitValue < 0.0)
        return false;

    string tail = trim(text.substr(close + 1));
    if (tail != "repeat" &&
        !(tail.rfind("repeat ", 0) == 0 && tail.find('=') == string::npos))
        return false;

    seconds = static_cast<float>(waitValue);
    repeatLine = tail;
    return true;
}

// ============================================================
// OPTIONAL OBJECT MEMORY / DELETE / ADD OBJECT
// ============================================================
// `say 2d/3d` saves the original object definition in memory.
// `delete "name"` removes the active object from the scene but keeps
// its saved definition. `add object "name"` restores that definition.
// These commands are additive and do not replace any existing syntax.

bool tfObjectIsActive(const string &name)
{
    for (const TFObject2D &obj : objects2D)
        if (obj.name == name)
            return true;

    for (const TFObject3D &obj : objects3D)
        if (obj.name == name)
            return true;

    return false;
}

void tfRemoveObjectRuntimeState(const string &name)
{
    gravityBodies.erase(name);
    gravityGrounded.erase(name);
    shapeDeformations.erase(deformationKey2D(name));
    shapeDeformations.erase(deformationKey3D(name));

    // Keep collision declarations: if the object is added again, its old
    // collision relationships become active automatically again.
}

bool tfDeleteObject(const string &name)
{
    bool removed = false;

    for (size_t i = 0; i < objects2D.size(); )
    {
        if (objects2D[i].name == name)
        {
            objects2D.erase(objects2D.begin() + static_cast<long>(i));
            removed = true;
            continue;
        }
        ++i;
    }

    for (size_t i = 0; i < objects3D.size(); )
    {
        if (objects3D[i].name == name)
        {
            objects3D.erase(objects3D.begin() + static_cast<long>(i));
            removed = true;
            continue;
        }
        ++i;
    }

    if (removed)
        tfRemoveObjectRuntimeState(name);

    return removed;
}

bool tfAddRememberedObject(const string &name)
{
    if (tfObjectIsActive(name))
    {
        error("Object already exists: \"" + name + "\".");
        return true;
    }

    auto it2D = objectMemory2D.find(name);
    if (it2D != objectMemory2D.end())
    {
        objects2D.push_back(it2D->second);
        sceneHas2D = true;
        lastCreatedObjectKey = deformationKey2D(name);
        return true;
    }

    auto it3D = objectMemory3D.find(name);
    if (it3D != objectMemory3D.end())
    {
        objects3D.push_back(it3D->second);
        sceneHas3D = true;
        lastCreatedObjectKey = deformationKey3D(name);
        return true;
    }

    error("Object memory not found: \"" + name + "\". The object must have been created earlier with say 2d or say 3d.");
    return true;
}

bool parseQuotedObjectNameCommand(const string &line,
                                  const string &keyword,
                                  string &name)
{
    string rest = trim(line.substr(keyword.size()));
    if (rest.size() < 2 || rest.front() != '\"' || rest.back() != '\"')
        return false;

    size_t close = rest.find('\"', 1);
    if (close == string::npos || close != rest.size() - 1)
        return false;

    name = rest.substr(1, close - 1);
    return !name.empty();
}

// ============================================================
// OPTIONAL FUNCTIONS / RETURN / BREAK / CONTINUE
// ============================================================

bool tfParseFunctionHeader(const string &line,
                           string &name,
                           vector<string> &parameters)
{
    string text = trim(line);
    if (text.rfind("function ", 0) != 0)
        return false;

    string rest = trim(text.substr(9));
    size_t open = rest.find('(');
    if (open == string::npos || rest.back() != ')')
        return false;

    name = trim(rest.substr(0, open));
    if (!validName(name))
        return false;

    string inside = trim(rest.substr(open + 1, rest.size() - open - 2));
    parameters.clear();
    if (inside.empty())
        return true;

    for (string part : splitParts(inside))
    {
        part = trim(part);
        if (!validName(part))
            return false;
        for (const string &existing : parameters)
        {
            if (existing == part)
                return false;
        }
        parameters.push_back(part);
    }

    return true;
}

bool tfParseFunctionCall(const string &line,
                         string &name,
                         vector<string> &arguments)
{
    string text = trim(line);
    size_t open = text.find('(');
    if (open == string::npos || text.empty() || text.back() != ')')
        return false;

    // Only a bare identifier followed by one balanced (...) group is a call.
    name = trim(text.substr(0, open));
    if (!validName(name))
        return false;

    int depth = 0;
    bool inQuotes = false;
    for (size_t i = open; i < text.size(); ++i)
    {
        char c = text[i];
        if (c == '"')
            inQuotes = !inQuotes;
        if (inQuotes)
            continue;
        if (c == '(')
            ++depth;
        else if (c == ')')
        {
            --depth;
            if (depth == 0 && i != text.size() - 1)
                return false;
            if (depth < 0)
                return false;
        }
    }

    if (depth != 0 || inQuotes)
        return false;

    string inside = text.substr(open + 1, text.size() - open - 2);
    arguments.clear();
    if (trim(inside).empty())
        return true;

    arguments = splitParts(inside);
    for (string &arg : arguments)
        arg = trim(arg);
    return true;
}

string tfResolveFunctionArgument(string value)
{
    value = trim(value);
    value = resolveListAccess(value);

    if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
        return value.substr(1, value.size() - 2);
    if (value.size() >= 2 && value.front() == '\'' && value.back() == '\'')
        return value.substr(1, value.size() - 2);

    value = replaceVariables(value);

    double numeric = 0.0;
    if (tryNumber(value, numeric))
        return formatNumber(numeric);

    try
    {
        CountParser parser(value);
        return formatNumber(parser.parse());
    }
    catch (...)
    {
    }

    return value;
}

bool tfCallFunction(const string &name,
                    const vector<string> &arguments,
                    string &returnValue)
{
    auto functionIt = tfFunctions.find(name);
    if (functionIt == tfFunctions.end())
        return false;

    const TFFunction function = functionIt->second;
    if (arguments.size() != function.parameters.size())
    {
        error("Function \"" + name + "\" requires " +
              to_string(function.parameters.size()) +
              " argument(s), but " + to_string(arguments.size()) + " were given.");
        return true;
    }

    // Parameters are local to this invocation. Existing variables with the
    // same names are temporarily hidden and restored afterward.
    vector<pair<string, bool>> hadOld;
    vector<string> oldValues;
    hadOld.reserve(function.parameters.size());
    oldValues.reserve(function.parameters.size());

    for (const string &parameter : function.parameters)
    {
        auto it = variables.find(parameter);
        hadOld.push_back({parameter, it != variables.end()});
        oldValues.push_back(it != variables.end() ? it->second : "");
    }

    for (size_t i = 0; i < function.parameters.size(); ++i)
        variables[function.parameters[i]] =
            tfResolveFunctionArgument(arguments[i]);

    TFControlSignal previousSignal = tfControlSignal;
    string previousReturn = tfReturnValue;
    tfControlSignal = TFControlSignal::None;
    tfReturnValue.clear();

    ++tfFunctionDepth;
    executeBlock(function.body, 0, 0);
    --tfFunctionDepth;

    TFControlSignal functionSignal = tfControlSignal;
    returnValue = tfReturnValue;

    tfControlSignal = previousSignal;
    tfReturnValue = previousReturn;

    for (size_t i = 0; i < function.parameters.size(); ++i)
    {
        const string &parameter = function.parameters[i];
        if (hadOld[i].second)
            variables[parameter] = oldValues[i];
        else
            variables.erase(parameter);
    }

    // A function consumes its own return signal. Break/continue are deliberately
    // not allowed to escape a function, because they belong to loops in the
    // current execution context rather than the caller's context.
    if (functionSignal == TFControlSignal::Break ||
        functionSignal == TFControlSignal::Continue)
    {
        error("break/continue cannot leave a function. Use return to leave the function.");
        returnValue.clear();
    }

    return true;
}

bool tfTryFunctionCallExpression(string value, string &result)
{
    value = trim(value);
    string name;
    vector<string> arguments;
    if (!tfParseFunctionCall(value, name, arguments))
        return false;

    if (name == "memory")
    {
        if (arguments.size() != 1)
            return false;
        // Memory is handled below, after persistent-memory helpers are declared.
        return false;
    }

    if (tfFunctions.count(name) == 0)
        return false;

    return tfCallFunction(name, arguments, result);
}

// ============================================================
// BASIC PERSISTENT MEMORY
// ============================================================

string tfSanitizeMemoryValue(string value)
{
    for (char &c : value)
    {
        if (c == '\t' || c == '\n' || c == '\r')
            c = ' ';
    }
    return value;
}

void tfLoadPersistentMemory()
{
    if (tfPersistentMemoryLoaded)
        return;

    tfPersistentMemoryLoaded = true;
    ifstream file(TF_MEMORY_FILE);
    if (!file)
        return;

    string line;
    while (getline(file, line))
    {
        size_t tab = line.find('\t');
        if (tab == string::npos)
            continue;

        string name = line.substr(0, tab);
        string value = line.substr(tab + 1);
        if (validName(name))
            tfPersistentMemory[name] = value;
    }
}

void tfSavePersistentMemory()
{
    ofstream file(TF_MEMORY_FILE, ios::trunc);
    if (!file)
    {
        error("Could not save TigerFlash memory file: \"" + TF_MEMORY_FILE + "\"");
        return;
    }

    for (const auto &entry : tfPersistentMemory)
        file << entry.first << '\t' << tfSanitizeMemoryValue(entry.second) << '\n';
}

bool tfExtractMemoryName(const string &line,
                         const string &prefix,
                         string &name)
{
    string rest = trim(line.substr(prefix.size()));
    if (rest.size() < 2 || rest.front() != '"' || rest.back() != '"')
        return false;

    size_t close = rest.find('"', 1);
    if (close == string::npos || close != rest.size() - 1)
        return false;

    name = rest.substr(1, close - 1);
    return validName(name);
}

string tfMemoryValue(const string &name)
{
    tfLoadPersistentMemory();
    auto it = tfPersistentMemory.find(name);
    if (it == tfPersistentMemory.end())
        return "null";
    return it->second;
}

// ============================================================
// EXTERNAL TIGERFLASH LIBRARIES
// ============================================================

static string tfShellQuote(const string &value)
{
    string result = "'";
    for (char c : value)
    {
        if (c == '\'')
            result += "'\\''";
        else
            result += c;
    }
    result += "'";
    return result;
}

bool tfHostRegisterSay3DOptionsHook(TFSay3DOptionsHook hook)
{
    if (!hook)
        return false;
    if (find(tfLibrarySay3DOptionsHooks.begin(),
             tfLibrarySay3DOptionsHooks.end(),
             hook) == tfLibrarySay3DOptionsHooks.end())
        tfLibrarySay3DOptionsHooks.push_back(hook);
    return true;
}

bool tfHostRegisterSay3DHook(TFSay3DHook hook)
{
    if (!hook)
        return false;
    if (find(tfLibrarySay3DHooks.begin(),
             tfLibrarySay3DHooks.end(),
             hook) == tfLibrarySay3DHooks.end())
        tfLibrarySay3DHooks.push_back(hook);
    return true;
}

bool tfHostRegisterCommandHook(TFLibraryCommandHook hook)
{
    if (!hook)
        return false;
    if (find(tfLibraryCommandHooks.begin(),
             tfLibraryCommandHooks.end(),
             hook) == tfLibraryCommandHooks.end())
        tfLibraryCommandHooks.push_back(hook);
    return true;
}

bool tfHostRegisterConditionHook(TFLibraryConditionHook hook)
{
    if (!hook)
        return false;
    if (find(tfLibraryConditionHooks.begin(),
             tfLibraryConditionHooks.end(),
             hook) == tfLibraryConditionHooks.end())
        tfLibraryConditionHooks.push_back(hook);
    return true;
}

bool tfHostSetObjectMotionBlur(const char *objectName, float value)
{
    if (!objectName)
        return false;

    TFObject3D *object = findObject3D(trim(objectName));
    if (!object)
        return false;

    if (!isfinite(value))
        value = 0.0f;

    object->motionBlur = max(0.0f, min(10.0f, value));

    if (!object->motionBlurPreviousInitialized)
    {
        object->motionBlurPreviousPosition = {object->x, object->y, object->z};
        object->motionBlurPreviousInitialized = true;
    }

    return true;
}

float tfHostGetObjectMotionBlur(const char *objectName)
{
    if (!objectName)
        return -1.0f;

    TFObject3D *object = findObject3D(trim(objectName));
    return object ? object->motionBlur : -1.0f;
}

bool tfHostObject3DExists(const char *objectName)
{
    if (!objectName)
        return false;
    return findObject3D(trim(objectName)) != nullptr;
}

void tfHostLibraryLog(const char *message)
{
    if (message)
        cout << "TigerFlash library: " << message << endl;
}

bool tfRunLibrarySay3DOptionsHooks(string &options)
{
    bool changed = false;

    for (TFSay3DOptionsHook hook : tfLibrarySay3DOptionsHooks)
    {
        if (!hook)
            continue;

        vector<char> output(max<size_t>(4096, options.size() * 2 + 256), '\0');
        if (hook(options.c_str(), output.data(), output.size()))
        {
            options = trim(string(output.data()));
            changed = true;
        }
    }

    return changed;
}

void tfRunLibrarySay3DHooks(const string &objectName,
                            const string &fullLine)
{
    for (TFSay3DHook hook : tfLibrarySay3DHooks)
    {
        if (hook)
            hook(objectName.c_str(), fullLine.c_str());
    }
}

bool tfRunLibraryCommandHooks(const string &line)
{
    for (TFLibraryCommandHook hook : tfLibraryCommandHooks)
    {
        if (hook && hook(line.c_str()))
            return true;
    }
    return false;
}

bool tfRunLibraryConditionHooks(const string &condition, bool &handled)
{
    handled = false;

    for (TFLibraryConditionHook hook : tfLibraryConditionHooks)
    {
        if (!hook)
            continue;

        bool hookHandled = false;
        bool result = hook(condition.c_str(), &hookHandled);
        if (hookHandled)
        {
            handled = true;
            return result;
        }
    }

    return false;
}

bool tfLoadTigerFlashLibrary(const string &requestedName)
{
    string name = trim(requestedName);

    if (name.size() >= 2 &&
        name.front() == '<' &&
        name.back() == '>')
        name = trim(name.substr(1, name.size() - 2));

    if (name.size() >= 2 &&
        name.front() == '"' &&
        name.back() == '"')
        name = trim(name.substr(1, name.size() - 2));

    if (!validName(name))
    {
        error("Invalid library name. Use: library <shaders>");
        return true;
    }

    if (tfLoadedLibraries.count(name) != 0)
        return true;

    const filesystem::path libraryDir = "library";
    const filesystem::path sourcePath = libraryDir / (name + ".cpp");
    const filesystem::path directSoPath = libraryDir / (name + ".so");
    const filesystem::path cacheDir = libraryDir / ".cache";
    const filesystem::path cachedSoPath = cacheDir / (name + ".so");

    filesystem::path loadPath;

    try
    {
        // Prefer library/<name>.cpp whenever it exists. The interpreter builds
        // a timestamped cache in library/.cache/, preventing an older manually
        // compiled library/<name>.so from silently overriding the current source.
        if (filesystem::exists(sourcePath) &&
            filesystem::is_regular_file(sourcePath))
        {
            filesystem::create_directories(cacheDir);

            bool needsCompile = !filesystem::exists(cachedSoPath);
            if (!needsCompile)
            {
                needsCompile =
                    filesystem::last_write_time(sourcePath) >
                    filesystem::last_write_time(cachedSoPath);
            }

            if (needsCompile)
            {
                string command =
                    "g++ -std=c++17 -O2 -fPIC -shared " +
                    tfShellQuote(sourcePath.string()) +
                    " -o " +
                    tfShellQuote(cachedSoPath.string());

                int result = system(command.c_str());
                if (result != 0)
                {
                    error("Could not compile library: \"" +
                          sourcePath.string() + "\"");
                    return true;
                }
            }

            loadPath = cachedSoPath;
        }
        else if (filesystem::exists(directSoPath) &&
                 filesystem::is_regular_file(directSoPath))
        {
            loadPath = directSoPath;
        }
        else
        {
            error("Library not found: \"" + name +
                  "\". Expected library/" + name +
                  ".cpp or library/" + name + ".so");
            return true;
        }
    }
    catch (const filesystem::filesystem_error &e)
    {
        error("Library filesystem error: " + string(e.what()));
        return true;
    }

    void *handle = dlopen(loadPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle)
    {
        const char *message = dlerror();
        error("Could not open library \"" + name + "\": " +
              (message ? string(message) : "unknown dlopen error"));
        return true;
    }

    dlerror();
    void *symbol = dlsym(handle, "tfLibraryInit");
    const char *symbolError = dlerror();

    if (symbolError || !symbol)
    {
        error("Library \"" + name +
              "\" does not export tfLibraryInit().");
        dlclose(handle);
        return true;
    }

    TFLibraryInitFunction init =
        reinterpret_cast<TFLibraryInitFunction>(symbol);

    TFLibraryAPI api;
    api.version = 1;
    api.registerSay3DOptionsHook = tfHostRegisterSay3DOptionsHook;
    api.registerSay3DHook = tfHostRegisterSay3DHook;
    api.registerCommandHook = tfHostRegisterCommandHook;
    api.registerConditionHook = tfHostRegisterConditionHook;
    api.setObjectMotionBlur = tfHostSetObjectMotionBlur;
    api.getObjectMotionBlur = tfHostGetObjectMotionBlur;
    api.object3DExists = tfHostObject3DExists;
    api.log = tfHostLibraryLog;

    if (!init(&api))
    {
        error("Library \"" + name +
              "\" rejected the TigerFlash library API.");
        dlclose(handle);
        return true;
    }

    tfLibraryHandles.push_back(handle);
    tfLoadedLibraries.insert(name);

    cout << "TigerFlash: library <" << name << "> loaded." << endl;
    return true;
}

// ============================================================
// MODULE / IMPORT
// ============================================================

bool tfImportModule(const string &path)
{
    ifstream file(path);
    if (!file)
    {
        error("Could not import module: \"" + path + "\"");
        return true;
    }

    string normalized = path;
    if (tfImportedModules.count(normalized) != 0)
        return true;
    tfImportedModules.insert(normalized);

    vector<string> importedScript;
    string raw;
    bool insideComment = false;

    while (getline(file, raw))
    {
        string remaining;
        if (consumeCommentLine(raw, insideComment, remaining))
        {
            raw = remaining;
            if (raw.empty())
                continue;
        }

        if (toLowerOutsideQuotes(trim(raw)) == "stop")
            break;

        importedScript.push_back(raw);
    }

    if (insideComment)
        error("Imported module has an unclosed comment: \"" + path + "\"");

    if (!importedScript.empty())
        executeBlock(importedScript, 0, 0);

    return true;
}

// ============================================================
// EXECUTE ONE COMMAND
// ============================================================
 
bool executeLine(string line)
{
    line = trim(line);
 
    if (line.empty())
    {
        error("Empty command: this line has no content to run. If you see this unexpectedly, check for stray blank lines or invisible characters.");
        return true;
    }

    // =========================
    // RETURN / BREAK / CONTINUE
    // =========================
    if (line == "break")
    {
        if (tfFunctionDepth > 0)
        {
            error("break cannot be used to leave a function. Use return instead.");
            return true;
        }
        if (tfLoopDepth <= 0)
        {
            error("break can only be used inside repeat.");
            return true;
        }
        tfControlSignal = TFControlSignal::Break;
        return true;
    }

    if (line == "continue")
    {
        if (tfFunctionDepth > 0)
        {
            error("continue cannot be used to leave a function. Use return instead.");
            return true;
        }
        if (tfLoopDepth <= 0)
        {
            error("continue can only be used inside repeat.");
            return true;
        }
        tfControlSignal = TFControlSignal::Continue;
        return true;
    }

    if (line == "return" || line.rfind("return ", 0) == 0)
    {
        if (tfFunctionDepth <= 0)
        {
            error("return can only be used inside a function.");
            return true;
        }

        string value = line == "return" ? "" : trim(line.substr(7));
        string functionResult;
        if (!value.empty() && tfTryFunctionCallExpression(value, functionResult))
            value = functionResult;
        else
        {
            value = resolveListAccess(value);
            value = replaceVariables(value);
            if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
                value = value.substr(1, value.size() - 2);
            else if (value.size() >= 2 && value.front() == '\'' && value.back() == '\'')
                value = value.substr(1, value.size() - 2);
            else
            {
                try
                {
                    CountParser parser(value);
                    value = formatNumber(parser.parse());
                }
                catch (...)
                {
                }
            }
        }

        tfReturnValue = value;
        tfControlSignal = TFControlSignal::Return;
        return true;
    }

    // =========================
    // MEMORY (OPTIONAL)
    // =========================
    // memory "name" = value
    // see result = memory("name")
    // delete memory "name"
    if (line.rfind("memory ", 0) == 0)
    {
        tfLoadPersistentMemory();
        string rest = trim(line.substr(7));
        size_t equal = rest.find('=');

        if (equal == string::npos)
        {
            error("Memory syntax error. Use: memory \"name\" = value");
            return true;
        }

        string namePart = trim(rest.substr(0, equal));
        string value = trim(rest.substr(equal + 1));
        if (namePart.size() < 2 || namePart.front() != '\"' || namePart.back() != '\"')
        {
            error("Memory name must use quotes. Example: memory \"score\" = 100");
            return true;
        }
        string name = namePart.substr(1, namePart.size() - 2);
        if (!validName(name))
        {
            error("Invalid memory name. Use letters, numbers and underscores.");
            return true;
        }

        value = resolveListAccess(value);
        if (value.size() >= 2 && value.front() == '\"' && value.back() == '\"')
            value = value.substr(1, value.size() - 2);
        else
        {
            string functionResult;
            if (tfTryFunctionCallExpression(value, functionResult))
                value = functionResult;
            else
            {
                value = replaceVariables(value);
                try
                {
                    CountParser parser(value);
                    value = formatNumber(parser.parse());
                }
                catch (...)
                {
                }
            }
        }

        tfPersistentMemory[name] = tfSanitizeMemoryValue(value);
        tfSavePersistentMemory();
        return true;
    }

    if (line.rfind("delete memory ", 0) == 0)
    {
        string name;
        if (!tfExtractMemoryName(line, "delete memory ", name))
        {
            error("Delete memory syntax error. Use: delete memory \"name\"");
            return true;
        }
        tfLoadPersistentMemory();
        tfPersistentMemory.erase(name);
        tfSavePersistentMemory();
        return true;
    }

    // =========================
    // IMPORT MODULE (OPTIONAL)
    // =========================
    if (line.rfind("import ", 0) == 0)
    {
        string path = trim(line.substr(7));
        if (path.size() < 2 || path.front() != '\"' || path.back() != '\"')
        {
            error("Import syntax error. Use: import \"module.tf\"");
            return true;
        }
        path = path.substr(1, path.size() - 2);
        return tfImportModule(path);
    }

    // =========================
    // EXTERNAL LIBRARY
    // =========================
    // Syntax: library <name>
    if (line.rfind("library ", 0) == 0)
    {
        string libraryName = trim(line.substr(8));
        return tfLoadTigerFlashLibrary(libraryName);
    }

    // =========================
    // FUNCTION CALL (OPTIONAL)
    // =========================
    {
        string functionName;
        vector<string> functionArguments;
        if (tfParseFunctionCall(line, functionName, functionArguments) &&
            tfFunctions.count(functionName) != 0)
        {
            string ignoredResult;
            tfCallFunction(functionName, functionArguments, ignoredResult);
            return true;
        }
    }

    // `wait(...)` is a scheduler hint, not a blocking sleep. A standalone
    // wait is consumed by the block parser when it is placed immediately
    // before a repeat. This keeps the graphics window responsive.
    if (isStandaloneWaitCommand(line))
        return true;


    // =========================
    // DELETE OBJECT (OPTIONAL)
    // =========================
    // delete "objectName"
    if (line.rfind("delete ", 0) == 0)
    {
        string name;
        if (!parseQuotedObjectNameCommand(line, "delete ", name))
        {
            error("Delete syntax error. Use: delete \"objectName\"");
            return true;
        }

        if (!tfObjectIsActive(name))
        {
            if (objectMemory2D.count(name) != 0 || objectMemory3D.count(name) != 0)
                error("Object is already deleted: \"" + name + "\". Use add object \"" + name + "\" to restore it.");
            else
                error("Object not found: \"" + name + "\". It must have been created earlier with say 2d or say 3d.");
            return true;
        }

        tfDeleteObject(name);
        return true;
    }

    // =========================
    // ADD OBJECT FROM MEMORY (OPTIONAL)
    // =========================
    // add object "objectName"
    if (line.rfind("add object ", 0) == 0)
    {
        string name;
        if (!parseQuotedObjectNameCommand(line, "add object ", name))
        {
            error("Add object syntax error. Use: add object \"objectName\"");
            return true;
        }

        return tfAddRememberedObject(name);
    }
 
    if (line.rfind("repeat ", 0) == 0 &&
        line.find('=') != string::npos)
    {
        executeRepeatAdjustment(line);
        return true;
    }
 
    
    // =========================
    // COLISION
    // =========================
    // Outside if: persistent physical collision.
    if (line.rfind("colision ", 0) == 0)
    {
        string first;
        string second;

        if (!parseCollisionSyntax(line, first, second))
        {
            error("Invalid collision syntax. Use: colision \"name1\" to \"name2\"");
            return true;
        }

        addPhysicalCollision(first, second);
        return true;
    }

    // =========================
    // SHAPE PART DEFORMATION
    // =========================
    if (line.rfind("shape p[", 0) == 0)
    {
        TFDeformation deformation;

        parseShapePartDeformation(
            line,
            deformation
        );

        return true;
    }

// =========================
    // SEE
    // =========================
 
    if (line.rfind("see ", 0) == 0)
    {
        string content = line.substr(4);
        size_t equal = content.find('=');
 
        if (equal == string::npos)
        {
            error("Invalid variable syntax. Expected: see <name> = <value>. Example: see health = 100");
            return true;
        }
 
        string name = content.substr(0, equal);
        string value = content.substr(equal + 1);
 
        name = trim(name);
        value = trim(value);
        value = resolveListAccess(value);
 
        if (!validName(name))
        {
            error("Invalid variable name. Names must start with a letter or underscore, and contain only letters, numbers and underscores (no spaces or symbols).");
            return true;
        }
 
        if (value == "null")
        {
            variables[name] = "null";
            return true;
        }
 
        if (value == "random")
        {
            variables[name] = randomNumber();
            return true;
        }
 
        if (isRandomText(value))
        {
            variables[name] = randomText();
            return true;
        }

        // Optional function-call expressions: see result = sum(2,3)
        // Optional persistent memory expression: see score = memory(\"score\")
        string functionResult;
        if (tfTryFunctionCallExpression(value, functionResult))
        {
            variables[name] = functionResult;
            return true;
        }

        if (value.rfind("memory(", 0) == 0 && value.size() >= 9 && value.back() == ')')
        {
            string inside = trim(value.substr(7, value.size() - 8));
            if (inside.size() >= 2 && inside.front() == '\"' && inside.back() == '\"')
            {
                string memoryName = inside.substr(1, inside.size() - 2);
                variables[name] = tfMemoryValue(memoryName);
                return true;
            }
        }
 
        try
        {
            CountParser parser(value);
            double result = parser.parse();
            variables[name] = formatNumber(result);
            return true;
        }
        catch (const exception &)
        {
        }
 
        if (value.size() >= 2 &&
            value.front() == '"' &&
            value.back() == '"')
        {
            variables[name] =
                value.substr(1, value.size() - 2);
            return true;
        }
 
        if (value.size() >= 2 &&
            value.front() == '\'' &&
            value.back() == '\'')
        {
            variables[name] =
                value.substr(1, value.size() - 2);
            return true;
        }
 
        error("Invalid variable value. Use a number, a quoted \"text\" value, true, false, null, or a valid expression.");
        return true;
    }
 
    // =========================
    // LIST
    // =========================
    // Criar:  list nome {item1; item2; item3}
    // Add:    list nome add: {item} posicao
    // Remove: list nome remove: [posicao]
    // Ler:    growl(list nome[posicao]) -> usado dentro de growl/see/if
    // Posição começa em 1 (a primeira coisa da lista é a posição 1).
 
    if (line.rfind("list ", 0) == 0)
    {
        string rest = trim(line.substr(5));
 
        size_t nameEnd = 0;
 
        while (nameEnd < rest.size() &&
               !isspace((unsigned char)rest[nameEnd]))
        {
            nameEnd++;
        }
 
        string name = rest.substr(0, nameEnd);
        string remainder = trim(rest.substr(nameEnd));
 
        if (!validName(name))
        {
            error("Invalid list name. Names must start with a letter or underscore, and contain only letters, numbers and underscores (no spaces or symbols).");
            return true;
        }
 
        // ---- list nome add: {valor} posicao ----
        if (remainder.rfind("add:", 0) == 0)
        {
            string addRest = trim(remainder.substr(4));
 
            if (addRest.empty() || addRest.front() != '{')
            {
                error("Invalid list add syntax. Use: list name add: {value} position");
                return true;
            }
 
            size_t closeBrace = addRest.find('}');
 
            if (closeBrace == string::npos)
            {
                error("Missing closing '}' in list add.");
                return true;
            }
 
            string valueText = addRest.substr(1, closeBrace - 1);
            string positionText = trim(addRest.substr(closeBrace + 1));
 
            string value = resolveListElementValue(valueText);
 
            double positionNumber = 0;
 
            if (!tryNumber(replaceVariables(resolveListAccess(positionText)), positionNumber) ||
                positionNumber != (long long)positionNumber ||
                positionNumber < 1)
            {
                error("List position must be a positive whole number (1, 2, 3, ...). Positions start counting at 1, not 0.");
                return true;
            }
 
            long long position = (long long)positionNumber;
            vector<string> &target = lists[name];
 
            if (position >= (long long)target.size() + 1)
            {
                while ((long long)target.size() + 1 < position)
                    target.push_back("");
 
                target.push_back(value);
            }
            else
            {
                target.insert(target.begin() + (position - 1), value);
            }
 
            return true;
        }
 
        // ---- list nome remove: [posicao] ----
        if (remainder.rfind("remove:", 0) == 0)
        {
            string removeRest = trim(remainder.substr(7));
 
            if (removeRest.size() < 2 ||
                removeRest.front() != '[' ||
                removeRest.back() != ']')
            {
                error("Invalid list remove syntax. Use: list name remove: [position]");
                return true;
            }
 
            string positionText =
                trim(removeRest.substr(1, removeRest.size() - 2));
 
            double positionNumber = 0;
 
            if (!tryNumber(replaceVariables(resolveListAccess(positionText)), positionNumber) ||
                positionNumber != (long long)positionNumber ||
                positionNumber < 1)
            {
                error("List position must be a positive whole number (1, 2, 3, ...). Positions start counting at 1, not 0.");
                return true;
            }
 
            if (!lists.count(name))
            {
                error("List not found: \"" + name + "\". Make sure it was created earlier with: list " + name + " = {...}");
                return true;
            }
 
            long long position = (long long)positionNumber;
            vector<string> &target = lists[name];
 
            if (position > (long long)target.size())
            {
                error("List position out of range. Positions start at 1 and must be within the list's current size.");
                return true;
            }
 
            target.erase(target.begin() + (position - 1));
            return true;
        }
 
        // ---- list nome {item1; item2; item3}  (criação) ----
        if (!remainder.empty() && remainder.front() == '{')
        {
            if (remainder.back() != '}')
            {
                error("Missing closing '}' in list creation.");
                return true;
            }
 
            string inner = remainder.substr(1, remainder.size() - 2);
            vector<string> parts = splitParts(inner);
            vector<string> values;
 
            for (string &part : parts)
            {
                part = trim(part);
 
                if (part.empty() && parts.size() == 1)
                    continue; // list nome {} -> lista vazia
 
                values.push_back(resolveListElementValue(part));
            }
 
            lists[name] = values;
            return true;
        }
 
        error("Invalid list syntax. Expected one of: list <name> = {value1, value2, ...} | list <name> add: {value} <position> | list <name> remove: [position]");
        return true;
    }
 
    // =========================
    // SAY 2D
    // =========================
 
    if (line.rfind("say 2d ", 0) == 0)
    {
        string tail = trim(line.substr(7));
        string object;
        string objectName;
        string color = "white";
        vector<double> inlinePosition;
        vector<double> inlineScale;
        bool hasPosition = false;
        bool hasScale = false;
 
        if (tail.size() < 2 || tail.front() != '"')
        {
            error("Invalid 2D syntax. Use: say 2d \"object\" as \"name\" color at(x,y) shape(x,y). position(x,y) is also accepted.");
            return true;
        }
 
        size_t closingQuote = tail.find('"', 1);
        if (closingQuote == string::npos)
        {
            error("Missing closing quote in 2D object.");
            return true;
        }
 
        object = tail.substr(1, closingQuote - 1);
        string options = trim(tail.substr(closingQuote + 1));
 
        if (!extractObjectName(options, objectName))
        {
            error("Invalid object name. Use: as \"name\"");
            return true;
        }
 
        bool positionToken = containsInlinePositionToken(options);
        if (positionToken && !extractInlinePosition(options, inlinePosition))
        {
            error("Invalid 2D position. Use at(x,y), for example: at(10,-2)");
            return true;
        }
        if (positionToken)
            hasPosition = true;
 
        if (extractInlineShape(options, inlineScale))
            hasScale = true;
 
        options = trim(options);
 
        if (!options.empty())
        {
            string parsedColor;
            if (extractInlineColor(options, parsedColor))
                color = parsedColor;
            else
            {
                error("Unknown 2D color.");
                return true;
            }
        }
 
        if (object == "random")
            object = randomShape2D();
 
        if (color == "random")
            color = randomColor();
 
        if (object != "cube" &&
            object != "sphere" &&
            object != "cone" &&
            object != "triangle")
        {
            error("Unknown 2D object. Use cube, sphere, cone or triangle.");
            return true;
        }
 
        if (!validColor(color))
        {
            error("Unknown 2D color.");
            return true;
        }
 
        if (objectName.empty())
            objectName = "object2d_" + to_string(objects2D.size() + 1);
 
        for (const TFObject2D &obj : objects2D)
        {
            if (obj.name == objectName)
            {
                error("Object name already exists: " + objectName);
                return true;
            }
        }
 
        TFObject2D obj;
        obj.name = objectName;
        obj.shape = object;
        obj.color = color;
 
        if (hasPosition)
        {
            if (inlinePosition.size() != 2)
            {
                error("2D position requires (x,y).");
                return true;
            }
            obj.x = (float)inlinePosition[0];
            obj.y = (float)inlinePosition[1];
        }
 
        if (hasScale)
        {
            if (inlineScale.size() != 2)
            {
                error("2D shape requires (x,y).");
                return true;
            }
            obj.scaleX = (float)inlineScale[0];
            obj.scaleY = (float)inlineScale[1];
        }
 
        objects2D.push_back(obj);
        objectMemory2D[objectName] = obj;
        lastCreatedObjectKey = deformationKey2D(objectName);
        sceneHas2D = true;
        return true;
    }
 
    // =========================
    // SAY 3D
    // =========================
 
    if (line.rfind("say 3d ", 0) == 0)
    {
        string tail = trim(line.substr(7));
        string object;
        string objectName;
        string color = "white";
        vector<double> inlinePosition;
        vector<double> inlineScale;
        bool hasPosition = false;
        bool hasScale = false;
 
        if (tail.size() < 2 || tail.front() != '"')
        {
            error("Invalid 3D syntax. Use: say 3d \"object\" as \"name\" color at(x,y,z) shape(x,y,z). position(x,y,z) is also accepted.");
            return true;
        }
 
        size_t closingQuote = tail.find('"', 1);
        if (closingQuote == string::npos)
        {
            error("Missing closing quote in 3D object.");
            return true;
        }
 
        object = tail.substr(1, closingQuote - 1);
        for (char &c : object)
            c = (char)tolower((unsigned char)c);

        if (object == "cristal") object = "crystal";
        else if (object == "capsula") object = "capsule";
        else if (object == "cilindro") object = "cylinder";
        else if (object == "diamante") object = "diamond";
        else if (object == "prisma") object = "prism";
        else if (object == "rocha") object = "rock";
        else if (object == "piramide") object = "triangle";

        string options = trim(tail.substr(closingQuote + 1));
        // External libraries can remove their own suffix syntax before the
        // core parser checks the remaining color/position/shape tokens.
        tfRunLibrarySay3DOptionsHooks(options);

 
        if (!extractObjectName(options, objectName))
        {
            error("Invalid object name. Use: as \"name\"");
            return true;
        }
 
        bool positionToken = containsInlinePositionToken(options);
        if (positionToken && !extractInlinePosition(options, inlinePosition))
        {
            error("Invalid 3D position. Use at(x,y,z), for example: at(0,3,0)");
            return true;
        }
        if (positionToken)
            hasPosition = true;
 
        if (extractInlineShape(options, inlineScale))
            hasScale = true;
 
        options = trim(options);
 
        if (!options.empty())
        {
            string parsedColor;
            if (extractInlineColor(options, parsedColor))
                color = parsedColor;
            else
            {
                error("Unknown 3D color.");
                return true;
            }
        }
 
        if (object == "random")
            object = randomShape();
 
        if (color == "random")
            color = randomColor();
 
        if (object != "cube" &&
            object != "sphere" &&
            object != "cone" &&
            object != "triangle" &&
            object != "cylinder" &&
            object != "capsule" &&
            object != "crystal" &&
            object != "diamond" &&
            object != "prism" &&
            object != "rock")
        {
            error("Unknown 3D object. Use cube, sphere, cone, triangle, cylinder, capsule, crystal, diamond, prism or rock.");
            return true;
        }
 
        if (!validColor(color))
        {
            error("Unknown 3D color.");
            return true;
        }
 
        if (objectName.empty())
            objectName = "object3d_" + to_string(objects3D.size() + 1);
 
        for (const TFObject3D &obj : objects3D)
        {
            if (obj.name == objectName)
            {
                error("Object name already exists: " + objectName);
                return true;
            }
        }
 
        TFObject3D obj;
        obj.name = objectName;
        obj.shape = object;
        obj.color = color;
 
        if (hasPosition)
        {
            if (inlinePosition.size() != 3)
            {
                error("3D position requires (x,y,z).");
                return true;
            }
            obj.x = (float)inlinePosition[0];
            obj.y = (float)inlinePosition[1];
            obj.z = (float)inlinePosition[2];
        }
 
        if (hasScale)
        {
            if (inlineScale.size() != 3)
            {
                error("3D shape requires (x,y,z).");
                return true;
            }
            obj.scaleX = (float)inlineScale[0];
            obj.scaleY = (float)inlineScale[1];
            obj.scaleZ = (float)inlineScale[2];
        }
 
        objects3D.push_back(obj);
        objectMemory3D[objectName] = obj;
        lastCreatedObjectKey = deformationKey3D(objectName);
        sceneHas3D = true;

        // The library receives the complete original command after creation.
        tfRunLibrarySay3DHooks(objectName, line);

        return true;
    }
 
    // ====================================================
    // GUI GRAVITY / BODY TYPE
    // gravity gui "point" = 300
    // gravity gui rigid body "point" = 300
    // gravity gui soft body (8) "point" = 300
    // ====================================================
    if (line.rfind("gravity gui ", 0) == 0)
    {
        string rest = trim(line.substr(12));
        bool soft = false;
        float softness = 10.0f;

        if (rest.rfind("rigid body ", 0) == 0)
            rest = trim(rest.substr(11));
        else if (rest.rfind("soft body ", 0) == 0)
        {
            soft = true;
            rest = trim(rest.substr(10));
            if (!rest.empty() && rest.front() == '(')
            {
                size_t close = rest.find(')');
                if (close == string::npos)
                {
                    error("GUI soft body requires a value like (8).");
                    return true;
                }
                double value = 10.0;
                if (!tryNumber(rest.substr(1, close - 1), value))
                {
                    error("GUI soft body value must be numeric.");
                    return true;
                }
                softness = (float)max(0.0, min(10.0, value));
                rest = trim(rest.substr(close + 1));
            }
        }

        if (rest.empty() || rest.front() != '"')
        {
            error("GUI gravity syntax: gravity gui [rigid body|soft body (0-10)] \"name\" = value");
            return true;
        }
        size_t closeName = rest.find('"', 1);
        if (closeName == string::npos)
        {
            error("Missing closing quote in GUI gravity name.");
            return true;
        }
        string name = rest.substr(1, closeName - 1);
        rest = trim(rest.substr(closeName + 1));
        if (rest.empty() || rest.front() != '=')
        {
            error("GUI gravity is missing '= value'.");
            return true;
        }
        double gravityValue = 0.0;
        if (!tryNumber(trim(rest.substr(1)), gravityValue))
        {
            error("GUI gravity value must be numeric.");
            return true;
        }

        TFGuiElement *gui = findGuiElement(name);
        if (!gui)
        {
            error("GUI gravity requires an existing GUI name: \"" + name + "\"");
            return true;
        }
        gui->physicsEnabled = true;
        gui->gravity = (float)gravityValue;
        gui->softBody = soft;
        gui->softness = softness;
        return true;
    }

    // ====================================================
    // GRAVITY / BODY TYPE
    // Examples:
    // gravity rigid body "ball" = 9.80665
    // gravity soft body (10) "ball" = 9.80665
    // gravity "ball" = 9.80665
    // ====================================================
    if (line.rfind("gravity ", 0) == 0)
    {
        string rest = trim(line.substr(8));
        TFGravityBody body;

        if (rest.rfind("rigid body ", 0) == 0)
        {
            body.type = TFBodyType::Rigid;
            rest = trim(rest.substr(11));
        }
        else if (rest.rfind("soft body ", 0) == 0)
        {
            body.type = TFBodyType::Soft;
            rest = trim(rest.substr(10));

            if (!rest.empty() && rest.front() == '(')
            {
                size_t close = rest.find(')');
                if (close == string::npos)
                {
                    error("Soft body requires a value like (10).");
                    return true;
                }

                double softness = 10.0;
                if (!tryNumber(rest.substr(1, close - 1), softness))
                {
                    error("Soft body value must be numeric.");
                    return true;
                }

                // Store softness by using the body gravity temporarily as an
                // internal marker. The actual gravity is assigned below.
                softness = max(0.0, min(10.0, softness));
                body.gravity = (float)softness;
                rest = trim(rest.substr(close + 1));
            }
        }

        if (rest.empty() || rest.front() != '"')
        {
            error("Gravity syntax error. Expected: gravity [soft body (softness)] \"<object>\" = <value>. Softness ranges 0-10 and only applies to soft bodies. Example: gravity soft body (10) \"jelly\" = 9.80665, or gravity \"rock\" = 9.80665 for a rigid body.");
            return true;
        }

        size_t closeName = rest.find('"', 1);
        if (closeName == string::npos)
        {
            error("Missing closing quote (\") after the object name in the 'gravity' command. Example: gravity \"rock\" = 9.80665");
            return true;
        }

        string objectName = rest.substr(1, closeName - 1);
        rest = trim(rest.substr(closeName + 1));

        if (rest.empty() || rest.front() != '=')
        {
            error("Gravity command is missing '= value'. Expected: gravity \"<object>\" = <value>, e.g. gravity \"rock\" = 9.80665");
            return true;
        }

        string gravityText = trim(rest.substr(1));
        double gravityValue = 0.0;
        if (!tryNumber(gravityText, gravityValue))
        {
            error("Gravity value must be a number (meters per second squared). Earth gravity is 9.80665; use 0 to disable falling for this object.");
            return true;
        }

        TFObject3D *object = findObject3D(objectName);
        if (!object)
        {
            error("Gravity requires a 3D object: \"" + objectName + "\" was not found among the 3D objects created so far. Gravity/physics only works on 3D scenes.");
            return true;
        }

        float softness = body.type == TFBodyType::Soft ? body.gravity : 0.0f;
        body.gravity = (float)max(0.0, gravityValue);

        TFGravityBody &stored = gravityBodies[objectName];

        // Gravity always starts from the object's current world coordinates.
        // This is important when the object was created with `at(...)`.
        stored.previousPosition = {object->x, object->y, object->z};
        stored.sweepStartY = object->y;
        stored.sweepStartInitialized = true;
        stored.startupHoldPosition = {object->x, object->y, object->z};
        stored.startupHoldRemaining = TF_COLLISION_STARTUP_HOLD_SECONDS;
        stored.initialized = true;

        stored.type = body.type;
        stored.gravity = body.gravity;
        // Softness is encoded separately using the dynamic spring target scale.
        if (stored.type == TFBodyType::Soft)
        {
            // Map 0..10 to the deformation mode through a compact field.
            // We keep it in flowX until the next physics tick initializes it.
            stored.softness = softness;
        }

        return true;
    }

    // ====================================================
    // CAMERA FOLLOW
    // camera follow "player" 0.08
    // The final value controls how quickly the camera catches the target.
    // Smaller values make the camera lag more behind the player.
    // ====================================================

    if (line.rfind("camera follow ", 0) == 0)
    {
        string rest = trim(line.substr(14));
        if (rest.empty() || rest.front() != '"')
        {
            error("Camera follow syntax error. Expected: camera follow \"<object>\" [speed]. Speed is optional (default 0.08, range 0.001-1.0). Example: camera follow \"player\" 0.08");
            return true;
        }

        size_t closeName = rest.find('\"', 1);
        if (closeName == string::npos)
        {
            error("Missing closing quote (\") after the object name in 'camera follow'. Example: camera follow \"player\" 0.08");
            return true;
        }

        string target = rest.substr(1, closeName - 1);
        string speedText = trim(rest.substr(closeName + 1));

        float smooth = 0.08f;
        if (!speedText.empty())
        {
            double value = 0.0;
            if (!tryNumber(speedText, value))
            {
                error("Camera follow speed must be a number between 0.001 and 1.0. Smaller values make the camera lag more behind the target.");
                return true;
            }
            smooth = static_cast<float>(max(0.001, min(1.0, value)));
        }

        bool found = false;
        for (const TFObject3D &obj : objects3D)
        {
            if (obj.name == target)
            {
                found = true;
                break;
            }
        }

        if (!found)
        {
            error("Object not found: \"" + target + "\". Make sure it was created earlier in the script (say 2d / say 3d) with this exact name, or use \"cam\" to target the camera.");
            return true;
        }

        tfCameraFollowTarget = target;
        tfCameraFollowSmooth = smooth;
        tfCameraFollowEnabled = true;
        return true;
    }

    // ====================================================
    // MOUSE LOOK (opt-in free camera look)
    // mouse look on
    // mouse look off
    //
    // OFF by default: moving the mouse does nothing to the camera unless
    // this is explicitly turned on, or the script binds camera rotation
    // itself (key d "cam" rotate(12,0,0), or a motion combo such as
    // key right_mouse + moviment up mouse "cam" rotate(-1,0,0)).
    // Turning this on gives classic FPS-style always-on mouse look instead.
    // ====================================================

    if (line == "mouse look on" || line == "mouse look off")
    {
        tfMouseLookEnabled = (line == "mouse look on");
        return true;
    }

    if (line.rfind("mouse look", 0) == 0)
    {
        error("Unknown 'mouse look' command \"" + line +
              "\". Use exactly: mouse look on   or   mouse look off");
        return true;
    }

    // ====================================================
    // OPTIONAL DIRECT OBJECT TRANSFORMS
    // These are additional syntaxes; the old commands remain untouched.
    //
    // "name" add moviment (x,y,z)
    // "name" rotate(x,y,z)
    // "name" shape(x,y,z)
    //
    // They execute immediately, so they can be used by repeat blocks, if
    // blocks, key actions and any other place where executeLine() runs.
    // ====================================================

    if (!line.empty() && line.front() == '"')
    {
        size_t closeName = line.find('"', 1);
        if (closeName != string::npos)
        {
            string target = line.substr(1, closeName - 1);
            string action = trim(line.substr(closeName + 1));

            // -----------------------------
            // "name" add moviment(x,y,z)
            // -----------------------------
            if (action.rfind("add moviment", 0) == 0)
            {
                string args = trim(action.substr(12));
                if (args.size() < 5 || args.front() != '(' || args.back() != ')')
                {
                    error("Direct movement must use: \"name\" add moviment (x,y,z)");
                    return true;
                }

                vector<string> parts = splitMovement(args.substr(1, args.size() - 2));
                if (parts.size() != 3)
                {
                    error("Direct movement requires exactly 3 values: (x,y,z)");
                    return true;
                }

                double dx = 0.0, dy = 0.0, dz = 0.0;
                if (!tryNumber(replaceVariables(parts[0]), dx) ||
                    !tryNumber(replaceVariables(parts[1]), dy) ||
                    !tryNumber(replaceVariables(parts[2]), dz))
                {
                    error("Direct movement values must be numeric: (x,y,z)");
                    return true;
                }

                if (target == "cam")
                {
                    tfCamera.position.x += static_cast<float>(dx);
                    tfCamera.position.y += static_cast<float>(dy);
                    tfCamera.position.z += static_cast<float>(dz);
                    tfApplyCameraRotation(tfCamera);
                    return true;
                }

                bool found = false;

                for (TFObject2D &obj : objects2D)
                {
                    if (obj.name != target) continue;
                    obj.x += static_cast<float>(dx);
                    obj.y += static_cast<float>(dy);
                    found = true;
                    break;
                }

                if (!found)
                {
                    for (TFObject3D &obj : objects3D)
                    {
                        if (obj.name != target) continue;

                        auto gravityIt = gravityBodies.find(obj.name);
                        if (gravityIt != gravityBodies.end())
                        {
                            tfMoveGravityBodyKinematic(obj,
                                {static_cast<float>(dx),
                                 static_cast<float>(dy),
                                 static_cast<float>(dz)});
                        }
                        else
                        {
                            obj.x += static_cast<float>(dx);
                            obj.y += static_cast<float>(dy);
                            obj.z += static_cast<float>(dz);
                        }
                        found = true;
                        break;
                    }
                }

                if (!found)
                    error("Object not found: \"" + target + "\".");

                return true;
            }

            // -----------------------------
            // "name" rotate(x,y,z)
            // -----------------------------
            if (action.rfind("rotate", 0) == 0)
            {
                string args = trim(action.substr(6));
                if (args.empty())
                    args = "()";

                float rx = 0.0f, ry = 0.0f, rz = 0.0f;
                if (args == "()")
                {
                    ry = 1.0f;
                }
                else if (args.size() >= 5 && args.front() == '(' && args.back() == ')')
                {
                    vector<string> parts = splitMovement(args.substr(1, args.size() - 2));
                    if (parts.size() != 3)
                    {
                        error("Direct rotation requires exactly 3 values: (x,y,z)");
                        return true;
                    }

                    double x = 0.0, y = 0.0, z = 0.0;
                    if (!tryNumber(replaceVariables(parts[0]), x) ||
                        !tryNumber(replaceVariables(parts[1]), y) ||
                        !tryNumber(replaceVariables(parts[2]), z))
                    {
                        error("Direct rotation values must be numeric: (x,y,z)");
                        return true;
                    }

                    rx = static_cast<float>(x);
                    ry = static_cast<float>(y);
                    rz = static_cast<float>(z);
                }
                else
                {
                    error("Direct rotation must use: \"name\" rotate(x,y,z) or \"name\" rotate()");
                    return true;
                }

                if (target == "cam")
                {
                    tfCameraRotationInitialized = true;
                    tfCameraRotation.x += rx;
                    tfCameraRotation.y += ry;
                    tfCameraRotation.z += rz;
                    if (IsWindowReady())
                        tfApplyCameraRotation(tfCamera);
                    return true;
                }

                bool found = false;
                for (TFObject2D &obj : objects2D)
                {
                    if (obj.name != target) continue;
                    obj.rotation += ry;
                    found = true;
                    break;
                }

                if (!found)
                {
                    for (TFObject3D &obj : objects3D)
                    {
                        if (obj.name != target) continue;
                        obj.rotationX += rx;
                        obj.rotationY += ry;
                        obj.rotationZ += rz;
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    // delete keeps the definition in objectMemory*. A repeat
                    // block may still contain rotate for that deleted object;
                    // ignore it silently until `add object` restores the object.
                    if (objectMemory2D.count(target) != 0 ||
                        objectMemory3D.count(target) != 0)
                        return true;

                    error("Object not found: \"" + target + "\".");
                }

                return true;
            }

            // -----------------------------
            // "name" shape(x,y[,z])
            // -----------------------------
            if (action.rfind("shape", 0) == 0)
            {
                string args = trim(action.substr(5));
                if (args.size() < 3 || args.front() != '(' || args.back() != ')')
                {
                    error("Direct shape must use: \"name\" shape(x,y) or \"name\" shape(x,y,z)");
                    return true;
                }

                vector<string> parts = splitMovement(args.substr(1, args.size() - 2));
                if (parts.size() != 2 && parts.size() != 3)
                {
                    error("Direct shape requires (x,y) for 2D or (x,y,z) for 3D.");
                    return true;
                }

                vector<double> values;
                for (const string &part : parts)
                {
                    double value = 0.0;
                    if (!tryNumber(replaceVariables(trim(part)), value))
                    {
                        error("Direct shape values must be numeric.");
                        return true;
                    }
                    values.push_back(value);
                }

                bool found = false;
                for (TFObject2D &obj : objects2D)
                {
                    if (obj.name != target) continue;
                    if (values.size() != 2)
                    {
                        error("2D shape requires exactly (x,y).");
                        return true;
                    }
                    obj.scaleX = static_cast<float>(values[0]);
                    obj.scaleY = static_cast<float>(values[1]);
                    found = true;
                    break;
                }

                if (!found)
                {
                    for (TFObject3D &obj : objects3D)
                    {
                        if (obj.name != target) continue;
                        if (values.size() != 3)
                        {
                            error("3D shape requires exactly (x,y,z).");
                            return true;
                        }
                        obj.scaleX = static_cast<float>(values[0]);
                        obj.scaleY = static_cast<float>(values[1]);
                        obj.scaleZ = static_cast<float>(values[2]);
                        found = true;
                        break;
                    }
                }

                if (!found)
                    error("Object not found: \"" + target + "\".");

                return true;
            }
        }
    }

    // ====================================================
    // DIRECT ROTATION
    // rotate "object" (x,y,z)
    // rotate "cam" (x,y,z)
    // ====================================================

    if (line.rfind("rotate ", 0) == 0)
    {
        string rest = trim(line.substr(7));
        if (rest.empty() || rest.front() != '"')
        {
            error("Rotation syntax error. Expected: rotate \"<object>\" (x,y,z). Example: rotate \"cube\" (0,1,0)");
            return true;
        }

        size_t closeName = rest.find('"', 1);
        if (closeName == string::npos)
        {
            error("Missing closing quote in rotation target.");
            return true;
        }

        string target = rest.substr(1, closeName - 1);
        string args = trim(rest.substr(closeName + 1));
        if (args.empty())
            args = "()";

        float rx = 0.0f, ry = 0.0f, rz = 0.0f;
        if (args == "()")
        {
            rx = 0.0f;
            ry = 1.0f;
            rz = 0.0f;
        }
        else if (args.front() == '(' && args.back() == ')')
        {
            vector<string> parts = splitMovement(args.substr(1, args.size() - 2));
            if (parts.size() != 3)
            {
                error("Rotation requires exactly 3 values in the form (x,y,z), e.g. (0,1,0). Fewer or more values were given.");
                return true;
            }

            double ax = 0.0, ay = 0.0, az = 0.0;
            if (!tryNumber(parts[0], ax) ||
                !tryNumber(parts[1], ay) ||
                !tryNumber(parts[2], az))
            {
                error("Rotation values must be numeric. Expected three numbers, e.g. rotate \"cube\" (0,1,0)");
                return true;
            }

            rx = static_cast<float>(ax);
            ry = static_cast<float>(ay);
            rz = static_cast<float>(az);
        }
        else
        {
            error("Rotation must use the format (x,y,z) as a per-axis rotation speed, e.g. rotate(0,1,0), or rotate() with no arguments for a small automatic Y spin.");
            return true;
        }

        if (target == "cam")
        {
            // Direct camera commands are applied immediately when a window
            // already exists, or preserved until the graphics window starts.
            tfCameraRotationInitialized = true;
            tfCameraRotation.x += rx;
            tfCameraRotation.y += ry;
            tfCameraRotation.z += rz;
            if (IsWindowReady())
                tfApplyCameraRotation(tfCamera);
            return true;
        }

        bool found = false;
        for (TFObject2D &obj : objects2D)
        {
            if (obj.name == target)
            {
                obj.rotation += ry;
                found = true;
                break;
            }
        }

        if (!found)
        {
            for (TFObject3D &obj : objects3D)
            {
                if (obj.name == target)
                {
                    obj.rotationX += rx;
                    obj.rotationY += ry;
                    obj.rotationZ += rz;
                    found = true;
                    break;
                }
            }
        }

        if (!found)
            error("Object not found: \"" + target + "\". Make sure it was created earlier in the script (say 2d / say 3d) with this exact name, or use \"cam\" to target the camera.");

        return true;
    }

    // ====================================================
    // KEY INPUT
    // Movimento:
    // key d "cubinho" add moviment (5,0,0)
    // key d "cam" add moviment (0,0,-0.3)
    // Rotacao:
    // key d "cubinho" rotate(0,1,0)
    // key d "cam" rotate(0,1,0)
    // key d "cam" rotate()
    // Comando:
    // key d growl("OI")
    // Tambem aceita todas as teclas do teclado, botoes/eixos de mouse e gamepad, touch e VR.
    // Tambem aceita combinacoes como: key left_mouse + moviment up mouse "cam" rotate(-1,0,0).
    // ====================================================
 
    if (line.rfind("key ", 0) == 0)
    {
        string rest = trim(line.substr(4));
 
        size_t firstSpace = rest.find_first_of(" \t");
        if (firstSpace == string::npos)
        {
            error("Invalid 'key' syntax. Expected: key <input> \"<object>\" <action>. Example: key d \"cube\" add moviment (5,0,0)   or   key d \"cube\" rotate(0,1,0)   or   key d growl(\"Hi\")");
            return true;
        }
 
        string control = trim(rest.substr(0, firstSpace));
        rest = trim(rest.substr(firstSpace + 1));
 
        string normalized = normalizeInputControl(control);
        if (keyboardKeyFromName(normalized) < 0 &&
            mouseButtonFromName(normalized) < 0 &&
            gamepadButtonFromName(normalized) < 0 &&
            gamepadAxisFromName(normalized) < 0 &&
            !tfIsMotionControlName(normalized))
        {
            error("Unknown input control: \"" + control + "\". Valid inputs: keyboard keys (a-z, 0-9, space, enter, escape, tab, arrows, f1-f12, shift, ctrl, alt...), mouse buttons (left_mouse/mouse1, right_mouse/mouse2, middle_mouse/mouse3), gamepad buttons (gamepad_a, gamepad_b, gamepad_x, gamepad_y, dpad_up/down/left/right, gamepad_l1/r1/l2/r2...), gamepad axes (left_x, left_y, right_x, right_y, left_trigger, right_trigger), or motion controls (mouse_move_up/down/left/right, joystick_up/down/left/right, right_joystick_up/down/left/right, vr_up/down/left/right/front/back).");
            return true;
        }
 
        TFBinding binding;
        binding.control = control;
 
        if (mouseButtonFromName(normalized) >= 0)
            binding.type = TFInputType::Mouse;
        else if (gamepadButtonFromName(normalized) >= 0)
            binding.type = TFInputType::Gamepad;
        else if (gamepadAxisFromName(normalized) >= 0 || tfIsMotionControlName(normalized))
            binding.type = TFInputType::Axis;
        else
            binding.type = TFInputType::Keyboard;
 
        // Helper: parse a 3-value vector from either (x,y,z) or ( x,y,z ).
        auto parseBindingVector = [&](const string &text,
                                      float &x, float &y, float &z,
                                      bool &usedDefault) -> bool
        {
            string value = trim(text);
            usedDefault = false;
 
            if (value == "()")
            {
                // Empty rotate() means a small continuous Y rotation.
                x = 0.0f;
                y = 1.0f;
                z = 0.0f;
                usedDefault = true;
                return true;
            }
 
            if (value.size() < 5 || value.front() != '(' || value.back() != ')')
                return false;
 
            string coords = value.substr(1, value.size() - 2);
            vector<string> parts = splitMovement(coords);
            if (parts.size() != 3)
                return false;
 
            double ax = 0.0, ay = 0.0, az = 0.0;
            if (!tryNumber(parts[0], ax) ||
                !tryNumber(parts[1], ay) ||
                !tryNumber(parts[2], az))
                return false;
 
            x = static_cast<float>(ax);
            y = static_cast<float>(ay);
            z = static_cast<float>(az);
            return true;
        };
 
        // ====================================================
        // TWO-INPUT MOTION BINDING
        // Example:
        // key left_mouse + moviment up mouse "cam" rotate(-1,0,0)
        // key shift + moviment front joystick "player" add moviment(0,0,-0.3)
        //
        // The direction is threshold-based. A tiny imprecise movement is
        // ignored; any movement beyond the threshold counts as the direction.
        // ====================================================
        if (rest.rfind("+ moviment", 0) == 0)
        {
            string motionRest = trim(rest.substr(string("+ moviment").size()));
            size_t directionEnd = motionRest.find_first_of(" \t");
            if (directionEnd == string::npos)
            {
                error("This motion binding is missing its direction and source. Expected: key <input> + moviment <direction> <source> \"<object>\" <action>, e.g. key right_mouse + moviment up mouse \"cam\" rotate(-1,0,0).");
                return true;
            }

            string direction = normalizeInputControl(motionRest.substr(0, directionEnd));
            motionRest = trim(motionRest.substr(directionEnd + 1));

            size_t sourceEnd = motionRest.find_first_of(" \t");
            if (sourceEnd == string::npos)
            {
                error("This motion binding is missing its source. Expected one of: mouse, joystick, right_joystick, touch, vr right after the direction, e.g. '+ moviment up mouse'.");
                return true;
            }

            string source = normalizeInputControl(motionRest.substr(0, sourceEnd));
            motionRest = trim(motionRest.substr(sourceEnd + 1));

            if (tfAxisDirectionFromName(direction) == TFAxisDirection::None)
            {
                error("Unknown motion direction. Valid directions are: up, down, left, right, front (or forward), back (or backward), any (or move/movement).");
                return true;
            }

            TFMotionSource motionSource = TFMotionSource::None;
            if (source == "mouse") motionSource = TFMotionSource::Mouse;
            else if (source == "joystick" || source == "gamepad" || source == "left_joystick") motionSource = TFMotionSource::Joystick;
            else if (source == "right_joystick") motionSource = TFMotionSource::Gamepad;
            else if (source == "touch") motionSource = TFMotionSource::Touch;
            else if (source == "vr" || source == "head" || source == "hmd") motionSource = TFMotionSource::VR;
            else
            {
                error("Unknown motion source. Valid sources are: mouse, joystick (or gamepad/left_joystick), right_joystick, touch, vr (or head/hmd).");
                return true;
            }

            float threshold = 0.15f;
            if (motionRest.rfind("threshold ", 0) == 0)
            {
                string thresholdRest = trim(motionRest.substr(10));
                size_t thresholdEnd = thresholdRest.find_first_of(" \t");
                string numberText = thresholdEnd == string::npos ? thresholdRest : thresholdRest.substr(0, thresholdEnd);
                double number = 0.0;
                if (!tryNumber(numberText, number))
                {
                    error("Motion threshold must be a number, e.g. 'threshold 0.2'. This controls how far the mouse/stick must move before the binding counts as active.");
                    return true;
                }
                threshold = max(0.001f, static_cast<float>(fabs(number)));
                motionRest = thresholdEnd == string::npos ? "" : trim(thresholdRest.substr(thresholdEnd + 1));
            }

            if (motionRest.empty() || motionRest.front() != '"')
            {
                error("Combined motion syntax error. Expected: key <input> + moviment <direction> <source> \"<object>\" <action>. Example: key right_mouse + moviment up mouse \"cam\" rotate(-1,0,0). Directions: up, down, left, right, front, back, any. Sources: mouse, joystick, right_joystick, touch, vr.");
                return true;
            }

            size_t closeName = motionRest.find('"', 1);
            if (closeName == string::npos)
            {
                error("Missing closing quote (\") after the object name in this combined key binding. Every object name must be wrapped in matching double quotes.");
                return true;
            }

            TFBinding combined;
            combined.control = control;
            combined.objectName = motionRest.substr(1, closeName - 1);
            combined.hasMotionCondition = true;
            combined.motionSource = motionSource;
            combined.motionDirection = direction;
            combined.motionThreshold = threshold;
            combined.deviceIndex = 0;

            string combinedAction = trim(motionRest.substr(closeName + 1));
            if (combinedAction.empty())
            {
                error("Combined key action cannot be empty. After the quoted object name, a command, 'add moviment (x,y,z)' or 'rotate(x,y,z)' is required.");
                return true;
            }

            if (mouseButtonFromName(normalized) >= 0)
                combined.type = TFInputType::Mouse;
            else if (gamepadButtonFromName(normalized) >= 0)
                combined.type = TFInputType::Gamepad;
            else if (gamepadAxisFromName(normalized) >= 0 || tfIsMotionControlName(normalized))
                combined.type = TFInputType::Axis;
            else
                combined.type = TFInputType::Keyboard;

            if (combinedAction.rfind("add moviment", 0) == 0)
            {
                string args = trim(combinedAction.substr(12));
                bool ignoredDefault = false;
                float dx = 0.0f, dy = 0.0f, dz = 0.0f;
                if (!parseBindingVector(args, dx, dy, dz, ignoredDefault))
                {
                    error("Combined movement must use the format (x,y,z), e.g. add moviment(0,0,-0.3). Parentheses and all three numbers are required.");
                    return true;
                }
                combined.dx = dx;
                combined.dy = dy;
                combined.dz = dz;
                combined.isMovement = true;
                combined.isRotation = false;
            }
            else if (combinedAction.rfind("rotate", 0) == 0)
            {
                string args = trim(combinedAction.substr(6));
                if (args.empty()) args = "()";
                bool usedDefault = false;
                float rx = 0.0f, ry = 0.0f, rz = 0.0f;
                if (!parseBindingVector(args, rx, ry, rz, usedDefault))
                {
                    error("Combined rotation must use the format (x,y,z), e.g. rotate(-1,0,0), or rotate() with no arguments for a small automatic spin.");
                    return true;
                }
                combined.rx = rx;
                combined.ry = ry;
                combined.rz = rz;
                combined.isMovement = false;
                combined.isRotation = true;
            }
            else
            {
                combined.action = combinedAction;
                combined.isMovement = false;
                combined.isRotation = false;
            }

            inputBindings.push_back(combined);
            return true;
        }

        // Natural syntax: key d "name" action...
        if (!rest.empty() && rest.front() == '"')
        {
            size_t closeName = rest.find('"', 1);
            if (closeName == string::npos)
            {
                error("Missing closing quote (\") after the object name in this 'key' command. Every object name must be wrapped in matching double quotes.");
                return true;
            }
 
            string possibleName = rest.substr(1, closeName - 1);
            string afterName = trim(rest.substr(closeName + 1));
 
            if (possibleName.empty())
            {
                error("Key target name cannot be empty. Expected a quoted object name right after the input, e.g. key d \"cube\" ...");
                return true;
            }
 
            if (afterName.rfind("add moviment", 0) == 0)
            {
                string args = trim(afterName.substr(12));
                if (!args.empty() && args.front() == '(')
                {
                    rest = "add moviment \"" + possibleName + "\" " + args;
                }
                else
                {
                    error("Movement must use the format (x,y,z), e.g. add moviment (5,0,-2). Parentheses and all three numbers are required.");
                    return true;
                }
            }
            else if (afterName.rfind("rotate", 0) == 0)
            {
                string args = trim(afterName.substr(6));
                if (args.empty())
                    args = "()";
                rest = "rotate \"" + possibleName + "\" " + args;
            }
            else
            {
                // Any other TigerFlash command stays a normal key action.
                binding.objectName = possibleName;
                binding.action = afterName;
                binding.isMovement = false;
                binding.isRotation = false;
                inputBindings.push_back(binding);
                return true;
            }
        }
 
        if (rest.rfind("add moviment", 0) == 0)
        {
            string movementRest = trim(rest.substr(12));
 
            if (movementRest.empty() || movementRest.front() != '"')
            {
                error("Movement syntax error. Expected: key <input> \"<object>\" add moviment (x,y,z). Example: key d \"cube\" add moviment (5,0,0)");
                return true;
            }
 
            size_t closeName = movementRest.find('"', 1);
            if (closeName == string::npos)
            {
                error("Missing closing quote (\") after the object name in this 'key' command. Every object name must be wrapped in matching double quotes.");
                return true;
            }
 
            string objectName = movementRest.substr(1, closeName - 1);
            string args = trim(movementRest.substr(closeName + 1));
 
            if (args.empty() || args.front() != '(' || args.back() != ')')
            {
                error("Movement must use the format (x,y,z), e.g. add moviment (5,0,-2). Parentheses and all three numbers are required.");
                return true;
            }
 
            bool ignoredDefault = false;
            float dx = 0.0f, dy = 0.0f, dz = 0.0f;
            if (!parseBindingVector(args, dx, dy, dz, ignoredDefault))
            {
                error("Movement coordinates must be numeric: expected three numbers in the form (x,y,z), e.g. (5,0,-2).");
                return true;
            }
 
            bool objectFound = (objectName == "cam");
            for (const TFObject2D &obj : objects2D)
                if (obj.name == objectName) objectFound = true;
            for (const TFObject3D &obj : objects3D)
                if (obj.name == objectName) objectFound = true;
 
            if (!objectFound)
            {
                error("Object not found: " + objectName);
                return true;
            }
 
            binding.objectName = objectName;
            binding.dx = dx;
            binding.dy = dy;
            binding.dz = dz;
            binding.isMovement = true;
            binding.isRotation = false;
        }
        else if (rest.rfind("rotate", 0) == 0)
        {
            string rotationRest = trim(rest.substr(6));
 
            if (rotationRest.empty() || rotationRest.front() != '"')
            {
                error("Rotation syntax error. Expected: key <input> \"<object>\" rotate(x,y,z), or rotate() with no arguments for a small automatic spin. Example: key d \"cube\" rotate(0,1,0)");
                return true;
            }
 
            size_t closeName = rotationRest.find('"', 1);
            if (closeName == string::npos)
            {
                error("Missing closing quote (\") after the object name in this 'key ... rotate' command. Every object name must be wrapped in matching double quotes.");
                return true;
            }
 
            string objectName = rotationRest.substr(1, closeName - 1);
            string args = trim(rotationRest.substr(closeName + 1));
            if (args.empty())
                args = "()";
 
            bool usedDefault = false;
            float rx = 0.0f, ry = 0.0f, rz = 0.0f;
            if (!parseBindingVector(args, rx, ry, rz, usedDefault))
            {
                error("Rotation must use the format (x,y,z) as a per-axis rotation speed, e.g. rotate(0,1,0), or rotate() with no arguments for a small automatic Y spin.");
                return true;
            }
 
            bool objectFound = (objectName == "cam");
            for (const TFObject2D &obj : objects2D)
                if (obj.name == objectName) objectFound = true;
            for (const TFObject3D &obj : objects3D)
                if (obj.name == objectName) objectFound = true;
 
            if (!objectFound)
            {
                error("Object not found: " + objectName);
                return true;
            }
 
            binding.objectName = objectName;
            binding.rx = rx;
            binding.ry = ry;
            binding.rz = rz;
            binding.isMovement = false;
            binding.isRotation = true;
        }
        else
        {
            // Qualquer comando TigerFlash pode ser ligado a uma entrada.
            if (rest.empty())
            {
                error("Key action cannot be empty. After the input (and optional \"object\"), a command, 'add moviment (x,y,z)' or 'rotate(x,y,z)' is required.");
                return true;
            }
 
            binding.action = rest;
            binding.isMovement = false;
            binding.isRotation = false;
        }
 
        inputBindings.push_back(binding);
        return true;
    }
 
    // ====================================================
    // COUNT ALERT GROWL
    // ====================================================
 
    if (line.rfind("count alert growl(", 0) == 0 &&
        line.size() >= 19 &&
        line.back() == ')')
    {
        string content =
            line.substr(18, line.size() - 19);
 
        executegrowl(content, true, true);
        return true;
    }
 
    // ====================================================
    // ALERT COUNT GROWL
    // ====================================================
 
    if (line.rfind("alert count growl(", 0) == 0 &&
        line.size() >= 19 &&
        line.back() == ')')
    {
        string content =
            line.substr(18, line.size() - 19);
 
        executegrowl(content, true, true);
        return true;
    }
 
    // =========================
    // COUNT GROWL
    // =========================
 
    if (line.rfind("count growl(", 0) == 0 &&
        line.size() >= 13 &&
        line.back() == ')')
    {
        string content =
            line.substr(12, line.size() - 13);
 
        executegrowl(content, true, false);
        return true;
    }
 
    // =========================
    // ALERT GROWL
    // =========================
 
    if (line.rfind("alert growl(", 0) == 0 &&
        line.size() >= 13 &&
        line.back() == ')')
    {
        string content =
            line.substr(12, line.size() - 13);
 
        executegrowl(content, false, true);
        return true;
    }
 
    // =========================
    // GROWL
    // =========================
 
    if (line.rfind("growl(", 0) == 0 &&
        line.size() >= 8 &&
        line.back() == ')')
    {
        string content =
            line.substr(6, line.size() - 7);
 
        executegrowl(content, false, false);
        return true;
    }
 
    // ====================================================
    // GUI ANIMATION COMMANDS
    // gui animate "name" move(x,y) speed 120 [loop]
    // gui animate "name" rotate(360) speed 180 [loop]
    // gui animate "name" scale(1.5,1.5) speed 1 [loop]
    // gui animate stop "name"
    //
    // These are real persistent animations: the command starts a state
    // machine and the main loop advances it every frame.
    // ====================================================
    if (line.rfind("gui animate stop ", 0) == 0)
    {
        string name = trim(line.substr(17));
        if (name.size() < 2 || name.front() != '"' || name.back() != '"')
        {
            error("Use: gui animate stop \"name\"");
            return true;
        }
        name = name.substr(1, name.size() - 2);

        for (size_t i = 0; i < guiAnimations.size(); )
        {
            if (guiAnimations[i].guiName == name)
                guiAnimations.erase(guiAnimations.begin() + static_cast<long>(i));
            else
                ++i;
        }
        return true;
    }

    if (line.rfind("gui animate ", 0) == 0)
    {
        string rest = trim(line.substr(12));
        if (rest.empty() || rest.front() != '"')
        {
            error("Use: gui animate \"name\" move(200,0) speed 120 loop");
            return true;
        }

        size_t closeName = rest.find('"', 1);
        if (closeName == string::npos)
        {
            error("Missing closing quote in gui animate.");
            return true;
        }

        string name = rest.substr(1, closeName - 1);
        rest = trim(rest.substr(closeName + 1));
        TFGuiElement *gui = findGuiElement(name);
        if (!gui)
        {
            error("GUI name not found: \"" + name + "\"");
            return true;
        }

        TFGuiAnimation anim;
        anim.guiName = name;
        anim.startX = gui->posX;
        anim.startY = gui->posY;
        anim.startRotation = gui->rotation;
        anim.startScaleX = gui->scaleX;
        anim.startScaleY = gui->scaleY;

        if (rest.rfind("move(", 0) == 0)
        {
            size_t endIndex = string::npos;
            string group = extractParenGroup(rest, 4, endIndex);
            if (endIndex == string::npos)
            {
                error("Use: gui animate \"name\" move(x,y) speed 120");
                return true;
            }

            vector<string> parts = splitMovement(group);
            if (parts.size() != 2)
            {
                error("GUI move animation requires (x,y).");
                return true;
            }

            double dx = 0.0, dy = 0.0;
            if (!tryNumber(replaceVariables(parts[0]), dx) ||
                !tryNumber(replaceVariables(parts[1]), dy))
            {
                error("GUI move animation values must be numeric.");
                return true;
            }

            anim.type = TFGuiAnimationType::Move;
            anim.targetX = gui->posX + static_cast<float>(dx);
            anim.targetY = gui->posY + static_cast<float>(dy);
            rest = trim(rest.substr(endIndex));
        }
        else if (rest.rfind("rotate(", 0) == 0)
        {
            size_t endIndex = string::npos;
            string group = extractParenGroup(rest, 6, endIndex);
            if (endIndex == string::npos)
            {
                error("Use: gui animate \"name\" rotate(360) speed 180");
                return true;
            }

            double angle = 0.0;
            if (!tryNumber(replaceVariables(trim(group)), angle))
            {
                error("GUI rotation animation angle must be numeric.");
                return true;
            }

            anim.type = TFGuiAnimationType::Rotate;
            anim.targetRotation = gui->rotation + static_cast<float>(angle);
            rest = trim(rest.substr(endIndex));
        }
        else if (rest.rfind("scale(", 0) == 0)
        {
            size_t endIndex = string::npos;
            string group = extractParenGroup(rest, 5, endIndex);
            if (endIndex == string::npos)
            {
                error("Use: gui animate \"name\" scale(1.5,1.5) speed 1");
                return true;
            }

            vector<string> parts = splitMovement(group);
            if (parts.size() != 2)
            {
                error("GUI scale animation requires (x,y).");
                return true;
            }

            double sx = 1.0, sy = 1.0;
            if (!tryNumber(replaceVariables(parts[0]), sx) ||
                !tryNumber(replaceVariables(parts[1]), sy))
            {
                error("GUI scale animation values must be numeric.");
                return true;
            }

            anim.type = TFGuiAnimationType::Scale;
            anim.targetScaleX = max(0.01f, static_cast<float>(sx));
            anim.targetScaleY = max(0.01f, static_cast<float>(sy));
            rest = trim(rest.substr(endIndex));
        }
        else
        {
            error("Unknown GUI animation. Use move(x,y), rotate(angle), or scale(x,y).");
            return true;
        }

        // Parse optional speed and loop flags in any order after the action.
        vector<string> tokens;
        string token;
        for (char c : rest)
        {
            if (isspace(static_cast<unsigned char>(c)))
            {
                if (!token.empty())
                {
                    tokens.push_back(token);
                    token.clear();
                }
            }
            else
            {
                token += c;
            }
        }
        if (!token.empty())
            tokens.push_back(token);

        for (size_t t = 0; t < tokens.size(); t++)
        {
            if (tokens[t] == "loop")
            {
                anim.loop = true;
                continue;
            }

            if (tokens[t] == "speed" && t + 1 < tokens.size())
            {
                double speed = 0.0;
                if (!tryNumber(replaceVariables(tokens[t + 1]), speed))
                {
                    error("GUI animation speed must be numeric.");
                    return true;
                }
                anim.speed = max(0.0001f, static_cast<float>(speed));
                ++t;
                continue;
            }
        }

        // Replace any previous animation on this GUI element, so reusing
        // the same name has deterministic behavior rather than stacking
        // multiple transforms on top of one another.
        for (size_t i = 0; i < guiAnimations.size(); )
        {
            if (guiAnimations[i].guiName == name)
                guiAnimations.erase(guiAnimations.begin() + static_cast<long>(i));
            else
                ++i;
        }

        guiAnimations.push_back(anim);
        return true;
    }

    // ====================================================
    // GUI TRANSFORM COMMANDS
    // gui move "name" (x,y)
    // gui rotate "name" angle
    // gui scale "name" (x,y)
    // These operate by GUI name.
    // ====================================================
    if (line.rfind("gui move ", 0) == 0)
    {
        string rest = trim(line.substr(9));
        if (rest.empty() || rest.front() != '"')
        {
            error("Use: gui move \"name\" (x,y)");
            return true;
        }
        size_t closeName = rest.find('"', 1);
        if (closeName == string::npos)
        {
            error("Missing closing quote in gui move.");
            return true;
        }
        string name = rest.substr(1, closeName - 1);
        string vec = trim(rest.substr(closeName + 1));
        if (vec.size() < 5 || vec.front() != '(' || vec.back() != ')')
        {
            error("GUI move requires (x,y).");
            return true;
        }
        vector<string> parts = splitMovement(vec.substr(1, vec.size() - 2));
        if (parts.size() != 2)
        {
            error("GUI move requires (x,y).");
            return true;
        }
        double x, y;
        if (!tryNumber(replaceVariables(parts[0]), x) || !tryNumber(replaceVariables(parts[1]), y))
        {
            error("GUI move values must be numeric.");
            return true;
        }
        TFGuiElement *gui = findGuiElement(name);
        if (!gui)
        {
            error("GUI name not found: \"" + name + "\"");
            return true;
        }
        gui->posX += (float)x;
        gui->posY += (float)y;
        return true;
    }

    if (line.rfind("gui rotate ", 0) == 0)
    {
        string rest = trim(line.substr(11));
        if (rest.empty() || rest.front() != '"')
        {
            error("Use: gui rotate \"name\" angle");
            return true;
        }
        size_t closeName = rest.find('"', 1);
        if (closeName == string::npos)
        {
            error("Missing closing quote in gui rotate.");
            return true;
        }
        string name = rest.substr(1, closeName - 1);
        double angle = 0.0;
        if (!tryNumber(replaceVariables(trim(rest.substr(closeName + 1))), angle))
        {
            error("GUI rotation angle must be numeric.");
            return true;
        }
        TFGuiElement *gui = findGuiElement(name);
        if (!gui)
        {
            error("GUI name not found: \"" + name + "\"");
            return true;
        }
        gui->rotation += (float)angle;
        return true;
    }

    if (line.rfind("gui scale ", 0) == 0)
    {
        string rest = trim(line.substr(10));
        if (rest.empty() || rest.front() != '"')
        {
            error("Use: gui scale \"name\" (x,y)");
            return true;
        }
        size_t closeName = rest.find('"', 1);
        if (closeName == string::npos)
        {
            error("Missing closing quote in gui scale.");
            return true;
        }
        string name = rest.substr(1, closeName - 1);
        string vec = trim(rest.substr(closeName + 1));
        if (vec.size() < 5 || vec.front() != '(' || vec.back() != ')')
        {
            error("GUI scale requires (x,y).");
            return true;
        }
        vector<string> parts = splitMovement(vec.substr(1, vec.size() - 2));
        if (parts.size() != 2)
        {
            error("GUI scale requires (x,y).");
            return true;
        }
        double x, y;
        if (!tryNumber(replaceVariables(parts[0]), x) || !tryNumber(replaceVariables(parts[1]), y))
        {
            error("GUI scale values must be numeric.");
            return true;
        }
        TFGuiElement *gui = findGuiElement(name);
        if (!gui)
        {
            error("GUI name not found: \"" + name + "\"");
            return true;
        }
        gui->scaleX = max(0.01f, gui->scaleX * (float)x);
        gui->scaleY = max(0.01f, gui->scaleY * (float)y);
        return true;
    }

    // ====================================================
    // GUI BUTTON CLICK BINDING
    // mouse click "button" <command>
    // ====================================================
    if (line.rfind("mouse click ", 0) == 0)
    {
        string rest = trim(line.substr(12));
        if (rest.empty() || rest.front() != '"')
        {
            error("Use: mouse click \"button_name\" <command>");
            return true;
        }
        size_t closeName = rest.find('"', 1);
        if (closeName == string::npos)
        {
            error("Missing closing quote in mouse click.");
            return true;
        }
        string name = rest.substr(1, closeName - 1);
        string action = trim(rest.substr(closeName + 1));
        TFGuiElement *gui = findGuiElement(name);
        if (!gui || !gui->isButton)
        {
            error("mouse click requires an existing GUI button: \"" + name + "\"");
            return true;
        }
        guiClickBindings.push_back({name, action});
        return true;
    }

    // =========================
    // GUI (on-screen text / image HUD)
    // =========================
    // gui text(<content>) shape(w,h,d) position(x,y,z) [as "name"]
    // gui image "<path>" shape(w,h,d) position(x,y,z) [as "name"]
// Absolute paths may point to ANY existing Linux directory, including spaces.
    //
    // Draws on top of both "say 2d" and "say 3d" scenes (screen-space,
    // not affected by the camera). "shape" sizes the element in pixels
    // (for text, h is the font size and w auto-shrinks it to fit);
    // "position" is where it appears on screen, with z as draw order
    // (higher z draws on top). "as \"name\"" is optional: reusing the same
    // name updates that element in place instead of stacking a new one,
    // which is how to show a live value (like growl, but on screen).
    if (line.rfind("gui ", 0) == 0)
    {
        string rest = trim(line.substr(4));
        TFGuiElement element;

        if (rest.rfind("button(", 0) == 0)
        {
            element.isButton = true;
            element.isImage = false;

            size_t endIndex;
            string content = extractParenGroup(rest, 6, endIndex);
            if (endIndex == string::npos)
            {
                error("Missing closing ')' in 'gui button(...)'. Example: gui button(PLAY) shape(160,50,0) position(20,20,0) as \"play\"");
                return true;
            }
            element.content = content;
            rest = trim(rest.substr(endIndex));
        }
        else if (rest.rfind("text(", 0) == 0)
        {
            element.isImage = false;

            size_t endIndex;
            string content = extractParenGroup(rest, 4, endIndex);

            if (endIndex == string::npos)
            {
                error("Missing closing ')' in 'gui text(...)'. Example: gui text(oi) shape(20,20,0) position(10,10,0)");
                return true;
            }

            element.content = content;
            rest = trim(rest.substr(endIndex));
        }
        else if (rest.rfind("image ", 0) == 0)
        {
            element.isImage = true;

            string afterImage = trim(rest.substr(6));

            if (afterImage.empty() || afterImage.front() != '"')
            {
                error("Invalid 'gui image' syntax: expected a quoted file path right after 'image'. Example: gui image \"assets/logo.png\" shape(64,64,0) position(10,10,0)");
                return true;
            }

            size_t closeQuote = afterImage.find('"', 1);

            if (closeQuote == string::npos)
            {
                error("Missing closing quote (\") after the file path in 'gui image'.");
                return true;
            }

            element.content = afterImage.substr(1, closeQuote - 1);
            rest = trim(afterImage.substr(closeQuote + 1));
        }
        else
        {
            error("Invalid 'gui' syntax. Use: gui text(<content>) ..., gui image \"<path>\" ..., or gui button(<content>) shape(w,h,d) position(x,y,z) as \"name\"");
            return true;
        }

        float sw = 0.0f, sh = 0.0f, sd = 0.0f;
        float px = 0.0f, py = 0.0f, pz = 0.0f;
        string vecError;

        bool hasShape = tfFindNamedVector(rest, "shape", sw, sh, sd, vecError);

        if (hasShape && !vecError.empty())
        {
            error(vecError);
            return true;
        }

        if (!hasShape)
        {
            error("'gui' requires a shape(w,h,d) to size it. Example: shape(20,20,0)");
            return true;
        }

        bool hasPosition = tfFindNamedVector(rest, "position", px, py, pz, vecError);

        if (hasPosition && !vecError.empty())
        {
            error(vecError);
            return true;
        }

        if (!hasPosition)
        {
            error("'gui' requires a position(x,y,z) to place it on screen. Example: position(10,10,0)");
            return true;
        }

        element.shapeW = sw;
        element.shapeH = sh;
        element.shapeD = sd;
        element.posX = px;
        element.posY = py;
        element.posZ = pz;

        string name;
        size_t asPos = rest.find(" as ");

        if (asPos != string::npos)
        {
            string afterAs = trim(rest.substr(asPos + 4));

            if (afterAs.size() < 2 || afterAs.front() != '"')
            {
                error("Invalid 'as \"name\"' in 'gui'. Example: as \"score_hud\"");
                return true;
            }

            size_t closeQuote = afterAs.find('"', 1);

            if (closeQuote == string::npos)
            {
                error("Missing closing quote (\") after 'as' in 'gui'.");
                return true;
            }

            name = afterAs.substr(1, closeQuote - 1);
        }

        if (name.empty())
            name = "gui_" + to_string(guiElements.size() + 1);

        element.name = name;

        bool updatedExisting = false;

        for (TFGuiElement &existing : guiElements)
        {
            if (existing.name != name)
                continue;

            bool contentChanged = existing.isImage != element.isImage ||
                                   existing.content != element.content;

            existing.isImage = element.isImage;
            existing.isButton = element.isButton;
            existing.content = element.content;
            existing.shapeW = element.shapeW;
            existing.shapeH = element.shapeH;
            existing.shapeD = element.shapeD;
            existing.posX = element.posX;
            existing.posY = element.posY;
            existing.posZ = element.posZ;
            if (!existing.physicsEnabled)
            {
                existing.rotation = element.rotation;
                existing.scaleX = element.scaleX;
                existing.scaleY = element.scaleY;
            }

            // Only force a texture reload if the image path actually
            // changed, so updating a HUD image's position every frame
            // doesn't reload it from disk every frame too.
            if (contentChanged)
            {
                existing.textureLoadAttempted = false;
                existing.textureLoadFailed = false;
            }

            updatedExisting = true;
            break;
        }

        if (!updatedExisting)
            guiElements.push_back(element);

        sceneHasGui = true;
        return true;
    }
 
    // Loaded libraries get the last chance to handle a custom command.
    if (tfRunLibraryCommandHooks(line))
        return true;

    error("Unknown TigerFlash command: \"" + line + "\". Check the spelling of the command, and make sure it starts at the beginning of the line (only indentation is allowed before it).");
    return true;
}
 
// ============================================================
// SECONDARY SCRIPT FLOW
// ============================================================
// `if` and `repeat` blocks are cooperative background tasks.
// They never execute an infinite loop on the main interpreter thread.
// The graphics/physics window stays alive while these tasks are polled
// once per frame with a strict command budget.
enum class TFBackgroundTaskType
{
    Conditional,
    Repeat,
    DelayedCommand
};

struct TFBackgroundTask
{
    TFBackgroundTaskType type = TFBackgroundTaskType::Conditional;
    string condition;
    vector<string> body;
    vector<string> elseBody;
    long long remaining = -1; // -1 = infinite
    bool lastCondition = false;

    // Optional repeat interval. Zero preserves the original per-frame repeat.
    // This is used only by the new optional `wait(...) repeat` syntax.
    float waitSeconds = 0.0f;
    float waitRemaining = 0.0f;
};

vector<TFBackgroundTask> tfBackgroundTasks;
// Timeline cursor for standalone wait() commands. Multiple waits in the
// top-level script are cumulative so: wait(0.1), delete, wait(0.1), add
// produces a visible 0.1s gap between delete and add.
float tfDelayedCommandCursor = 0.0f;

static constexpr int TF_BACKGROUND_COMMAND_BUDGET = 2048;

vector<string> tfExtractBlockLines(const vector<string> &script,
                                   size_t bodyStart,
                                   size_t bodyEnd)
{
    vector<string> body;
    if (bodyStart >= script.size() || bodyStart >= bodyEnd)
        return body;

    int bodyIndent = indentationLevel(script[bodyStart]);
    body.reserve(bodyEnd - bodyStart);

    for (size_t i = bodyStart; i < bodyEnd; ++i)
    {
        string line = script[i];
        if (trim(line).empty())
            continue;

        int indent = indentationLevel(line);
        if (indent < bodyIndent)
            continue;

        // Remove the block's common indentation. Nested indentation is kept.
            size_t p = 0;
        int removed = 0;
        while (p < line.size() && removed < bodyIndent)
        {
            if (line[p] == ' ')
            {
                ++p;
                ++removed;
            }
            else if (line[p] == '\t')
            {
                ++p;
                ++removed;
            }
            else
            {
                break;
            }
        }
        // Normalize each background command once at registration time.
        // The old scheduler trimmed and lower-cased every command on every
        // frame, which was unnecessary work for persistent `if`/`repeat`
        // blocks. Quoted text remains untouched by toLowerOutsideQuotes().
        body.push_back(toLowerOutsideQuotes(trim(line.substr(p))));
    }

    return body;
}

void tfRegisterBackgroundConditional(const string &condition,
                                     const vector<string> &body,
                                     const vector<string> &elseBody)
{
    TFBackgroundTask task;
    task.type = TFBackgroundTaskType::Conditional;
    task.condition = condition;
    task.body = body;
    task.elseBody = elseBody;
    task.remaining = -1;
    tfBackgroundTasks.push_back(task);
}

void tfRegisterBackgroundRepeat(long long count,
                                const vector<string> &body,
                                float waitSeconds = 0.0f)
{
    TFBackgroundTask task;
    task.type = TFBackgroundTaskType::Repeat;
    task.remaining = count;
    task.body = body;
    task.waitSeconds = max(0.0f, waitSeconds);
    task.waitRemaining = task.waitSeconds;
    tfBackgroundTasks.push_back(task);
}

void tfRegisterDelayedCommand(const string &command, float waitSeconds)
{
    TFBackgroundTask task;
    task.type = TFBackgroundTaskType::DelayedCommand;
    task.body = {command};

    const float delay = max(0.0f, waitSeconds);
    tfDelayedCommandCursor += delay;
    task.waitSeconds = tfDelayedCommandCursor;
    task.waitRemaining = tfDelayedCommandCursor;
    tfBackgroundTasks.push_back(task);
}

// Execute commands inside a background block without creating a
// synchronous infinite loop. Nested `if` blocks are evaluated immediately
// inside the current frame, while nested `repeat` blocks remain protected
// from recursion/hangs and are skipped here. Top-level repeats are handled
// by the cooperative scheduler below.
size_t tfFindBackgroundVectorBlockEnd(const vector<string> &body,
                                      size_t start,
                                      int baseIndent)
{
    size_t i = start;

    while (i < body.size())
    {
        if (body[i].empty())
        {
            ++i;
            continue;
        }

        int indent = indentationLevel(body[i]);
        if (indent <= baseIndent)
            break;

        ++i;
    }

    return i;
}

size_t tfExecuteBackgroundBodyRange(const vector<string> &body,
                                     size_t start,
                                     size_t end,
                                     int baseIndent,
                                     int &commandBudget)
{
    size_t i = start;

    while (i < end && commandBudget > 0)
    {
        const string &raw = body[i];
        if (raw.empty())
        {
            ++i;
            continue;
        }

        int indent = indentationLevel(raw);
        if (indent < baseIndent)
            return i;

        string line = raw.substr(static_cast<size_t>(indent));

        if (line == "else" && indent == baseIndent)
            return i;

        // A nested if is handled recursively, regardless of its indentation.
        // This fixes the old behavior where every nested `if` was silently
        // ignored by the background scheduler.
        if (line.rfind("if ", 0) == 0)
        {
            const int controlIndent = indent;
            string condition = trim(line.substr(3));

            size_t childStart = i + 1;
            while (childStart < end && body[childStart].empty())
                ++childStart;

            if (childStart >= end ||
                indentationLevel(body[childStart]) <= controlIndent)
            {
                ++i;
                continue;
            }

            int childIndent = indentationLevel(body[childStart]);
            size_t childEnd = tfFindBackgroundVectorBlockEnd(
                body, childStart, controlIndent);

            size_t afterIf = childEnd;
            size_t elseStart = end;
            size_t elseEnd = end;

            if (afterIf < end &&
                indentationLevel(body[afterIf]) == controlIndent &&
                body[afterIf] == "else")
            {
                elseStart = afterIf + 1;
                while (elseStart < end && body[elseStart].empty())
                    ++elseStart;

                if (elseStart < end &&
                    indentationLevel(body[elseStart]) > controlIndent)
                {
                    elseEnd = tfFindBackgroundVectorBlockEnd(
                        body, elseStart, controlIndent);
                }
                else
                {
                    elseStart = end;
                    elseEnd = end;
                }
            }

            if (evaluateCondition(condition))
            {
                tfExecuteBackgroundBodyRange(
                    body, childStart, childEnd, childIndent, commandBudget);
            }
            else if (elseStart < end)
            {
                int elseIndent = indentationLevel(body[elseStart]);
                tfExecuteBackgroundBodyRange(
                    body, elseStart, elseEnd, elseIndent, commandBudget);
            }

            if (tfControlSignal != TFControlSignal::None)
                return (elseStart < end) ? max(childEnd, elseEnd) : childEnd;

            i = (elseStart < end) ? max(childEnd, elseEnd) : childEnd;
            continue;
        }

        // Nested repeats are still protected from synchronous recursion. The
        // top-level scheduler is responsible for cooperative repeat tasks.
        if (line == "repeat" ||
            (line.rfind("repeat ", 0) == 0 &&
             line.find('=') == string::npos))
        {
            const int controlIndent = indent;
            size_t childStart = i + 1;
            while (childStart < end && body[childStart].empty())
                ++childStart;

            if (childStart < end &&
                indentationLevel(body[childStart]) > controlIndent)
            {
                i = tfFindBackgroundVectorBlockEnd(
                    body, childStart, controlIndent);
            }
            else
            {
                ++i;
            }
            continue;
        }

        // Ordinary command: remove only the command's indentation. In normal
        // background bodies indent is usually zero; nested blocks arrive here
        // only for ordinary statements beneath a handled nested control block.
        executeLine(line);
        --commandBudget;
        ++i;

        if (tfControlSignal != TFControlSignal::None)
            return i;
    }

    return i;
}

void tfExecuteBackgroundBody(const vector<string> &body,
                             int &commandBudget)
{
    tfExecuteBackgroundBodyRange(
        body, 0, body.size(), 0, commandBudget);
}

void tfUpdateBackgroundScript(float)
{
    if (tfBackgroundTasks.empty())
        return;

    int commandBudget = TF_BACKGROUND_COMMAND_BUDGET;

    for (size_t i = 0; i < tfBackgroundTasks.size() && commandBudget > 0; )
    {
        TFBackgroundTask &task = tfBackgroundTasks[i];

        if (task.type == TFBackgroundTaskType::Conditional)
        {
            bool active = evaluateCondition(task.condition);
            task.lastCondition = active;

            if (active)
                tfExecuteBackgroundBody(task.body, commandBudget);
            else if (!task.elseBody.empty())
                tfExecuteBackgroundBody(task.elseBody, commandBudget);

            ++i;
            continue;
        }

        if (task.type == TFBackgroundTaskType::DelayedCommand)
        {
            task.waitRemaining -= GetFrameTime();
            if (task.waitRemaining > 0.0f)
            {
                ++i;
                continue;
            }

            if (!task.body.empty())
            {
                executeLine(task.body.front());
                --commandBudget;
            }

            tfBackgroundTasks.erase(
                tfBackgroundTasks.begin() + static_cast<long>(i));
            continue;
        }

        // Cooperative repeat. With no wait(), behavior remains exactly the
        // old one-iteration-per-frame behavior. When wait(...) was supplied,
        // only the interval between iterations changes.
        if (task.remaining == 0)
        {
            tfBackgroundTasks.erase(tfBackgroundTasks.begin() + static_cast<long>(i));
            continue;
        }

        if (task.waitSeconds > 0.0f)
        {
            task.waitRemaining -= GetFrameTime();
            if (task.waitRemaining > 0.0f)
            {
                ++i;
                continue;
            }

            // Preserve the remaining fraction instead of accumulating frame
            // drift when the timer crosses zero.
            task.waitRemaining = task.waitSeconds;
        }

        ++tfLoopDepth;
        tfControlSignal = TFControlSignal::None;
        tfExecuteBackgroundBody(task.body, commandBudget);
        --tfLoopDepth;

        if (tfControlSignal == TFControlSignal::Break)
        {
            tfControlSignal = TFControlSignal::None;
            tfBackgroundTasks.erase(tfBackgroundTasks.begin() + static_cast<long>(i));
            continue;
        }

        if (tfControlSignal == TFControlSignal::Continue)
        {
            tfControlSignal = TFControlSignal::None;
            if (task.remaining > 0)
                --task.remaining;
            ++i;
            continue;
        }

        if (task.remaining > 0)
            --task.remaining;

        if (task.remaining == 0)
        {
            tfBackgroundTasks.erase(tfBackgroundTasks.begin() + static_cast<long>(i));
            continue;
        }

        ++i;
    }
}

// ============================================================
// EXECUTE BLOCK
// ============================================================

size_t findBlockEnd(const vector<string> &script,
                    size_t start,
                    int blockIndent)
{
    size_t i = start;

    while (i < script.size())
    {
        if (trim(script[i]).empty())
        {
            ++i;
            continue;
        }

        int indent = indentationLevel(script[i]);
        if (indent <= blockIndent)
            break;

        ++i;
    }

    return i;
}

// Used only for ordinary commands inside a background body. It deliberately
// does not contain the old synchronous infinite-loop behavior.
size_t executeBlockImmediate(const vector<string> &script,
                             size_t start,
                             int baseIndent)
{
    size_t i = start;

    while (i < script.size())
    {
        string raw = script[i];
        if (trim(raw).empty())
        {
            ++i;
            continue;
        }

        int indent = indentationLevel(raw);
        if (indent < baseIndent)
            return i;

        string line = toLowerOutsideQuotes(trim(raw));
        if (indent == baseIndent && line == "else")
            return i;

        if (indent == baseIndent &&
            (line == "repeat" ||
             (line.rfind("repeat ", 0) == 0 && line.find('=') == string::npos) ||
             line.rfind("if ", 0) == 0))
        {
            // Nested control flow is handled cooperatively by the scheduler.
            // Never execute it synchronously here.
            ++i;
            size_t bodyStart = i;
            while (bodyStart < script.size() && trim(script[bodyStart]).empty())
                ++bodyStart;
            if (bodyStart < script.size() && indentationLevel(script[bodyStart]) > baseIndent)
            {
                size_t bodyEnd = findBlockEnd(script, bodyStart, baseIndent);
                i = bodyEnd;
                if (i < script.size() &&
                    toLowerOutsideQuotes(trim(script[i])) == "else" &&
                    indentationLevel(script[i]) == baseIndent)
                {
                    ++i;
                    while (i < script.size() && trim(script[i]).empty())
                        ++i;
                    if (i < script.size() && indentationLevel(script[i]) > baseIndent)
                        i = findBlockEnd(script, i, baseIndent);
                }
            }
            continue;
        }

        executeLine(line);
        ++i;
    }

    return i;
}

size_t executeBlock(const vector<string> &script,
                    size_t start,
                    int baseIndent)
{
    size_t i = start;

    while (i < script.size())
    {
        string raw = script[i];

        if (trim(raw).empty())
        {
            ++i;
            continue;
        }

        int indent = indentationLevel(raw);
        if (indent < baseIndent)
            return i;

        string line = toLowerOutsideQuotes(trim(raw));

        if (indent == baseIndent && line == "else")
            return i;

        // ------------------------------------------------------------
        // FUNCTION DEFINITION (OPTIONAL)
        // ------------------------------------------------------------
        if (indent == baseIndent && line.rfind("function ", 0) == 0)
        {
            string functionName;
            vector<string> parameters;
            if (!tfParseFunctionHeader(line, functionName, parameters))
            {
                error("Invalid function syntax. Use: function name(a,b)");
                ++i;
                continue;
            }

            size_t bodyStart = i + 1;
            while (bodyStart < script.size() && trim(script[bodyStart]).empty())
                ++bodyStart;

            if (bodyStart >= script.size() ||
                indentationLevel(script[bodyStart]) <= baseIndent)
            {
                error("function requires an indented block right after it.");
                ++i;
                continue;
            }

            size_t bodyEnd = findBlockEnd(script, bodyStart, baseIndent);
            vector<string> body = tfExtractBlockLines(script, bodyStart, bodyEnd);
            if (body.empty())
            {
                error("function requires an indented block containing at least one command.");
            }
            else
            {
                TFFunction function;
                function.parameters = parameters;
                function.body = body;
                tfFunctions[functionName] = function;
            }

            i = bodyEnd;
            continue;
        }

        if (tfControlSignal != TFControlSignal::None)
            return i;

        // ------------------------------------------------------------
        // IF
        // ------------------------------------------------------------
        // Normal conditions are evaluated once, in source order.
        // Only the two live physics sensors remain frame-polled:
        //     if colision "a" to "b"
        //     if gravity "object"
        // This prevents a normal `if` from silently becoming a 60 FPS loop.
        if (indent == baseIndent && line.rfind("if ", 0) == 0)
        {
            string condition = trim(line.substr(3));
            size_t bodyStart = i + 1;
            while (bodyStart < script.size() && trim(script[bodyStart]).empty())
                ++bodyStart;

            if (bodyStart >= script.size() ||
                indentationLevel(script[bodyStart]) <= baseIndent)
            {
                error("'if' requires an indented block right after it.");
                ++i;
                continue;
            }

            size_t bodyEnd = findBlockEnd(script, bodyStart, baseIndent);
            vector<string> body = tfExtractBlockLines(script, bodyStart, bodyEnd);
            vector<string> elseBody;
            size_t afterIf = bodyEnd;

            if (afterIf < script.size() &&
                indentationLevel(script[afterIf]) == baseIndent &&
                toLowerOutsideQuotes(trim(script[afterIf])) == "else")
            {
                size_t elseStart = afterIf + 1;
                while (elseStart < script.size() && trim(script[elseStart]).empty())
                    ++elseStart;

                if (elseStart < script.size() &&
                    indentationLevel(script[elseStart]) > baseIndent)
                {
                    size_t elseEnd = findBlockEnd(script, elseStart, baseIndent);
                    elseBody = tfExtractBlockLines(script, elseStart, elseEnd);
                    afterIf = elseEnd;
                }
                else
                {
                    error("'else' requires an indented block right after it.");
                    afterIf = afterIf + 1;
                }
            }

            if (body.empty())
            {
                error("'if' requires an indented block containing at least one command.");
            }
            else
            {
                const bool isLiveSensor =
                    condition.rfind("colision ", 0) == 0 ||
                    condition.rfind("gravity ", 0) == 0;

                if (isLiveSensor)
                {
                    // Physics sensors must be re-evaluated while the game runs.
                    tfRegisterBackgroundConditional(condition, body, elseBody);
                }
                else
                {
                    // Ordinary `if` executes exactly once, here and now,
                    // after all preceding source lines have taken effect.
                    const bool active = evaluateCondition(condition);
                    if (active)
                    {
                        executeBlock(body, 0, 0);
                    }
                    else if (!elseBody.empty())
                    {
                        executeBlock(elseBody, 0, 0);
                    }
                }
            }

            if (tfControlSignal != TFControlSignal::None)
                return afterIf;

            i = afterIf;
            continue;
        }

        // ------------------------------------------------------------
        // REPEAT N / REPEAT: cooperative background task.
        // One iteration is executed per frame, so even infinite repeat
        // can never stop the graphics window from running.
        // ------------------------------------------------------------
        float compactRepeatWait = 0.0f;
        string compactRepeatLine;
        bool hasCompactWaitRepeat =
            parseWaitRepeatCommand(line, compactRepeatWait, compactRepeatLine);

        if (indent == baseIndent &&
            (line == "repeat" ||
             (line.rfind("repeat ", 0) == 0 && line.find('=') == string::npos) ||
             hasCompactWaitRepeat))
        {
            long long amount = -1;
            float repeatWaitSeconds = compactRepeatWait;

            // New optional forms:
            //     wait(1)
            //     repeat
            //         ...
            // or: wait(1) repeat
            //     ...
            // Existing repeat syntax is unchanged when no wait() is supplied.
            if (!hasCompactWaitRepeat && i > 0)
            {
                size_t prev = i;
                while (prev > 0)
                {
                    --prev;
                    string previousLine = trim(script[prev]);
                    if (previousLine.empty())
                        continue;

                    float parsedWait = 0.0f;
                    if (parseWaitCommand(toLowerOutsideQuotes(previousLine), parsedWait))
                        repeatWaitSeconds = parsedWait;
                    break;
                }
            }

            string repeatCommandLine = hasCompactWaitRepeat
                ? compactRepeatLine
                : line;

            if (repeatCommandLine != "repeat")
            {
                string amountText = trim(repeatCommandLine.substr(7));
                double parsed = 0.0;
                if (!tryNumber(amountText, parsed) ||
                    parsed < 0.0 ||
                    parsed != static_cast<long long>(parsed))
                {
                    error("'repeat' requires a non-negative whole number or no number for infinite background repetition.");
                    ++i;
                    continue;
                }
                amount = static_cast<long long>(parsed);
            }

            size_t bodyStart = i + 1;
            while (bodyStart < script.size() && trim(script[bodyStart]).empty())
                ++bodyStart;

            if (bodyStart >= script.size() ||
                indentationLevel(script[bodyStart]) <= baseIndent)
            {
                error("'repeat' requires an indented block right after it.");
                ++i;
                continue;
            }

            size_t bodyEnd = findBlockEnd(script, bodyStart, baseIndent);
            vector<string> body = tfExtractBlockLines(script, bodyStart, bodyEnd);

            if (body.empty())
                error("'repeat' requires an indented block containing at least one command.");
            else
                tfRegisterBackgroundRepeat(amount, body, repeatWaitSeconds);

            i = bodyEnd;
            continue;
        }

        // Ordinary setup command executes once before the window opens.
        if (tfControlSignal != TFControlSignal::None)
            return i;

        if (indent == baseIndent)
        {
            float waitSeconds = 0.0f;
            if (parseWaitCommand(line, waitSeconds))
            {
                // `wait(...)` before repeat remains the optional repeat
                // interval and is consumed by the repeat parser above.
                // When the next top-level line is an ordinary command, the
                // wait is now a real cooperative delay before that command.
                size_t next = i + 1;
                while (next < script.size() && trim(script[next]).empty())
                    ++next;

                if (next < script.size())
                {
                    string nextLine = toLowerOutsideQuotes(trim(script[next]));
                    if (nextLine == "repeat" ||
                        (nextLine.rfind("repeat ", 0) == 0 &&
                         nextLine.find('=') == string::npos))
                    {
                        ++i;
                        continue;
                    }

                    // For any other top-level command, delay that command
                    // cooperatively so the graphics window stays responsive.
                    tfRegisterDelayedCommand(script[next], waitSeconds);
                    i = next;
                    continue;
                }

                ++i;
                continue;
            }

            executeLine(line);
            ++i;
            continue;
        }

        error("Unexpected indentation: this line is indented further than expected.");
        ++i;
    }

    return i;
}

// ============================================================
// MAIN
// ============================================================
 
// ============================================================
// TIGERFLASH COMMENTS
// ============================================================
//
// Comentarios usam aspas duplas e podem ocupar quantas linhas
// quiser. O comentario começa quando o primeiro caractere,
// ignorando espacos/tabs, for " e termina na proxima ".
//
// Exemplos:
//
// "isto e um comentario"
// "
// comentario de varias linhas
// pode ter qualquer texto aqui
// stop
// growl(123)
// "
//
// Conteudo de comentario nunca entra no script para execucao.
//
// Importante: aspas usadas dentro de comandos normais, como:
// growl("ola")
// see nome = "Victor"
// continuam funcionando normalmente, porque o comentario precisa
// começar no inicio da linha (depois de espacos/tabs).
// ============================================================
 
bool isCommentLineStart(const string &line)
{
    string value = trim(line);
    return !value.empty() && value.front() == '"';
}

// A quoted object name can now appear at the beginning of a line for the
// optional direct-transform syntax:
//     "player" add moviment (1,0,0)
//     "player" rotate(0,1,0)
//     "player" shape(2,2,2)
// Those lines must NOT be swallowed by the quotation-based comment system.
bool isDirectObjectTransformLine(const string &line)
{
    size_t leading = line.find_first_not_of(" \t");
    if (leading == string::npos || line[leading] != '"')
        return false;

    size_t closingQuote = line.find('"', leading + 1);
    if (closingQuote == string::npos)
        return false;

    string action = toLowerOutsideQuotes(trim(line.substr(closingQuote + 1)));
    return action.rfind("add moviment", 0) == 0 ||
           action.rfind("rotate", 0) == 0 ||
           action.rfind("shape", 0) == 0;
}

// Comments start only at the first non-space character. A one-line comment
// can now be followed by another command on the same line, and multi-line
// comments still work. Quoted strings inside normal commands remain intact.
bool consumeCommentLine(const string &line,
                         bool &insideComment,
                         string &remaining)
{
    remaining.clear();
    size_t pos = 0;

    if (insideComment)
    {
        size_t closingQuote = line.find('"');
        if (closingQuote == string::npos)
            return true;

        insideComment = false;
        pos = closingQuote + 1;
    }
    else
    {
        size_t leading = line.find_first_not_of(" \t");
        if (leading == string::npos || line[leading] != '"')
            return false;

        // IMPORTANT: a direct object transform also begins with a quoted
        // object name. Let it reach executeLine() instead of treating it as
        // a comment. The old quotation-comment syntax remains unchanged for
        // every other line that starts with a quote.
        if (isDirectObjectTransformLine(line))
            return false;

        size_t closingQuote = line.find('"', leading + 1);
        if (closingQuote == string::npos)
        {
            insideComment = true;
            return true;
        }

        pos = closingQuote + 1;
    }

    remaining = trim(line.substr(pos));
    return true;
}


// ============================================================
// EXECUTE BLOCK BUILDER
// ============================================================

static string tfCppEscapeString(const string &value)
{
    string out;
    out.reserve(value.size() + 16);

    for (unsigned char c : value)
    {
        switch (c)
        {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\0': out += "\\0"; break;
            default:
                if (c < 32)
                {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\x%02X", c);
                    out += buf;
                }
                else
                {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }

    return out;
}

static bool tfParseLibraryDirectiveForBuild(const string &line,
                                            string &name)
{
    string text = trim(line);
    string lower = toLowerOutsideQuotes(text);

    if (lower.rfind("library ", 0) != 0)
        return false;

    name = trim(text.substr(8));

    if (name.size() >= 2 && name.front() == '<' && name.back() == '>')
        name = trim(name.substr(1, name.size() - 2));

    if (name.size() >= 2 && name.front() == '"' && name.back() == '"')
        name = trim(name.substr(1, name.size() - 2));

    return validName(name);
}

static string tfBuildLibraryIdentifier(const string &name)
{
    string id = "tfLibraryInit_";
    for (unsigned char c : name)
    {
        if (isalnum(c) || c == '_')
            id.push_back(static_cast<char>(c));
        else
            id.push_back('_');
    }
    return id;
}

static void tfRunEmbeddedBackgroundTasks()
{
    if (tfBackgroundTasks.empty())
        return;

    int commandBudget = TF_BACKGROUND_COMMAND_BUDGET * 16;

    for (size_t i = 0;
         i < tfBackgroundTasks.size() && commandBudget > 0; )
    {
        TFBackgroundTask &task = tfBackgroundTasks[i];

        if (task.type == TFBackgroundTaskType::Conditional)
        {
            bool active = evaluateCondition(task.condition);
            task.lastCondition = active;

            if (active)
                tfExecuteBackgroundBody(task.body, commandBudget);
            else if (!task.elseBody.empty())
                tfExecuteBackgroundBody(task.elseBody, commandBudget);

            tfBackgroundTasks.erase(
                tfBackgroundTasks.begin() + static_cast<long>(i));
            continue;
        }

        if (task.remaining == -1)
        {
            tfBackgroundTasks.erase(
                tfBackgroundTasks.begin() + static_cast<long>(i));
            continue;
        }

        while (task.remaining > 0 && commandBudget > 0)
        {
            tfExecuteBackgroundBody(task.body, commandBudget);
            --task.remaining;
        }

        tfBackgroundTasks.erase(
            tfBackgroundTasks.begin() + static_cast<long>(i));
    }

    if (!tfBackgroundTasks.empty())
    {
        error("Embedded executable command budget exhausted before all background commands finished.");
        tfBackgroundTasks.clear();
    }
}

// ------------------------------------------------------------
// Build diagnostics helpers.
// These exist so a failed `execute block()` always tells the user WHY it
// failed (missing g++, missing static raylib, missing system libs) instead
// of silently leaving only the generated .cpp behind, which is what used
// to happen: system() calls inherit whatever stdout/stderr the IDE gives
// the process, and many IDEs/launchers swallow that output entirely.
// ------------------------------------------------------------
static bool tfCommandExists(const string &command)
{
    string probe = command + " --version > /dev/null 2>&1";
    return system(probe.c_str()) == 0;
}

static bool tfFindRaylibStaticLib(filesystem::path &outPath)
{
    static const char *candidates[] = {
        "/usr/local/lib/libraylib.a",
        "/usr/lib/libraylib.a",
        "/usr/lib/x86_64-linux-gnu/libraylib.a",
        "/usr/local/lib/x86_64-linux-gnu/libraylib.a",
        "/usr/lib/aarch64-linux-gnu/libraylib.a",
    };

    for (const char *candidate : candidates)
    {
        std::error_code ec;
        if (filesystem::exists(candidate, ec) &&
            filesystem::is_regular_file(candidate, ec))
        {
            outPath = candidate;
            return true;
        }
    }
    return false;
}

static string tfReadLogFile(const filesystem::path &path)
{
    ifstream in(path);
    if (!in)
        return string();
    return string((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
}

static int tfRunBuildCommandWithProgress(const string &command,
                                        const string &phase,
                                        double estimatedSeconds)
{
    estimatedSeconds = max(1.0, estimatedSeconds);

    atomic<bool> finished(false);
    const auto startTime = chrono::steady_clock::now();

    thread progressThread([&]()
    {
        while (!finished.load())
        {
            const double elapsed = chrono::duration<double>(
                chrono::steady_clock::now() - startTime).count();

            // This is intentionally an estimate, not a fake exact compiler
            // percentage. It approaches 95% while the real compiler works and
            // becomes 100% only when the command actually finishes.
            const double curve = 1.0 - exp(-elapsed / estimatedSeconds);
            int percent = 5 + static_cast<int>(curve * 90.0);
            percent = max(5, min(95, percent));

            cout << "\rTigerFlash: Carregando executavel... "
                 << setw(3) << percent << "% | "
                 << phase << " | "
                 << fixed << setprecision(1) << elapsed << "s | estimativa ~"
                 << setprecision(1) << estimatedSeconds << "s"
                 << flush;

            this_thread::sleep_for(chrono::milliseconds(250));
        }
    });

    const int result = system(command.c_str());

    finished.store(true);
    if (progressThread.joinable())
        progressThread.join();

    cout << "\r" << string(180, ' ') << "\r" << flush;
    return result;
}

static double tfEstimateBuildSeconds(size_t generatedCppBytes,
                                     size_t libraryCount)
{
    // Rough first-build estimate based on the generated source size and the
    // number of external TigerFlash libraries that must be compiled.
    double estimate = 1.5 +
                      static_cast<double>(generatedCppBytes) / (256.0 * 1024.0) +
                      static_cast<double>(libraryCount) * 1.5;
    return max(2.0, min(60.0, estimate));
}

static bool tfBuildExecutableBlock(const vector<string> &sourceScript,
                                   const string &scriptPath)
{
    if (sourceScript.empty())
    {
        error("execute block() requires TigerFlash code before the previous stop.");
        return false;
    }

    filesystem::path inputPath(scriptPath);
    if (inputPath.empty())
    {
        error("execute block() requires a .tf file on disk; stdin scripts cannot be built yet.");
        return false;
    }

    // ------------------------------------------------------------
    // IMPORTANT:
    // Do NOT copy the whole IDE/runtime into the generated .cpp.
    // The old builder read the IDE source into runtimeText and then tried
    // to cut it at runTigerFlashInput(). That made the generated source
    // fragile (and could cut inside the builder's own string literal).
    //
    // The native program is now generated as a small translation unit that
    // includes the already-existing TigerFlash runtime source. g++ compiles
    // that included source as part of this translation unit, so the resulting
    // executable does not depend on the IDE while running.
    // ------------------------------------------------------------

    filesystem::path runtimeSource;
    try
    {
        const filesystem::path compiledSource(__FILE__);

        auto tryFile = [&](const filesystem::path &candidate) -> bool
        {
            try
            {
                if (!candidate.empty() &&
                    filesystem::exists(candidate) &&
                    filesystem::is_regular_file(candidate))
                {
                    runtimeSource = filesystem::absolute(candidate);
                    return true;
                }
            }
            catch (...)
            {
            }
            return false;
        };

        if (compiledSource.is_absolute())
            tryFile(compiledSource);

        if (runtimeSource.empty() && !tfLauncherExecutablePath.empty())
        {
            filesystem::path exePath(tfLauncherExecutablePath);
            filesystem::path exeDir = exePath.parent_path();

            tryFile(exeDir / compiledSource.filename());

            if (runtimeSource.empty())
            {
                filesystem::path siblingSource =
                    exeDir / (exePath.stem().string() + ".cpp");
                tryFile(siblingSource);
            }
        }

        if (runtimeSource.empty() && !inputPath.parent_path().empty())
            tryFile(inputPath.parent_path() / compiledSource.filename());

        if (runtimeSource.empty())
            tryFile(filesystem::current_path() / compiledSource.filename());

        if (runtimeSource.empty())
        {
            filesystem::path exePath("/proc/self/exe");
            std::error_code ec;
            filesystem::path resolvedExe = filesystem::read_symlink(exePath, ec);
            if (!ec && !resolvedExe.empty())
            {
                filesystem::path exeDir = resolvedExe.parent_path();
                tryFile(exeDir / compiledSource.filename());

                if (runtimeSource.empty())
                    tryFile(exeDir / (resolvedExe.stem().string() + ".cpp"));
            }
        }
    }
    catch (const filesystem::filesystem_error &e)
    {
        error("execute block() could not resolve the TigerFlash runtime source: " +
              string(e.what()));
        return false;
    }

    if (runtimeSource.empty())
    {
        error(
            "execute block() could not find the TigerFlash runtime source. "
            "Keep the IDE .cpp beside the TigerFlash executable or compile "
            "with an absolute source path.");
        return false;
    }

    vector<string> programLines;
    vector<string> libraries;
    unordered_set<string> seenLibraries;

    for (const string &raw : sourceScript)
    {
        string libraryName;
        if (tfParseLibraryDirectiveForBuild(raw, libraryName))
        {
            if (seenLibraries.insert(libraryName).second)
                libraries.push_back(libraryName);
            continue; // The embedded executable initializes libraries directly.
        }

        programLines.push_back(raw);
    }

    const filesystem::path libraryDir = inputPath.parent_path() / "library";
    vector<filesystem::path> librarySources;

    for (const string &libraryName : libraries)
    {
        filesystem::path source = libraryDir / (libraryName + ".cpp");
        filesystem::path directSo = libraryDir / (libraryName + ".so");

        if (filesystem::exists(source) && filesystem::is_regular_file(source))
        {
            librarySources.push_back(source);
            continue;
        }

        if (filesystem::exists(directSo) && filesystem::is_regular_file(directSo))
        {
            error("execute block() currently needs library/" + libraryName +
                  ".cpp to statically include that library in the executable. "
                  "A standalone .so is not bundled yet.");
            return false;
        }

        error("execute block() could not find library source: \"" +
              source.string() + "\"");
        return false;
    }

    const filesystem::path outputCpp =
        inputPath.parent_path() / (inputPath.stem().string() + "_executable.cpp");
    const filesystem::path outputExe =
        inputPath.parent_path() / (inputPath.stem().string() + "_executable");

    ofstream generated(outputCpp);
    if (!generated)
    {
        error("execute block() could not create generated C++: \"" +
              outputCpp.string() + "\"");
        return false;
    }

    const string runtimeInclude = tfCppEscapeString(runtimeSource.string());

    generated << "// ============================================================\n";
    generated << "// TIGERFLASH NATIVE EXECUTABLE\n";
    generated << "// Generated from: " << inputPath.filename().string() << "\n";
    generated << "// The runtime is compiled here as an included translation unit.\n";
    generated << "// This file has no IDE input loop and starts only this script.\n";
    generated << "// ============================================================\n\n";
    generated << "#define TF_EXECUTABLE_BUILD 1\n";
    generated << "#define main tf_embedded_original_main\n";
    generated << "#include \"" << runtimeInclude << "\"\n";
    generated << "#undef main\n\n";

    generated << "// Embedded library entry points.\n";
    for (const string &libraryName : libraries)
    {
        generated << "extern \"C\" bool "
                   << tfBuildLibraryIdentifier(libraryName)
                   << "(const TFLibraryAPI *api);\n";
    }
    generated << "\n";

    generated << "static vector<string> tfEmbeddedBuildScript()\n";
    generated << "{\n";
    generated << "    vector<string> script;\n";
    generated << "    script.reserve(" << programLines.size() << ");\n";

    for (const string &line : programLines)
    {
        generated << "    script.emplace_back(\""
                   << tfCppEscapeString(line)
                   << "\");\n";
    }

    generated << "    return script;\n";
    generated << "}\n\n";

    generated << "static void tfBuildInitializeEmbeddedLibraries()\n";
    generated << "{\n";
    generated << "    TFLibraryAPI api;\n";
    generated << "    api.version = 1;\n";
    generated << "    api.registerSay3DOptionsHook = tfHostRegisterSay3DOptionsHook;\n";
    generated << "    api.registerSay3DHook = tfHostRegisterSay3DHook;\n";
    generated << "    api.registerCommandHook = tfHostRegisterCommandHook;\n";
    generated << "    api.registerConditionHook = tfHostRegisterConditionHook;\n";
    generated << "    api.setObjectMotionBlur = tfHostSetObjectMotionBlur;\n";
    generated << "    api.getObjectMotionBlur = tfHostGetObjectMotionBlur;\n";
    generated << "    api.object3DExists = tfHostObject3DExists;\n";
    generated << "    api.log = tfHostLibraryLog;\n";

    for (const string &libraryName : libraries)
    {
        generated << "    if (!" << tfBuildLibraryIdentifier(libraryName)
                   << "(&api))\n";
        generated << "    {\n";
        generated << "        error(\"Embedded library rejected the TigerFlash API: "
                   << tfCppEscapeString(libraryName) << "\");\n";
        generated << "        std::exit(20);\n";
        generated << "    }\n";
    }

    generated << "}\n\n";

    generated << "int main()\n";
    generated << "{\n";
    generated << "    tfBuildInitializeEmbeddedLibraries();\n";
    generated << "    vector<string> script = tfEmbeddedBuildScript();\n";
    generated << "    if (!script.empty())\n";
    generated << "        executeBlock(script, 0, 0);\n";
    generated << "\n";
    generated << "    if (sceneHas2D || sceneHas3D || sceneHasGui)\n";
    generated << "        runGraphicsWindow();\n";
    generated << "    else\n";
    generated << "        tfRunEmbeddedBackgroundTasks();\n";
    generated << "\n";
    generated << "    return 0;\n";
    generated << "}\n";
    generated.close();

    cout << "TigerFlash: execute block() source generated: "
         << filesystem::absolute(outputCpp).string() << endl;
    cout << "TigerFlash: Carregando executavel... preparando compilacao." << endl;

    if (!tfCommandExists("g++"))
    {
        error(
            "execute block() nao encontrou o compilador \"g++\" no PATH deste "
            "processo. O .cpp foi gerado em \"" + outputCpp.string() + "\", mas "
            "nada foi compilado.");
        return false;
    }

    filesystem::path raylibStaticLib;
    if (!tfFindRaylibStaticLib(raylibStaticLib))
    {
        error(
            "execute block() nao encontrou libraylib.a em nenhum local padrao. "
            "O .cpp foi gerado em \"" + outputCpp.string() + "\", mas o link "
            "nao pode ser feito sem o raylib estatico.");
        return false;
    }

    const filesystem::path tempDir =
        filesystem::temp_directory_path() /
        ("tigerflash_execute_block_" +
         to_string(static_cast<long long>(getpid())));

    try
    {
        filesystem::create_directories(tempDir);
    }
    catch (const filesystem::filesystem_error &e)
    {
        error("execute block() could not create its temporary build directory: " +
              string(e.what()));
        return false;
    }

    vector<filesystem::path> objects;

    for (size_t i = 0; i < librarySources.size(); ++i)
    {
        const string &libraryName = libraries[i];
        filesystem::path objectPath = tempDir / (libraryName + ".o");
        filesystem::path compileLog = tempDir / (libraryName + "_compile.log");

        string command =
            string("g++ -std=c++17 -O2 -c ") +
            "-DtfLibraryInit=" + tfBuildLibraryIdentifier(libraryName) +
            " " + tfShellQuote(librarySources[i].string()) +
            " -o " + tfShellQuote(objectPath.string()) +
            " > " + tfShellQuote(compileLog.string()) + " 2>&1";

        const double libraryEstimate = max(1.5,
            tfEstimateBuildSeconds(
                static_cast<size_t>(filesystem::file_size(librarySources[i])),
                0));

        int result = tfRunBuildCommandWithProgress(
            command,
            "biblioteca " + to_string(i + 1) + "/" +
                to_string(librarySources.size()),
            libraryEstimate);
        if (result != 0)
        {
            string log = tfReadLogFile(compileLog);
            filesystem::path savedLog = inputPath.parent_path() /
                (libraryName + "_compile_error.log");
            try
            {
                filesystem::copy_file(
                    compileLog,
                    savedLog,
                    filesystem::copy_options::overwrite_existing);
            }
            catch (...)
            {
            }

            error("execute block() failed while compiling library: \"" +
                  librarySources[i].string() + "\". Saida completa do g++ " +
                  "salva em: \"" + savedLog.string() + "\"" +
                  (log.empty() ? string() : ("\n" + log)));
            try { filesystem::remove_all(tempDir); } catch (...) {}
            return false;
        }

        objects.push_back(objectPath);
    }

    const filesystem::path linkLog = tempDir / "link.log";

    string linkCommand =
        "g++ -std=c++17 -O2 " + tfShellQuote(outputCpp.string());

    for (const filesystem::path &objectPath : objects)
        linkCommand += " " + tfShellQuote(objectPath.string());

    linkCommand +=
        " " + tfShellQuote(raylibStaticLib.string()) +
        " -lGL -lm -lpthread -ldl -lrt -lX11" +
        " -o " + tfShellQuote(outputExe.string()) +
        " > " + tfShellQuote(linkLog.string()) + " 2>&1";

    const size_t generatedCppBytes = [&]() -> size_t
    {
        std::error_code ec;
        const uintmax_t bytes = filesystem::file_size(outputCpp, ec);
        return ec ? static_cast<size_t>(0) : static_cast<size_t>(bytes);
    }();

    const double linkEstimate = max(1.5,
        tfEstimateBuildSeconds(generatedCppBytes, libraries.size()) * 0.45);

    const int linkResult = tfRunBuildCommandWithProgress(
        linkCommand,
        "linkando C++ e bibliotecas",
        linkEstimate);
    const string linkLogContent = tfReadLogFile(linkLog);

    filesystem::path savedLinkLog;
    if (linkResult != 0)
    {
        savedLinkLog = inputPath.parent_path() /
            (inputPath.stem().string() + "_link_error.log");
        try
        {
            filesystem::copy_file(
                linkLog,
                savedLinkLog,
                filesystem::copy_options::overwrite_existing);
        }
        catch (...)
        {
        }
    }

    try
    {
        filesystem::remove_all(tempDir);
    }
    catch (...)
    {
    }

    if (linkResult != 0)
    {
        error("execute block() failed while creating the executable ("
              "comando: " + linkCommand + "). Generated C++ was kept at: \"" +
              outputCpp.string() + "\". Saida completa do g++/linker salva em: \"" +
              savedLinkLog.string() + "\"" +
              (linkLogContent.empty() ? string() : ("\n" + linkLogContent)));
        return false;
    }

    cout << "TigerFlash: execute block() generated "
         << outputCpp.filename().string() << endl;
    cout << "TigerFlash: executable created: "
         << filesystem::absolute(outputExe).string() << endl;

    return true;
}


// ============================================================
// APK BLOCK / PORTABLE ANDROID RUNTIME
// `apk block()` does NOT compile C++ or search for an Android SDK/NDK on the
// user's machine. The TigerFlash Android interpreter is precompiled once for
// ARM64 and shipped in android_runtime/runtime_template/. This command only
// packages the current .tf script and project assets around that runtime.
// ============================================================

static filesystem::path tfFindPortableAndroidRuntimeKit(const filesystem::path &inputPath)
{
    vector<filesystem::path> candidates;

    if (!inputPath.empty() && !inputPath.parent_path().empty())
        candidates.push_back(inputPath.parent_path() / "android_runtime");

    if (!tfLauncherExecutablePath.empty())
        candidates.push_back(filesystem::path(tfLauncherExecutablePath).parent_path() /
                             "android_runtime");

    candidates.push_back(filesystem::current_path() / "android_runtime");

    for (const filesystem::path &candidate : candidates)
    {
        std::error_code ec;
        if (filesystem::exists(candidate / "runtime_template", ec) &&
            filesystem::is_directory(candidate / "runtime_template", ec) &&
            filesystem::exists(candidate / "tools" / "package_apk.sh", ec) &&
            filesystem::is_regular_file(candidate / "tools" / "package_apk.sh", ec))
        {
            return filesystem::absolute(candidate);
        }
    }

    return {};
}

static bool tfCopyDirectoryContents(const filesystem::path &source,
                                    const filesystem::path &destination)
{
    if (source.empty())
        return false;

    try
    {
        std::error_code ec;
        if (!filesystem::exists(source, ec) ||
            !filesystem::is_directory(source, ec))
        {
            return false;
        }

        filesystem::create_directories(destination, ec);
        if (ec)
            return false;

        for (const auto &entry : filesystem::recursive_directory_iterator(source, ec))
        {
            if (ec)
                return false;

            const filesystem::path relative =
                filesystem::relative(entry.path(), source, ec);
            if (ec)
                return false;

            const filesystem::path target = destination / relative;

            if (entry.is_directory(ec))
            {
                filesystem::create_directories(target, ec);
                if (ec)
                    return false;
            }
            else if (entry.is_regular_file(ec))
            {
                filesystem::create_directories(target.parent_path(), ec);
                if (ec)
                    return false;

                filesystem::copy_file(
                    entry.path(),
                    target,
                    filesystem::copy_options::overwrite_existing,
                    ec);

                if (ec)
                    return false;
            }
        }

        return true;
    }
    catch (...)
    {
        return false;
    }
}

static string tfAndroidPackageIdentifier(const string &stem)
{
    string id;
    id.reserve(stem.size());

    for (char c : stem)
    {
        if (isalnum(static_cast<unsigned char>(c)))
            id.push_back(static_cast<char>(tolower(static_cast<unsigned char>(c))));
        else
            id.push_back('_');
    }

    while (!id.empty() && id.front() == '_')
        id.erase(id.begin());

    if (id.empty())
        id = "app";

    if (!isalpha(static_cast<unsigned char>(id.front())))
        id = "app_" + id;

    return id;
}

static bool tfBuildApkBlock(const vector<string> &sourceScript,
                            const string &scriptPath)
{
    if (sourceScript.empty())
    {
        error("apk block() requires TigerFlash code before the previous stop.");
        return false;
    }

    filesystem::path inputPath(scriptPath);
    if (inputPath.empty())
    {
        error("apk block() requires a .tf file on disk; stdin scripts cannot be built yet.");
        return false;
    }

    const filesystem::path kit = tfFindPortableAndroidRuntimeKit(inputPath);
    if (kit.empty())
    {
        error(
            "apk block() could not find the portable TigerFlash Android runtime. "
            "Expected android_runtime/runtime_template/ and "
            "android_runtime/tools/package_apk.sh beside the TigerFlash executable "
            "or project. This build does not require Android Studio, SDK or NDK "
            "on the user's machine; the Android runtime must be prepared once "
            "by the TigerFlash developer and shipped with the IDE.");
        return false;
    }

    const filesystem::path templateDir = kit / "runtime_template";
    const filesystem::path packager = kit / "tools" / "package_apk.sh";
    const filesystem::path outputApk =
        inputPath.parent_path() / (inputPath.stem().string() + ".apk");

    const filesystem::path buildDir =
        filesystem::temp_directory_path() /
        ("tigerflash_portable_apk_" +
         to_string(static_cast<long long>(getpid())));

    const filesystem::path scriptFile = buildDir / "main.tf";
    const filesystem::path assetsDir = buildDir / "assets";
    const filesystem::path resourcesDir = buildDir / "resources";

    try
    {
        filesystem::remove_all(buildDir);
        filesystem::create_directories(assetsDir);
        filesystem::create_directories(resourcesDir);

        ofstream scriptOut(scriptFile);
        if (!scriptOut)
        {
            error("apk block() could not create the temporary Android script: \"" +
                  scriptFile.string() + "\"");
            return false;
        }

        for (const string &line : sourceScript)
            scriptOut << line << '\n';
        scriptOut.close();

        tfCopyDirectoryContents(inputPath.parent_path() / "assets", assetsDir);
        tfCopyDirectoryContents(inputPath.parent_path() / "resources", resourcesDir);

        const string packageId =
            "com.tigerflash." + tfAndroidPackageIdentifier(inputPath.stem().string());
        const string label = inputPath.stem().string().empty()
            ? string("TigerFlash")
            : inputPath.stem().string();

        string command =
            tfShellQuote(packager.string()) +
            " --template " + tfShellQuote(templateDir.string()) +
            " --script " + tfShellQuote(scriptFile.string()) +
            " --assets " + tfShellQuote(assetsDir.string()) +
            " --resources " + tfShellQuote(resourcesDir.string()) +
            " --output " + tfShellQuote(outputApk.string()) +
            " --package " + tfShellQuote(packageId) +
            " --label " + tfShellQuote(label);

        cout << "TigerFlash: apk block() using portable Android runtime." << endl;
        cout << "TigerFlash: criando APK ARM64..." << endl;

        const int result = tfRunBuildCommandWithProgress(
            command,
            "empacotando runtime + script + assets",
            4.0);

        try { filesystem::remove_all(buildDir); } catch (...) {}

        if (result != 0)
        {
            error(
                "apk block() failed inside the portable Android packager. "
                "The IDE did not invoke Android Studio/SDK/NDK. Check "
                "android_runtime/tools/package_apk.sh and the packaged runtime.");
            return false;
        }

        std::error_code ec;
        if (!filesystem::exists(outputApk, ec) ||
            !filesystem::is_regular_file(outputApk, ec))
        {
            error(
                "apk block() finished without producing the expected APK: \"" +
                outputApk.string() + "\"");
            return false;
        }

        cout << "TigerFlash: APK created: "
             << filesystem::absolute(outputApk).string() << endl;
        return true;
    }
    catch (const filesystem::filesystem_error &e)
    {
        try { filesystem::remove_all(buildDir); } catch (...) {}
        error("apk block() failed while preparing its portable Android build: " +
              string(e.what()));
        return false;
    }
}

#if defined(PLATFORM_ANDROID) || defined(__ANDROID__)
static bool tfReadAndroidAssetText(const string &assetName, string &output)
{
    output.clear();

    struct android_app *app = GetAndroidApp();
    if (!app || !app->activity || !app->activity->assetManager)
        return false;

    AAssetManager *manager = app->activity->assetManager;
    AAsset *asset = AAssetManager_open(manager, assetName.c_str(), AASSET_MODE_BUFFER);
    if (!asset)
        return false;

    const off_t length = AAsset_getLength(asset);
    if (length > 0)
    {
        output.resize(static_cast<size_t>(length));
        const int64_t readBytes = AAsset_read(asset, output.data(), static_cast<size_t>(length));
        if (readBytes < 0 || static_cast<off_t>(readBytes) != length)
        {
            AAsset_close(asset);
            output.clear();
            return false;
        }
    }

    AAsset_close(asset);
    return true;
}
#endif

static int runTigerFlashInput(istream &input)
{
    vector<string> script;
    string line;
    bool insideComment = false;
 
    while (getline(input, line))
    {
        // Remove comments before execution. A command after a closed
        // one-line comment on the same source line is preserved.
        string remaining;
        if (consumeCommentLine(line, insideComment, remaining))
        {
            line = remaining;
            if (line.empty())
                continue;
        }

        string command = toLowerOutsideQuotes(trim(line));

        // `execute block()` and `apk block()` are intentionally directives
        // AFTER `stop`. They build only the script that was just executed.
        if ((tfExecuteBlockArmed || tfApkBlockArmed) &&
            command == "execute block()")
        {
            tfBuildExecutableBlock(tfLastExecutedScript, tfCurrentScriptPath);
            tfExecuteBlockArmed = false;
            tfApkBlockArmed = false;

            // For graphical scripts the stop already executed the scene, but
            // the window is deliberately delayed by one command so the build
            // can finish BEFORE the 3D/2D window opens.
            if (tfPendingSceneWindow)
            {
                runGraphicsWindow();
                tfPendingSceneWindow = false;
            }
            continue;
        }

        if ((tfExecuteBlockArmed || tfApkBlockArmed) &&
            command == "apk block()")
        {
            tfBuildApkBlock(tfLastExecutedScript, tfCurrentScriptPath);
            tfExecuteBlockArmed = false;
            tfApkBlockArmed = false;

            if (tfPendingSceneWindow)
            {
                runGraphicsWindow();
                tfPendingSceneWindow = false;
            }
            continue;
        }

        // A normal command after stop keeps the original behavior: show the
        // finished scene before accepting the next TigerFlash command.
        if (tfPendingSceneWindow)
        {
            runGraphicsWindow();
            tfPendingSceneWindow = false;
        }

        // Any ordinary command after a stop starts a new script section and
        // therefore cancels the pending build directive.
        if ((tfExecuteBlockArmed || tfApkBlockArmed) &&
            command != "execute block()" &&
            command != "apk block()")
        {
            tfExecuteBlockArmed = false;
            tfApkBlockArmed = false;
        }

        if (command != "stop")
        {
            script.push_back(line);
            continue;
        }
 
        if (!script.empty())
        {
            executeBlock(script, 0, 0);
            tfLastExecutedScript = script;
            tfExecuteBlockArmed = true;
            tfApkBlockArmed = true;
        }
        else
        {
            tfLastExecutedScript.clear();
            tfExecuteBlockArmed = false;
            tfApkBlockArmed = false;
        }

        if (sceneHas2D || sceneHas3D || sceneHasGui)
        {
            // Delay the graphical window until after the next line is known.
            // This lets `execute block()` build the executable first.
            tfPendingSceneWindow = true;
        }
        else if (!tfBackgroundTasks.empty())
        {
            // A pure console program has no graphics loop to poll the
            // cooperative `if` scheduler. Run the registered conditions once
            // here so `if` works in ordinary non-graphical TigerFlash scripts
            // as well. This path is intentionally finite and budgeted.
            int commandBudget = TF_BACKGROUND_COMMAND_BUDGET * 16;

            for (size_t i = 0;
                 i < tfBackgroundTasks.size() && commandBudget > 0; )
            {
                TFBackgroundTask &task = tfBackgroundTasks[i];

                if (task.type == TFBackgroundTaskType::Conditional)
                {
                    bool active = evaluateCondition(task.condition);
                    task.lastCondition = active;

                    if (active)
                        tfExecuteBackgroundBody(task.body, commandBudget);
                    else if (!task.elseBody.empty())
                        tfExecuteBackgroundBody(task.elseBody, commandBudget);

                    tfBackgroundTasks.erase(
                        tfBackgroundTasks.begin() + static_cast<long>(i));
                    continue;
                }

                if (task.remaining == -1)
                {
                    // An infinite repeat has no natural console termination.
                    // Leave it unexecuted rather than freezing the interpreter.
                    tfBackgroundTasks.erase(
                        tfBackgroundTasks.begin() + static_cast<long>(i));
                    continue;
                }

                while (task.remaining > 0 && commandBudget > 0)
                {
                    tfExecuteBackgroundBody(task.body, commandBudget);
                    --task.remaining;
                }

                tfBackgroundTasks.erase(
                    tfBackgroundTasks.begin() + static_cast<long>(i));
            }

            if (!tfBackgroundTasks.empty())
            {
                error("Console control-flow budget exhausted before all background commands finished.");
                tfBackgroundTasks.clear();
            }
        }
 
        script.clear();
        variables.clear();
        lists.clear();
        objects2D.clear();
        objects3D.clear();
        objectMemory2D.clear();
        objectMemory3D.clear();
        inputBindings.clear();
        tfBackgroundTasks.clear();
        tfFunctions.clear();
        tfImportedModules.clear();
        tfControlSignal = TFControlSignal::None;
        tfReturnValue.clear();
        tfFunctionDepth = 0;
        tfLoopDepth = 0;
        physicalCollisionPairs.clear();
        cameraCollisionPairs.clear();
        tfCameraFollowEnabled = false;
        tfCameraFollowTarget.clear();
        tfCameraFollowSmooth = 0.08f;
        // The camera itself, and whether free mouse-look is on, must not
        // leak from one script/scene into the next.
        tfMouseLookEnabled = false;
        tfCameraRotationInitialized = false;
        shapeDeformations.clear();
        gravityBodies.clear();
        gravityGrounded.clear();
        tfSmoothedClusterSize3D.clear();
        tfSmoothedClusterSize2D.clear();
        lastCreatedObjectKey.clear();
        // GUI textures were already freed (tfUnloadGuiTextures, inside
        // runGraphicsWindow) before the window closed; this just drops
        // the now-stale CPU-side element list.
        guiElements.clear();
        guiAnimations.clear();
        guiCollisionPairs.clear();
        guiClickBindings.clear();
        sceneHasGui = false;
        sceneHas2D = false;
        sceneHas3D = false;
    }
 
    // When a .tf file reaches EOF without an explicit `stop`, execute the
    // remaining commands automatically. Interactive stdin keeps its original
    // behavior: `stop` still executes the current scene immediately.
    if (!script.empty())
    {
        executeBlock(script, 0, 0);

        if (sceneHas2D || sceneHas3D || sceneHasGui)
        {
            runGraphicsWindow();
        }
        else if (!tfBackgroundTasks.empty())
        {
            int commandBudget = TF_BACKGROUND_COMMAND_BUDGET * 16;

            for (size_t i = 0;
                 i < tfBackgroundTasks.size() && commandBudget > 0; )
            {
                TFBackgroundTask &task = tfBackgroundTasks[i];

                if (task.type == TFBackgroundTaskType::Conditional)
                {
                    bool active = evaluateCondition(task.condition);
                    task.lastCondition = active;

                    if (active)
                        tfExecuteBackgroundBody(task.body, commandBudget);
                    else if (!task.elseBody.empty())
                        tfExecuteBackgroundBody(task.elseBody, commandBudget);

                    tfBackgroundTasks.erase(
                        tfBackgroundTasks.begin() + static_cast<long>(i));
                    continue;
                }

                if (task.remaining == -1)
                {
                    tfBackgroundTasks.erase(
                        tfBackgroundTasks.begin() + static_cast<long>(i));
                    continue;
                }

                while (task.remaining > 0 && commandBudget > 0)
                {
                    tfExecuteBackgroundBody(task.body, commandBudget);
                    --task.remaining;
                }

                tfBackgroundTasks.erase(
                    tfBackgroundTasks.begin() + static_cast<long>(i));
            }

            if (!tfBackgroundTasks.empty())
            {
                error("Console control-flow budget exhausted before all background commands finished.");
                tfBackgroundTasks.clear();
            }
        }
    }

    // If stop was the final command, there was no following line on which to
    // trigger the delayed graphics window. Open it here.
    if (tfPendingSceneWindow)
    {
        runGraphicsWindow();
        tfPendingSceneWindow = false;
    }

    // Comentario aberto ate o fim do arquivo: nao executa nada.
    if (insideComment)
        error("A comment was opened with \" but never closed. Add a matching closing \" somewhere after it so the comment block ends properly.");
 
    return 0;
}


// ============================================================
// TIGERFLASH FILE ENTRY POINT
// ============================================================
// Normal terminal/IDE usage:
//     TigerFlash programa.tf
//
// TigerFlash source files are intentionally restricted to .tf.
// Without a filename, stdin remains supported for pipes and interactive use:
//     cat programa.tf | TigerFlash
// ============================================================
int main(int argc, char *argv[])
{
    if (argc > 2)
    {
        cerr << "TigerFlash: use apenas um arquivo .tf por execucao.\n";
        cerr << "Uso: " << argv[0] << " arquivo.tf\n";
        return 2;
    }

#if defined(PLATFORM_ANDROID) || defined(__ANDROID__)
    // On Android, raylib's NativeActivity calls the same main() with argc == 1.
    // The portable runtime reads the TigerFlash program from APK assets/main.tf.
    if (argc == 1)
    {
        string androidScript;
        if (tfReadAndroidAssetText("main.tf", androidScript))
        {
            tfCurrentScriptPath = "main.tf";
            istringstream input(androidScript);
            return runTigerFlashInput(input);
        }
    }
#endif

    // No filename: preserve the original stdin mode on desktop.
    if (argc == 1)
    {
        tfCurrentScriptPath.clear();
        return runTigerFlashInput(cin);
    }

    string fileName = argv[1];

    // Help is deliberately not treated as a script filename.
    if (fileName == "--help" || fileName == "-h")
    {
        cout << "TigerFlash - interpretador de arquivos .tf\n\n";
        cout << "Uso:\n";
        cout << "  " << argv[0] << " arquivo.tf\n";
        cout << "  cat arquivo.tf | " << argv[0] << "\n\n";
        cout << "Build nativo (depois de stop): execute block()\n";
        cout << "Build APK Android (depois de stop): apk block()\n";
        cout << "Somente arquivos com extensao .tf sao aceitos.\n";
        return 0;
    }

    // Only .tf files are accepted. The comparison is case-insensitive.
    auto dot = fileName.find_last_of('.');
    string extension = (dot == string::npos) ? "" : fileName.substr(dot);
    transform(extension.begin(), extension.end(), extension.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });

    if (extension != ".tf")
    {
        cerr << "TigerFlash: arquivo recusado: \"" << fileName << "\"\n";
        cerr << "TigerFlash aceita somente arquivos .tf.\n";
        return 3;
    }

    // Resolve the script path before changing the working directory.
    // This makes relative resources (images, modules, etc.) resolve relative
    // to the .tf file when TigerFlash is launched by Geany, VS Code or a shell.
    try
    {
        tfLauncherExecutablePath =
            filesystem::absolute(filesystem::path(argv[0])).string();
    }
    catch (...)
    {
        tfLauncherExecutablePath.clear();
    }
    filesystem::path scriptPath;
    try
    {
        scriptPath = filesystem::absolute(filesystem::path(fileName));
    }
    catch (const filesystem::filesystem_error &e)
    {
        cerr << "TigerFlash: nao foi possivel resolver o caminho do arquivo: "
             << e.what() << "\n";
        return 4;
    }

    if (!filesystem::exists(scriptPath) ||
        !filesystem::is_regular_file(scriptPath))
    {
        cerr << "TigerFlash: arquivo .tf nao encontrado: "
             << fileName << "\n";
        return 5;
    }

    ifstream file(scriptPath);
    if (!file.is_open())
    {
        cerr << "TigerFlash: nao foi possivel abrir o arquivo .tf: "
             << fileName << "\n";
        return 6;
    }

    // IDEs normally start the program with their project directory as cwd.
    // Move the interpreter cwd to the script directory so commands using
    // relative paths behave the same when double-run from different IDEs.
    const filesystem::path scriptDirectory = scriptPath.parent_path();
    if (!scriptDirectory.empty())
    {
        string directory = scriptDirectory.string();
        if (chdir(directory.c_str()) != 0)
        {
            cerr << "TigerFlash: aviso: nao foi possivel usar a pasta do .tf "
                 << "como diretorio de trabalho.\n";
        }
    }

    tfCurrentScriptPath = scriptPath.string();
    cout << "TigerFlash: executando " << scriptPath.filename().string() << "\n";
    return runTigerFlashInput(file);
}
