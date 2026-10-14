#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// Runtime-only integration: the learning/training code is intentionally absent.
static constexpr int N_GROUPS = 512;
static constexpr int GROUP_SIZE = 32;
static constexpr int N_NEURONS = N_GROUPS * GROUP_SIZE;
static constexpr int N_IN = 33;
static constexpr int N_ACT = 12;
static constexpr int K_MIN = 3;
static constexpr int K_MAX = 512;
static constexpr float ROUTER_BETA = 6.0f;
static constexpr float ROUTER_MASS = 0.60f;
static constexpr float PHYS_DT = 1.0f / 240.0f;
static constexpr int SUBSTEPS = 12;
static constexpr float CTRL_DT = PHYS_DT * SUBSTEPS;
static constexpr float ORB_PICK_R = 0.75f;
static constexpr float kTwoPi = 6.28318530718f;
static const char* AI_MEMORY_FILE = "cerebro_memoria.bin";

static std::mt19937 g_rng(12345);
static float randn() { static std::normal_distribution<float> d(0.f, 1.f); return d(g_rng); }
static float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }

struct V3 { float x=0,y=0,z=0; };
static inline V3 operator+(V3 a,V3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
static inline V3 operator-(V3 a,V3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static inline V3 operator*(V3 a,float s){return {a.x*s,a.y*s,a.z*s};}
static inline V3 operator*(float s,V3 a){return a*s;}
static inline float dot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static inline V3 cross(V3 a,V3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static inline float len(V3 a){return std::sqrt(dot(a,a));}
static inline V3 norm(V3 a){float l=len(a);return l>1e-9f?a*(1.f/l):V3{};}
static inline V3 rotAxis(V3 v,V3 k,float a){float c=std::cos(a),s=std::sin(a);return v*c+cross(k,v)*s+k*(dot(k,v)*(1.f-c));}
static inline Vector3 toRl(V3 v){return {v.x,v.y,v.z};}
static inline Color rgba(int r,int g,int b,int a=255){return {(unsigned char)r,(unsigned char)g,(unsigned char)b,(unsigned char)a};}

struct Act {                       // resultado de uma passagem pelo cerebro
    int   nGroups = 0, nActive = 0;
    int   groups[K_MAX];
    int   idx[K_MAX * GROUP_SIZE];
    float hn[K_MAX * GROUP_SIZE];  // ativacoes normalizadas dos neuronios ativos
    float scale = 0;
    float mu[N_ACT];
    float value = 0;
};

struct Brain {
    std::vector<float> router, Win, bias, Wout, V;
    std::vector<uint32_t> useCount;
    uint64_t steps = 0, episodes = 0, orbs = 0;
    float emaReturn = 0;

    std::vector<float> score; std::vector<int> order;

    void init() {
        router.assign(N_GROUPS * N_IN, 0.f);
        for (int g = 0; g < N_GROUPS; g++) {
            float n = 0;
            for (int i = 0; i < N_IN; i++) { float v = randn(); router[g * N_IN + i] = v; n += v * v; }
            n = std::sqrt(n);
            for (int i = 0; i < N_IN; i++) router[g * N_IN + i] /= n;
        }
        Win.resize((size_t)N_NEURONS * N_IN);
        for (auto& w : Win) w = randn() / std::sqrt((float)N_IN) * 1.2f;
        bias.resize(N_NEURONS);
        for (auto& b : bias) b = randn() * 0.1f;
        Wout.assign((size_t)N_NEURONS * N_ACT, 0.f);
        V.assign(N_NEURONS, 0.f);
        useCount.assign(N_GROUPS, 0);
        steps = episodes = orbs = 0; emaReturn = 0;
        prepare();
    }
    void prepare() { score.assign(N_GROUPS, 0.f); order.resize(N_GROUPS); std::iota(order.begin(), order.end(), 0); }

    // Passagem para frente ESPARSA: so calcula os grupos escolhidos pelo roteador.
    void forward(const float* x, Act& a) {
        float mx = -1e30f;
        for (int g = 0; g < N_GROUPS; g++) {
            const float* r = &router[g * N_IN]; float s = 0;
            for (int i = 0; i < N_IN; i++) s += r[i] * x[i];
            score[g] = s; if (s > mx) mx = s;
        }
        float Z = 0;
        for (int g = 0; g < N_GROUPS; g++) Z += std::exp(ROUTER_BETA * (score[g] - mx));
        std::partial_sort(order.begin(), order.begin() + K_MAX, order.end(),
                          [&](int p, int q) { return score[p] > score[q]; });
        float cum = 0; int k = 0;
        while (k < K_MAX) {                       // usa so os grupos necessarios
            int g = order[k]; cum += std::exp(ROUTER_BETA * (score[g] - mx)) / Z;
            a.groups[k++] = g;
            if (k >= K_MIN && cum >= ROUTER_MASS) break;
        }
        a.nGroups = k;
        int n = 0; float sq = 0;
        for (int gi = 0; gi < k; gi++) {
            int g = a.groups[gi];
            for (int u = 0; u < GROUP_SIZE; u++) {
                int j = g * GROUP_SIZE + u;
                const float* w = &Win[(size_t)j * N_IN];
                float p = bias[j];
                for (int i = 0; i < N_IN; i++) p += w[i] * x[i];
                if (p > 0.f) { a.idx[n] = j; a.hn[n] = p; sq += p * p; n++; }   // ReLU esparso
            }
        }
        a.nActive = n; a.scale = 1.f / std::sqrt(sq + 0.05f);
        for (int kk = 0; kk < N_ACT; kk++) a.mu[kk] = 0;
        a.value = 0;
        for (int t = 0; t < n; t++) {
            float h = a.hn[t] * a.scale; a.hn[t] = h; int j = a.idx[t];
            const float* wo = &Wout[(size_t)j * N_ACT];
            for (int kk = 0; kk < N_ACT; kk++) a.mu[kk] += wo[kk] * h;
            a.value += V[j] * h;
        }
    }

    template <class T>
    static bool rd(std::ifstream& f, std::vector<T>& v) {
        if (v.empty()) return true;
        f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
        return static_cast<bool>(f);
    }

    bool load(const char* path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        char magic[8]; int32_t dims[4];
        f.read(magic, 8); f.read((char*)dims, sizeof dims);
        if (!f || std::memcmp(magic, "CBR3D001", 8) != 0 || dims[0] != N_GROUPS || dims[1] != GROUP_SIZE ||
            dims[2] != N_IN || dims[3] != N_ACT) {
            f.close();
            std::printf("[memoria] arquivo incompativel; guardando como .old e criando cerebro novo\n");
            std::error_code ec; std::filesystem::rename(path, std::string(path) + ".old", ec);
            return false;
        }
        f.read((char*)&steps, 8); f.read((char*)&episodes, 8); f.read((char*)&orbs, 8); f.read((char*)&emaReturn, 4);
        router.resize(N_GROUPS * N_IN); Win.resize((size_t)N_NEURONS * N_IN); bias.resize(N_NEURONS);
        Wout.resize((size_t)N_NEURONS * N_ACT); V.resize(N_NEURONS); useCount.resize(N_GROUPS);
        bool ok = rd(f, router) && rd(f, Win) && rd(f, bias) && rd(f, Wout) && rd(f, V) && rd(f, useCount);
        if (ok) prepare();
        return ok;
    }
};

#include "raylib.h"
#include "raymath.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>

// ============================================================================
// Config.h
// ============================================================================
namespace Config {
    inline constexpr int    SCREEN_WIDTH     = 1280;
    inline constexpr int    SCREEN_HEIGHT    = 720;
    inline constexpr int    TARGET_FPS       = 60;
    inline constexpr const char* WINDOW_TITLE = "BACKROOMS - Nível 94 : UNREAL LIGHTING & PROCEDURAL TERRAIN";

    inline constexpr int    TERRAIN_RES      = 256;
    inline constexpr float  TERRAIN_SIZE     = 350.0f;
    inline constexpr float  HILL_HEIGHT      = 14.0f;
    inline constexpr float  HILL_FREQUENCY   = 0.0075f;
    inline constexpr float  HILL_FLAT_CUT    = 0.20f;
    inline constexpr float  HILL_POWER       = 1.45f;
}

// ============================================================================
// NoiseUtils.h
// ============================================================================
namespace Noise {

inline float Hash2D(int x, int y) {
    uint32_t h = static_cast<uint32_t>(x) * 374761393u + static_cast<uint32_t>(y) * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return (static_cast<float>(h & 0x00FFFFFFu) / static_cast<float>(0x00FFFFFFu)) * 2.0f - 1.0f;
}

inline float SmoothStep01(float t) { return t * t * (3.0f - 2.0f * t); }
inline float Lerp(float a, float b, float t) { return a + (b - a) * t; }

inline float ValueNoise2D(float x, float y) {
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float tx = SmoothStep01(x - static_cast<float>(x0));
    float ty = SmoothStep01(y - static_cast<float>(y0));

    float v00 = Hash2D(x0, y0);
    float v10 = Hash2D(x1, y0);
    float v01 = Hash2D(x0, y1);
    float v11 = Hash2D(x1, y1);

    float ix0 = Lerp(v00, v10, tx);
    float ix1 = Lerp(v01, v11, tx);
    return Lerp(ix0, ix1, ty);
}

inline float FBM(float x, float y, int octaves = 5, float lacunarity = 2.0f, float gain = 0.5f) {
    float sum  = 0.0f;
    float amp  = 0.5f;
    float freq = 1.0f;
    float norm = 0.0f;
    for (int i = 0; i < octaves; i++) {
        sum  += ValueNoise2D(x * freq, y * freq) * amp;
        norm += amp;
        freq *= lacunarity;
        amp  *= gain;
    }
    return (norm > 0.0f) ? (sum / norm) : sum;
}

inline float Saturate(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

inline uint32_t XorShift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

inline float RandomFloat01(uint32_t& state) {
    return static_cast<float>(XorShift32(state) & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
}

inline float RandomRange(uint32_t& state, float lo, float hi) {
    return lo + (hi - lo) * RandomFloat01(state);
}

} // namespace Noise

// ============================================================================
// Physics & Collision
// ============================================================================
namespace Physics {

struct Cylinder {
    Vector3 base{};
    float radius = 0.4f;
    float height = 1.75f;
};

inline float SampleHeightBilinear(const std::vector<float>& heights, int res, float worldSize, float x, float z) {
    if (heights.empty() || res < 2) return 0.0f;

    float half = worldSize * 0.5f;
    float gx = ((x + half) / worldSize) * static_cast<float>(res - 1);
    float gz = ((z + half) / worldSize) * static_cast<float>(res - 1);

    gx = std::clamp(gx, 0.0f, static_cast<float>(res - 1));
    gz = std::clamp(gz, 0.0f, static_cast<float>(res - 1));

    int ix0 = static_cast<int>(std::floor(gx));
    int iz0 = static_cast<int>(std::floor(gz));
    int ix1 = std::min(ix0 + 1, res - 1);
    int iz1 = std::min(iz0 + 1, res - 1);

    float fx = gx - static_cast<float>(ix0);
    float fz = gz - static_cast<float>(iz0);

    float h00 = heights[iz0 * res + ix0];
    float h10 = heights[iz0 * res + ix1];
    float h01 = heights[iz1 * res + ix0];
    float h11 = heights[iz1 * res + ix1];

    float hx0 = Noise::Lerp(h00, h10, fx);
    float hx1 = Noise::Lerp(h01, h11, fx);
    return Noise::Lerp(hx0, hx1, fz);
}

inline Vector3 SampleTerrainNormal(const std::vector<float>& heights, int res, float worldSize, float x, float z) {
    float step = worldSize / static_cast<float>(std::max(res - 1, 1));
    float hL = SampleHeightBilinear(heights, res, worldSize, x - step, z);
    float hR = SampleHeightBilinear(heights, res, worldSize, x + step, z);
    float hD = SampleHeightBilinear(heights, res, worldSize, x, z - step);
    float hU = SampleHeightBilinear(heights, res, worldSize, x, z + step);

    Vector3 n = { hL - hR, 2.0f * step, hD - hU };
    return Vector3Normalize(n);
}

} // namespace Physics

// ============================================================================
// Terrain System with Unreal-like Procedural Shading & Lighting Simulation
// ============================================================================
class Terrain {
public:
    void Generate(uint32_t& rngState) {
        BuildHeightArray();
        BuildMesh();
        BuildProps();
        ScatterProps(rngState);
    }

    void Unload() {
        if (modelLoaded) { UnloadModel(model); modelLoaded = false; }
        if (propsLoaded) { UnloadModel(bushModel); UnloadModel(rockModel); propsLoaded = false; }
    }

    void Draw(Shader shader) {
        if (!modelLoaded) return;
        model.materials[0].shader = shader;
        DrawModel(model, { 0.0f, 0.0f, 0.0f }, 1.0f, WHITE);

        // Draw Sun Effect in the sky
        Vector3 sunDir = Vector3Normalize({ 0.6f, 0.8f, 0.4f });
        Vector3 sunPos = { sunDir.x * 300.0f, sunDir.y * 300.0f, sunDir.z * 300.0f };
        DrawSphere(sunPos, 28.0f, Color{ 255, 255, 220, 255 });
        DrawSphere(sunPos, 55.0f, Color{ 255, 240, 180, 120 }); // Outer glow halo
        DrawSphere(sunPos, 90.0f, Color{ 255, 220, 140, 60 });  // Large atmospheric bloom
    }

    void DrawProps(Shader shader) {
        if (!propsLoaded) return;
        bushModel.materials[0].shader = shader;
        rockModel.materials[0].shader = shader;

        for (const auto& p : props) {
            if (p.type == 0) {
                DrawModelEx(bushModel, p.position, { 0.0f, 1.0f, 0.0f }, p.rotY, { p.scale, p.scale * 0.85f, p.scale }, Color{ 76, 132, 64, 255 });
            } else {
                DrawModelEx(rockModel, p.position, { 0.0f, 1.0f, 0.0f }, p.rotY, { p.scale, p.scale * 0.7f, p.scale }, Color{ 110, 115, 120, 255 });
            }
        }
    }

    float GetHeight(float x, float z) const {
        return Physics::SampleHeightBilinear(heights, resolution, worldSize, x, z);
    }

    int Resolution() const { return resolution; }
    float WorldSize() const { return worldSize; }

private:
    struct PropInstance {
        Vector3 position;
        float scale;
        float rotY;
        int type;
    };

    std::vector<float> heights;
    int resolution = 0;
    float worldSize = 0.0f;
    Model model{};
    bool modelLoaded = false;
    Model bushModel{};
    Model rockModel{};
    bool propsLoaded = false;
    std::vector<PropInstance> props;

    float HeightFunction(float x, float z) const {
        float n = Noise::FBM(x * Config::HILL_FREQUENCY, z * Config::HILL_FREQUENCY, 5, 2.0f, 0.5f);
        float raised = (n * 0.5f + 0.5f) - Config::HILL_FLAT_CUT;
        if (raised < 0.0f) raised = 0.0f;
        float norm = raised / (1.0f - Config::HILL_FLAT_CUT);
        return std::pow(norm, Config::HILL_POWER) * Config::HILL_HEIGHT;
    }

    void BuildHeightArray() {
        resolution = Config::TERRAIN_RES;
        worldSize = Config::TERRAIN_SIZE;
        heights.assign(static_cast<size_t>(resolution) * static_cast<size_t>(resolution), 0.0f);
        float half = worldSize * 0.5f;
        for (int z = 0; z < resolution; z++) {
            for (int x = 0; x < resolution; x++) {
                float wx = -half + worldSize * (static_cast<float>(x) / static_cast<float>(resolution - 1));
                float wz = -half + worldSize * (static_cast<float>(z) / static_cast<float>(resolution - 1));
                heights[z * resolution + x] = HeightFunction(wx, wz);
            }
        }
    }

    void BuildMesh() {
        int res = resolution;
        Mesh mesh{};
        mesh.vertexCount = res * res;
        mesh.triangleCount = (res - 1) * (res - 1) * 2;

        mesh.vertices = (float*)MemAlloc(mesh.vertexCount * 3 * sizeof(float));
        mesh.normals = (float*)MemAlloc(mesh.vertexCount * 3 * sizeof(float));
        mesh.texcoords = (float*)MemAlloc(mesh.vertexCount * 2 * sizeof(float));
        mesh.colors = (unsigned char*)MemAlloc(mesh.vertexCount * 4 * sizeof(unsigned char));
        mesh.indices = (unsigned short*)MemAlloc(mesh.triangleCount * 3 * sizeof(unsigned short));

        float half = worldSize * 0.5f;
        Vector3 sunDir = Vector3Normalize({ 0.6f, 0.8f, 0.4f });

        for (int z = 0; z < res; z++) {
            for (int x = 0; x < res; x++) {
                int idx = z * res + x;
                float wx = -half + worldSize * (static_cast<float>(x) / static_cast<float>(res - 1));
                float wz = -half + worldSize * (static_cast<float>(z) / static_cast<float>(res - 1));
                float h = heights[idx];

                mesh.vertices[idx * 3 + 0] = wx;
                mesh.vertices[idx * 3 + 1] = h;
                mesh.vertices[idx * 3 + 2] = wz;

                Vector3 n = Physics::SampleTerrainNormal(heights, res, worldSize, wx, wz);
                mesh.normals[idx * 3 + 0] = n.x;
                mesh.normals[idx * 3 + 1] = n.y;
                mesh.normals[idx * 3 + 2] = n.z;

                mesh.texcoords[idx * 2 + 0] = static_cast<float>(x) / static_cast<float>(res - 1);
                mesh.texcoords[idx * 2 + 1] = static_cast<float>(z) / static_cast<float>(res - 1);

                float microDetail = Noise::FBM(wx * 0.4f, wz * 0.4f, 4, 2.2f, 0.45f);
                float slope = 1.0f - n.y;

                float r = 55.0f + microDetail * 25.0f;
                float g = 125.0f + microDetail * 35.0f;
                float b = 45.0f + microDetail * 15.0f;

                if (slope > 0.35f) {
                    float rockBlend = Noise::Saturate((slope - 0.35f) * 3.0f);
                    r = Noise::Lerp(r, 105.0f, rockBlend);
                    g = Noise::Lerp(g, 110.0f, rockBlend);
                    b = Noise::Lerp(b, 115.0f, rockBlend);
                }

                float ndotl = Vector3DotProduct(n, sunDir);
                float lighting = std::clamp(ndotl * 0.75f + 0.35f, 0.15f, 1.0f);

                r *= lighting;
                g *= lighting;
                b *= lighting;

                mesh.colors[idx * 4 + 0] = static_cast<unsigned char>(std::clamp(r, 0.0f, 255.0f));
                mesh.colors[idx * 4 + 1] = static_cast<unsigned char>(std::clamp(g, 0.0f, 255.0f));
                mesh.colors[idx * 4 + 2] = static_cast<unsigned char>(std::clamp(b, 0.0f, 255.0f));
                mesh.colors[idx * 4 + 3] = 255;
            }
        }

        int ii = 0;
        for (int z = 0; z < res - 1; z++) {
            for (int x = 0; x < res - 1; x++) {
                unsigned short i00 = z * res + x;
                unsigned short i10 = z * res + x + 1;
                unsigned short i01 = (z + 1) * res + x;
                unsigned short i11 = (z + 1) * res + x + 1;

                mesh.indices[ii++] = i00; mesh.indices[ii++] = i01; mesh.indices[ii++] = i10;
                mesh.indices[ii++] = i10; mesh.indices[ii++] = i01; mesh.indices[ii++] = i11;
            }
        }

        UploadMesh(&mesh, false);
        model = LoadModelFromMesh(mesh);
        modelLoaded = true;
    }

    void BuildProps() {
        bushModel = LoadModelFromMesh(GenMeshSphere(1.0f, 6, 6));
        rockModel = LoadModelFromMesh(GenMeshCube(1.0f, 1.0f, 1.0f));
        propsLoaded = true;
    }

    void ScatterProps(uint32_t& rngState) {
        props.clear();
        float half = worldSize * 0.45f;
        for (int i = 0; i < 220; i++) {
            float x = Noise::RandomRange(rngState, -half, half);
            float z = Noise::RandomRange(rngState, -half, half);
            float h = GetHeight(x, z);

            PropInstance p;
            p.position = { x, h, z };
            p.scale = Noise::RandomRange(rngState, 0.3f, 1.4f);
            p.rotY = Noise::RandomRange(rngState, 0.0f, 360.0f);
            p.type = (Noise::RandomFloat01(rngState) > 0.45f) ? 0 : 1;
            props.push_back(p);
        }
    }
};


// ============================================================================
// Stair environment used by the AI and rendered in the world.
// ============================================================================
struct StairStep { Vector3 center{}; float width=2.5f; float depth=0.9f; float height=0.25f; };
static const Terrain* gTerrain = nullptr;
static std::vector<StairStep> gStairs;
static float WorldGroundHeight(float x, float z) {
    float h = gTerrain ? gTerrain->GetHeight(x,z) : 0.f;
    for (const StairStep& s : gStairs) {
        if (std::fabs(x-s.center.x) <= s.width*0.5f && std::fabs(z-s.center.z) <= s.depth*0.5f)
            h = std::max(h, s.height);
    }
    return h;
}
static std::function<float(float,float)> gGroundHeight = WorldGroundHeight;

// ================================ CRIATURA ===================================
// Corpo: X = frente, Y = cima, Z = lateral. Quadrupede: 4 patas; bipede: 2 pernas digitigradas.
static const float MASS = 10.f, INERT_X = 0.26f, INERT_Y = 0.88f, INERT_Z = 0.72f;
static const float L1 = 0.35f, L2 = 0.35f;
static const float TORSO_LX = 0.9f, TORSO_LY = 0.2f, TORSO_LZ = 0.5f;
static const V3 HIP[4] = {{0.38f, -0.05f, 0.2f}, {0.38f, -0.05f, -0.2f}, {-0.38f, -0.05f, 0.2f}, {-0.38f, -0.05f, -0.2f}};
static const float KN = 3000.f, CN = 100.f, KT = 2000.f, CT = 40.f, MU = 1.0f;
static constexpr float SPHERE_STATIC_FRICTION = 1.05f;
static constexpr float SPHERE_TANGENT_STIFFNESS = 240.f;
static constexpr float SPHERE_TANGENT_DAMPING = 30.f;
static const float SERVO_RATE = 15.f;
static const float BALANCE_BODY_CLEARANCE = 0.603f; // places the neutral feet near load-bearing contact, avoiding launch impulses
static constexpr float FOOT_COLLISION_RADIUS = 0.065f; // quadruped toe collider
// Digitigrade anatomy: the calcaneal heel is raised, the metatarsus slopes
// down/forward, and only the small toe pad normally bears the load.
static constexpr float BIPED_HOCK_BACK = 0.040f;
static constexpr float BIPED_HOCK_UP = 0.055f;
static constexpr float BIPED_METATARSAL_FORWARD = 0.075f;
static constexpr float BIPED_METATARSAL_DROP = 0.030f;
static constexpr float BIPED_TOE_FORWARD = 0.190f;
static constexpr float BIPED_TOE_DROP = 0.095f;
static constexpr float BIPED_HOCK_COLLISION_RADIUS = 0.023f;
static constexpr float BIPED_METATARSAL_COLLISION_RADIUS = 0.027f;
static constexpr float BIPED_TOE_COLLISION_RADIUS = 0.034f;
static constexpr float BIPED_FOOT_WIDTH = 0.105f;

struct Creature {
    bool biped = false; // morphology only; policy still controls every active joint
    V3 pos, vel, ax{1, 0, 0}, ay{0, 1, 0}, az{0, 0, 1}, wb;
    float q[N_ACT], cmd[N_ACT];
    V3 fbPrev[4], kneePrev[4]; bool fbInit = false; bool floorContact = false;
    bool torsoSphereContactPrev = false;
    V3 torsoSphereAnchor{};
    bool contact[4]; float anchX[4], anchZ[4], fn[4];
    // Contact material points on the moving sphere; this is physical static
    // friction, not a scripted balancing controller.
    V3 sphereAnchor[4][9]; // biped: hock/metatarsus/toe x outer/centre/inner rows
    bool sphereAnchorActive[4][9]{};
    float soleAnchorX[4][9]{}, soleAnchorZ[4][9]{};
    bool soleGroundContact[4][9]{};

    int limbCount() const { return biped ? 2 : 4; }
    int activeActionCount() const { return limbCount() * 3; }
    V3 hipLocal(int l) const {
        if (biped) return {0.0f, -0.24f, (l == 0) ? 0.14f : -0.14f};
        return HIP[l];
    }
    V3 torsoDimensions() const {
        // A digitigrade biped has a taller, narrower trunk; the quadruped keeps
        // its original horizontal torso dimensions.
        return biped ? V3{0.40f, 0.68f, 0.32f}
                     : V3{TORSO_LX, TORSO_LY, TORSO_LZ};
    }

    void reset(float yaw) {
        pos = {0, biped ? 1.000f : 0.69f, 0}; vel = {}; wb = {};
        ax = {std::cos(yaw), 0, -std::sin(yaw)}; ay = {0, 1, 0}; az = {std::sin(yaw), 0, std::cos(yaw)};
        for (int i = 0; i < N_ACT; i++) q[i] = cmd[i] = 0;
        for (int l = 0; l < 4; l++) {
            contact[l] = false; fn[l] = 0; anchX[l] = anchZ[l] = 0;
            for (int slot = 0; slot < 9; ++slot) {
                sphereAnchor[l][slot] = {};
                sphereAnchorActive[l][slot] = false;
                soleAnchorX[l][slot] = soleAnchorZ[l][slot] = 0.f;
                soleGroundContact[l][slot] = false;
            }
            fbPrev[l] = kneePrev[l] = {};
        }
        fbInit = false;
        floorContact = false;
        torsoSphereContactPrev = false;
        torsoSphereAnchor = {};
    }
    V3 toWorld(V3 b) const { return ax * b.x + ay * b.y + az * b.z; }
    V3 toBody(V3 w) const { return {dot(w, ax), dot(w, ay), dot(w, az)}; }

    // Limb kinematics. Walking keeps its original mapping; sphere training uses
    // the full neural action range instead of narrow, inward-only joint clamps.
    // q is already in [-1, 1] because it is the tanh output of the policy.
    void legPoints(int l, V3& knee, V3& foot, bool sphereMode = false) const {
        const V3 hip = hipLocal(l);
        const float hipAction = q[l * 3];
        const float abductionAction = q[l * 3 + 1];
        const float kneeAction = q[l * 3 + 2];
        const float side = hip.z >= 0.f ? 1.f : -1.f;
        float th, ph, kn, terminalAngle;

        if (biped) {
            // Digitigrade biped: each leg has independent hip pitch, lateral
            // abduction and knee flexion. The toe endpoint extends forward from
            // the hock; all six joint targets remain neural-policy outputs.
            th = 1.35f * hipAction;
            ph = 0.85f * abductionAction;
            kn = 0.30f + 0.95f * kneeAction; // broad travel around a digitigrade hock rest angle
            terminalAngle = th - kn;          // raised heel / backward hock, toes forward
        } else if (sphereMode) {
            // Broad continuous quadruped joint travel, signed inward/outward roll.
            th = 1.40f * hipAction;
            ph = 1.00f * abductionAction;
            kn = 0.75f * (kneeAction + 1.0f);
            terminalAngle = th + (hip.x > 0.f ? kn : -kn);
        } else {
            // Preserve the original quadruped walking articulation model.
            th = hipAction * 0.6f;
            ph = 0.1f + abductionAction * 0.35f;
            kn = 0.7f + kneeAction * 0.6f;
            terminalAngle = th - kn;
        }

        // Biped segments are slightly shorter so the hock stays lifted above
        // the ground while the toes can reach it. The quadruped keeps its old lengths.
        const float upperLength = biped ? 0.32f : L1;
        const float lowerLength = biped ? 0.32f : L2;
        const float x1 = upperLength * std::sin(th), y1 = -upperLength * std::cos(th);
        const float x2 = x1 + lowerLength * std::sin(terminalAngle);
        const float y2 = y1 - lowerLength * std::cos(terminalAngle);
        const float cp = std::cos(ph), sp = std::sin(ph);
        knee = hip + V3{x1, y1 * cp, -y1 * sp * side};
        foot = hip + V3{x2, y2 * cp, -y2 * sp * side};
        // In biped mode this endpoint is the ankle/hock. The foot is a separate,
        // finite sole extending both backward (heel) and forward (toes).
    }

    // Three local points along a digitigrade foot: raised calcaneus, sloped
    // metatarsus, toe pad. These points are shared by rendering and physics.
    void bipedFootPoints(int l, V3& heel, V3& middle, V3& toe, bool sphereMode = false) const {
        V3 knee, ankle;
        legPoints(l, knee, ankle, sphereMode);
        // The heel/calcaneus is behind AND above the ankle; it is not a flat
        // load-bearing heel. The metatarsus angles down and forward to a compact
        // toe pad. That raised heel + toe-only neutral support is the key
        // digitigrade geometry (not a flat palmigrade sole).
        heel   = ankle + V3{-BIPED_HOCK_BACK, BIPED_HOCK_UP, 0.f};
        middle = ankle + V3{ BIPED_METATARSAL_FORWARD, -BIPED_METATARSAL_DROP, 0.f};
        toe    = ankle + V3{ BIPED_TOE_FORWARD, -BIPED_TOE_DROP, 0.f};
    }

    // Closest point from the sphere centre to the oriented torso box. This is a
    // collision primitive only: it does not apply any preferred orientation.
    bool torsoSphereContact(V3 center, float radius, V3& point, V3& normal,
                            float& penetration) const {
        const V3 dimensions = torsoDimensions();
        const float hx = dimensions.x * 0.5f;
        const float hy = dimensions.y * 0.5f;
        const float hz = dimensions.z * 0.5f;
        const V3 toCenter = center - pos;
        const float lx = dot(toCenter, ax);
        const float ly = dot(toCenter, ay);
        const float lz = dot(toCenter, az);
        float cx = clampf(lx, -hx, hx);
        float cy = clampf(ly, -hy, hy);
        float cz = clampf(lz, -hz, hz);
        point = pos + ax * cx + ay * cy + az * cz;
        V3 delta = point - center;
        float d = len(delta);

        // If the sphere centre has numerically entered the box, select its
        // nearest face so the constraint still has a valid outward normal.
        if (d < 1e-6f) {
            const float dx = hx - std::fabs(lx);
            const float dy = hy - std::fabs(ly);
            const float dz = hz - std::fabs(lz);
            if (dx <= dy && dx <= dz) {
                cx = (lx >= 0.f ? hx : -hx);
                point = pos + ax * cx + ay * clampf(ly, -hy, hy) + az * clampf(lz, -hz, hz);
                normal = ax * (lx >= 0.f ? 1.f : -1.f);
            } else if (dy <= dz) {
                cy = (ly >= 0.f ? hy : -hy);
                point = pos + ax * clampf(lx, -hx, hx) + ay * cy + az * clampf(lz, -hz, hz);
                normal = ay * (ly >= 0.f ? 1.f : -1.f);
            } else {
                cz = (lz >= 0.f ? hz : -hz);
                point = pos + ax * clampf(lx, -hx, hx) + ay * clampf(ly, -hy, hy) + az * cz;
                normal = az * (lz >= 0.f ? 1.f : -1.f);
            }
            d = len(point - center);
        } else {
            normal = delta * (1.f / d);
        }
        penetration = std::max(0.f, radius - d);
        return penetration > 0.f;
    }

    void physics(float dt, bool sphereMode = false, V3 sphereCenter = {},
                 V3 sphereVelocity = {}, float sphereRadius = 1.7f,
                 V3 sphereAngularVelocity = {}) {
        // Motor servos only follow the policy's requested joint values. There is
        // no programmed balance pose or corrective controller.
        const float servoRate = sphereMode ? 10.0f : SERVO_RATE;
        const float al = 1.f - std::exp(-servoRate * dt);
        for (int i = 0; i < N_ACT; i++) q[i] += (cmd[i] - q[i]) * al;

        V3 F{0, -9.81f * MASS, 0}, T{};
        const V3 wW = toWorld(wb);
        bool wasSphereAnchorActive[4][9]{};
        std::memcpy(wasSphereAnchorActive, sphereAnchorActive, sizeof sphereAnchorActive);
        bool sphereAnchorUsedNow[4][9]{};
        bool soleGroundUsedNow[4][9]{};
        V3 kneeB[4], footB[4], kneeVelB[4], footVelB[4];

        auto applyCapsuleContact = [&](V3 localA, V3 localB, V3 velocityA, V3 velocityB,
                                       float colliderRadius, int leg, bool lowerSegment, int anchorSlot) {
            const V3 a = pos + toWorld(localA);
            const V3 b = pos + toWorld(localB);
            const V3 ab = b - a;
            const float ab2 = dot(ab, ab);
            const float t = ab2 > 1e-9f ? clampf(dot(sphereCenter - a, ab) / ab2, 0.f, 1.f) : 0.f;
            const V3 point = a + ab * t;
            const V3 localPointVelocity = velocityA * (1.f - t) + velocityB * t;
            const V3 r = point - pos;
            const V3 fromCenter = point - sphereCenter;
            const float distance = len(fromCenter);
            const float collisionRadius = sphereRadius + colliderRadius;
            if (distance >= collisionRadius || distance <= 1e-7f) return;

            const V3 normal = fromCenter * (1.f / distance);
            const V3 pointVelocity = vel + cross(wW, r) + toWorld(localPointVelocity);
            const V3 surfaceVelocity = sphereVelocity + cross(sphereAngularVelocity, normal * sphereRadius);
            const V3 relativeVelocity = pointVelocity - surfaceVelocity;
            const float vn = dot(relativeVelocity, normal);
            const float penetration = collisionRadius - distance;
            constexpr float SPHERE_KN = 2600.f;
            constexpr float SPHERE_CN = 65.f;
            constexpr float SPHERE_MAX_NORMAL_FORCE = 300.f;
            const float fnLocal = clampf(SPHERE_KN * penetration - SPHERE_CN * vn,
                                         0.f, SPHERE_MAX_NORMAL_FORCE);
            if (fnLocal <= 0.f) return;

            V3 force = normal * fnLocal;
            const V3 tangentVelocity = relativeVelocity - normal * vn;
            // Every limb segment has dynamic Coulomb-like friction. The distal
            // part of a lower limb additionally keeps a material-point anchor so
            // a planted foot can resist sliding instead of being reset each frame.
            const bool useStaticAnchor = lowerSegment && anchorSlot >= 0 && anchorSlot < 9
                                      && (biped || t >= 0.70f);
            if (useStaticAnchor) {
                const V3 footSurfacePoint = point - normal * colliderRadius;
                // A separate persistent material point for each heel/centre/toe row.
                if (!wasSphereAnchorActive[leg][anchorSlot]) {
                    sphereAnchor[leg][anchorSlot] = norm(footSurfacePoint - sphereCenter) * sphereRadius;
                }
                const V3 anchorWorld = sphereCenter + sphereAnchor[leg][anchorSlot];
                V3 tangentError = footSurfacePoint - anchorWorld;
                tangentError = tangentError - normal * dot(tangentError, normal);
                V3 tangentForce = tangentError * (-SPHERE_TANGENT_STIFFNESS)
                                + tangentVelocity * (-SPHERE_TANGENT_DAMPING);
                const float tangentMagnitude = len(tangentForce);
                const float maxFriction = SPHERE_STATIC_FRICTION * fnLocal;
                if (tangentMagnitude > maxFriction && tangentMagnitude > 1e-6f) {
                    tangentForce = tangentForce * (maxFriction / tangentMagnitude);
                    const V3 newAnchor = footSurfacePoint - sphereCenter;
                    sphereAnchor[leg][anchorSlot] = norm(newAnchor) * sphereRadius;
                }
                sphereAnchorUsedNow[leg][anchorSlot] = true;
                force = force + tangentForce;
                contact[leg] = true;
                fn[leg] += fnLocal;
            } else {
                // Sliding friction for shins/thighs that brush the sphere. This
                // does not constrain joints or create self-collisions.
                V3 tangentForce = tangentVelocity * (-28.f);
                const float maxDynamicFriction = 0.82f * fnLocal;
                const float tangentMagnitude = len(tangentForce);
                if (tangentMagnitude > maxDynamicFriction && tangentMagnitude > 1e-6f)
                    tangentForce = tangentForce * (maxDynamicFriction / tangentMagnitude);
                force = force + tangentForce;
            }
            F = F + force;
            T = T + cross(r, force);
        };

        auto applyGroundPoint = [&](V3 localPoint, V3 localPointVelocity, float colliderRadius) {
            const V3 r = toWorld(localPoint);
            const V3 point = pos + r;
            const float groundY = gGroundHeight ? gGroundHeight(point.x, point.z) : 0.f;
            const float penetration = colliderRadius + groundY - point.y;
            const V3 pointVelocity = vel + cross(wW, r) + toWorld(localPointVelocity);
            if (penetration <= 0.f && !(point.y < groundY + 0.015f && pointVelocity.y < 0.f)) return;
            if (penetration > 0.f) floorContact = true;
            constexpr float GROUND_KN = 2600.f;
            constexpr float GROUND_CN = 90.f;
            const float fnGround = clampf(GROUND_KN * std::max(0.f, penetration)
                                          - GROUND_CN * pointVelocity.y, 0.f, 450.f);
            if (fnGround > 0.f) {
                const V3 force{0.f, fnGround, 0.f};
                F = F + force;
                T = T + cross(r, force);
                floorContact = true;
            }
        };

        auto applyWalkingSoleContact = [&](V3 localPoint, V3 localPointVelocity,
                                           float colliderRadius, int leg, int slot) {
            const V3 r = toWorld(localPoint);
            const V3 point = pos + r;
            const float groundY = gGroundHeight ? gGroundHeight(point.x, point.z) : 0.f;
            const float penetration = colliderRadius + groundY - point.y;
            const V3 pointVelocity = vel + cross(wW, r) + toWorld(localPointVelocity);
            if (penetration <= 0.f && !(point.y < groundY + colliderRadius + 0.015f && pointVelocity.y < 0.f))
                return;
            const float fnSole = clampf(KN * std::max(0.f, penetration) - CN * pointVelocity.y,
                                        0.f, 450.f);
            if (fnSole <= 0.f) return;
            if (!soleGroundContact[leg][slot]) {
                soleAnchorX[leg][slot] = point.x;
                soleAnchorZ[leg][slot] = point.z;
            }
            float fx = -KT * (point.x - soleAnchorX[leg][slot]) - CT * pointVelocity.x;
            float fz = -KT * (point.z - soleAnchorZ[leg][slot]) - CT * pointVelocity.z;
            const float fmax = MU * fnSole;
            const float fm = std::sqrt(fx * fx + fz * fz);
            if (fm > fmax && fm > 1e-6f) {
                const float scale = fmax / fm;
                fx *= scale; fz *= scale;
                soleAnchorX[leg][slot] = point.x + fx / KT;
                soleAnchorZ[leg][slot] = point.z + fz / KT;
            }
            const V3 force{fx, fnSole, fz};
            F = F + force;
            T = T + cross(r, force);
            soleGroundUsedNow[leg][slot] = true;
            contact[leg] = true;
            fn[leg] += fnSole;
            floorContact = true;
        };

        for (int l = 0; l < limbCount(); l++) {
            legPoints(l, kneeB[l], footB[l], sphereMode);
            kneeVelB[l] = fbInit ? (kneeB[l] - kneePrev[l]) * (1.f / dt) : V3{};
            footVelB[l] = fbInit ? (footB[l] - fbPrev[l]) * (1.f / dt) : V3{};
            kneePrev[l] = kneeB[l];
            fbPrev[l] = footB[l];
            fn[l] = 0.f;
            contact[l] = false;
            if (sphereMode) {
                // Upper leg and shin are capsules. A biped's separate heel-to-toe
                // sole has three parallel contact rows (inside/centre/outside).
                // No self-collision or agent-agent collision is introduced.
                applyCapsuleContact(hipLocal(l), kneeB[l], V3{}, kneeVelB[l], 0.050f, l, false, -1);
                if (biped) {
                    applyCapsuleContact(kneeB[l], footB[l], kneeVelB[l], footVelB[l],
                                        0.045f, l, false, -1);
                    V3 heel, middle, toe;
                    bipedFootPoints(l, heel, middle, toe, sphereMode);
                    const V3 digitigradePoints[3] = {heel, middle, toe};
                    const float digitigradeRadii[3] = {
                        BIPED_HOCK_COLLISION_RADIUS,
                        BIPED_METATARSAL_COLLISION_RADIUS,
                        BIPED_TOE_COLLISION_RADIUS
                    };
                    const float lateralRows[3] = {-BIPED_FOOT_WIDTH * 0.28f, 0.f,
                                                   BIPED_FOOT_WIDTH * 0.28f};
                    for (int segment = 0; segment < 3; ++segment) {
                        for (int row = 0; row < 3; ++row) {
                            const int slot = segment * 3 + row;
                            const V3 p = digitigradePoints[segment] + V3{0.f, 0.f, lateralRows[row]};
                            // Nine compact contact pads on an angled digitigrade
                            // foot. No heel-to-toe flat capsule: only the toe row
                            // normally reaches the support surface.
                            applyCapsuleContact(p, p, footVelB[l], footVelB[l],
                                                digitigradeRadii[segment], l, true, slot);
                            applyGroundPoint(p, footVelB[l], digitigradeRadii[segment]);
                        }
                    }
                    applyGroundPoint(kneeB[l], kneeVelB[l], 0.055f);
                    // The hock endpoint is part of the shin collider and may
                    // strike the floor after a fall; it is not a standing sole.
                } else {
                    applyCapsuleContact(kneeB[l], footB[l], kneeVelB[l], footVelB[l],
                                        FOOT_COLLISION_RADIUS, l, true, 0);
                    // A fallen agent still collides with the ground plane.
                    applyGroundPoint(kneeB[l], kneeVelB[l], 0.055f);
                    applyGroundPoint(footB[l], footVelB[l], FOOT_COLLISION_RADIUS);
                }
            } else if (biped) {
                // A biped's sloped metatarsus and toe pads contact the ground separately.
                // In neutral posture, the elevated heel should not act as a flat support.
                V3 heel, middle, toe;
                bipedFootPoints(l, heel, middle, toe, false);
                const V3 digitigradePoints[3] = {heel, middle, toe};
                const float digitigradeRadii[3] = {
                    BIPED_HOCK_COLLISION_RADIUS,
                    BIPED_METATARSAL_COLLISION_RADIUS,
                    BIPED_TOE_COLLISION_RADIUS
                };
                const float lateralRows[3] = {-BIPED_FOOT_WIDTH * 0.28f, 0.f,
                                               BIPED_FOOT_WIDTH * 0.28f};
                // Full physical width comes from separate small pads across the
                // foot, while the raised hock remains out of contact in neutral stance.
                for (int segment = 0; segment < 3; ++segment) {
                    for (int row = 0; row < 3; ++row) {
                        const int slot = segment * 3 + row;
                        const V3 p = digitigradePoints[segment] + V3{0.f, 0.f, lateralRows[row]};
                        applyWalkingSoleContact(p, footVelB[l], digitigradeRadii[segment], l, slot);
                    }
                }
            } else {
                const V3 r = toWorld(footB[l]);
                const V3 pf = pos + r;
                const float groundY = gGroundHeight ? gGroundHeight(pf.x, pf.z) : 0.f;
                const float penetration = groundY - pf.y;
                if (penetration > 0.f) {
                    const V3 vf = vel + cross(wW, r) + toWorld(footVelB[l]);
                    float f_n = KN * penetration - CN * vf.y;
                    if (f_n < 0.f) f_n = 0.f;
                    if (!contact[l]) { contact[l] = true; anchX[l] = pf.x; anchZ[l] = pf.z; }
                    float fx = -KT * (pf.x - anchX[l]) - CT * vf.x;
                    float fz = -KT * (pf.z - anchZ[l]) - CT * vf.z;
                    const float fmax = MU * f_n;
                    const float fm = std::sqrt(fx * fx + fz * fz);
                    if (fm > fmax) {
                        const float scale = fm > 1e-6f ? fmax / fm : 0.f;
                        fx *= scale; fz *= scale;
                        anchX[l] = pf.x + fx / KT; anchZ[l] = pf.z + fz / KT;
                    }
                    const V3 force{fx, f_n, fz};
                    F = F + force; T = T + cross(r, force); fn[l] = f_n;
                } else contact[l] = false;
            }
        }
        // Commit per-row contact history after all three biped sole rows were tested.
        if (sphereMode) {
            for (int l = 0; l < limbCount(); ++l)
                for (int slot = 0; slot < 9; ++slot)
                    sphereAnchorActive[l][slot] = sphereAnchorUsedNow[l][slot];
        } else {
            for (int l = 0; l < limbCount(); ++l)
                for (int slot = 0; slot < 9; ++slot)
                    soleGroundContact[l][slot] = soleGroundUsedNow[l][slot];
        }

        // The torso is a real oriented box collider against the sphere. Besides
        // normal force, it receives tangential friction when sliding on the shell.
        // This is environment contact only: no agent-agent or self-collision.
        if (sphereMode) {
            const bool wasTorsoContact = torsoSphereContactPrev;
            torsoSphereContactPrev = false;
            V3 point{}, normal{}; float penetration = 0.f;
            if (torsoSphereContact(sphereCenter, sphereRadius, point, normal, penetration)) {
                const V3 r = point - pos;
                const V3 pointVelocity = vel + cross(wW, r);
                const V3 surfaceVelocity = sphereVelocity + cross(sphereAngularVelocity, point - sphereCenter);
                const V3 relativeVelocity = pointVelocity - surfaceVelocity;
                const float vn = dot(relativeVelocity, normal);
                constexpr float BODY_KN = 4000.f;
                constexpr float BODY_CN = 145.f;
                constexpr float BODY_MAX_FORCE = 600.f;
                const float fnBody = clampf(BODY_KN * penetration - BODY_CN * vn,
                                            0.f, BODY_MAX_FORCE);
                V3 force = normal * fnBody;
                if (!wasTorsoContact) torsoSphereAnchor = norm(point - sphereCenter) * sphereRadius;
                const V3 anchorWorld = sphereCenter + torsoSphereAnchor;
                V3 tangentError = point - anchorWorld;
                tangentError = tangentError - normal * dot(tangentError, normal);
                const V3 tangentVelocity = relativeVelocity - normal * vn;
                V3 tangentForce = tangentError * (-420.f) + tangentVelocity * (-42.f);
                const float frictionLimit = 0.90f * fnBody;
                const float tangentMagnitude = len(tangentForce);
                if (tangentMagnitude > frictionLimit && tangentMagnitude > 1e-6f) {
                    tangentForce = tangentForce * (frictionLimit / tangentMagnitude);
                    torsoSphereAnchor = norm(point - sphereCenter) * sphereRadius;
                }
                force = force + tangentForce;
                torsoSphereContactPrev = true;
                F = F + force;
                T = T + cross(r, force);
            }
        } else {
            torsoSphereContactPrev = false;
        }
        if (sphereMode) {
            // Oriented-box contact with the floor catches the torso if the policy
            // falls off the ball. This is environment collision, not self-collision.
            const V3 dimensions = torsoDimensions();
            const V3 localCorner{
                ax.y >= 0.f ? -dimensions.x * 0.5f : dimensions.x * 0.5f,
                ay.y >= 0.f ? -dimensions.y * 0.5f : dimensions.y * 0.5f,
                az.y >= 0.f ? -dimensions.z * 0.5f : dimensions.z * 0.5f
            };
            const V3 r = toWorld(localCorner);
            const V3 point = pos + r;
            const V3 pointVelocity = vel + cross(wW, r);
            const float groundY = gGroundHeight ? gGroundHeight(point.x, point.z) : 0.f;
            if (point.y < groundY + 0.015f && (point.y < groundY || pointVelocity.y < 0.f)) {
                const float groundY2 = gGroundHeight ? gGroundHeight(point.x, point.z) : 0.f;
                const float penetration = std::max(0.f, groundY2 - point.y);
                if (penetration > 0.f) floorContact = true;
                const float fnFloor = clampf(3800.f * penetration - 150.f * pointVelocity.y,
                                             0.f, 800.f);
                if (fnFloor > 0.f) {
                    const V3 force{0.f, fnFloor, 0.f};
                    F = F + force;
                    T = T + cross(r, force);
                    floorContact = true;
                }
            }
        }

        fbInit = true;
        vel = vel + F * (dt / MASS);
        pos = pos + vel * dt;
        const V3 Tb = toBody(T);
        wb.x += Tb.x / INERT_X * dt;
        wb.y += Tb.y / INERT_Y * dt;
        wb.z += Tb.z / INERT_Z * dt;
        wb = wb * (1.f / (1.f + 0.3f * dt));
        V3 Om = toWorld(wb) * dt;
        const float ang = len(Om);
        if (ang > 1e-9f) {
            const V3 k = Om * (1.f / ang);
            ax = rotAxis(ax, k, ang); ay = rotAxis(ay, k, ang);
        }
        ax = norm(ax);
        az = norm(cross(ax, ay));
        if (len(az) < 1e-6f) {
            // Choose a fallback perpendicular to ax. A fixed world-Z fallback
            // can be parallel to ax and collapse the whole body basis to zero.
            const V3 seed = std::fabs(ax.y) < 0.90f ? V3{0.f, 1.f, 0.f}
                                                    : V3{0.f, 0.f, 1.f};
            az = norm(cross(ax, seed));
        }
        ay = norm(cross(az, ax));
        az = norm(cross(ax, ay));

        if (sphereMode) {
            // Position-level non-penetration pass for fast falls. Forces handle
            // ordinary contact; this only removes deep overlap/tunnelling and
            // removes velocity directed into the sphere instead of adding bounce.
            for (int pass = 0; pass < 3; ++pass) {
                bool corrected = false;
                for (int l = 0; l < limbCount(); ++l) {
                    V3 kB, fB; legPoints(l, kB, fB, sphereMode);
                    V3 segA[5] = {hipLocal(l), kB, {}, {}, {}};
                    V3 segB[5] = {kB, fB, {}, {}, {}};
                    float segRadius[5] = {0.050f, biped ? 0.045f : FOOT_COLLISION_RADIUS, 0.f, 0.f, 0.f};
                    int segCount = 2;
                    if (biped) {
                        V3 heel, middle, toe;
                        bipedFootPoints(l, heel, middle, toe, sphereMode);
                        // Position-level anti-tunnelling uses three small rounded
                        // points. There is deliberately no broad flat sole collider.
                        segA[2] = segB[2] = heel;
                        segA[3] = segB[3] = middle;
                        segA[4] = segB[4] = toe;
                        segRadius[2] = BIPED_HOCK_COLLISION_RADIUS;
                        segRadius[3] = BIPED_METATARSAL_COLLISION_RADIUS;
                        segRadius[4] = BIPED_TOE_COLLISION_RADIUS;
                        segCount = 5;
                    }
                    for (int seg = 0; seg < segCount; ++seg) {
                        const V3 a = pos + toWorld(segA[seg]);
                        const V3 b = pos + toWorld(segB[seg]);
                        const V3 ab = b - a;
                        const float ab2 = dot(ab, ab);
                        const float t = ab2 > 1e-9f ? clampf(dot(sphereCenter - a, ab) / ab2, 0.f, 1.f) : 0.f;
                        const V3 point = a + ab * t;
                        const V3 delta = point - sphereCenter;
                        const float d = len(delta);
                        const float penetration = sphereRadius + segRadius[seg] - d;
                        if (d > 1e-7f && penetration > 0.01f) {
                            const V3 n = delta * (1.f / d);
                            pos = pos + n * std::min(0.08f, penetration - 0.004f);
                            const V3 surfaceVelocity = sphereVelocity + cross(sphereAngularVelocity, n * sphereRadius);
                            const float vn = dot(vel - surfaceVelocity, n);
                            if (vn < 0.f) vel = vel - n * vn;
                            corrected = true;
                        }
                        // Capsule endpoints are also constrained against the floor.
                        const V3 endpointA = pos + toWorld(segA[seg]);
                        const V3 endpointB = pos + toWorld(segB[seg]);
                        const float floorPenA = segRadius[seg] - endpointA.y;
                        const float floorPenB = segRadius[seg] - endpointB.y;
                        const float floorPen = std::max(floorPenA, floorPenB);
                        if (floorPen > 0.004f) {
                            pos.y += std::min(0.08f, floorPen - 0.003f);
                            if (vel.y < 0.f) vel.y = 0.f;
                            floorContact = true;
                            corrected = true;
                        }
                    }
                }
                V3 boxPoint{}, boxNormal{}; float boxPen = 0.f;
                if (torsoSphereContact(sphereCenter, sphereRadius, boxPoint, boxNormal, boxPen)
                    && boxPen > 0.01f) {
                    pos = pos + boxNormal * std::min(0.08f, boxPen - 0.004f);
                    const V3 surfaceVelocity = sphereVelocity + cross(sphereAngularVelocity, boxPoint - sphereCenter);
                    const float vn = dot(vel - surfaceVelocity, boxNormal);
                    if (vn < 0.f) vel = vel - boxNormal * vn;
                    corrected = true;
                }
                const V3 dimensions = torsoDimensions();
                const V3 lowestLocal{
                    ax.y >= 0.f ? -dimensions.x * 0.5f : dimensions.x * 0.5f,
                    ay.y >= 0.f ? -dimensions.y * 0.5f : dimensions.y * 0.5f,
                    az.y >= 0.f ? -dimensions.z * 0.5f : dimensions.z * 0.5f
                };
                const V3 lowestPoint = pos + toWorld(lowestLocal);
                const float groundY3 = gGroundHeight ? gGroundHeight(lowestPoint.x, lowestPoint.z) : 0.f;
                if (lowestPoint.y < groundY3 + 0.004f) {
                    pos.y += std::min(0.08f, groundY3 + 0.004f - lowestPoint.y);
                    if (vel.y < 0.f) vel.y = 0.f;
                    floorContact = true;
                    corrected = true;
                }
                if (!corrected) break;
            }
        }
    }
};


// ============================================================================
// AI runtime: inference only. No Actor-Critic, no rollout, no cloning, no save.
// ============================================================================
struct OrbTarget { V3 p{}; bool active=false; };

class GameAI {
public:
    Brain brain;
    Creature cr;
    OrbTarget orbs[6];
    Act act{};
    float obs[N_IN]{};
    int targetIndex=0;
    int collected=0;
    int lastActive=0;
    int lastGroups=0;
    bool loaded=false;
    bool menu=true;
    bool networkView=false;
    bool manualControl=false;
    int selectedGroup=0;
    bool selectedGroupPinned=false;
    float decisionTimer=0.f;
    float invertedTimer=0.f;
    float resetNotice=0.f;
    int resetCount=0;
    float spawnX=0.f, spawnZ=15.f;

    // Cached 2D neural-network visualization. The hidden texture contains all
    // 512 x 32 = 16,384 neurons; the weight textures summarize connections by group.
    Texture2D neuronMap{};
    Texture2D routerMap{};
    Texture2D winMap{};
    Texture2D woutMap{};
    std::vector<Color> neuronPixels;
    bool networkMapsReady=false;

    bool Load() {
        loaded = std::filesystem::exists(AI_MEMORY_FILE) && brain.load(AI_MEMORY_FILE);
        if (!loaded) std::printf("[IA] memoria nao encontrada: %s\n", AI_MEMORY_FILE);
        return loaded;
    }

    static Color WeightColor(float w) {
        const float a = std::fabs(w);
        if (a < 0.008f) return rgba(22, 27, 35, 255);
        const float t = clampf(a * 360.f, 18.f, 235.f);
        if (w > 0.f) return rgba((int)t, (int)(24.f + 0.72f*t), 12, 255); // excitatory sign
        return rgba(8, (int)(28.f + 0.52f*t), (int)t, 255);              // inhibitory sign
    }

    static Texture2D MakeWeightTexture(int width, int height,
                                        const std::function<float(int,int)>& getWeight) {
        Image img = GenImageColor(width, height, rgba(20,24,30));
        if (img.data) {
            Color* px = static_cast<Color*>(img.data);
            for (int y=0; y<height; ++y)
                for (int x=0; x<width; ++x)
                    px[y*width+x] = WeightColor(getWeight(x,y));
        }
        Texture2D tex = LoadTextureFromImage(img);
        UnloadImage(img);
        if (tex.id) SetTextureFilter(tex, TEXTURE_FILTER_POINT);
        return tex;
    }

    void InitNetworkView() {
        Image nodes = GenImageColor(N_GROUPS, GROUP_SIZE, rgba(24,29,37));
        neuronMap = LoadTextureFromImage(nodes);
        UnloadImage(nodes);
        if (neuronMap.id) SetTextureFilter(neuronMap, TEXTURE_FILTER_POINT);
        neuronPixels.assign((size_t)N_GROUPS * GROUP_SIZE, rgba(24,29,37));
        if (!loaded || brain.router.size() != (size_t)N_GROUPS*N_IN ||
            brain.Win.size() != (size_t)N_NEURONS*N_IN ||
            brain.Wout.size() != (size_t)N_NEURONS*N_ACT) return;

        routerMap = MakeWeightTexture(N_IN, N_GROUPS, [this](int x,int y) {
            return brain.router[(size_t)y*N_IN+x];
        });
        // Exact per-neuron weight maps, not group averages. Within each row
        // (group), each of the 32 neurons gets its own block of synapse columns.
        // This keeps texture dimensions GPU-friendly while preserving every weight.
        winMap = MakeWeightTexture(N_IN*GROUP_SIZE, N_GROUPS, [this](int x,int y) {
            const int input=x/GROUP_SIZE;
            const int unit=x%GROUP_SIZE;
            const int neuron=y*GROUP_SIZE+unit;
            return brain.Win[(size_t)neuron*N_IN+input];
        });
        woutMap = MakeWeightTexture(N_ACT*GROUP_SIZE, N_GROUPS, [this](int x,int y) {
            const int unit=x/N_ACT;
            const int action=x%N_ACT;
            const int neuron=y*GROUP_SIZE+unit;
            return brain.Wout[(size_t)neuron*N_ACT+action];
        });
        networkMapsReady = neuronMap.id && routerMap.id && winMap.id && woutMap.id;
        RefreshNeuronMap();
    }

    void RefreshNeuronMap() {
        if (!neuronMap.id || neuronPixels.size() != (size_t)N_GROUPS*GROUP_SIZE) return;
        bool chosen[N_GROUPS]{};
        for (int k=0;k<act.nGroups;k++) if (act.groups[k]>=0 && act.groups[k]<N_GROUPS)
            chosen[act.groups[k]]=true;
        for (int u=0;u<GROUP_SIZE;u++) {
            for (int g=0;g<N_GROUPS;g++) {
                neuronPixels[(size_t)u*N_GROUPS+g] = chosen[g] ? rgba(76,39,20) : rgba(24,29,37);
            }
        }
        for (int k=0;k<act.nActive;k++) {
            const int j=act.idx[k];
            if (j<0 || j>=N_NEURONS) continue;
            const float intensity=clampf(act.hn[k]*75.f,35.f,255.f);
            neuronPixels[(size_t)(j%GROUP_SIZE)*N_GROUPS+(j/GROUP_SIZE)] =
                rgba(255,(int)(70.f+0.55f*intensity), (int)(18.f+0.12f*intensity));
        }
        UpdateTexture(neuronMap, neuronPixels.data());
    }

    void UnloadNetworkView() {
        if (neuronMap.id) UnloadTexture(neuronMap);
        if (routerMap.id) UnloadTexture(routerMap);
        if (winMap.id) UnloadTexture(winMap);
        if (woutMap.id) UnloadTexture(woutMap);
        neuronMap={}; routerMap={}; winMap={}; woutMap={};
        networkMapsReady=false;
    }

    void Reset(float x,float z) {
        spawnX=x; spawnZ=z;
        const float gy = gGroundHeight ? gGroundHeight(x,z) : 0.f;
        cr.biped=false;
        cr.reset(0.f);
        cr.pos={x,gy+0.69f,z};
        targetIndex=0; collected=0;
        for(int i=0;i<6;i++) orbs[i].active=(i==0);
        decisionTimer=0.f;
        invertedTimer=0.f;
    }

    void RotateBasis(V3 axis, float angle) {
        axis=norm(axis);
        if (len(axis)<1e-6f) return;
        cr.ax=norm(rotAxis(cr.ax,axis,angle));
        cr.ay=norm(rotAxis(cr.ay,axis,angle));
        cr.az=norm(rotAxis(cr.az,axis,angle));
    }

    void ClearPhysicsHistory() {
        cr.vel={}; cr.wb={}; cr.fbInit=false; cr.floorContact=false;
        cr.torsoSphereContactPrev=false;
        for (int l=0;l<4;l++) {
            cr.contact[l]=false; cr.fn[l]=0.f; cr.anchX[l]=cr.anchZ[l]=0.f;
            cr.fbPrev[l]={}; cr.kneePrev[l]={};
            for (int slot=0;slot<9;slot++) {
                cr.sphereAnchorActive[l][slot]=false;
                cr.soleGroundContact[l][slot]=false;
                cr.soleAnchorX[l][slot]=cr.soleAnchorZ[l][slot]=0.f;
            }
        }
    }

    void ManualTransform(float dt) {
        // In manual mode the policy/physics is paused so precise transforms do not fight the controller.
        V3 forward=cr.ax; forward.y=0.f; forward=norm(forward);
        V3 side=cr.az; side.y=0.f; side=norm(side);
        V3 move{};
        if (IsKeyDown(KEY_W)) move=move+forward;
        if (IsKeyDown(KEY_S)) move=move-forward;
        if (IsKeyDown(KEY_D)) move=move+side;
        if (IsKeyDown(KEY_A)) move=move-side;
        if (IsKeyDown(KEY_R)) move.y+=1.f;
        if (IsKeyDown(KEY_F)) move.y-=1.f;
        if (len(move)>1e-6f) cr.pos=cr.pos+norm(move)*(4.0f*dt);
        cr.vel={}; cr.wb={}; cr.fbInit=false;
        const float turn=1.4f*dt;
        if (IsKeyDown(KEY_Q)) RotateBasis({0,1,0},-turn);
        if (IsKeyDown(KEY_E)) RotateBasis({0,1,0}, turn);
        if (IsKeyDown(KEY_T)) RotateBasis(cr.az,turn);
        if (IsKeyDown(KEY_G)) RotateBasis(cr.az,-turn);
        if (IsKeyDown(KEY_Z)) RotateBasis(cr.ax,turn);
        if (IsKeyDown(KEY_X)) RotateBasis(cr.ax,-turn);
    }

    int NearestActiveOrb() const {
        int best=-1; float bd=1e30f;
        for(int i=0;i<6;i++) if(orbs[i].active){
            V3 d=orbs[i].p-cr.pos; float dd=d.x*d.x+d.z*d.z;
            if(dd<bd){bd=dd;best=i;}
        }
        return best;
    }

    void BuildObs() {
        const V3 vb=cr.toBody(cr.vel);
        obs[0]=vb.x/1.5f; obs[1]=vb.y/1.5f; obs[2]=vb.z/1.5f;
        obs[3]=cr.wb.x/3.f; obs[4]=cr.wb.y/3.f; obs[5]=cr.wb.z/3.f;
        obs[6]=cr.ax.y; obs[7]=cr.ay.y; obs[8]=cr.az.y;
        const float gy=gGroundHeight?gGroundHeight(cr.pos.x,cr.pos.z):0.f;
        obs[9]=(cr.pos.y-(gy+0.66f))*3.f;
        const int ni=NearestActiveOrb();
        V3 d=ni>=0?orbs[ni].p-cr.pos:V3{1,0,0}; d.y=0;
        float dist=len(d); V3 db=cr.toBody(dist>1e-6f?d*(1.f/dist):V3{1,0,0});
        obs[10]=db.x; obs[11]=db.z; obs[12]=clampf(dist/6.f,0.f,1.5f);
        for(int i=0;i<N_ACT;i++) obs[13+i]=cr.q[i];
        for(int i=0;i<4;i++) obs[25+i]=cr.contact[i]?1.f:0.f;
        obs[29]=std::sin(kTwoPi*1.5f*decisionTimer);
        obs[30]=std::cos(kTwoPi*1.5f*decisionTimer);
        obs[31]=std::sin(kTwoPi*3.f*decisionTimer);
        obs[32]=std::cos(kTwoPi*3.f*decisionTimer);
    }

    void Decide() {
        if(!loaded) return;
        BuildObs();
        brain.forward(obs,act);
        for(int i=0;i<N_ACT;i++) cr.cmd[i]=std::tanh(act.mu[i]);
        lastActive=act.nActive;
        lastGroups=act.nGroups;
        if (!selectedGroupPinned && act.nGroups > 0) selectedGroup = act.groups[0];
        RefreshNeuronMap();
    }

    void Update(float dt) {
        if (resetNotice > 0.f) resetNotice = std::max(0.f, resetNotice-dt);
        if (manualControl || !loaded) return;
        decisionTimer += dt;
        // Same 20 Hz policy cadence used by the original trained walker.
        static float accumulator=0.f;
        accumulator += dt;
        while(accumulator >= CTRL_DT){
            accumulator -= CTRL_DT;
            Decide();
            for(int i=0;i<SUBSTEPS;i++) cr.physics(PHYS_DT,false);
        }
        // If the torso's local up axis stays pointing down, restart only the AI.
        // Manual transform mode deliberately bypasses this so the user can inspect rotations.
        if (cr.ay.y < -0.30f) invertedTimer += dt;
        else invertedTimer = std::max(0.f, invertedTimer-dt*2.0f);
        if (invertedTimer >= 0.40f) {
            Reset(spawnX,spawnZ);
            resetCount++;
            resetNotice=2.0f;
            std::printf("[IA] caiu de ponta-cabeca; reiniciada no ponto inicial. Total: %d\n", resetCount);
            return;
        }
        for(int i=0;i<6;i++) if(orbs[i].active){
            V3 d=orbs[i].p-cr.pos; d.y=0;
            if(d.x*d.x+d.z*d.z < ORB_PICK_R*ORB_PICK_R){
                orbs[i].active=false; collected++;
                const int next=(i+1)%6;
                orbs[next].active=true; targetIndex=next;
                break;
            }
        }
        if(collected>=6){
            collected=0; for(int i=0;i<6;i++) orbs[i].active=(i==0); targetIndex=0;
        }
    }
};

static void BuildStairsAndOrbs(GameAI& ai) {
    gStairs.clear();
    // Three broad staircases. Each top receives an AI target orb.
    const Vector3 bases[6] = {
        {-12,0,-18},{10,0,-2},{-18,0,10},{15,0,18},{-4,0,28},{24,0,-20}
    };
    const int counts[6]={8,9,8,10,9,8};
    for(int s=0;s<6;s++){
        const float sx=bases[s].x, sz=bases[s].z;
        const bool alongX=(s==1||s==3||s==5);
        const float dir= (s==3)?-1.f:1.f;
        for(int k=0;k<counts[s];k++){
            StairStep st;
            st.width=3.0f; st.depth=1.0f; st.height=0.25f*(k+1);
            if(alongX){ st.center={sx+dir*(k+0.5f),0,sz}; }
            else      { st.center={sx,0,sz+dir*(k+0.5f)}; }
            gStairs.push_back(st);
        }
        // Put each target on the top step.
        const int last=counts[s]-1;
        const float ox=alongX?sx+dir*(last+0.5f):sx;
        const float oz=alongX?sz:sz+dir*(last+0.5f);
        const float gy=WorldGroundHeight(ox,oz);
        ai.orbs[s].p={ox,gy+0.9f,oz};
    }
}

static void DrawStairsAndOrbs(const GameAI& ai) {
    for(const StairStep& s:gStairs){
        const float terrainY=gTerrain?gTerrain->GetHeight(s.center.x,s.center.z):0.f;
        DrawCube({s.center.x,terrainY+s.height*0.5f,s.center.z},s.width,s.height,s.depth,rgba(82,86,92));
        DrawCubeWires({s.center.x,terrainY+s.height*0.5f,s.center.z},s.width,s.height,s.depth,rgba(150,155,165));
    }
    for(int i=0;i<6;i++){
        const OrbTarget&o=ai.orbs[i];
        Color c=o.active?rgba(255,210,65):rgba(90,75,35,100);
        DrawSphere(toRl(o.p),0.32f,c);
        DrawSphereWires(toRl(o.p),0.38f,12,18,rgba(255,235,130,o.active?220:70));
        if(o.active) DrawLine3D(toRl(o.p),toRl(o.p+V3{0,0.75f,0}),rgba(255,230,100,140));
    }
}

static void DrawAI(const GameAI& ai) {
    const Creature& c=ai.cr;
    float m[16]={c.ax.x,c.ax.y,c.ax.z,0,c.ay.x,c.ay.y,c.ay.z,0,c.az.x,c.az.y,c.az.z,0,c.pos.x,c.pos.y,c.pos.z,1};
    rlPushMatrix(); rlMultMatrixf(m);
    DrawCube({0,0,0},0.9f,0.2f,0.5f,rgba(210,220,235));
    DrawCubeWires({0,0,0},0.9f,0.2f,0.5f,rgba(255,170,70));
    rlPopMatrix();
    for(int l=0;l<4;l++){
        V3 k,f;c.legPoints(l,k,f,false);
        V3 h=c.pos+c.toWorld(c.hipLocal(l)), kw=c.pos+c.toWorld(k), fw=c.pos+c.toWorld(f);
        DrawCylinderEx(toRl(h),toRl(kw),0.055f,0.045f,8,rgba(170,190,220));
        DrawCylinderEx(toRl(kw),toRl(fw),0.045f,0.032f,8,rgba(120,145,180));
        DrawSphere(toRl(kw),0.055f,WHITE);
        DrawSphere(toRl(fw),0.065f,c.contact[l]?rgba(90,255,120):rgba(255,100,100));
    }
}

static void DrawAIMenu(const GameAI& ai) {
    if(!ai.menu) return;
    DrawRectangle(16,78,390,310,rgba(5,8,12,220));
    DrawRectangleLines(16,78,390,310,rgba(255,170,70,220));
    DrawText("IA // CEREBRO ESPARSO",30,92,20,WHITE);
    DrawText(ai.loaded?"ESTADO: ATIVA":"ESTADO: MEMORIA AUSENTE",30,122,15,ai.loaded?rgba(110,255,150):RED);
    DrawText(TextFormat("Neuronios ocultos: %d",N_NEURONS),30,146,14,LIGHTGRAY);
    DrawText(TextFormat("Ativos: %d  | grupos: %d",ai.lastActive,ai.lastGroups),30,166,14,LIGHTGRAY);
    DrawText(TextFormat("Orbes: %d / 6",ai.collected),30,186,14,rgba(255,215,70));
    DrawText(TextFormat("Alvo atual: %d",ai.targetIndex+1),30,206,14,rgba(255,215,70));
    DrawText("Treinamento: DESATIVADO",30,226,14,rgba(255,170,70));
    DrawText(ai.manualControl?"MODO: EDITAR IA (fisica pausada)":"MODO: IA autonoma",30,246,13,ai.manualControl?rgba(255,190,80):LIGHTGRAY);
    if (ai.resetNotice>0.f) DrawText("IA reiniciada apos ficar de ponta-cabeca",30,266,12,rgba(255,130,90));
    DrawText(TextFormat("Reinicios automaticos: %d",ai.resetCount),30,286,12,LIGHTGRAY);
    DrawText("I editar | N rede 2D | M painel",30,306,12,GRAY);
    DrawText("Editar: WASD mover, R/F subir/descer",30,324,12,GRAY);
    DrawText("Q/E girar | T/G inclinar | Z/X rolar",30,342,12,GRAY);
    DrawText("Backspace = voltar ao inicio (em editar)",30,360,12,GRAY);
}

static void DrawNeuralNetwork2D(const GameAI& ai) {
    if (!ai.networkView) return;

    const int sw = GetScreenWidth();
    const int sh = GetScreenHeight();
    const float now = (float)GetTime();

    DrawRectangle(8, 64, sw - 16, sh - 72, rgba(3, 5, 9, 248));
    DrawRectangleLines(8, 64, sw - 16, sh - 72, rgba(255, 155, 65, 240));

    DrawText("CEREBRO DA IA // ARQUITETURA REAL + SINAPSES ATIVANDO", 22, 75, 19, WHITE);
    DrawText("Sem linhas: cada ponto representa uma sinapse/contribuicao. Brilho = atividade no quadro atual.",
             22, 100, 11, LIGHTGRAY);

    // ------------------------------------------------------------------------
    // This is the actual architecture used by Brain::forward():
    // 33 inputs -> 512 router groups -> 512 x 32 hidden neurons -> 12 outputs.
    // Nothing is averaged into a fake layer here. The 16,384 hidden neurons
    // are laid out individually, preserving the exact group/unit topology.
    // ------------------------------------------------------------------------
    const float top = 145.f;
    const float bottom = (float)sh - 65.f;
    const float H = bottom - top;

    const float inputX = 35.f;
    const float routerX = 115.f;
    const float hiddenX = 235.f;
    const float hiddenW = std::min(690.f, (float)sw * 0.57f);
    const float outputX = hiddenX + hiddenW + 48.f;
    const float detailX = outputX + 78.f;
    const float detailW = std::max(220.f, (float)sw - detailX - 20.f);

    const int selectedGroup = std::clamp(ai.selectedGroup, 0, N_GROUPS - 1);

    DrawText("33 ENTRADAS", 19, 126, 10, rgba(130, 205, 255));
    DrawText("ROUTER", 91, 126, 10, rgba(255, 180, 90));
    DrawText("512 GRUPOS x 32 = 16.384 NEURONIOS", hiddenX, 126, 10, rgba(255, 190, 95));
    DrawText("12 SAIDAS", outputX - 8.f, 126, 10, rgba(130, 205, 255));

    // Backgrounds for the exact layers.
    DrawRectangle((int)(inputX - 9), (int)top, 24, (int)H, rgba(10, 18, 27));
    DrawRectangle((int)(routerX - 9), (int)top, 24, (int)H, rgba(20, 16, 12));
    DrawRectangle((int)(hiddenX - 7), (int)top, (int)hiddenW + 14, (int)H, rgba(9, 12, 18));
    DrawRectangle((int)(outputX - 9), (int)top, 24, (int)H, rgba(10, 18, 27));

    // 33 input neurons.
    for (int i = 0; i < N_IN; ++i) {
        const float y = top + ((float)i + 0.5f) * H / N_IN;
        const float v = ai.obs[i];
        const float mag = clampf(std::fabs(v), 0.f, 1.f);
        const float pulse = 0.5f + 0.5f * std::sin(now * 5.0f + i * 0.71f);
        Color c = v >= 0.f
            ? rgba(45, (int)(100.f + 145.f * mag), 255, 255)
            : rgba(255, (int)(75.f + 100.f * mag), 55, 255);
        DrawCircleV({inputX, y}, 2.4f + mag * 2.2f * pulse, c);
        DrawText(TextFormat("%02d", i), 18, (int)y - 4, 8, LIGHTGRAY);
    }

    // 512 router groups. Bright groups are the actual groups selected by the
    // sparse router in Brain::forward().
    bool activeGroup[N_GROUPS]{};
    for (int k = 0; k < ai.act.nGroups; ++k) {
        if (ai.act.groups[k] >= 0 && ai.act.groups[k] < N_GROUPS)
            activeGroup[ai.act.groups[k]] = true;
    }

    for (int g = 0; g < N_GROUPS; ++g) {
        const float y = top + ((float)g + 0.5f) * H / N_GROUPS;
        const float score = (ai.brain.score.empty() ? 0.f : ai.brain.score[(size_t)g]);
        const float intensity = clampf(std::fabs(score) * 0.45f, 0.f, 1.f);
        Color c = activeGroup[g]
            ? rgba(255, (int)(90.f + 150.f * intensity), 30, 255)
            : rgba(70, 48, 28, (unsigned char)(80.f + 80.f * intensity));
        DrawCircleV({routerX, y}, activeGroup[g] ? 2.3f : 1.15f, c);
    }

    // Exact 512 x 32 hidden topology. Every dot is one actual neuron.
    // 512 columns are groups, 32 rows are neurons inside each group.
    const float colStep = hiddenW / N_GROUPS;
    const float rowStep = H / GROUP_SIZE;
    for (int g = 0; g < N_GROUPS; ++g) {
        const bool groupOn = activeGroup[g];
        for (int u = 0; u < GROUP_SIZE; ++u) {
            const int j = g * GROUP_SIZE + u;
            const float x = hiddenX + ((float)g + 0.5f) * colStep;
            const float y = top + ((float)u + 0.5f) * rowStep;

            float activation = 0.f;
            for (int k = 0; k < ai.act.nActive; ++k) {
                if (ai.act.idx[k] == j) {
                    activation = ai.act.hn[k];
                    break;
                }
            }

            if (activation > 0.f) {
                const float pulse = 0.55f + 0.45f * std::sin(now * 7.0f + j * 0.013f);
                const float glow = clampf(activation * 11.f, 0.f, 1.f);
                DrawCircleV({x, y}, 1.1f + 1.3f * glow * pulse, rgba(255, 125 + (int)(110.f * glow), 25, 255));
                if (glow > 0.25f)
                    DrawCircleV({x, y}, 3.0f + 2.0f * glow * pulse, rgba(255, 110, 25, 45));
            } else if (groupOn) {
                DrawCircleV({x, y}, 0.65f, rgba(110, 62, 30, 170));
            } else {
                DrawCircleV({x, y}, 0.42f, rgba(38, 44, 54, 150));
            }
        }
    }

    // Group separators make the exact 512-group structure visible without
    // using connection lines between neurons.
    for (int g = 0; g <= N_GROUPS; g += 16) {
        const float x = hiddenX + (float)g * colStep;
        DrawRectangle((int)x, (int)top, 1, (int)H, rgba(150, 110, 70, g % 32 == 0 ? 55 : 25));
    }

    // Selected group frame. It contains exactly 32 neurons.
    const float selectedX = hiddenX + selectedGroup * colStep;
    DrawRectangleLines((int)selectedX, (int)top,
                       std::max(2, (int)std::ceil(colStep)), (int)H,
                       rgba(255, 240, 150, 235));

    // 12 actual output neurons.
    for (int a = 0; a < N_ACT; ++a) {
        const float y = top + ((float)a + 0.5f) * H / N_ACT;
        const float v = ai.act.mu[a];
        const float mag = clampf(std::fabs(v), 0.f, 1.f);
        Color c = v >= 0.f
            ? rgba(255, (int)(90.f + 145.f * mag), 30, 255)
            : rgba(40, (int)(120.f + 90.f * mag), 255, 255);
        DrawCircleV({outputX, y}, 3.0f + 2.0f * mag, c);
        DrawText(TextFormat("a%02d", a), (int)outputX + 9, (int)y - 5, 9, LIGHTGRAY);
    }

    // ------------------------------------------------------------------------
    // SYNAPSES: no connection lines. Instead, active synapses are rendered as
    // moving particles along their mathematical source->target path. This
    // preserves direction and activation while keeping the display readable.
    // ------------------------------------------------------------------------
    auto drawSynapseParticles = [&](Vector2 src, Vector2 dst, float contribution, float phase) {
        const float mag = std::fabs(contribution);
        if (mag < 0.035f) return;
        const float strength = clampf(mag * 4.5f, 0.f, 1.f);
        const int count = 2 + (int)(strength * 7.f);
        const float speed = 0.45f + strength * 1.8f;
        const float flow = std::fmod(now * speed + phase, 1.f);
        for (int p = 0; p < count; ++p) {
            float t = std::fmod(flow + (float)p / (float)count, 1.f);
            // Slight curve so the particles do not look like one continuous line.
            const float arc = std::sin(t * 3.14159265f) * (contribution >= 0.f ? 5.f : -5.f);
            Vector2 q = {
                src.x + (dst.x - src.x) * t,
                src.y + (dst.y - src.y) * t + arc
            };
            const unsigned char alpha = (unsigned char)(70.f + 180.f * strength);
            Color c = contribution >= 0.f
                ? rgba(255, (int)(90.f + 130.f * strength), 25, alpha)
                : rgba(35, (int)(105.f + 100.f * strength), 255, alpha);
            DrawCircleV(q, 1.0f + 1.8f * strength, c);
        }
    };

    // Show the strongest currently active neurons. For each one, show its
    // strongest input synapses and output synapses as particles, never as lines.
    std::vector<std::pair<float, int>> strongest;
    strongest.reserve((size_t)ai.act.nActive);
    for (int k = 0; k < ai.act.nActive; ++k)
        strongest.emplace_back(ai.act.hn[k], ai.act.idx[k]);
    const int showNeurons = std::min(18, (int)strongest.size());
    if (showNeurons > 0) {
        std::partial_sort(strongest.begin(), strongest.begin() + showNeurons, strongest.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    for (int n = 0; n < showNeurons; ++n) {
        const int j = strongest[(size_t)n].second;
        const int g = j / GROUP_SIZE;
        const int u = j % GROUP_SIZE;
        const float nx = hiddenX + ((float)g + 0.5f) * colStep;
        const float ny = top + ((float)u + 0.5f) * rowStep;

        int bestIn[5] = {-1,-1,-1,-1,-1};
        float bestInV[5] = {0,0,0,0,0};
        for (int i = 0; i < N_IN; ++i) {
            const float w = ai.loaded ? ai.brain.Win[(size_t)j * N_IN + i] : 0.f;
            const float c = ai.obs[i] * w;
            for (int q = 0; q < 5; ++q) {
                if (std::fabs(c) > std::fabs(bestInV[q])) {
                    for (int z = 4; z > q; --z) {
                        bestIn[z] = bestIn[z-1];
                        bestInV[z] = bestInV[z-1];
                    }
                    bestIn[q] = i;
                    bestInV[q] = c;
                    break;
                }
            }
        }
        for (int q = 0; q < 5; ++q) if (bestIn[q] >= 0) {
            const int i = bestIn[q];
            const float iy = top + ((float)i + 0.5f) * H / N_IN;
            drawSynapseParticles({inputX + 5.f, iy}, {nx, ny}, bestInV[q], i * 0.11f + j * 0.0007f);
        }

        int bestOut[4] = {-1,-1,-1,-1};
        float bestOutV[4] = {0,0,0,0};
        for (int a = 0; a < N_ACT; ++a) {
            const float w = ai.loaded ? ai.brain.Wout[(size_t)j * N_ACT + a] : 0.f;
            const float c = strongest[(size_t)n].first * w;
            for (int q = 0; q < 4; ++q) {
                if (std::fabs(c) > std::fabs(bestOutV[q])) {
                    for (int z = 3; z > q; --z) {
                        bestOut[z] = bestOut[z-1];
                        bestOutV[z] = bestOutV[z-1];
                    }
                    bestOut[q] = a;
                    bestOutV[q] = c;
                    break;
                }
            }
        }
        for (int q = 0; q < 4; ++q) if (bestOut[q] >= 0) {
            const int a = bestOut[q];
            const float oy = top + ((float)a + 0.5f) * H / N_ACT;
            drawSynapseParticles({nx, ny}, {outputX - 5.f, oy}, bestOutV[q], a * 0.17f + j * 0.0011f);
        }
    }

    // Selected group inspector: the actual 32 neurons and their 33 input /
    // 12 output synapses. This is a compact microscope for the exact group.
    DrawRectangle((int)detailX - 10, 118, (int)detailW + 5, (int)H + 35, rgba(9, 13, 20, 245));
    DrawRectangleLines((int)detailX - 10, 118, (int)detailW + 5, (int)H + 35, rgba(90, 110, 130, 180));
    const bool routerActive = activeGroup[selectedGroup];
    DrawText(TextFormat("GRUPO %03d / 512", selectedGroup), (int)detailX, 128, 12, WHITE);
    DrawText(routerActive ? "ROUTER: ATIVO" : "ROUTER: inativo",
             (int)detailX, 147, 9, routerActive ? rgba(255, 185, 75) : GRAY);
    DrawText(TextFormat("score %.3f", ai.brain.score.empty() ? 0.f : ai.brain.score[(size_t)selectedGroup]),
             (int)detailX, 163, 9, LIGHTGRAY);

    const float miniY = 190.f;
    const float miniW = detailW - 12.f;
    const float inputCell = miniW / N_IN;
    DrawText("SINAPSES DAS 33 ENTRADAS -> 32 NEURONIOS", (int)detailX, 180, 8, rgba(255, 180, 90));
    for (int u = 0; u < GROUP_SIZE; ++u) {
        const int j = selectedGroup * GROUP_SIZE + u;
        float activation = 0.f;
        for (int k = 0; k < ai.act.nActive; ++k)
            if (ai.act.idx[k] == j) { activation = ai.act.hn[k]; break; }
        const float y = miniY + u * 7.1f;
        for (int i = 0; i < N_IN; ++i) {
            const float c = ai.loaded ? ai.obs[i] * ai.brain.Win[(size_t)j * N_IN + i] : 0.f;
            const float mag = clampf(std::fabs(c) * 4.f, 0.f, 1.f);
            if (mag < 0.04f) continue;
            Color col = c >= 0.f ? rgba(255, (int)(75 + 160 * mag), 25, (unsigned char)(70 + 185 * mag))
                                 : rgba(30, (int)(90 + 130 * mag), 255, (unsigned char)(70 + 185 * mag));
            DrawCircleV({detailX + i * inputCell + inputCell * 0.5f, y}, 1.2f + mag * 1.4f, col);
        }
        if (activation > 0.f)
            DrawCircleV({detailX + miniW + 2.f, y}, 2.f + clampf(activation * 5.f, 0.f, 3.f), rgba(255, 145, 35, 230));
    }

    const float outY = miniY + GROUP_SIZE * 7.1f + 24.f;
    DrawText("SINAPSES 32 NEURONIOS -> 12 SAIDAS", (int)detailX, (int)outY - 13, 8, rgba(100, 190, 255));
    for (int u = 0; u < GROUP_SIZE; ++u) {
        const int j = selectedGroup * GROUP_SIZE + u;
        float activation = 0.f;
        for (int k = 0; k < ai.act.nActive; ++k)
            if (ai.act.idx[k] == j) { activation = ai.act.hn[k]; break; }
        if (activation <= 0.f) continue;
        const float y = outY + u * 7.1f;
        for (int a = 0; a < N_ACT; ++a) {
            const float c = activation * ai.brain.Wout[(size_t)j * N_ACT + a];
            const float mag = clampf(std::fabs(c) * 5.f, 0.f, 1.f);
            if (mag < 0.04f) continue;
            Color col = c >= 0.f ? rgba(255, (int)(80 + 150 * mag), 25, (unsigned char)(75 + 180 * mag))
                                 : rgba(30, (int)(95 + 125 * mag), 255, (unsigned char)(75 + 180 * mag));
            DrawCircleV({detailX + a * (miniW / N_ACT) + 4.f, y}, 1.3f + mag * 1.5f, col);
        }
    }

    DrawText("LARANJA = excitacao / contribuicao positiva", 22, sh - 48, 9, rgba(255, 180, 90));
    DrawText("AZUL = contribuicao negativa | brilho/pulso = sinapse ativa", 22, sh - 32, 9, rgba(110, 195, 255));
    DrawText("Clique no grupo | [ e ] selecionam grupo | N fecha", sw - 315, sh - 32, 9, LIGHTGRAY);
}


// ============================================================================
// Main
// ============================================================================
int main(){
    g_rng.seed(std::random_device{}());
    InitWindow(1280,720,"BACKROOMS - IA INTEGRADA");
    SetTargetFPS(60); DisableCursor();

    Camera3D camera{};
    camera.up={0,1,0}; camera.fovy=65.f; camera.projection=CAMERA_PERSPECTIVE;
    uint32_t rngState=777;
    Terrain terrain; terrain.Generate(rngState); gTerrain=&terrain;

    // Build the environment before positioning the AI, because stair tops are part of the ground query.
    GameAI ai;
    BuildStairsAndOrbs(ai);
    ai.Load();
    ai.Reset(0,15);
    // Rebuild orb positions after Reset, preserving the stair targets.
    BuildStairsAndOrbs(ai);
    for(int i=0;i<6;i++) ai.orbs[i].active=(i==0);
    ai.InitNetworkView();

    Shader shader=LoadShader(0,0);
    RenderTexture2D motionBlurTarget=LoadRenderTexture(1280,720);

    Physics::Cylinder player{};
    player.base={0,terrain.GetHeight(0,0),15}; player.radius=.4f; player.height=1.75f;
    float cameraPitch=0,cameraYaw=0,velocityY=0,walkTimer=0; bool isGrounded=true;
    const float gravity=22.f,jumpForce=7.5f;

    while(!WindowShouldClose()){
        float dt=std::min(GetFrameTime(),0.033f);
        if(IsKeyPressed(KEY_M)) ai.menu=!ai.menu;
        if(IsKeyPressed(KEY_N)) ai.networkView=!ai.networkView;
        if(ai.networkView && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            const Vector2 mp=GetMousePosition();
            const float viewTop=151.f, viewH=std::max(300,GetScreenHeight()-260);
            const float viewX=210.f, viewW=std::min(642.f,(float)GetScreenWidth()-615.f);
            if(mp.x>=viewX && mp.x<viewX+viewW && mp.y>=viewTop && mp.y<viewTop+viewH) {
                ai.selectedGroup=std::clamp((int)((mp.x-viewX)/viewW*N_GROUPS),0,N_GROUPS-1);
                ai.selectedGroupPinned=true;
            }
        }
        if(ai.networkView && IsKeyPressed(KEY_LEFT_BRACKET)) {
            ai.selectedGroup=(ai.selectedGroup+N_GROUPS-1)%N_GROUPS; ai.selectedGroupPinned=true;
        }
        if(ai.networkView && IsKeyPressed(KEY_RIGHT_BRACKET)) {
            ai.selectedGroup=(ai.selectedGroup+1)%N_GROUPS; ai.selectedGroupPinned=true;
        }
        if(IsKeyPressed(KEY_I)) {
            ai.manualControl=!ai.manualControl;
            ai.invertedTimer=0.f;
            if (!ai.manualControl) ai.ClearPhysicsHistory();
        }
        if(ai.manualControl && IsKeyPressed(KEY_BACKSPACE)) ai.Reset(ai.spawnX,ai.spawnZ);

        Vector2 md=GetMouseDelta(); cameraYaw-=md.x*.0025f; cameraPitch-=md.y*.0025f;
        cameraPitch=std::clamp(cameraPitch,-1.52f,1.52f);
        Vector3 forward={std::sin(cameraYaw),0,std::cos(cameraYaw)};
        Vector3 right={-std::cos(cameraYaw),0,std::sin(cameraYaw)};
        Vector3 move={};
        if(ai.manualControl) {
            ai.ManualTransform(dt);
        } else {
            if(IsKeyDown(KEY_W)) move=Vector3Add(move,forward);
            if(IsKeyDown(KEY_S)) move=Vector3Subtract(move,forward);
            if(IsKeyDown(KEY_D)) move=Vector3Add(move,right);
            if(IsKeyDown(KEY_A)) move=Vector3Subtract(move,right);
            float speed=IsKeyDown(KEY_LEFT_SHIFT)?7.5f:4.2f;
            if(Vector3LengthSqr(move)>0.001f){move=Vector3Normalize(move);player.base.x+=move.x*speed*dt;player.base.z+=move.z*speed*dt;}
            float terrainH=terrain.GetHeight(player.base.x,player.base.z);
            if(isGrounded&&IsKeyPressed(KEY_SPACE)){velocityY=jumpForce;isGrounded=false;}
            if(!isGrounded){velocityY-=gravity*dt;player.base.y+=velocityY*dt;if(player.base.y<=terrainH){player.base.y=terrainH;velocityY=0;isGrounded=true;}}
            else player.base.y=terrainH;
            if(isGrounded&&Vector3LengthSqr(move)>0.001f){float bs=IsKeyDown(KEY_LEFT_SHIFT)?14.f:9.f;walkTimer+=dt*bs;}
            else {walkTimer+=dt*1.8f;}
        }
        float bobX=(Vector3LengthSqr(move)>0.001f)?std::cos(walkTimer*.5f)*.045f:std::sin(walkTimer*.6f)*.015f;
        float bobY=(Vector3LengthSqr(move)>0.001f)?std::sin(walkTimer)*.07f:std::cos(walkTimer)*.02f;
        camera.position={player.base.x,player.base.y+1.6f+bobY,player.base.z};
        camera.target={camera.position.x+std::sin(cameraYaw+bobX)*std::cos(cameraPitch),camera.position.y+std::sin(cameraPitch),camera.position.z+std::cos(cameraYaw+bobX)*std::cos(cameraPitch)};

        ai.Update(dt);

        BeginTextureMode(motionBlurTarget);
        ClearBackground(Color{185,215,245,255});
        BeginMode3D(camera);
        terrain.Draw(shader);
        terrain.DrawProps(shader);
        DrawStairsAndOrbs(ai);
        DrawAI(ai);
        EndMode3D();
        EndTextureMode();

        BeginDrawing();
        ClearBackground(BLACK);
        DrawTextureRec(motionBlurTarget.texture,{0,0,(float)motionBlurTarget.texture.width,(float)-motionBlurTarget.texture.height},{0,0},Fade(WHITE,.65f));
        DrawText("BACKROOMS LEVEL 94 // IA INTEGRADA",20,20,18,RAYWHITE);
        DrawText(ai.manualControl?"EDITAR IA: WASD mover | R/F altura | Q/E yaw | T/G pitch | Z/X roll | I sair":"WASD jogador | SHIFT correr | ESPACO pular | I editar IA | N rede neural | M painel",20,48,14,LIGHTGRAY);
        DrawAIMenu(ai);
        DrawNeuralNetwork2D(ai);
        DrawFPS(GetScreenWidth()-90,20);
        EndDrawing();
    }

    ai.UnloadNetworkView();
    UnloadRenderTexture(motionBlurTarget); terrain.Unload(); UnloadShader(shader); CloseWindow();
    return 0;
}
