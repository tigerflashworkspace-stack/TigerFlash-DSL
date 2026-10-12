// SIHTL - Sistema de Iluminacao Hibrida por Tubos de Luz (prototipo leve, CPU + raylib).
// Compilacao Linux (com raylib instalado):
// g++ -O3 -march=native -std=c++17 -pthread sihtl_raylib.cpp -o sihtl_raylib /usr/local/lib/libraylib.a -lGL -lm -ldl -lrt -lX11
//
// Controles:
//   WASD / setas       mover camera
//   Botao direito      segurar e arrastar para olhar ao redor
//   Botao esquerdo     clicar e arrastar uma esfera/caixa no plano horizontal
//   1 / 2 / 3          qualidade leve / equilibrada / alta
//   N                  ligar/desligar denoiser guiado por geometria
//   I                  ligar/desligar iluminacao indireta (tubos de luz)
//   B                  rebatimentos de luz: 1 / 2 / 3
//   [ / ]              subdivisao maxima dos tubos (menos / mais)
//   - / =              limite de erro dos tubos (mais fino / mais grosso)
//   T                  mostrar pontos de impacto dos tubos (depuracao)
//   F11                tela cheia
//   R                  reiniciar camera e objetos
//   ESC                sair
//
// Base: ray tracing em CPU com lightmaps, sombras reutilizadas e reflexo unico.
// Camada SIHTL (nova): cada luz emite 6 tubos (4 laterais, 1 superior, 1 inferior).
// Cada tubo e testado com 5 raios reais (centro + 4 cantos). A superficie encontrada
// preve onde os cantos deveriam bater; se a previsao falha (erro > limite), o tubo e
// dividido em 4. Tubos confiaveis viram uma fonte secundaria que carrega a energia
// refletida (conservada na divisao). A iluminacao indireta e acumulada em lightmaps de
// baixa resolucao e interpolada, sem custo extra por pixel durante o movimento da camera.

#include "raylib.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
constexpr float SIHTL_PI = 3.14159265358979323846f;
constexpr float INF = 1.0e30f;

struct V3 {
    float x = 0, y = 0, z = 0;
    constexpr V3() = default;
    constexpr V3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
    V3 operator+(V3 b) const { return {x+b.x, y+b.y, z+b.z}; }
    V3 operator-(V3 b) const { return {x-b.x, y-b.y, z-b.z}; }
    V3 operator*(float s) const { return {x*s, y*s, z*s}; }
    V3 operator/(float s) const { return {x/s, y/s, z/s}; }
    V3& operator+=(V3 b) { x+=b.x; y+=b.y; z+=b.z; return *this; }
};
float dot(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
float length(V3 v) { return std::sqrt(dot(v,v)); }
inline V3 normalize(V3 v) {
    const float len2 = dot(v, v);
    if (len2 <= 1e-16f) return V3(0,1,0);
    // Uma inversa de raiz quadrada, seguida de multiplicacoes, evita tres divisoes.
    return v * (1.0f / std::sqrt(len2));
}
V3 hadamard(V3 a, V3 b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; }
V3 reflect(V3 d, V3 n) { return d - n*(2.0f*dot(d,n)); }
float clampf(float x, float a, float b) { return std::max(a, std::min(b, x)); }
V3 clampv(V3 c, float a, float b) { return {clampf(c.x,a,b),clampf(c.y,a,b),clampf(c.z,a,b)}; }

// Expoentes de brilho dos materiais sao inteiros (12, 18, 28, 52, 68, 78, 87).
// Exponenciacao por quadrados e muito mais barata que chamar pow() da libm por pixel.
inline float fastPowInteger(float base, int exponent) {
    float result = 1.0f;
    base = clampf(base, 0.0f, 1.0f);
    while (exponent > 0) {
        if (exponent & 1) result *= base;
        base *= base;
        exponent >>= 1;
    }
    return result;
}

enum class ObjectType { Sphere, Box };
struct Object {
    ObjectType type;
    std::string name;
    V3 center;
    float radius = 0.5f;
    V3 halfSize;
    V3 albedo;
    float roughness = 0.5f;
    float metalness = 0.0f;
};
struct Light { V3 pos, color; float power; };

// Lightmaps store expensive, view-independent diffuse illumination and baked
// visibility. Camera movement reuses these values instead of tracing shadow
// rays again for every pixel.
struct LightValue {
    V3 irradiance{};
    V3 indirect{};   // SIHTL: luz refletida por tubos secundarios
    std::array<float, 2> visibility{{0.0f, 0.0f}};
};
struct LightMap {
    int width = 0, height = 0;
    std::vector<LightValue> texels;
    LightValue sample(float u, float v) const {
        if (width <= 0 || height <= 0 || texels.empty()) return {};
        u = clampf(u, 0.0f, 1.0f);
        v = clampf(v, 0.0f, 1.0f);
        const float fx = u * static_cast<float>(width - 1);
        const float fy = v * static_cast<float>(height - 1);
        const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
        const int x1 = std::min(x0 + 1, width - 1), y1 = std::min(y0 + 1, height - 1);
        const float tx = fx - static_cast<float>(x0), ty = fy - static_cast<float>(y0);
        const LightValue& a = texels[static_cast<size_t>(y0) * width + x0];
        const LightValue& b = texels[static_cast<size_t>(y0) * width + x1];
        const LightValue& c = texels[static_cast<size_t>(y1) * width + x0];
        const LightValue& d = texels[static_cast<size_t>(y1) * width + x1];
        const auto mixValue = [](const LightValue& q, const LightValue& r, float t) {
            LightValue out;
            out.irradiance = q.irradiance * (1.0f - t) + r.irradiance * t;
            out.indirect = q.indirect * (1.0f - t) + r.indirect * t;
            for (size_t k = 0; k < out.visibility.size(); ++k)
                out.visibility[k] = q.visibility[k] * (1.0f - t) + r.visibility[k] * t;
            return out;
        };
        return mixValue(mixValue(a, b, tx), mixValue(c, d, tx), ty);
    }
};
// Fonte secundaria criada por um tubo confiavel: um "retalho" de superficie que
// reemite a energia refletida (modelo difuso de Lambert).
struct LightPatch {
    V3 p, n;       // posicao e normal do impacto
    V3 flux;       // fluxo refletido (ja multiplicado pelo albedo)
    float soft;    // suavizacao ~ area coberta pelo tubo (evita pontos quentes)
    int bounce;    // 1 = primeiro rebatimento
    int depth;     // nivel de subdivisao do tubo que gerou o retalho
};
struct SihtlConfig {
    bool indirect = true;
    int maxDepth = 3;          // subdivisoes maximas por tubo
    float errorLimit = 0.04f;  // erro relativo maximo da previsao do tubo
    int bounces = 1;           // 1..3
    int maxPatches = 300;      // orcamento de memoria/tempo (fontes secundarias)
    int stride = 4;            // lightmap indireto = resolucao / stride
    float minContribution = 0.00002f; // descarta contribuicoes irrelevantes
};
struct LightingCache {
    std::vector<LightPatch> patches;
    LightMap floor, backWall, leftWall, rightWall;
    std::array<LightMap, 3> spheres;
    std::array<LightMap, 6> boxFaces;
    bool valid = false;
};
enum class Surface { Floor, Wall, Side, Sphere, Box };
struct Hit {
    bool valid = false;
    float t = INF;
    V3 p, n, albedo;
    float roughness = 0.8f, metalness = 0;
    Surface surface = Surface::Floor;
    int objectIndex = -1;
};
struct Camera {
    V3 pos{0.0f, 1.25f, -4.6f};
    float yaw = 0.0f;
    float pitch = -0.04f;
};
const std::array<V3, 4> originalCenters = {
    V3(-1.05f,0.83f,2.4f), V3(0.75f,0.58f,3.15f),
    V3(0.05f,0.34f,1.25f), V3(2.0f,0.625f,2.4f)
};
std::vector<Object> objects = {
    {ObjectType::Sphere, "Esfera vermelha", {-1.05f,0.83f,2.4f}, 0.83f, {}, {0.78f,0.12f,0.075f}, 0.22f, 0.04f},
    {ObjectType::Sphere, "Esfera azul metalica", {0.75f,0.58f,3.15f}, 0.58f, {}, {0.08f,0.34f,0.72f}, 0.13f, 0.55f},
    {ObjectType::Sphere, "Esfera dourada", {0.05f,0.34f,1.25f}, 0.34f, {}, {0.82f,0.72f,0.43f}, 0.48f, 0.02f},
    {ObjectType::Box, "Caixa verde", {2.0f,0.625f,2.4f}, 0.0f, {0.35f,0.625f,0.35f}, {0.38f,0.65f,0.37f}, 0.32f, 0.05f}
};
const std::array<Light,2> lights = {{
    {{-2.1f,3.65f,0.8f},{1.0f,0.82f,0.62f},23.0f},
    {{2.3f,2.9f,3.2f},{0.58f,0.73f,1.0f},14.0f}
}};
const std::array<V3,4> lightSampleOffsets = {{
    V3(-0.70f,-0.35f,-0.70f), V3(0.70f,0.35f,0.70f),
    V3(-0.70f,0.35f,0.70f), V3(0.70f,-0.35f,-0.70f)
}};
int lightSamples = 2;
SihtlConfig sihtl;
bool showTubes = false;
// Estatisticas da ultima construcao dos tubos (escritas pelo worker, lidas pela UI).
std::atomic<int> statTubes{0}, statRays{0}, statPatches{0}, statMs{0}, statLeafMaxDepth{0};
std::atomic<long long> statRayEquivalent{0};
std::atomic<float> statEnergyKept{0.0f};
Camera camera;
int renderW = 240, renderH = 135;
int qualityMode = 0; // 0 leve, 1 equilibrada, 2 alta
int maxAccumSamples = 1;
bool denoiserEnabled = false;
bool fullscreen = false;
bool sceneDirty = true;
bool displayDirty = true;
int selectedObject = -1;
int draggingObject = -1;
float dragPlaneY = 0.5f;
V3 dragOffset{};
int accumulatedSamples = 0;
uint32_t frameIndex = 0;

Texture2D renderTexture{};
std::vector<V3> accumulationBuffer;
std::vector<V3> denoisedBuffer;
std::vector<Color> pixelBuffer;
std::vector<Hit> hitGuideBuffer;

std::shared_ptr<const LightingCache> activeLightingCache;
std::thread lightingBakeThread;
std::atomic<uint64_t> lightingGeneration{0};
std::atomic<bool> lightingBakeRunning{false};
std::atomic<bool> lightingBakeCompleted{false};
bool lightingNeedsBake = false; // true while lightmaps no longer match object positions

// A single persistent worker coalesces object-move requests. The UI thread never
// joins/recreates a baking thread on every mouse event.
std::mutex lightingWorkMutex;
std::condition_variable lightingWorkCv;
std::vector<Object> pendingLightingObjects;
int pendingLightingSamples = 1;
SihtlConfig pendingSihtl;
uint64_t pendingLightingGeneration = 0;
bool lightingWorkPending = false;
bool stopLightingWorker = false;
void lightingWorkerLoop();
void startLightingWorker();
void stopLightingWorkerAndJoin();
void requestLightingBake();

V3 checker(float x, float z) {
    const int k = (static_cast<int>(std::floor(x*1.35f)) + static_cast<int>(std::floor(z*1.35f))) & 1;
    return k ? V3(0.20f,0.23f,0.27f) : V3(0.36f,0.38f,0.40f);
}

void acceptHit(Hit& best, float t, V3 p, V3 n, V3 albedo, float rough, float metal,
               Surface surface, int objectIndex, float maxT) {
    if (t > 0.001f && t < best.t && t < maxT) {
        best.valid = true; best.t=t; best.p=p; best.n=n; best.albedo=albedo;
        best.roughness=rough; best.metalness=metal; best.surface=surface; best.objectIndex=objectIndex;
    }
}

Hit traceSceneIn(const std::vector<Object>& objs, V3 origin, V3 direction, float maxT=INF) {
    Hit best; best.t = maxT;

    // Chao quadriculado, y=0.
    if (std::fabs(direction.y) > 1e-6f) {
        const float t = -origin.y/direction.y;
        if (t > 0.001f && t < best.t) {
            const V3 p = origin + direction*t;
            acceptHit(best,t,p,{0,1,0},checker(p.x,p.z),0.72f,0.0f,Surface::Floor,-1,maxT);
        }
    }

    // Parede de fundo z=6.8.
    if (std::fabs(direction.z) > 1e-6f) {
        const float t = (6.8f-origin.z)/direction.z;
        if (t > 0.001f && t < best.t) {
            const V3 p=origin+direction*t;
            if (p.y>=0 && std::fabs(p.x)<5.0f)
                acceptHit(best,t,p,{0,0,-1},{0.30f,0.34f,0.40f},0.88f,0,Surface::Wall,-1,maxT);
        }
    }

    // Paredes laterais.
    if (std::fabs(direction.x) > 1e-6f) {
        for (float side : {-4.5f,4.5f}) {
            const float t=(side-origin.x)/direction.x;
            if (t > 0.001f && t < best.t) {
                const V3 p=origin+direction*t;
                if (p.y>=0 && p.z>-8 && p.z<7)
                    acceptHit(best,t,p,{-((side>0)?1.0f:-1.0f),0,0},{0.23f,0.29f,0.36f},0.82f,0,Surface::Side,-1,maxT);
            }
        }
    }

    for (int i=0;i<static_cast<int>(objs.size());++i) {
        const Object& ob=objs[i];
        if (ob.type == ObjectType::Sphere) {
            const V3 oc=origin-ob.center;
            const float b=dot(oc,direction), c=dot(oc,oc)-ob.radius*ob.radius;
            const float disc=b*b-c;
            if (disc>=0) {
                float t=-b-std::sqrt(disc);
                if (t<0.001f) t=-b+std::sqrt(disc);
                if (t>0.001f && t<best.t && t<maxT) {
                    const V3 p=origin+direction*t;
                    acceptHit(best,t,p,normalize(p-ob.center),ob.albedo,ob.roughness,ob.metalness,Surface::Sphere,i,maxT);
                }
            }
        } else {
            const V3 lo=ob.center-ob.halfSize, hi=ob.center+ob.halfSize;
            float tMin=-INF, tMax=INF;
            V3 nMin{}, nMax{};
            bool possible=true;
            for (int axis=0;axis<3;++axis) {
                const float o = axis==0?origin.x:(axis==1?origin.y:origin.z);
                const float d = axis==0?direction.x:(axis==1?direction.y:direction.z);
                const float low = axis==0?lo.x:(axis==1?lo.y:lo.z);
                const float high= axis==0?hi.x:(axis==1?hi.y:hi.z);
                if (std::fabs(d)<1e-7f) {
                    if (o<low || o>high) { possible=false; break; }
                } else {
                    float a=(low-o)/d, b=(high-o)/d;
                    V3 na{},nb{};
                    if(axis==0){na.x=-1;nb.x=1;} else if(axis==1){na.y=-1;nb.y=1;} else {na.z=-1;nb.z=1;}
                    if(a>b){std::swap(a,b);std::swap(na,nb);}
                    if(a>tMin){tMin=a;nMin=na;}
                    if(b<tMax){tMax=b;nMax=nb;}
                    if(tMin>tMax){possible=false;break;}
                }
            }
            if(possible) {
                const float t = tMin>0.001f?tMin:tMax;
                const V3 n = tMin>0.001f?nMin:nMax;
                if(t>0.001f && t<best.t && t<maxT)
                    acceptHit(best,t,origin+direction*t,n,ob.albedo,ob.roughness,ob.metalness,Surface::Box,i,maxT);
            }
        }
    }
    return best;
}

// Versao para a thread principal: usa a cena atual.
Hit traceScene(V3 origin, V3 direction, float maxT=INF) {
    return traceSceneIn(objects, origin, direction, maxT);
}

V3 sky(V3 d) {
    const float t=clampf(d.y*0.5f+0.5f,0,1);
    return V3(0.035f,0.055f,0.09f)*(1-t)+V3(0.28f,0.42f,0.62f)*t;
}

// Caminho especializado para raios de sombra. As luzes ficam dentro da sala,
// acima do chao; portanto, depois de deslocar o raio da superficie, apenas os
// quatro objetos podem bloquear a luz. Evitamos testar novamente o chao, as
// paredes, materiais e normais em centenas de milhares de raios secundarios.
bool occludedByObjects(const std::vector<Object>& sceneObjects, V3 origin, V3 direction, float maxT) {
    for (const Object& ob : sceneObjects) {
        if (ob.type == ObjectType::Sphere) {
            const V3 oc = origin - ob.center;
            const float b = dot(oc, direction);
            const float c = dot(oc, oc) - ob.radius*ob.radius;
            const float disc = b*b - c;
            if (disc < 0.0f) continue;
            const float root = std::sqrt(disc);
            float t = -b - root;
            if (t <= 0.001f) t = -b + root;
            if (t > 0.001f && t < maxT) return true;
        } else {
            const V3 lo = ob.center - ob.halfSize;
            const V3 hi = ob.center + ob.halfSize;
            float tNear = -INF, tFar = INF;
            bool possible = true;
            for (int axis=0; axis<3; ++axis) {
                const float o = axis==0 ? origin.x : (axis==1 ? origin.y : origin.z);
                const float d = axis==0 ? direction.x : (axis==1 ? direction.y : direction.z);
                const float low = axis==0 ? lo.x : (axis==1 ? lo.y : lo.z);
                const float high = axis==0 ? hi.x : (axis==1 ? hi.y : hi.z);
                if (std::fabs(d) < 1e-7f) {
                    if (o < low || o > high) { possible=false; break; }
                } else {
                    float a=(low-o)/d, b=(high-o)/d;
                    if (a>b) std::swap(a,b);
                    tNear=std::max(tNear,a);
                    tFar=std::min(tFar,b);
                    if (tNear>tFar) { possible=false; break; }
                }
            }
            if (possible) {
                const float t = tNear>0.001f ? tNear : tFar;
                if (t>0.001f && t<maxT) return true;
            }
        }
    }
    return false;
}

LightValue bakeLightingAt(V3 p, V3 normal, const std::vector<Object>& sceneObjects,
                          int samples) {
    LightValue result;
    const int sampleCount = std::max(1, std::min(samples, 2));
    for (size_t li = 0; li < lights.size(); ++li) {
        const Light& light = lights[li];
        for (int si = 0; si < sampleCount; ++si) {
            V3 lp = light.pos;
            if (sampleCount > 1) {
                const V3 off = lightSampleOffsets[static_cast<size_t>(si)];
                lp += V3(off.x*0.20f, off.y*0.12f, off.z*0.20f);
            }
            const V3 toLight = lp - p;
            const float dist2 = dot(toLight, toLight);
            const float dist = std::sqrt(std::max(dist2, 1e-12f));
            const V3 ld = toLight * (1.0f / dist);
            const float nd = std::max(0.0f, dot(normal, ld));
            if (nd <= 0.0f) continue;
            if (occludedByObjects(sceneObjects, p + normal*0.012f, ld, dist - 0.025f)) continue;
            const float diffuse = nd * light.power / (dist2 + 1.2f);
            result.irradiance += light.color * (diffuse / static_cast<float>(sampleCount));
            result.visibility[li] += 1.0f / static_cast<float>(sampleCount);
        }
    }
    return result;
}


// ============================================================================
// SIHTL - tubos de luz
// Um tubo e uma piramide de direcoes parametrizada numa face de um cubo
// (u,v em [u0,u1]x[v0,v1]). Os 6 tubos iniciais cobrem a esfera inteira:
// 4 laterais, 1 superior e 1 inferior.
// ============================================================================
struct Frame { V3 f, tu, tv; };
struct Tube {
    Frame fr;
    float u0, u1, v0, v1;
    V3 flux;     // energia que o tubo transporta (conservada ao subdividir)
    int depth;
    int bounce;
};
struct Emitter {
    V3 origin;
    bool cosine = false;   // false: fonte pontual isotropica; true: reemissao difusa
    V3 n{0,1,0};
};

inline float cornerSolidAngle(float u, float v) {
    return std::atan2(u*v, std::sqrt(1.0f + u*u + v*v));
}
inline float tubeSolidAngle(const Tube& t) {
    return cornerSolidAngle(t.u1,t.v1) - cornerSolidAngle(t.u0,t.v1)
         - cornerSolidAngle(t.u1,t.v0) + cornerSolidAngle(t.u0,t.v0);
}
inline V3 tubeDir(const Frame& fr, float u, float v) {
    return normalize(fr.f + fr.tu*u + fr.tv*v);
}
// Peso de energia de uma regiao: angulo solido (x cos para superficies difusas).
inline float tubeWeight(const Emitter& em, const Tube& t) {
    const float om = std::max(0.0f, tubeSolidAngle(t));
    if (!em.cosine) return om;
    const V3 d = tubeDir(t.fr, 0.5f*(t.u0+t.u1), 0.5f*(t.v0+t.v1));
    return om * std::max(0.0f, dot(d, em.n));
}

struct BuildCounters {
    int tubes = 0, rays = 0;
    long long rayEquivalent = 0;
    int leafMaxDepth = 0;
};

// Erro de previsao: o centro do tubo define um plano local; os cantos deveriam
// bater nesse plano. A diferenca entre previsto e real mede a incerteza.
inline float cornerError(V3 origin, V3 dir, const Hit& rep, const Hit& actual) {
    if (rep.valid != actual.valid) return 1.0f;
    if (!rep.valid) return 0.0f;
    if (rep.surface != actual.surface || rep.objectIndex != actual.objectIndex) return 1.0f;
    if (rep.surface != Surface::Sphere && dot(rep.n, actual.n) < 0.98f) return 1.0f;
    const float den = dot(dir, rep.n);
    if (std::fabs(den) < 1e-4f) return 1.0f;
    const float tPred = dot(rep.p - origin, rep.n) / den;
    const V3 pPred = origin + dir * tPred;
    return length(actual.p - pPred) / std::max(rep.t, 0.25f);
}

// Processa um tubo. Retorna true se foi subdividido (filhos adicionados em `queue`).
bool processTube(const std::vector<Object>& scene, const Emitter& em, const Tube& tube,
                 const SihtlConfig& cfg, int depthCap, bool allowSplit,
                 std::vector<Tube>& queue, std::vector<LightPatch>& out, BuildCounters& bc) {
    ++bc.tubes;
    const float cu = 0.5f*(tube.u0+tube.u1), cv = 0.5f*(tube.v0+tube.v1);
    const V3 dirs[5] = {
        tubeDir(tube.fr, cu, cv),
        tubeDir(tube.fr, tube.u0, tube.v0), tubeDir(tube.fr, tube.u1, tube.v0),
        tubeDir(tube.fr, tube.u0, tube.v1), tubeDir(tube.fr, tube.u1, tube.v1)
    };
    Hit hits[5];
    int firstValid = -1;
    for (int k = 0; k < 5; ++k) {
        hits[k] = traceSceneIn(scene, em.origin, dirs[k]);
        if (hits[k].valid && firstValid < 0) firstValid = k;
    }
    bc.rays += 5;
    if (firstValid < 0) return false;                 // a energia do tubo escapa da cena

    const int rk = hits[0].valid ? 0 : firstValid;    // referencia: centro, se possivel
    const Hit& rep = hits[rk];
    float uncertainty = 0.0f;
    for (int k = 0; k < 5; ++k) {
        if (k == rk) continue;
        uncertainty = std::max(uncertainty, cornerError(em.origin, dirs[k], rep, hits[k]));
    }

    const bool wantSplit = uncertainty > cfg.errorLimit && tube.depth < depthCap;
    if (wantSplit && allowSplit) {
        Tube kids[4];
        const float mu = cu, mv = cv;
        const float U[3] = {tube.u0, mu, tube.u1}, Vv[3] = {tube.v0, mv, tube.v1};
        float wsum = 0.0f;
        for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) {
            Tube& k = kids[a*2+b];
            k = tube; k.u0=U[a]; k.u1=U[a+1]; k.v0=Vv[b]; k.v1=Vv[b+1]; k.depth=tube.depth+1;
            wsum += tubeWeight(em, k);
        }
        if (wsum > 1e-9f) {
            for (Tube& k : kids) {
                k.flux = tube.flux * (tubeWeight(em, k) / wsum);   // conserva a energia do pai
                queue.push_back(k);
            }
            return true;
        }
    }

    // Folha: o tubo e considerado confiavel (ou chegou ao limite). Cria a fonte secundaria.
    LightPatch lp;
    lp.p = rep.p; lp.n = rep.n;
    lp.flux = hadamard(tube.flux, rep.albedo);
    const float cosI = std::max(0.2f, std::fabs(dot(dirs[rk], rep.n)));
    const float area = tubeSolidAngle(tube) * rep.t * rep.t / cosI;
    lp.soft = 0.06f + 0.2f * area;
    lp.bounce = tube.bounce;
    lp.depth = tube.depth;
    out.push_back(lp);
    bc.leafMaxDepth = std::max(bc.leafMaxDepth, tube.depth);
    bc.rayEquivalent += 1LL << (2 * (depthCap - tube.depth));   // raios uniformes equivalentes
    return false;
}

// Seis tubos iniciais de uma luz: +x, -x, +z, -z (laterais), +y (superior), -y (inferior).
inline void seedPointLight(const Light& L, std::vector<Tube>& out) {
    const Frame frames[6] = {
        {{ 1,0,0},{0,1,0},{0,0,1}}, {{-1,0,0},{0,1,0},{0,0,1}},
        {{0,0, 1},{1,0,0},{0,1,0}}, {{0,0,-1},{1,0,0},{0,1,0}},
        {{0, 1,0},{1,0,0},{0,0,1}}, {{0,-1,0},{1,0,0},{0,0,1}}
    };
    Emitter em; em.origin = L.pos;
    std::array<Tube,6> t;
    float wsum = 0;
    for (int i = 0; i < 6; ++i) {
        t[i] = Tube{frames[i], -1,1,-1,1, V3{}, 0, 1};
        wsum += tubeWeight(em, t[i]);
    }
    // Fluxo total na convencao do renderizador: intensidade = power * cor (por esferorradiano).
    const V3 total = L.color * (L.power * 4.0f * SIHTL_PI);
    for (int i = 0; i < 6; ++i) {
        t[i].flux = total * (tubeWeight(em, t[i]) / wsum);
        out.push_back(t[i]);
    }
}

// Cinco tubos de reemissao difusa no hemisferio de uma fonte secundaria.
inline void seedHemisphere(const LightPatch& s, std::vector<Tube>& out) {
    const V3 n = s.n;
    const V3 helper = std::fabs(n.y) < 0.9f ? V3(0,1,0) : V3(1,0,0);
    V3 t = normalize(V3(helper.y*n.z-helper.z*n.y, helper.z*n.x-helper.x*n.z, helper.x*n.y-helper.y*n.x));
    V3 b = V3(n.y*t.z-n.z*t.y, n.z*t.x-n.x*t.z, n.x*t.y-n.y*t.x);
    Emitter em; em.cosine = true; em.n = n;
    std::array<Tube,5> k = {
        Tube{{n, t, b},        -1,1,-1,1, V3{}, 0, s.bounce+1},
        Tube{{t, b, n},        -1,1, 0,1, V3{}, 0, s.bounce+1},
        Tube{{V3(0,0,0)-t, b, n}, -1,1, 0,1, V3{}, 0, s.bounce+1},
        Tube{{b, t, n},        -1,1, 0,1, V3{}, 0, s.bounce+1},
        Tube{{V3(0,0,0)-b, t, n}, -1,1, 0,1, V3{}, 0, s.bounce+1},
    };
    float wsum = 0;
    for (const Tube& x : k) wsum += tubeWeight(em, x);
    if (wsum <= 1e-9f) return;
    for (Tube& x : k) { x.flux = s.flux * (tubeWeight(em, x) / wsum); out.push_back(x); }
}

// Constroi todas as fontes secundarias (tubos primarios + rebatimentos).
bool buildLightTubes(const std::vector<Object>& scene, const SihtlConfig& cfg,
                     std::vector<LightPatch>& patches, uint64_t generation) {
    BuildCounters bc;
    patches.clear();
    const size_t cap = static_cast<size_t>(std::max(6, cfg.maxPatches));
    const size_t cap1 = cfg.bounces > 1 ? cap * 6 / 10 : cap;
    V3 emitted{};

    // Nivel 1: tubos saindo das luzes. Fila FIFO = refinamento por niveis (justo entre tubos).
    for (const Light& L : lights) {
        emitted += L.color * (L.power * 4.0f * SIHTL_PI);
        Emitter em; em.origin = L.pos;
        std::vector<Tube> queue;
        seedPointLight(L, queue);
        for (size_t head = 0; head < queue.size(); ++head) {
            if (lightingGeneration.load(std::memory_order_relaxed) != generation) return false;
            const Tube tube = queue[head];
            const size_t pending = queue.size() - head - 1;
            const bool allowSplit = patches.size() + pending + 4 <= cap1 / lights.size() * (&L - lights.data() + 1);
            processTube(scene, em, tube, cfg, cfg.maxDepth, allowSplit, queue, patches, bc);
        }
    }
    const size_t level1End = patches.size();
    float kept = 0, total = std::max(1e-6f, emitted.x + emitted.y + emitted.z);
    for (const LightPatch& p : patches) kept += p.flux.x + p.flux.y + p.flux.z;

    // Niveis seguintes: so as fontes mais energeticas reemitem (as demais sao encerradas).
    size_t prevBegin = 0, prevEnd = level1End;
    for (int b = 2; b <= cfg.bounces; ++b) {
        const size_t remaining = cap > patches.size() ? cap - patches.size() : 0;
        if (remaining < 5 || prevEnd <= prevBegin) break;
        const size_t share = remaining / static_cast<size_t>(cfg.bounces - b + 1);
        size_t K = std::min(prevEnd - prevBegin, share / 5);
        if (K == 0) break;
        std::vector<size_t> idx;
        for (size_t i = prevBegin; i < prevEnd; ++i) idx.push_back(i);
        auto lum = [&](size_t i) { const V3& f = patches[i].flux; return f.x+f.y+f.z; };
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                          [&](size_t a, size_t c) { return lum(a) > lum(c); });
        const size_t newBegin = patches.size();
        for (size_t k = 0; k < K; ++k) {
            if (lightingGeneration.load(std::memory_order_relaxed) != generation) return false;
            const LightPatch src = patches[idx[k]];
            if (lum(idx[k]) < 1e-3f) break;   // contribuicao pouco relevante: encerra
            Emitter em; em.origin = src.p + src.n * 0.015f; em.cosine = true; em.n = src.n;
            std::vector<Tube> queue;
            seedHemisphere(src, queue);
            for (size_t head = 0; head < queue.size(); ++head)
                { const Tube tc = queue[head]; processTube(scene, em, tc, cfg, 0, false, queue, patches, bc); }
        }
        prevBegin = newBegin; prevEnd = patches.size();
    }

    statTubes.store(bc.tubes, std::memory_order_relaxed);
    statRays.store(bc.rays, std::memory_order_relaxed);
    statPatches.store(static_cast<int>(patches.size()), std::memory_order_relaxed);
    statRayEquivalent.store(bc.rayEquivalent, std::memory_order_relaxed);
    statLeafMaxDepth.store(bc.leafMaxDepth, std::memory_order_relaxed);
    statEnergyKept.store(kept / total, std::memory_order_relaxed);
    return true;
}

// Soma a luz refletida pelas fontes secundarias em um ponto receptor.
V3 gatherIndirect(V3 q, V3 nq, const std::vector<LightPatch>& patches,
                  const std::vector<Object>& scene, float minContribution) {
    V3 sum{};
    for (const LightPatch& s : patches) {
        const V3 d = s.p - q;
        const float d2 = dot(d, d);
        if (d2 < 1e-6f) continue;
        const float dist = std::sqrt(d2);
        const V3 w = d * (1.0f / dist);
        const float cr = dot(nq, w);
        if (cr <= 0.0f) continue;
        const float ce = -dot(s.n, w);
        if (ce <= 0.0f) continue;
        const V3 c = s.flux * (cr * ce / (SIHTL_PI * (d2 + s.soft)));
        if (std::max(c.x, std::max(c.y, c.z)) < minContribution) continue;
        if (occludedByObjects(scene, q + nq*0.012f, w, dist - 0.03f)) continue;
        sum += c;
    }
    return sum;
}

template <typename PositionNormalFn>
bool fillLightMap(LightMap& map, int width, int height, PositionNormalFn positionNormal,
                  const std::vector<Object>& sceneObjects, int samples, uint64_t generation,
                  const std::vector<LightPatch>* patches = nullptr, const SihtlConfig* cfg = nullptr) {
    map.width = width;
    map.height = height;
    map.texels.assign(static_cast<size_t>(width) * height, LightValue{});
    for (int y = 0; y < height; ++y) {
        if (lightingGeneration.load(std::memory_order_relaxed) != generation) return false;
        const float v = height > 1 ? static_cast<float>(y) / static_cast<float>(height - 1) : 0.0f;
        for (int x = 0; x < width; ++x) {
            const float u = width > 1 ? static_cast<float>(x) / static_cast<float>(width - 1) : 0.0f;
            V3 p, n;
            positionNormal(u, v, p, n);
            map.texels[static_cast<size_t>(y) * width + x] = bakeLightingAt(p, n, sceneObjects, samples);
        }
    }
    // Iluminacao indireta e de baixa frequencia: calculada numa grade `stride` vezes mais
    // grossa e interpolada. Custo = fontes secundarias x texels grossos (nao x pixels).
    if (patches && cfg && !patches->empty() && width > 1 && height > 1) {
        const int S = std::max(1, cfg->stride);
        const int cw = (width - 1 + S - 1) / S + 1, ch = (height - 1 + S - 1) / S + 1;
        LightMap coarse;
        coarse.width = cw; coarse.height = ch;
        coarse.texels.assign(static_cast<size_t>(cw) * ch, LightValue{});
        for (int y = 0; y < ch; ++y) {
            if (lightingGeneration.load(std::memory_order_relaxed) != generation) return false;
            const float v = static_cast<float>(std::min(y * S, height - 1)) / static_cast<float>(height - 1);
            for (int x = 0; x < cw; ++x) {
                const float u = static_cast<float>(std::min(x * S, width - 1)) / static_cast<float>(width - 1);
                V3 p, n;
                positionNormal(u, v, p, n);
                coarse.texels[static_cast<size_t>(y) * cw + x].indirect =
                    gatherIndirect(p + n*0.012f, n, *patches, sceneObjects, cfg->minContribution);
            }
        }
        // Os nos grossos nao caem exatamente nas bordas; reamostra pelo mesmo (u,v) normalizado.
        for (int y = 0; y < height; ++y) {
            const float v = static_cast<float>(y) / static_cast<float>(height - 1);
            for (int x = 0; x < width; ++x) {
                const float u = static_cast<float>(x) / static_cast<float>(width - 1);
                // Posicao do texel na grade grossa (em unidades de no).
                const float gx = clampf(u * static_cast<float>(width - 1) / S, 0.0f, static_cast<float>(cw - 1));
                const float gy = clampf(v * static_cast<float>(height - 1) / S, 0.0f, static_cast<float>(ch - 1));
                const int x0 = static_cast<int>(gx), y0 = static_cast<int>(gy);
                const int x1 = std::min(x0 + 1, cw - 1), y1 = std::min(y0 + 1, ch - 1);
                const float tx = gx - x0, ty = gy - y0;
                const V3& a = coarse.texels[static_cast<size_t>(y0)*cw + x0].indirect;
                const V3& b = coarse.texels[static_cast<size_t>(y0)*cw + x1].indirect;
                const V3& c = coarse.texels[static_cast<size_t>(y1)*cw + x0].indirect;
                const V3& d = coarse.texels[static_cast<size_t>(y1)*cw + x1].indirect;
                map.texels[static_cast<size_t>(y)*width + x].indirect =
                    (a*(1-tx) + b*tx)*(1-ty) + (c*(1-tx) + d*tx)*ty;
            }
        }
    }
    return true;
}

bool buildLightingCache(LightingCache& cache, const std::vector<Object>& sceneObjects,
                        int samples, uint64_t generation, const SihtlConfig& cfg) {
    const auto t0 = std::chrono::steady_clock::now();
    if (cfg.indirect) {
        if (!buildLightTubes(sceneObjects, cfg, cache.patches, generation)) return false;
    } else {
        statTubes = 0; statRays = 0; statPatches = 0; statRayEquivalent = 0; statEnergyKept = 0.0f;
    }
    const std::vector<LightPatch>* gi = cfg.indirect ? &cache.patches : nullptr;
    const SihtlConfig* gc = &cfg;
    // Room lightmaps are baked once per object/lighting change. The grid is
    // independent of camera resolution, so camera motion performs no shadow rays.
    if (!fillLightMap(cache.floor, 301, 301,
        [](float u, float v, V3& p, V3& n) {
            p = {-12.0f + 24.0f*u, 0.0f, -12.0f + 24.0f*v}; n = {0,1,0};
        }, sceneObjects, samples, generation, gi, gc)) return false;
    if (!fillLightMap(cache.backWall, 201, 161,
        [](float u, float v, V3& p, V3& n) {
            p = {-5.0f + 10.0f*u, 8.0f*v, 6.8f}; n = {0,0,-1};
        }, sceneObjects, samples, generation, gi, gc)) return false;
    if (!fillLightMap(cache.leftWall, 301, 161,
        [](float u, float v, V3& p, V3& n) {
            p = {-4.5f, 8.0f*v, -8.0f + 15.0f*u}; n = {1,0,0};
        }, sceneObjects, samples, generation, gi, gc)) return false;
    if (!fillLightMap(cache.rightWall, 301, 161,
        [](float u, float v, V3& p, V3& n) {
            p = {4.5f, 8.0f*v, -8.0f + 15.0f*u}; n = {-1,0,0};
        }, sceneObjects, samples, generation, gi, gc)) return false;

    // Sphere maps use latitude/longitude coordinates on the actual surface.
    for (size_t oi = 0; oi < 3; ++oi) {
        const Object& ob = sceneObjects[oi];
        if (!fillLightMap(cache.spheres[oi], 129, 65,
            [&ob](float u, float v, V3& p, V3& n) {
                const float phi = (u - 0.5f) * 2.0f * SIHTL_PI;
                const float theta = v * SIHTL_PI;
                n = {std::sin(theta)*std::sin(phi), std::cos(theta), std::sin(theta)*std::cos(phi)};
                p = ob.center + n*ob.radius;
            }, sceneObjects, samples, generation, gi, gc)) return false;
    }

    // Six compact maps for the six faces of the box.
    for (int face = 0; face < 6; ++face) {
        const Object& ob = sceneObjects[3];
        if (!fillLightMap(cache.boxFaces[static_cast<size_t>(face)], 65, 65,
            [&ob, face](float u, float v, V3& p, V3& n) {
                const V3 c = ob.center, h = ob.halfSize;
                switch (face) {
                    case 0: n={-1,0,0}; p={c.x-h.x,c.y-h.y+2*h.y*v,c.z-h.z+2*h.z*u}; break;
                    case 1: n={ 1,0,0}; p={c.x+h.x,c.y-h.y+2*h.y*v,c.z-h.z+2*h.z*u}; break;
                    case 2: n={0,-1,0}; p={c.x-h.x+2*h.x*u,c.y-h.y,c.z-h.z+2*h.z*v}; break;
                    case 3: n={0, 1,0}; p={c.x-h.x+2*h.x*u,c.y+h.y,c.z-h.z+2*h.z*v}; break;
                    case 4: n={0,0,-1}; p={c.x-h.x+2*h.x*u,c.y-h.y+2*h.y*v,c.z-h.z}; break;
                    default:n={0,0, 1}; p={c.x-h.x+2*h.x*u,c.y-h.y+2*h.y*v,c.z+h.z}; break;
                }
            }, sceneObjects, samples, generation, gi, gc)) return false;
    }
    cache.valid = true;
    statMs.store(static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count()), std::memory_order_relaxed);
    return true;
}

LightValue sampleLightingCache(const Hit& h, const LightingCache* cache) {
    if (!cache || !cache->valid) return {};
    switch (h.surface) {
        case Surface::Floor:
            return cache->floor.sample((h.p.x + 12.0f)/24.0f, (h.p.z + 12.0f)/24.0f);
        case Surface::Wall:
            return cache->backWall.sample((h.p.x + 5.0f)/10.0f, h.p.y/8.0f);
        case Surface::Side:
            if (h.n.x > 0.0f) return cache->leftWall.sample((h.p.z + 8.0f)/15.0f, h.p.y/8.0f);
            return cache->rightWall.sample((h.p.z + 8.0f)/15.0f, h.p.y/8.0f);
        case Surface::Sphere: {
            if (h.objectIndex < 0 || h.objectIndex >= 3) return {};
            const float phi = std::atan2(h.n.x, h.n.z);
            const float u = phi/(2.0f*SIHTL_PI) + 0.5f;
            const float v = std::acos(clampf(h.n.y,-1.0f,1.0f))/SIHTL_PI;
            return cache->spheres[static_cast<size_t>(h.objectIndex)].sample(u,v);
        }
        case Surface::Box: {
            if (h.objectIndex < 0 || h.objectIndex >= static_cast<int>(objects.size())) return {};
            const Object& ob = objects[static_cast<size_t>(h.objectIndex)];
            const V3 q = h.p - ob.center;
            int face; float u, v;
            if (std::fabs(h.n.x) >= std::fabs(h.n.y) && std::fabs(h.n.x) >= std::fabs(h.n.z)) {
                face = h.n.x < 0 ? 0 : 1;
                u = (q.z/ob.halfSize.z + 1.0f)*0.5f;
                v = (q.y/ob.halfSize.y + 1.0f)*0.5f;
            } else if (std::fabs(h.n.y) >= std::fabs(h.n.z)) {
                face = h.n.y < 0 ? 2 : 3;
                u = (q.x/ob.halfSize.x + 1.0f)*0.5f;
                v = (q.z/ob.halfSize.z + 1.0f)*0.5f;
            } else {
                face = h.n.z < 0 ? 4 : 5;
                u = (q.x/ob.halfSize.x + 1.0f)*0.5f;
                v = (q.y/ob.halfSize.y + 1.0f)*0.5f;
            }
            return cache->boxFaces[static_cast<size_t>(face)].sample(u,v);
        }
    }
    return {};
}

LightValue approximateLighting(const Hit& h) {
    LightValue result;
    for (size_t li=0; li<lights.size(); ++li) {
        const Light& light = lights[li];
        const V3 toLight = light.pos - h.p;
        const float dist2 = dot(toLight,toLight);
        const V3 ld = normalize(toLight);
        const float nd = std::max(0.0f,dot(h.n,ld));
        if (nd <= 0.0f) continue;
        result.irradiance += light.color * (nd*light.power/(dist2+1.2f));
        result.visibility[li] = 1.0f;
    }
    return result;
}

V3 shadeSurface(V3 direction, const Hit& h, const LightingCache* cache,
                bool dynamicLighting, bool useLightingCache) {
    if (!h.valid) return sky(direction);
    const V3 p=h.p, n=h.n, al=h.albedo, view=direction*(-1.0f);
    // While an object is actively moving, recompute direct lighting and shadows
    // for the current geometry so its shadow follows the object immediately.
    // After the drag, use a cheap unshadowed estimate until the asynchronous
    // lightmap bake is ready; once stable, camera motion only samples the cache.
    const LightValue baked = dynamicLighting
        ? bakeLightingAt(p, n, objects, lightSamples)
        : ((useLightingCache && cache && cache->valid)
            ? sampleLightingCache(h,cache) : approximateLighting(h));
    // SIHTL: luz indireta vinda dos tubos secundarios. Durante o arraste, os valores do
    // cache anterior sao reaproveitados (variam suavemente), evitando piscadas.
    V3 indirect{};
    if (sihtl.indirect) {
        if (dynamicLighting) { if (cache && cache->valid) indirect = sampleLightingCache(h,cache).indirect; }
        else indirect = baked.indirect;
    }
    const float ambient = sihtl.indirect ? 0.045f : 0.085f;
    V3 color = hadamard(al, V3(ambient,ambient,ambient) + baked.irradiance + indirect);

    // Highlights remain view-dependent, but use cached visibility: no shadow
    // rays are traced while looking around the fixed scene.
    for (size_t li=0; li<lights.size(); ++li) {
        if (baked.visibility[li] <= 0.0f) continue;
        const Light& light=lights[li];
        const V3 toLight=light.pos-p;
        const V3 ld=normalize(toLight);
        if (dot(n,ld) <= 0.0f) continue;
        const V3 halfV=normalize(ld+view);
        const float specBase=std::max(0.0f,dot(n,halfV));
        const int shininess=std::max(5,static_cast<int>(100.0f*(1.0f-h.roughness)+0.5f));
        const float spec=fastPowInteger(specBase,shininess)*(0.12f+h.metalness*0.65f)*baked.visibility[li];
        color += hadamard(V3(spec,spec,spec),light.color);
    }
    if(h.surface==Surface::Sphere && p.y<1.1f) color=color*0.92f;
    return clampv(color,0,3.0f);
}

V3 shade(V3 origin, V3 direction, int depth, const LightingCache* cache,
          bool dynamicLighting, bool useLightingCache);

V3 shadeHit(V3 direction, const Hit& h, int depth, const LightingCache* cache,
            bool dynamicLighting, bool useLightingCache) {
    if (!h.valid) return sky(direction);
    V3 color = shadeSurface(direction,h,cache,dynamicLighting,useLightingCache);
    // Keep a single secondary reflection ray. Its hit uses the same baked
    // lighting, so moving the camera changes reflection geometry but not the
    // expensive illumination/shadow solution.
    if(depth==0 && (h.metalness>0.08f || h.roughness<0.42f)) {
        V3 rd=normalize(reflect(direction,h.n));
        if(h.roughness>0.05f) {
            const V3 tangentRaw{-rd.z,0.0f,rd.x};
            V3 tangent=length(tangentRaw)>1e-5f?normalize(tangentRaw):V3(1,0,0);
            const V3 bitangent=normalize({
                rd.y*tangent.z-rd.z*tangent.y,
                rd.z*tangent.x-rd.x*tangent.z,
                rd.x*tangent.y-rd.y*tangent.x
            });
            rd=normalize(rd+tangent*(h.roughness*0.16f)+bitangent*(h.roughness*0.10f));
        }
        const V3 reflected=shade(h.p+h.n*0.012f,rd,1,cache,dynamicLighting,useLightingCache);
        const float fresnel=clampf(h.metalness*0.72f+(1-h.roughness)*0.18f,0,0.82f);
        color=color*(1-fresnel)+reflected*fresnel;
    }
    return clampv(color,0,3.0f);
}

V3 shade(V3 origin, V3 direction, int depth, const LightingCache* cache,
          bool dynamicLighting, bool useLightingCache) {
    const Hit h=traceScene(origin,direction);
    return shadeHit(direction,h,depth,cache,dynamicLighting,useLightingCache);
}

void lightingWorkerLoop() {
    for (;;) {
        std::vector<Object> sceneSnapshot;
        int samplesSnapshot = 1;
        SihtlConfig cfgSnapshot;
        uint64_t generation = 0;
        {
            std::unique_lock<std::mutex> lock(lightingWorkMutex);
            lightingWorkCv.wait(lock, [] { return stopLightingWorker || lightingWorkPending; });
            if (stopLightingWorker) return;
            sceneSnapshot = pendingLightingObjects;
            samplesSnapshot = pendingLightingSamples;
            cfgSnapshot = pendingSihtl;
            generation = pendingLightingGeneration;
            lightingWorkPending = false;
        }

        auto built = std::make_shared<LightingCache>();
        const bool ok = buildLightingCache(*built, sceneSnapshot, samplesSnapshot, generation, cfgSnapshot);
        {
            // Serialize publication with requests. An older bake can never
            // overwrite the cache for a newer object position.
            std::lock_guard<std::mutex> lock(lightingWorkMutex);
            if (ok && !stopLightingWorker &&
                lightingGeneration.load(std::memory_order_relaxed) == generation) {
                std::shared_ptr<const LightingCache> published = built;
                std::atomic_store(&activeLightingCache, published);
                lightingBakeCompleted.store(true, std::memory_order_relaxed);
            }
            if (!lightingWorkPending &&
                lightingGeneration.load(std::memory_order_relaxed) == generation) {
                lightingBakeRunning.store(false, std::memory_order_relaxed);
            }
        }
    }
}

void startLightingWorker() {
    lightingBakeThread = std::thread(lightingWorkerLoop);
}

void stopLightingWorkerAndJoin() {
    {
        std::lock_guard<std::mutex> lock(lightingWorkMutex);
        stopLightingWorker = true;
        lightingWorkPending = false;
        lightingGeneration.fetch_add(1, std::memory_order_relaxed);
    }
    lightingWorkCv.notify_one();
    if (lightingBakeThread.joinable()) lightingBakeThread.join();
}

void requestLightingBake() {
    std::vector<Object> snapshot = objects;
    {
        std::lock_guard<std::mutex> lock(lightingWorkMutex);
        const uint64_t generation = lightingGeneration.fetch_add(1, std::memory_order_relaxed) + 1;
        pendingLightingObjects = std::move(snapshot);
        pendingLightingSamples = lightSamples;
        pendingSihtl = sihtl;
        if (draggingObject >= 0) {
            // Refinamento adaptativo ao movimento: enquanto arrasta, usa menos tubos.
            pendingSihtl.maxDepth = std::max(1, sihtl.maxDepth - 1);
            pendingSihtl.maxPatches = std::max(60, sihtl.maxPatches / 3);
            pendingSihtl.bounces = 1;
        }
        pendingLightingGeneration = generation;
        lightingWorkPending = true;
        lightingBakeRunning.store(true, std::memory_order_relaxed);
        lightingBakeCompleted.store(false, std::memory_order_relaxed);
    }
    lightingWorkCv.notify_one();
}

V3 cameraForward() {
    return normalize({std::sin(camera.yaw)*std::cos(camera.pitch),
                      std::sin(camera.pitch),
                      std::cos(camera.yaw)*std::cos(camera.pitch)});
}
struct CameraBasis {
    V3 forward, right, up;
    float aspect, scale, firstPixelX, pixelStepX;
};
inline CameraBasis makeCameraBasis() {
    const V3 f=cameraForward();
    const V3 r=normalize({f.z,0,-f.x});
    const V3 u=normalize({f.y*r.z-f.z*r.y, f.z*r.x-f.x*r.z, f.x*r.y-f.y*r.x});
    const float aspect=static_cast<float>(renderW)/static_cast<float>(renderH);
    const float scale=std::tan(SIHTL_PI/6.0f);
    return {f,r,u,aspect,scale,(1.0f/static_cast<float>(renderW)-1.0f)*aspect*scale,
            (2.0f*aspect*scale)/static_cast<float>(renderW)};
}
inline V3 cameraRayFromBasis(float x, float y, const CameraBasis& b) {
    const float px=(2.0f*(x+0.5f)/renderW-1.0f)*b.aspect*b.scale;
    const float py=(1.0f-2.0f*(y+0.5f)/renderH)*b.scale;
    return normalize(b.forward+b.right*px+b.up*py);
}
V3 cameraRay(float x, float y) {
    return cameraRayFromBasis(x,y,makeCameraBasis());
}

void resetAccumulation() {
    // Toda renderizacao substitui todos os pixels; zerar o buffer aqui seria
    // uma passada completa extra a cada movimento sem qualquer beneficio visual.
    accumulatedSamples=0;
    sceneDirty=true;
    displayDirty=true;
}

void setRenderSize(int width, int height) {
    width=std::max(120,width);
    height=std::max(68,height);
    if (renderTexture.id && width==renderW && height==renderH) {
        resetAccumulation();
        return;
    }
    renderW=width;
    renderH=height;
    if(renderTexture.id) UnloadTexture(renderTexture);
    Image image=GenImageColor(renderW,renderH,Color{8,13,21,255});
    renderTexture=LoadTextureFromImage(image);
    UnloadImage(image);
    SetTextureFilter(renderTexture,TEXTURE_FILTER_BILINEAR);
    const size_t count=static_cast<size_t>(renderW)*renderH;
    accumulationBuffer.assign(count,V3{});
    denoisedBuffer.assign(count,V3{});
    hitGuideBuffer.assign(count,Hit{});
    pixelBuffer.assign(count,Color{8,13,21,255});
    resetAccumulation();
}

void setQuality(int mode) {
    qualityMode=std::max(0,std::min(2,mode));
    // Presets preservados. A resolucao nao cai durante a movimentacao da camera.
    if(qualityMode==0) { lightSamples=1; sihtl.maxDepth=3; sihtl.bounces=1; sihtl.maxPatches=300;  setRenderSize(240,135); }
    else if(qualityMode==1) { lightSamples=2; sihtl.maxDepth=4; sihtl.bounces=2; sihtl.maxPatches=600;  setRenderSize(320,180); }
    else { lightSamples=2; sihtl.maxDepth=5; sihtl.bounces=2; sihtl.maxPatches=1200; setRenderSize(480,270); }
    requestLightingBake();
}

bool guideDiscontinuity(const Hit& a, const Hit& b) {
    if(a.valid != b.valid) return true;
    if(!a.valid) return false;
    if(a.surface != b.surface || a.objectIndex != b.objectIndex) return true;
    const float depthDelta=std::fabs(a.t-b.t);
    if(depthDelta > std::max(0.055f,std::min(a.t,b.t)*0.022f)) return true;
    if(dot(a.n,b.n) < 0.91f) return true;
    // Nao trate mudancas do xadrez do chao como bordas geometricas.
    return false;
}

bool isGuideEdge(int x, int y) {
    const size_t i=static_cast<size_t>(y)*renderW+x;
    const Hit& center=hitGuideBuffer[i];
    if(x>0 && guideDiscontinuity(center,hitGuideBuffer[i-1])) return true;
    if(x+1<renderW && guideDiscontinuity(center,hitGuideBuffer[i+1])) return true;
    if(y>0 && guideDiscontinuity(center,hitGuideBuffer[i-renderW])) return true;
    if(y+1<renderH && guideDiscontinuity(center,hitGuideBuffer[i+renderW])) return true;
    return false;
}

void renderRows(int yBegin, int yEnd, int yStride, const CameraBasis& basis, const LightingCache* cache,
                bool dynamicLighting, bool useLightingCache) {
    // Cada linha avanca os raios por incrementos constantes. Linhas intercaladas
    // distribuem melhor o custo entre threads (ceu, objetos e chao custam diferente).
    const V3 step=basis.right*basis.pixelStepX;
    for (int y=yBegin; y<yEnd; y+=yStride) {
        const float py=(1.0f-2.0f*(static_cast<float>(y)+0.5f)/renderH)*basis.scale;
        V3 rayStart=basis.forward+basis.right*basis.firstPixelX+basis.up*py;
        const size_t row=static_cast<size_t>(y)*renderW;
        for (int x=0; x<renderW; ++x) {
            const size_t i=row+static_cast<size_t>(x);
            const V3 ray=normalize(rayStart);
            rayStart += step;
            const Hit hit=traceScene(camera.pos,ray);
            if (denoiserEnabled) hitGuideBuffer[i]=hit;
            // Escreve diretamente no buffer final, sem copiar toda a imagem depois.
            accumulationBuffer[i]=shadeHit(ray,hit,0,cache,dynamicLighting,useLightingCache);
        }
    }
}

void renderOneSample() {
    ++frameIndex;
    // Camera trigonometry is shared across the whole frame.
    const CameraBasis basis=makeCameraBasis();
    const std::shared_ptr<const LightingCache> cacheOwner = std::atomic_load(&activeLightingCache);
    const LightingCache* cache = cacheOwner ? cacheOwner.get() : nullptr;
    // A current cache is reused for every camera move. After an object changes,
    // compute current shadows live until the latest cache is ready, preventing
    // stale shadows or a lighting pop when the mouse is released.
    const bool dynamicLighting = lightingNeedsBake;
    const bool useLightingCache = !lightingNeedsBake;
    unsigned int n=std::thread::hardware_concurrency();
    if (n==0) n=1;
    // Paraleliza ate oito workers e espalha as linhas para balancear o custo.
    n=std::max(1u,std::min(8u,n));
    n=std::min(n,static_cast<unsigned int>(std::max(1,renderH)));
    if (n==1 || renderH<48) {
        renderRows(0,renderH,1,basis,cache,dynamicLighting,useLightingCache);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(n-1);
        for (unsigned int t=1;t<n;++t) {
            workers.emplace_back([&,t]() {
                renderRows(static_cast<int>(t),renderH,static_cast<int>(n),basis,cache,dynamicLighting,useLightingCache);
            });
        }
        renderRows(0,renderH,static_cast<int>(n),basis,cache,dynamicLighting,useLightingCache);
        for (auto& worker:workers) worker.join();
    }

    accumulatedSamples=1;
    sceneDirty=false;
    displayDirty=true;
}

void applyDenoiser() {
    if(!denoiserEnabled) return;

    // Filtro geometrico conservador: nunca mistura superficies diferentes,
    // silhuetas ou pixels com normais muito distintas. O modo padrao fica OFF
    // porque as amostras ja sao deterministicas e preservar nitidez e prioridade.
    for(int y=0;y<renderH;++y) {
        for(int x=0;x<renderW;++x) {
            const size_t i=static_cast<size_t>(y)*renderW+x;
            const Hit& h=hitGuideBuffer[i];
            const V3 center=accumulationBuffer[i];
            if(isGuideEdge(x,y) || !h.valid) {
                denoisedBuffer[i]=center;
                continue;
            }
            V3 sum=center*2.0f;
            float total=2.0f;
            for(int oy=-1;oy<=1;++oy) for(int ox=-1;ox<=1;++ox) {
                if(ox==0 && oy==0) continue;
                const int nx=x+ox, ny=y+oy;
                if(nx<0 || nx>=renderW || ny<0 || ny>=renderH) continue;
                const size_t ni=static_cast<size_t>(ny)*renderW+nx;
                const Hit& nh=hitGuideBuffer[ni];
                if(!nh.valid || nh.surface!=h.surface || nh.objectIndex!=h.objectIndex) continue;
                if(std::fabs(nh.t-h.t)>std::max(0.035f,h.t*0.012f)) continue;
                if(dot(nh.n,h.n)<0.96f) continue;
                const V3 c=accumulationBuffer[ni];
                const V3 delta=c-center;
                const float diff=dot(delta,delta);
                const float w=(ox==0 || oy==0?0.22f:0.12f)/(1.0f+18.0f*diff);
                sum+=c*w;
                total+=w;
            }
            denoisedBuffer[i]=sum/total;
        }
    }
}

unsigned char toByte(float x) {
    // Tabela de gama calculada uma unica vez: evita tres chamadas a pow() por pixel.
    static const std::array<unsigned char,4097> gammaLut=[] {
        std::array<unsigned char,4097> table{};
        for(size_t i=0;i<table.size();++i) {
            const float value=static_cast<float>(i)/4096.0f;
            table[i]=static_cast<unsigned char>(clampf(std::pow(value,0.85f)*255.0f+0.5f,0,255));
        }
        return table;
    }();
    const int index=static_cast<int>(clampf(x,0.0f,1.0f)*4096.0f+0.5f);
    return gammaLut[static_cast<size_t>(std::min(4096,index))];
}
void uploadDisplay() {
    if(denoiserEnabled) applyDenoiser();

    // Reforco de nitidez moderado para reduzir a suavizacao da ampliacao bilinear.
    const std::vector<V3>& displayBuffer=denoiserEnabled?denoisedBuffer:accumulationBuffer;
    for(int y=0;y<renderH;++y) {
        for(int x=0;x<renderW;++x) {
            const size_t i=static_cast<size_t>(y)*renderW+x;
            const V3 center=displayBuffer[i];
            V3 neighbors{};
            int count=0;
            if(x>0) { neighbors+=displayBuffer[i-1]; ++count; }
            if(x+1<renderW) { neighbors+=displayBuffer[i+1]; ++count; }
            if(y>0) { neighbors+=displayBuffer[i-renderW]; ++count; }
            if(y+1<renderH) { neighbors+=displayBuffer[i+renderW]; ++count; }
            const V3 average=count>0?neighbors/static_cast<float>(count):center;
            const V3 crisp=clampv(center+(center-average)*0.10f,0.0f,3.0f);
            pixelBuffer[i]=Color{toByte(crisp.x),toByte(crisp.y),toByte(crisp.z),255};
        }
    }
    UpdateTexture(renderTexture,pixelBuffer.data());
    displayDirty=false;
}

bool intersectHorizontalPlane(V3 origin, V3 dir, float planeY, V3& out) {
    if(std::fabs(dir.y)<1e-5f) return false;
    const float t=(planeY-origin.y)/dir.y;
    if(t<=0.001f) return false;
    out=origin+dir*t;
    return true;
}
V3 mouseRay() {
    const float x=static_cast<float>(GetMouseX())*renderW/static_cast<float>(std::max(1,GetScreenWidth()));
    const float y=static_cast<float>(GetMouseY())*renderH/static_cast<float>(std::max(1,GetScreenHeight()));
    return cameraRay(x,y);
}

void resetScene() {
    camera=Camera{};
    for(size_t i=0;i<objects.size();++i) objects[i].center=originalCenters[i];
    selectedObject=-1;
    draggingObject=-1;
    lightingNeedsBake = true;
    resetAccumulation();
    requestLightingBake();
}

void handleInput() {
    bool moved=false;
    const float dt=std::min(GetFrameTime(),0.05f);
    float forwardInput=0, sideInput=0;
    if(IsKeyDown(KEY_W)||IsKeyDown(KEY_UP)) forwardInput+=1;
    if(IsKeyDown(KEY_S)||IsKeyDown(KEY_DOWN)) forwardInput-=1;
    if(IsKeyDown(KEY_D)||IsKeyDown(KEY_RIGHT)) sideInput+=1;
    if(IsKeyDown(KEY_A)||IsKeyDown(KEY_LEFT)) sideInput-=1;
    if(forwardInput!=0 || sideInput!=0) {
        const float speed=(IsKeyDown(KEY_LEFT_SHIFT)?4.0f:2.2f)*dt;
        const float fx=std::sin(camera.yaw), fz=std::cos(camera.yaw);
        const float len=std::sqrt(forwardInput*forwardInput+sideInput*sideInput);
        forwardInput/=len; sideInput/=len;
        camera.pos.x += speed*(forwardInput*fx+sideInput*fz);
        camera.pos.z += speed*(forwardInput*fz-sideInput*fx);
        camera.pos.x=clampf(camera.pos.x,-4.0f,4.0f);
        camera.pos.z=clampf(camera.pos.z,-7.5f,5.5f);
        moved=true;
    }

    if(IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
        const Vector2 delta=GetMouseDelta();
        if(std::fabs(delta.x)>0.01f || std::fabs(delta.y)>0.01f) {
            camera.yaw += delta.x*0.0035f;
            camera.pitch=clampf(camera.pitch-delta.y*0.0035f,-1.15f,1.15f);
            moved=true;
        }
    }

    if(IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        const Hit h=traceScene(camera.pos,mouseRay());
        if(h.valid && h.objectIndex>=0) {
            selectedObject=h.objectIndex;
            draggingObject=h.objectIndex;
            dragPlaneY=objects[h.objectIndex].center.y;
            V3 planePoint;
            if(intersectHorizontalPlane(camera.pos,mouseRay(),dragPlaneY,planePoint)) {
                dragOffset=V3(objects[h.objectIndex].center.x-planePoint.x,0,
                              objects[h.objectIndex].center.z-planePoint.z);
            } else draggingObject=-1;
        } else {
            selectedObject=-1;
            draggingObject=-1;
        }
    }
    if(IsMouseButtonDown(MOUSE_BUTTON_LEFT) && draggingObject>=0) {
        V3 planePoint;
        if(intersectHorizontalPlane(camera.pos,mouseRay(),dragPlaneY,planePoint)) {
            Object& ob=objects[draggingObject];
            const float nx=clampf(planePoint.x+dragOffset.x,-3.7f,3.7f);
            const float nz=clampf(planePoint.z+dragOffset.z,-6.2f,5.7f);
            if(std::fabs(nx-ob.center.x)>0.001f || std::fabs(nz-ob.center.z)>0.001f) {
                ob.center.x=nx;
                ob.center.z=nz;
                moved=true;
                lightingNeedsBake=true;
                // Start refreshing lighting on this movement event, not on mouse-up.
                // The persistent worker coalesces rapid pointer events without blocking UI.
                requestLightingBake();
            }
        }
    }
    if(IsMouseButtonReleased(MOUSE_BUTTON_LEFT) && draggingObject>=0) {
        // Keep the cache marked dirty until the latest asynchronous bake is published.
        // The visible lighting was updated live throughout the drag.
        draggingObject=-1;
        requestLightingBake();   // refaz com a qualidade completa dos tubos
    }

    if(IsKeyPressed(KEY_ONE)) setQuality(0);
    if(IsKeyPressed(KEY_TWO)) setQuality(1);
    if(IsKeyPressed(KEY_THREE)) setQuality(2);
    if(IsKeyPressed(KEY_N)) {
        denoiserEnabled=!denoiserEnabled;
        displayDirty=true;
        // Guias geometricos so sao calculados quando o denoiser esta ativo.
        if(denoiserEnabled) sceneDirty=true;
    }
    {
        bool changed=false;
        if(IsKeyPressed(KEY_I)) { sihtl.indirect=!sihtl.indirect; changed=true; }
        if(IsKeyPressed(KEY_B)) { sihtl.bounces = sihtl.bounces>=3 ? 1 : sihtl.bounces+1; changed=true; }
        if(IsKeyPressed(KEY_LEFT_BRACKET))  { sihtl.maxDepth=std::max(0,sihtl.maxDepth-1); changed=true; }
        if(IsKeyPressed(KEY_RIGHT_BRACKET)) { sihtl.maxDepth=std::min(6,sihtl.maxDepth+1); changed=true; }
        if(IsKeyPressed(KEY_MINUS)) { sihtl.errorLimit=std::max(0.005f,sihtl.errorLimit*0.7f); changed=true; }
        if(IsKeyPressed(KEY_EQUAL)) { sihtl.errorLimit=std::min(0.5f,sihtl.errorLimit*1.4f); changed=true; }
        if(IsKeyPressed(KEY_T)) { showTubes=!showTubes; }
        if(changed) { requestLightingBake(); moved=true; }
    }
    if(IsKeyPressed(KEY_R)) resetScene();
    if(IsKeyPressed(KEY_F11)) {
        if(fullscreen) { ToggleFullscreen(); fullscreen=false; }
        else { ToggleFullscreen(); fullscreen=true; }
    }
    if(moved) resetAccumulation();
}

void drawTubeImpacts() {
    // Depuracao: cada ponto e o impacto de um tubo confiavel. Cores = energia refletida;
    // regioes com pontos mais densos sao onde o refinamento adaptativo subdividiu.
    const std::shared_ptr<const LightingCache> owner = std::atomic_load(&activeLightingCache);
    if(!owner || !owner->valid) return;
    const CameraBasis b=makeCameraBasis();
    const float W=static_cast<float>(GetScreenWidth()), H=static_cast<float>(GetScreenHeight());
    for(const LightPatch& p:owner->patches) {
        const V3 d=p.p-camera.pos;
        const float z=dot(d,b.forward);
        if(z<0.1f) continue;
        const float sx=(0.5f+0.5f*dot(d,b.right)/(z*b.aspect*b.scale))*W;
        const float sy=(0.5f-0.5f*dot(d,b.up)/(z*b.scale))*H;
        if(sx<0||sx>W||sy<0||sy>H) continue;
        const float e=clampf((p.flux.x+p.flux.y+p.flux.z)*0.6f,0.0f,1.0f);
        const Color c{static_cast<unsigned char>(80+175*e),static_cast<unsigned char>(220-120*e),
                      static_cast<unsigned char>(255-200*e),230};
        DrawCircle(static_cast<int>(sx),static_cast<int>(sy),2.0f+3.0f*(p.bounce==1?1.0f:0.4f),c);
    }
}

void drawOverlay() {
    DrawRectangle(12,12,430,168,Fade(Color{6,12,20,255},0.88f));
    DrawRectangleLines(12,12,430,168,Color{70,95,120,220});
    DrawText("SIHTL  -  TUBOS DE LUZ (CPU)",24,21,16,Color{225,237,248,255});
    const char* qualityName=qualityMode==0?"LEVE":(qualityMode==1?"EQUILIBRADA":"ALTA");
    DrawText(TextFormat("Resolucao: %d x %d | FPS: %d | Qualidade: %s",renderW,renderH,GetFPS(),qualityName),24,45,12,Color{170,195,218,255});
    DrawText(TextFormat("Luz pre-calculada | Amostras: %d | Denoiser: %s%s",lightSamples,denoiserEnabled?"ON":"OFF",lightingBakeRunning.load()?" | atualizando luz...":""),24,63,12,Color{170,195,218,255});
    DrawText(TextFormat("Indireta: %s | Rebatimentos: %d | Subdiv. max: %d | Erro: %.3f",
             sihtl.indirect?"ON":"OFF",sihtl.bounces,sihtl.maxDepth,sihtl.errorLimit),24,86,12,Color{140,230,180,255});
    const int rays=statRays.load(), tubes=statTubes.load(), patches=statPatches.load();
    const long long eq=statRayEquivalent.load();
    DrawText(TextFormat("Tubos: %d | Raios reais: %d | Fontes secundarias: %d | Bake: %d ms",tubes,rays,patches,statMs.load()),24,104,12,Color{140,230,180,255});
    if(rays>0 && eq>0)
        DrawText(TextFormat("Raios uniformes equivalentes: %lld  ->  %.1fx menos | Energia 1o rebatimento: %.0f%%",
                 eq,static_cast<double>(eq)/static_cast<double>(std::max(1,rays)),statEnergyKept.load()*100.0f),24,122,12,Color{140,230,180,255});
    if(selectedObject>=0 && selectedObject<static_cast<int>(objects.size()))
        DrawText(TextFormat("Selecionado: %s",objects[selectedObject].name.c_str()),24,146,12,Color{255,217,130,255});
    else
        DrawText("Clique e arraste uma esfera ou a caixa",24,146,12,Color{170,195,218,255});

    if(showTubes) drawTubeImpacts();

    const int barY=GetScreenHeight()-43;
    DrawRectangle(0,barY,GetScreenWidth(),43,Fade(Color{5,10,17,255},0.88f));
    DrawText("WASD/setas: camera   |   Botao direito: olhar   |   Botao esquerdo: mover objetos   |   T: ver tubos",12,barY+6,12,Color{220,231,241,255});
    DrawText("1/2/3 qualidade | I indireta | B rebatimentos | [ ] subdivisao | - = erro | N denoiser | R reiniciar | F11",12,barY+23,11,Color{160,183,205,255});
}
} // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT);
    InitWindow(1280,720,"SIHTL - Tubos de Luz - CPU");
    SetWindowMinSize(640,400);
    SetTargetFPS(60);
    startLightingWorker();
    setQuality(0);

    while(!WindowShouldClose()) {
        handleInput();
        if(lightingBakeCompleted.exchange(false, std::memory_order_relaxed)) {
            lightingNeedsBake = false;
            resetAccumulation();
        }
        if(sceneDirty || accumulatedSamples<maxAccumSamples) renderOneSample();
        if(displayDirty) uploadDisplay();

        BeginDrawing();
        ClearBackground(Color{5,8,13,255});
        const Rectangle source{0,0,static_cast<float>(renderW),static_cast<float>(renderH)};
        const Rectangle dest{0,0,static_cast<float>(GetScreenWidth()),static_cast<float>(GetScreenHeight())};
        DrawTexturePro(renderTexture,source,dest,Vector2{0,0},0.0f,WHITE);
        drawOverlay();
        EndDrawing();
    }

    stopLightingWorkerAndJoin();
    if(renderTexture.id) UnloadTexture(renderTexture);
    CloseWindow();
    return 0;
}
