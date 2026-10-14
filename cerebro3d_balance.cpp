// =============================================================================
//  CEREBRO ESPARSO 3D  -  uma criatura de 4 patas aprende a caminhar ate as
//  recompensas (orbes) que mudam de lugar.   C++17 + raylib, arquivo unico.
//
//  COMO FUNCIONA
//  * Cerebro com 16384 neuronios (512 grupos x 32), mas SPARSE: um "roteador"
//    escolhe, a cada instante, so os grupos necessarios (de 3 a 12 grupos,
//    conforme a situacao exigir) e so esses neuronios sao calculados.
//    Neuronios com ativacao <= 0 tambem nao entram na conta nem no aprendizado.
//  * Aprendizado: Actor-Critic (A2C + GAE) so atualizando os neuronios ativos.
//  * Memoria: o arquivo "cerebro_memoria.bin" e regravado a cada 5 s (e ao sair)
//    e e lido automaticamente ao abrir o programa, entao ele continua de onde parou.
//  * Fisica 3D propria (tronco com 6 graus de liberdade + 4 patas de 3 juntas,
//    contato com atrito), sem bibliotecas externas alem da raylib.
//
//  ENXAME (novo)
//  * Com o aprendizado LIGADO, o cerebro atual e clonado: ficam N_AGENTS IAs (a
//    original + clones) treinando ao mesmo tempo, cada uma no seu proprio mundo e
//    na sua propria thread, cada uma com um nivel de exploracao diferente.
//  * A cada "geracao" (GEN_EPISODES episodios por IA) elas sao comparadas pelo
//    retorno medio. O cerebro da melhor e guardado como "melhor forma" (e e ele que
//    vai para cerebro_memoria.bin) e as piores passam a ser copias da melhor.
//  * Ao DESLIGAR o aprendizado (tecla L) sobra UMA so IA, com o melhor cerebro.
//
//  COMPILAR (Linux):
//    g++ cerebro3d.cpp -o cerebro -O2 -std=c++17 -pthread -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
//  Windows (MinGW):
//    g++ cerebro3d.cpp -o cerebro.exe -O2 -std=c++17 -pthread -lraylib -lopengl32 -lgdi32 -lwinmm
//  macOS:
//    g++ cerebro3d.cpp -o cerebro -O2 -std=c++17 -pthread -lraylib -framework OpenGL -framework Cocoa -framework IOKit
//
//  (MinGW: precisa da versao "posix threads"; se nao tiver, ponha MULTITHREAD 0 abaixo)
//
//  EXECUTAR:   ./cerebro                 (com janela 3D)
//              ./cerebro --headless 120  (treina 120 s sem janela, bem rapido)
//
//  TECLAS:  CIMA/BAIXO velocidade da simulacao | L liga/desliga aprendizado (enxame)
//           E alterna equilibrio na esfera/caminhada | C alterna recompensa de corrida
//           TAB troca a IA exibida
//           S salvar agora | ESPACO pausa | mouse esquerdo gira camera | roda = zoom
// =============================================================================
#define MULTITHREAD 1     // 1 = cada IA do enxame roda numa thread; 0 = tudo numa thread so

#include "raylib.h"
#include "rlgl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#if MULTITHREAD
#include <thread>
#endif

// ============================ CONFIGURACAO ===================================
static const char* MEM_FILE = "cerebro_memoria.bin";
static const char* BALANCE_MEM_FILE = "cerebro_equilibrio_v13.bin";
static const char* BIPED_MEM_FILE = "cerebro_bipede_digitigrado_v15.bin";
static const char* BIPED_BALANCE_MEM_FILE = "cerebro_bipede_equilibrio_digitigrado_v15.bin";
static const char* RUN_MEM_FILE = "cerebro_corrida.bin";
static const char* BIPED_RUN_MEM_FILE = "cerebro_bipede_corrida.bin";

constexpr int   N_GROUPS   = 512;                   // grupos (colunas corticais)
constexpr int   GROUP_SIZE = 32;                    // neuronios por grupo
constexpr int   N_NEURONS  = N_GROUPS * GROUP_SIZE; // total de neuronios
constexpr int   N_IN       = 33;                    // sensores
constexpr int   N_ACT      = 12;                    // atuadores: 4 patas x 3, ou 2 pernas x 3 no bípede
constexpr int   K_MIN      = 3;                     // minimo de grupos ativos
constexpr int   K_MAX      = 512;                    // maximo de grupos ativos
constexpr float ROUTER_BETA = 6.0f;                 // "foco" do roteador
constexpr float ROUTER_MASS = 0.60f;                // massa de atencao que basta cobrir

constexpr float PHYS_DT   = 1.0f / 240.0f;          // passo da fisica
constexpr int   SUBSTEPS  = 12;                     // fisica por decisao do cerebro
constexpr float CTRL_DT   = PHYS_DT * SUBSTEPS;     // 20 decisoes por segundo
constexpr int   MAX_EP_STEPS = 600;                 // 30 s por episodio
constexpr int   N_ORBS    = 1;                      // recompensas simultaneas
constexpr float ORB_PICK_R = 0.75f;

constexpr int   ROLLOUT_T = 128;                    // lote menor: atualiza a politica a cada 6,4 s simulados
constexpr float GAMMA = 0.98f, LAMBDA = 0.95f;
constexpr float LR_ACTOR = 0.35f, LR_CRITIC = 0.5f, LR_HIDDEN = 0.03f;
constexpr float GCLIP = 5.0f, VCLIP = 5.0f;

// ---- enxame de IAs ----
constexpr int    N_AGENTS      = 15;     // total de IAs treinando juntas (a original + clones)
constexpr int    N_REPLACE     = 8;     // quantas das piores viram copia da melhor a cada geracao
constexpr size_t GEN_EPISODES  = 30;    // episodios por IA em cada geracao (janela de comparacao)
constexpr size_t FINAL_MIN_EPS = 8;     // minimo de episodios p/ uma IA concorrer ao desligar o aprendizado
constexpr size_t RECENT_MAX    = 40;    // quantos episodios recentes entram na media do score
static_assert(N_AGENTS >= 2 && N_REPLACE >= 1 && N_REPLACE < N_AGENTS, "configuracao do enxame invalida");

constexpr float kTwoPi = 6.28318530718f;

// ============================ UTILIDADES =====================================
static thread_local std::mt19937 g_rng(12345);   // thread_local: cada IA do enxame tem a sua sequencia
static float randn() { static thread_local std::normal_distribution<float> d(0.f, 1.f); return d(g_rng); }
static float rnd01() { return std::uniform_real_distribution<float>(0.f, 1.f)(g_rng); }
static float rndRange(float a, float b) { return a + (b - a) * rnd01(); }
static float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }

struct V3 { float x = 0, y = 0, z = 0; };
static inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static inline float len(V3 a) { return std::sqrt(dot(a, a)); }
static inline V3 norm(V3 a) { float l = len(a); return l > 1e-9f ? a * (1.f / l) : V3{}; }
static inline V3 rotAxis(V3 v, V3 k, float a) {
    float c = std::cos(a), s = std::sin(a);
    return v * c + cross(k, v) * s + k * (dot(k, v) * (1.f - c));
}
static inline Vector3 toRl(V3 v) { return Vector3{v.x, v.y, v.z}; }
static inline Color rgba(int r, int g, int b, int a = 255) {
    return Color{(unsigned char)r, (unsigned char)g, (unsigned char)b, (unsigned char)a};
}
static void hud(int x, int y, int size, Color c, const char* fmt, ...) {
    char buf[300]; va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    DrawText(buf, x, y, size, c);
}

// ================================ CEREBRO ====================================
struct Act {                       // resultado de uma passagem pelo cerebro
    int   nGroups = 0, nActive = 0;
    int   groups[K_MAX];
    int   idx[K_MAX * GROUP_SIZE];
    float hn[K_MAX * GROUP_SIZE];  // ativacoes normalizadas dos neuronios ativos
    float scale = 0;
    float mu[N_ACT];
    float value = 0;
};

struct Sample {                    // um instante guardado para aprender
    float x[N_IN];
    std::vector<int>   idx;
    std::vector<float> hn;
    float scale = 0, eps[N_ACT], value = 0, reward = 0, adv = 0, ret = 0;
    bool  done = false;
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

    // Actor-Critic: so os neuronios que estavam ativos sao modificados.
    void train(std::vector<Sample>& B, int n, float lastV, float sigma, float lrScale = 1.f) {
        float adv = 0;
        for (int t = n - 1; t >= 0; t--) {
            float nv = (t == n - 1) ? lastV : B[t + 1].value;
            float nt = B[t].done ? 0.f : 1.f;
            float delta = B[t].reward + GAMMA * nv * nt - B[t].value;
            adv = delta + GAMMA * LAMBDA * nt * adv;
            B[t].adv = adv; B[t].ret = adv + B[t].value;
        }
        float m = 0, v = 0;
        for (int t = 0; t < n; t++) m += B[t].adv;
        m /= n;
        for (int t = 0; t < n; t++) { float d = B[t].adv - m; v += d * d; }
        float sd = std::sqrt(v / n) + 1e-6f;
        for (int t = 0; t < n; t++) B[t].adv = (B[t].adv - m) / sd;

        const float la = LR_ACTOR * lrScale / n;
        const float lc = LR_CRITIC * lrScale / n;
        const float lh = LR_HIDDEN * lrScale / n;
        for (int t = 0; t < n; t++) {
            Sample& s = B[t];
            float g[N_ACT];
            for (int k = 0; k < N_ACT; k++) g[k] = clampf(s.adv * s.eps[k] / sigma, -GCLIP, GCLIP);
            float verr = clampf(s.ret - s.value, -VCLIP, VCLIP);
            for (size_t a = 0; a < s.idx.size(); a++) {
                int j = s.idx[a]; float h = s.hn[a];
                float* wo = &Wout[(size_t)j * N_ACT];
                float dh = 0;
                for (int k = 0; k < N_ACT; k++) dh += wo[k] * g[k];
                float dpre = (dh + 0.5f * V[j] * verr) * s.scale;
                for (int k = 0; k < N_ACT; k++) wo[k] += la * g[k] * h;
                V[j] += lc * verr * h;
                float* wi = &Win[(size_t)j * N_IN];
                float st = lh * dpre;
                for (int i = 0; i < N_IN; i++) wi[i] += st * s.x[i];
                bias[j] += st;
            }
        }
    }

    // ----------------------------- MEMORIA EM ARQUIVO --------------------------
    template <class T> static void wr(std::ofstream& f, const std::vector<T>& v) { f.write((const char*)v.data(), v.size() * sizeof(T)); }
    template <class T> static bool rd(std::ifstream& f, std::vector<T>& v) { f.read((char*)v.data(), v.size() * sizeof(T)); return (bool)f; }

    bool save(const char* path) const {
        namespace fs = std::filesystem;
        std::string tmp = std::string(path) + ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary);
            if (!f) return false;
            const char magic[8] = {'C', 'B', 'R', '3', 'D', '0', '0', '1'};
            int32_t dims[4] = {N_GROUPS, GROUP_SIZE, N_IN, N_ACT};
            f.write(magic, 8); f.write((const char*)dims, sizeof dims);
            f.write((const char*)&steps, 8); f.write((const char*)&episodes, 8); f.write((const char*)&orbs, 8);
            f.write((const char*)&emaReturn, 4);
            wr(f, router); wr(f, Win); wr(f, bias); wr(f, Wout); wr(f, V); wr(f, useCount);
            if (!f) return false;
        }
        std::error_code ec;
        fs::rename(tmp, path, ec);
        if (ec) { fs::remove(path, ec); fs::rename(tmp, path, ec); }   // Windows
        return !ec;
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
            const float penetration = colliderRadius - point.y;
            const V3 pointVelocity = vel + cross(wW, r) + toWorld(localPointVelocity);
            if (penetration <= 0.f && !(point.y < 0.015f && pointVelocity.y < 0.f)) return;
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
            const float penetration = colliderRadius - point.y;
            const V3 pointVelocity = vel + cross(wW, r) + toWorld(localPointVelocity);
            if (penetration <= 0.f && !(point.y < colliderRadius + 0.015f && pointVelocity.y < 0.f))
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
                const float penetration = -pf.y;
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
            if (point.y < 0.015f && (point.y < 0.f || pointVelocity.y < 0.f)) {
                const float penetration = std::max(0.f, -point.y);
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
                if (lowestPoint.y < 0.004f) {
                    pos.y += std::min(0.08f, 0.004f - lowestPoint.y);
                    if (vel.y < 0.f) vel.y = 0.f;
                    floorContact = true;
                    corrected = true;
                }
                if (!corrected) break;
            }
        }
    }
};

// ================================ SIMULACAO ==================================
struct Orb { V3 p; };

struct Sim {
    Brain brain;
    Creature cr;
    Orb orbs[N_ORBS];
    std::vector<Sample> buf;
    int nbuf = 0, sub = 0, epSteps = 0, epOrbs = 0, orbTimer = 0;
    float epReturn = 0, simT = 0;
    bool learning = true, sampleValid = false, dirty = false;
    bool balanceMode = false;
    bool speedRewardMode = false; // caminhada: recompensa extra por velocidade horizontal
    uint64_t learningUpdates = 0; // atualizacoes Actor-Critic nesta sessao
    // Cada agente tem sua propria esfera/plataforma em movimento.
    V3 arenaOrigin{}, spherePos{0, 2.05f, 0}, sphereVel{}, sphereAngularVelocity{};
    V3 sphereDir{1, 0, 0}, sphereTargetDir{1, 0, 0};
    const float sphereRadius = 1.7f;
    float sphereClock = 0.f, spherePhase = 0.f, sphereTurnTimer = 0.f, sphereMoveSpeed = 0.85f;
    float offSphereSeconds = 0.f;
    Act act; float xobs[N_IN]; float lastEps[N_ACT], lastAction[N_ACT];
    // estatisticas
    float lastReturn = 0, avgOrbsEp = 0, avgLen = 0, avgActiveFrac = 0, sigma = 0.3f;
    float stableSeconds = 0.f, bestStableSeconds = 0.f, balanceQuality = 0.f;
    double balancePoints = 0.0;
    int   lastGroups = 0, lastActive = 0, lastEpOrbs = 0;
    std::deque<float> retHist;
    float heat[N_GROUPS] = {};
    std::deque<V3> trail;
    // acumuladores p/ modo headless
    double accRet = 0, accOrbs = 0, accLen = 0; int accN = 0;
    // enxame
    float sigmaMul = 1.f;               // multiplicador da exploracao desta IA (diversidade)
    uint32_t rngSeed = 1;               // estado do gerador aleatorio desta IA entre os blocos de simulacao
    std::deque<float> recent;           // retornos dos ultimos episodios (para comparar as IAs)

    // media dos retornos recentes (ou NO_SCORE se ainda nao tem episodios suficientes)
    float score(size_t minEps) const {
        if (recent.empty() || recent.size() < minEps) return -1e30f;
        float t = 0; for (float v : recent) t += v;
        return t / (float)recent.size();
    }

    void setupBuffers() {
        buf.resize(ROLLOUT_T);
        for (auto& s : buf) { s.idx.reserve(K_MAX * GROUP_SIZE); s.hn.reserve(K_MAX * GROUP_SIZE); }
    }
    // cria esta IA como clone de um cerebro ja existente
    void initClone(const Brain& b, float sMul, uint32_t seed) {
        setupBuffers(); brain = b; sigmaMul = sMul; rngSeed = seed; resetEpisode();
    }
    // troca o cerebro por uma copia de outro (recomeca o episodio e a janela de score)
    void adopt(const Brain& b) { brain = b; nbuf = 0; recent.clear(); resetEpisode(); }

    float bodyClearance() const { return cr.biped ? 0.988f : BALANCE_BODY_CLEARANCE; }
    const char* memoryFile() const {
        if (balanceMode)
            return cr.biped ? BIPED_BALANCE_MEM_FILE : BALANCE_MEM_FILE;
        if (speedRewardMode)
            return cr.biped ? BIPED_RUN_MEM_FILE : RUN_MEM_FILE;
        return cr.biped ? BIPED_MEM_FILE : MEM_FILE;
    }

    void init() {
        setupBuffers();
        if (std::filesystem::exists(MEM_FILE) && brain.load(MEM_FILE))
            std::printf("[memoria] caminhada carregada: %llu passos, %llu episodios, %llu orbes coletados\n",
                        (unsigned long long)brain.steps, (unsigned long long)brain.episodes, (unsigned long long)brain.orbs);
        else { brain.init(); std::printf("[memoria] cerebro novo criado (%d neuronios)\n", N_NEURONS); }
        resetEpisode();
    }

    void spawnOrb(int i, V3 around) {
        float a = rnd01() * kTwoPi, r = rndRange(2.5f, 6.5f);
        orbs[i].p = {around.x + std::cos(a) * r, 0.6f, around.z + std::sin(a) * r};
    }

    void resetBalanceSphere() {
        sphereClock = 0.f;
        spherePhase = rndRange(0.f, kTwoPi);
        // Start at the exact height for this phase. Previously the first update
        // changed Y instantly, producing a huge fake velocity and spinning respawns.
        const float initialY = 2.05f + 0.16f * std::sin(spherePhase)
                                      + 0.05f * std::sin(spherePhase * 0.61f);
        spherePos = {arenaOrigin.x, initialY, arenaOrigin.z};
        sphereVel = {};
        sphereAngularVelocity = {};
        sphereTurnTimer = rndRange(3.0f, 5.0f);
        float a = rndRange(0.f, kTwoPi);
        sphereDir = sphereTargetDir = {std::cos(a), 0.f, std::sin(a)};
        const float practice = clampf(bestStableSeconds / 8.f, 0.f, 1.f);
        // Difficulty follows measured balance performance, not elapsed compute
        // steps. A struggling policy gets time to discover support before the
        // sphere becomes harder; no joint action is prescribed by this curriculum.
        sphereMoveSpeed = rndRange(0.30f + 0.30f * practice,
                                   0.45f + 0.55f * practice);
        offSphereSeconds = 0.f;
    }

    void updateBalanceSphere(float dt) {
        V3 previous = spherePos;
        sphereClock += dt;
        sphereTurnTimer -= dt;
        if (sphereTurnTimer <= 0.f) {
            float a = rndRange(0.f, kTwoPi);
            sphereTargetDir = {std::cos(a), 0.f, std::sin(a)};
            sphereTurnTimer = rndRange(3.0f, 5.0f);
            const float practice = clampf(bestStableSeconds / 8.f, 0.f, 1.f);
            sphereMoveSpeed = rndRange(0.30f + 0.30f * practice,
                                       0.45f + 0.55f * practice);
        }
        // Smooth steering, but keep this agent's sphere inside its own lane.
        // Without a boundary, independently moving spheres eventually cross and
        // overlap other agents' worlds because no inter-sphere collision solver exists.
        float steer = 1.f - std::exp(-0.9f * dt);
        sphereDir = norm(sphereDir * (1.f - steer) + sphereTargetDir * steer);
        V3 laneOffset = spherePos - arenaOrigin;
        laneOffset.y = 0.f;
        constexpr float ARENA_RADIUS = 0.60f;
        const float laneDistance = len(laneOffset);
        if (laneDistance > ARENA_RADIUS - 0.10f && laneDistance > 1e-6f) {
            const V3 outward = laneOffset * (1.f / laneDistance);
            const float outwardMotion = dot(sphereDir, outward);
            if (outwardMotion > 0.f) {
                sphereDir = norm(sphereDir - outward * (2.f * outwardMotion));
                sphereTargetDir = sphereDir;
            }
        }
        const float u = clampf(sphereClock / 0.70f, 0.f, 1.f);
        const float ramp = u * u * (3.f - 2.f * u); // smoothstep: zero initial velocity
        spherePos.x += sphereDir.x * sphereMoveSpeed * ramp * dt;
        spherePos.z += sphereDir.z * sphereMoveSpeed * ramp * dt;
        // Numerical safety: keep the centre inside the lane even on a large step.
        laneOffset = spherePos - arenaOrigin;
        laneOffset.y = 0.f;
        const float movedDistance = len(laneOffset);
        if (movedDistance > ARENA_RADIUS) {
            const V3 inward = laneOffset * (-1.f / movedDistance);
            spherePos.x = arenaOrigin.x - inward.x * ARENA_RADIUS;
            spherePos.z = arenaOrigin.z - inward.z * ARENA_RADIUS;
            sphereDir = norm(sphereDir - inward * (2.f * dot(sphereDir, inward)));
            sphereTargetDir = sphereDir;
        }
        const float startY = 2.05f + 0.16f * std::sin(spherePhase)
                                     + 0.05f * std::sin(spherePhase * 0.61f);
        const float movingY = 2.05f + 0.16f * std::sin(sphereClock * 0.80f + spherePhase)
                                      + 0.05f * std::sin(sphereClock * 1.45f + spherePhase * 0.61f);
        spherePos.y = startY + (movingY - startY) * ramp;
        sphereVel = (spherePos - previous) * (1.f / dt);
        // Roll the sphere as it translates: omega = up x velocity / radius.
        // Existing foot anchors rotate with the surface, so standing still is not
        // equivalent to being carried on a non-rotating platform.
        sphereAngularVelocity = cross(V3{0.f, 1.f, 0.f}, V3{sphereVel.x, 0.f, sphereVel.z})
                                * (1.f / sphereRadius);
        const float omega = len(sphereAngularVelocity);
        if (omega > 1e-8f) {
            const V3 axis = sphereAngularVelocity * (1.f / omega);
            for (int l = 0; l < cr.limbCount(); ++l)
                for (int slot = 0; slot < 9; ++slot)
                    if (cr.sphereAnchorActive[l][slot])
                        cr.sphereAnchor[l][slot] = rotAxis(cr.sphereAnchor[l][slot], axis, omega * dt);
            if (cr.torsoSphereContactPrev)
                cr.torsoSphereAnchor = rotAxis(cr.torsoSphereAnchor, axis, omega * dt);
        }
    }

    void resetEpisode() {
        if (balanceMode) resetBalanceSphere();
        // Balance episodes always restart from an upright, reproducible pose.
        // Random yaw is harmless, but a fixed yaw makes debugging and comparison easier.
        cr.reset(balanceMode ? 0.f : rnd01() * kTwoPi);
        if (balanceMode) {
            // Place the selected morphology so its neutral toes start near the sphere.
            cr.pos = {spherePos.x, spherePos.y + sphereRadius + bodyClearance(), spherePos.z};
        }
        for (int i = 0; i < N_ORBS; i++) spawnOrb(i, cr.pos);
        epSteps = 0; epOrbs = 0; epReturn = 0; orbTimer = 0; sub = 0; trail.clear();
        stableSeconds = 0.f;
        balanceQuality = 0.f;
        offSphereSeconds = 0.f;
        sampleValid = false;
    }
    int nearestOrb() const {
        int b = 0; float bd = 1e30f;
        for (int i = 0; i < N_ORBS; i++) {
            float dx = orbs[i].p.x - cr.pos.x, dz = orbs[i].p.z - cr.pos.z, d = dx * dx + dz * dz;
            if (d < bd) { bd = d; b = i; }
        }
        return b;
    }

    void buildObs(float* x) const {
        const Creature& c = cr;
        V3 sensedVelocity = balanceMode ? (c.vel - sphereVel) : c.vel;
        V3 vb = c.toBody(sensedVelocity);
        x[0] = vb.x / 1.5f; x[1] = vb.y / 1.5f; x[2] = vb.z / 1.5f;
        x[3] = c.wb.x / 3.f; x[4] = c.wb.y / 3.f; x[5] = c.wb.z / 3.f;
        x[6] = c.ax.y; x[7] = c.ay.y; x[8] = c.az.y;                 // "cima" visto do corpo
        if (balanceMode) {
            const V3 fromSphere = c.pos - spherePos;
            const float radialDistance = len(fromSphere);
            x[9] = clampf((radialDistance - (sphereRadius + bodyClearance())) / 0.75f, -1.5f, 1.5f);
            const V3 toCenter = radialDistance > 1e-6f ? (spherePos - c.pos) * (1.f / radialDistance) : V3{};
            const V3 centerInBody = c.toBody(toCenter);
            x[10] = centerInBody.x;
            x[11] = centerInBody.y;
            x[12] = centerInBody.z;
        } else {
            x[9] = (c.pos.y - 0.66f) * 3.f;
            V3 d = orbs[nearestOrb()].p - c.pos; d.y = 0;
            float dist = len(d); V3 db = c.toBody(dist > 1e-6f ? d * (1.f / dist) : V3{});
            x[10] = db.x; x[11] = db.z; x[12] = clampf(dist / 6.f, 0.f, 1.5f);
        }
        for (int i = 0; i < N_ACT; i++) x[13 + i] = c.q[i];
        for (int l = 0; l < 4; l++) {
            // In balance mode, provide graded contact pressure instead of a binary
            // flag. It lets the policy distinguish barely-touching from firm support.
            x[25 + l] = balanceMode ? clampf(c.fn[l] / 40.f, 0.f, 1.5f)
                                    : (c.contact[l] ? 1.f : 0.f);
        }
        if (balanceMode) {
            V3 platformMotion = c.toBody(sphereVel);
            x[29] = clampf(platformMotion.x / 1.5f, -1.f, 1.f);
            x[30] = clampf(platformMotion.z / 1.5f, -1.f, 1.f);
            x[31] = clampf(platformMotion.y / 1.0f, -1.f, 1.f);
            x[32] = clampf((len(c.pos - spherePos) - sphereRadius - bodyClearance()) / 0.75f, -1.f, 1.f);
        } else {
            x[29] = std::sin(kTwoPi * 1.5f * simT); x[30] = std::cos(kTwoPi * 1.5f * simT);
            x[31] = std::sin(kTwoPi * 3.0f * simT); x[32] = std::cos(kTwoPi * 3.0f * simT);
        }
    }

    float currentSigma() const {
        const float decay = std::exp(-(float)brain.steps / 200000.f);
        // For balance, keep policy noise smaller than the old OU-noise amplitude.
        // Noise is sampled independently in decide(), as required by the score gradient;
        // the joint servos provide the physical smoothing. Walking keeps its old schedule.
        return sigmaMul * (balanceMode ? (0.28f + 0.08f * decay)
                                       : (0.10f + 0.25f * decay));
    }

    void decide() {
        buildObs(xobs);
        brain.forward(xobs, act);
        sigma = currentSigma();
        // The policy itself chooses every joint target; no balancing reflex is added.
        // IMPORTANT: the Actor-Critic estimator below assumes independent standard
        // Gaussian action noise. The old strongly-correlated OU process was recorded
        // as if it were IID noise, biasing the policy gradient and often stalling learning.
        // Servo filtering in Creature::physics keeps the physical motion smooth.
        for (int k = 0; k < N_ACT; k++) {
            if (k >= cr.activeActionCount()) {
                lastEps[k] = 0.f;
                lastAction[k] = 0.f;
                cr.cmd[k] = 0.f;
                continue;
            }
            const float e = learning ? randn() : 0.f;
            lastEps[k] = e; // exact standardized Gaussian sample used by the policy
            lastAction[k] = std::tanh(act.mu[k] + (learning ? sigma * e : 0.f));
            cr.cmd[k] = lastAction[k];
        }
        for (int g = 0; g < N_GROUPS; g++) heat[g] *= 0.92f;
        for (int i = 0; i < act.nGroups; i++) { heat[act.groups[i]] = 1.f; brain.useCount[act.groups[i]]++; }
        lastGroups = act.nGroups; lastActive = act.nActive;
        avgActiveFrac += ((float)act.nActive / N_NEURONS - avgActiveFrac) * 0.01f;
        sampleValid = false;
        if (learning) {
            Sample& s = buf[nbuf];
            std::memcpy(s.x, xobs, sizeof xobs);
            s.idx.assign(act.idx, act.idx + act.nActive);
            s.hn.assign(act.hn, act.hn + act.nActive);
            s.scale = act.scale; s.value = act.value;
            std::memcpy(s.eps, lastEps, sizeof lastEps);
            sampleValid = true;
        }
    }

    void finish() {
        float energy = 0;
        for (int k = 0; k < N_ACT; k++) energy += lastAction[k] * lastAction[k];

        V3 relativeVelocity = balanceMode ? (cr.vel - sphereVel) : cr.vel;
        const float horizontalSpeed = std::sqrt(relativeVelocity.x * relativeVelocity.x + relativeVelocity.z * relativeVelocity.z);
        const float angularSpeed = len(cr.wb);
        // Smooth posture signal in balance mode: a tilted or even inverted body
        // still receives a graded signal that can favour recovery. The old hard
        // cutoff gave zero posture reward as soon as the body tipped too far.
        const float upright = balanceMode
            ? clampf((cr.ay.y + 0.50f) / 1.50f, 0.f, 1.f)
            : clampf((cr.ay.y - 0.35f) / 0.65f, 0.f, 1.f);
        const float angularControl = std::exp(-0.8f * angularSpeed);
        const float driftControl = std::exp(-0.55f * horizontalSpeed);
        const float bodyRadialDistance = balanceMode ? len(cr.pos - spherePos) : 0.f;
        const float bodyGap = balanceMode ? (bodyRadialDistance - sphereRadius) : 0.f;
        // This describes proximity to the sphere surface from any accessible
        // direction; it is not a target position at the crown/top.
        const float bodySupportQuality = balanceMode
            ? std::exp(-1.6f * std::fabs(bodyGap - bodyClearance()))
            : clampf((cr.pos.y - 0.25f) / 0.40f, 0.f, 1.f);
        int footContacts = 0;
        for (int l = 0; l < cr.limbCount(); ++l) if (cr.contact[l]) ++footContacts;
        const float contactRatio = (float)footContacts / (float)cr.limbCount();

        // Reward physical support anywhere on the sphere. There is deliberately
        // no reward for being near the crown and no target foot placement.
        float supportForceQuality = 0.f;
        if (balanceMode) {
            const float expectedLegLoad = MASS * 9.81f / (float)cr.limbCount();
            for (int l = 0; l < cr.limbCount(); ++l)
                supportForceQuality += clampf(cr.fn[l] / expectedLegLoad, 0.f, 1.f);
            supportForceQuality /= (float)cr.limbCount();
        }
        balanceQuality = upright * angularControl * driftControl * bodySupportQuality
                         * (balanceMode ? (0.25f + 0.75f * contactRatio) : 1.f);
        const bool stable = cr.ay.y > 0.90f && angularSpeed < 0.90f &&
                            horizontalSpeed < 0.55f && bodySupportQuality > 0.55f &&
                            (!balanceMode || footContacts >= 2);

        float reward = 0.f;
        if (balanceMode) {
            stableSeconds = stable ? stableSeconds + CTRL_DT : 0.f;
            bestStableSeconds = std::max(bestStableSeconds, stableSeconds);
            const float streakBonus = stable ? std::min(stableSeconds, 10.f) : 0.f;
            // Reward actual support, upright posture and low relative drift.
            // There is no preferred spot on the sphere and no corrective pose.
            const float rewardRate = 0.65f * upright
                                   + 0.45f * angularControl
                                   + 0.40f * driftControl
                                   + 1.20f * bodySupportQuality
                                   + 1.80f * supportForceQuality
                                   + 2.40f * contactRatio
                                   + 0.80f * balanceQuality
                                   + (stable ? 1.0f + 0.12f * streakBonus : 0.f);
            // Meaningful effort cost discourages needless joint flailing without
            // telling the network what any joint should do.
            reward = CTRL_DT * (rewardRate - 0.015f * energy);
            balancePoints += CTRL_DT * (3.f * balanceQuality
                                      + 4.f * supportForceQuality
                                      + 6.f * contactRatio
                                      + 0.5f * streakBonus);
            const bool lostPlatform = footContacts == 0
                || bodyRadialDistance > sphereRadius + 2.0f
                || cr.pos.y < spherePos.y - 0.35f;
            offSphereSeconds = lostPlatform ? offSphereSeconds + CTRL_DT
                                            : std::max(0.f, offSphereSeconds - 2.f * CTRL_DT);
        } else {
            stableSeconds = 0.f;
            int ni = nearestOrb();
            V3 d = orbs[ni].p - cr.pos; d.y = 0; V3 dir = norm(d);
            float vt = cr.vel.x * dir.x + cr.vel.z * dir.z;
            if (speedRewardMode) {
                // Corrida: recompensa a velocidade horizontal real, sem exigir que
                // a criatura esteja indo diretamente para a orbe. O teto evita que
                // velocidades extremas dominem todo o aprendizado; postura ereta
                // multiplica o ganho para evitar que simplesmente se jogue no chao.
                const float rewardedSpeed = clampf(horizontalSpeed, 0.f, 4.0f);
                const float speedRewardRate = 3.0f * rewardedSpeed * (0.15f + 0.85f * upright);
                reward = CTRL_DT * (speedRewardRate + 0.45f * upright
                                    - 0.20f * angularSpeed) - 0.002f * energy;
            } else {
                // Recompensa original de caminhada em direcao a orbe, preservada.
                reward = CTRL_DT * (2.5f * clampf(vt, -1.f, 2.f) + 0.4f * cr.ay.y
                                    - 0.15f * angularSpeed) - 0.002f * energy;
            }
            for (int i = 0; i < N_ORBS; i++) {                    // coleta recompensas
                float dx = orbs[i].p.x - cr.pos.x, dz = orbs[i].p.z - cr.pos.z;
                if (dx * dx + dz * dz < ORB_PICK_R * ORB_PICK_R) {
                    reward += 4.f; epOrbs++; brain.orbs++; spawnOrb(i, cr.pos);
                }
            }
            if (++orbTimer >= 240) {
                orbTimer = 0;
                spawnOrb((int)(rnd01() * N_ORBS) % N_ORBS, cr.pos);
            }
        }
        epSteps++;
        trail.push_back(cr.pos); if (trail.size() > 300) trail.pop_front();

        bool bad = !std::isfinite(cr.pos.x + cr.pos.y + cr.pos.z + cr.vel.x + cr.vel.y + cr.vel.z);
        // Do not reset just because the torso tips: recovering from a bad pose is
        // part of the task. In balance mode, terminate only after losing support
        // for long enough or physically moving well away/falling below the sphere.
        const bool outsideSphere = balanceMode &&
            (bodyRadialDistance > sphereRadius + 2.0f || cr.pos.y < spherePos.y - 0.35f);
        bool fell = bad || (!balanceMode && (cr.pos.y < 0.30f || cr.ay.y < 0.5f))
                    || (balanceMode && (offSphereSeconds > 0.55f || outsideSphere || cr.floorContact));
        bool timeout = epSteps >= MAX_EP_STEPS;
        if (fell) reward -= 2.f;
        epReturn += reward;

        if (sampleValid) { buf[nbuf].reward = reward; buf[nbuf].done = fell || timeout; nbuf++; brain.steps++; dirty = true; }
        if (fell || timeout) {
            lastReturn = epReturn; lastEpOrbs = epOrbs;
            brain.emaReturn = brain.episodes ? brain.emaReturn * 0.98f + epReturn * 0.02f : epReturn;
            avgOrbsEp += ((float)epOrbs - avgOrbsEp) * 0.05f;
            avgLen += ((float)epSteps - avgLen) * 0.05f;
            retHist.push_back(epReturn); if (retHist.size() > 160) retHist.pop_front();
            recent.push_back(epReturn); if (recent.size() > RECENT_MAX) recent.pop_front();
            accRet += epReturn; accOrbs += epOrbs; accLen += epSteps; accN++;
            brain.episodes++;
            resetEpisode();
        }
        if (learning && nbuf >= ROLLOUT_T) {                      // hora de aprender
            float lastV = 0;
            { Act tmp; float xo[N_IN]; buildObs(xo); brain.forward(xo, tmp); lastV = tmp.value; }
            // The sphere is a high-sensitivity task; take smaller gradient steps
            // so the policy can improve gradually instead of instantly undoing its stance.
            brain.train(buf, nbuf, lastV, sigma, balanceMode ? 0.75f : 1.f);
            learningUpdates++;
            nbuf = 0;
        }
    }

    void tick() {
        if (sub == 0) decide();
        if (balanceMode) updateBalanceSphere(PHYS_DT);
        cr.physics(PHYS_DT, balanceMode, spherePos, sphereVel, sphereRadius, sphereAngularVelocity);
        simT += PHYS_DT;
        if (++sub == SUBSTEPS) { sub = 0; finish(); }
    }
    void setLearning(bool on) { learning = on; nbuf = 0; sampleValid = false; }
    void setBalanceMode(bool on) {
        if (balanceMode == on) return;
        balanceMode = on;
        nbuf = 0;
        sampleValid = false;
        learningUpdates = 0;
        stableSeconds = 0.f;
        const char* path = memoryFile();
        if (std::filesystem::exists(path) && brain.load(path)) {
            std::printf("[memoria] tarefa %s carregada: %s\n", on ? "equilibrio" : "caminhada", path);
        } else {
            brain.init();
            if (on) {
                // Random output weights are NOT a balancing controller: they give a
                // fresh policy a measurable range of actions to explore. The agent
                // still must learn which actions earn reward.
                for (float& w : brain.Wout) w = randn() * 0.06f;
            }
            if (brain.save(path)) std::printf("[memoria] nova memoria da tarefa criada: %s\n", path);
        }
        dirty = false;
        resetEpisode();
    }

    void setSpeedRewardMode(bool on, const Brain& seedBrain) {
        if (speedRewardMode == on || balanceMode) return;
        speedRewardMode = on;
        nbuf = 0;
        sampleValid = false;
        learningUpdates = 0;
        recent.clear();
        retHist.clear();
        accRet = accOrbs = accLen = 0.0;
        accN = 0;
        avgOrbsEp = avgLen = lastReturn = 0.f;
        lastEpOrbs = epOrbs = 0;

        const char* path = memoryFile();
        if (std::filesystem::exists(path) && brain.load(path)) {
            std::printf("[memoria] recompensa %s carregada: %s\n",
                        on ? "corrida" : "caminhada", path);
        } else {
            // A nova tarefa parte da politica de caminhada existente, em vez de
            // comecar do zero. Assim, o novo objetivo pode ensinar a correr a uma IA
            // que ja aprendeu a controlar o corpo.
            brain = seedBrain;
            if (brain.save(path))
                std::printf("[memoria] politica inicial para recompensa %s: %s\n",
                            on ? "corrida" : "caminhada", path);
        }
        dirty = false;
        resetEpisode();
    }

    void setBipedMode(bool on) {
        if (cr.biped == on) return;
        cr.biped = on;
        nbuf = 0;
        sampleValid = false;
        learningUpdates = 0;
        recent.clear();
        retHist.clear();
        stableSeconds = bestStableSeconds = balanceQuality = 0.f;
        balancePoints = 0.0;
        const char* path = memoryFile();
        if (std::filesystem::exists(path) && brain.load(path)) {
            std::printf("[memoria] morfologia %s carregada: %s\n", on ? "bipede digitigrado" : "quadrupede", path);
        } else {
            brain.init();
            // Fresh morphology gets an untrained, exploratory policy. No joint
            // poses or balance reflexes are inserted here.
            for (float& w : brain.Wout) w = randn() * 0.06f;
            if (brain.save(path)) std::printf("[memoria] nova memoria morfologica: %s\n", path);
        }
        dirty = false;
        resetEpisode();
    }
    bool save() { bool ok = brain.save(memoryFile()); if (ok) dirty = false; return ok; }
    int groupsUsed() const { int n = 0; for (auto c : brain.useCount) if (c) n++; return n; }
};

// ================================== ENXAME ===================================
static constexpr float NO_SCORE = -1e30f;

struct Swarm {
    std::vector<std::unique_ptr<Sim>> ag;     // ag[0] e a IA original; so tem 1 elemento com o aprendizado desligado
    int shown = 0, generation = 0;
    Brain best;                               // "melhor forma" registrada (ou o ponto de partida, se ainda nao verificada)
    float bestScore = NO_SCORE;               // NO_SCORE = ainda nao verificada por nenhuma geracao
    uint64_t ticks = 0;                       // passos de fisica somados de todas as IAs

    Sim& cur() { return *ag[shown]; }
    const Sim& cur() const { return *ag[shown]; }
    bool learning() const { return ag.size() > 1; }
    bool balanceMode() const { return !ag.empty() && ag[0]->balanceMode; }
    bool speedRewardMode() const { return !ag.empty() && ag[0]->speedRewardMode; }
    bool bipedMode() const { return !ag.empty() && ag[0]->cr.biped; }
    bool dirty() const { for (auto& a : ag) if (a->dirty) return true; return false; }

    void arrangeBalanceArenas() {
        if (ag.empty()) return;
        const int cols = std::min((int)ag.size(), 5);
        const int rows = ((int)ag.size() + cols - 1) / cols;
        for (int i = 0; i < (int)ag.size(); ++i) {
            const int col = i % cols, row = i / cols;
            ag[i]->arenaOrigin = {(col - (cols - 1) * 0.5f) * 7.0f,
                                  0.f,
                                  (row - (rows - 1) * 0.5f) * 7.0f};
        }
    }

    void setBalanceMode(bool on) {
        if (ag.empty() || balanceMode() == on) return;
        // Save the task being left to its own file before switching objectives.
        save();
        if (on) arrangeBalanceArenas();
        else for (auto& a : ag) a->arenaOrigin = {};
        for (auto& a : ag) {
            a->setBalanceMode(on);
            a->recent.clear();
            a->retHist.clear(); // nao mistura graficos/medias de tarefas diferentes
        }
        // A partir daqui, o enxame compara todas as IAs usando a nova recompensa.
        best = ag[0]->brain;
        bestScore = NO_SCORE;
        generation = 0;
    }

    void setSpeedRewardMode(bool on) {
        if (ag.empty() || balanceMode() || speedRewardMode() == on) return;
        // Salva a tarefa anterior e usa seu melhor cerebro como ponto de partida
        // caso ainda nao exista uma memoria especifica para corrida.
        save();
        Brain seed = (learning() && bestScore > NO_SCORE) ? best : ag[0]->brain;
        for (auto& a : ag) a->setSpeedRewardMode(on, seed);
        for (auto& a : ag) {
            a->recent.clear();
            a->retHist.clear();
        }
        best = ag[0]->brain;
        bestScore = NO_SCORE;
        generation = 0;
        std::printf("[recompensa] modo caminhada: %s\n", on ? "CORRIDA (mais velocidade = mais recompensa)" : "CAMINHADA (ir ate a orbe)");
    }

    void setBipedMode(bool on) {
        if (ag.empty() || bipedMode() == on) return;
        // Preserve the best policy for the old body before selecting the separate
        // memory for this new morphology. The task (walking vs sphere balance) stays unchanged.
        save();
        for (auto& a : ag) {
            a->setBipedMode(on);
            a->recent.clear();
            a->retHist.clear();
            a->accRet = a->accOrbs = a->accLen = 0.0;
            a->accN = 0;
        }
        best = ag[0]->brain;
        bestScore = NO_SCORE;
        generation = 0;
        if (balanceMode()) arrangeBalanceArenas();
    }

    void init() {
        ag.clear(); ag.emplace_back(new Sim()); ag[0]->init();
        shown = 0; generation = 0;
        if (ag[0]->learning) spawnClones();
    }

    // aprendizado ligado: a IA atual vira N_AGENTS IAs
    void spawnClones() {
        static const float MUL[5] = {1.0f, 0.7f, 1.4f, 0.85f, 1.2f};   // exploracao de cada IA
        best = ag[0]->brain; bestScore = NO_SCORE; generation = 0;       // ponto de partida guardado
        ag[0]->sigmaMul = MUL[0]; ag[0]->rngSeed = g_rng(); ag[0]->recent.clear();
        while ((int)ag.size() < N_AGENTS) {
            int i = (int)ag.size();
            ag.emplace_back(new Sim());
            ag[i]->initClone(ag[0]->brain, MUL[i % 5], g_rng());
            ag[i]->balanceMode = ag[0]->balanceMode; // clones inherit the current task
            ag[i]->speedRewardMode = ag[0]->speedRewardMode;
            ag[i]->cr.biped = ag[0]->cr.biped;       // and the selected morphology
        }
        if (ag[0]->balanceMode) arrangeBalanceArenas();
        for (auto& a : ag) a->resetEpisode();
        std::printf("[enxame] %d IAs treinando (original + %d clones)\n", N_AGENTS, N_AGENTS - 1);
    }

    // aprendizado desligado: fica so a melhor IA
    void collapse() {
        const Brain* chosen = &best; float cs = bestScore;
        const char* why = bestScore > NO_SCORE ? "melhor forma registrada" : "ponto de partida (nenhuma geracao completa ainda)";
        for (auto& a : ag) {
            float sc = a->score(FINAL_MIN_EPS);
            if (sc > cs) { cs = sc; chosen = &a->brain; why = "IA em treino com o melhor score"; }
        }
        Sim& s0 = *ag[0];
        if (chosen != &s0.brain) s0.brain = *chosen;      // (copia antes de apagar os clones)
        if (cs > NO_SCORE) std::printf("[enxame] ficou 1 IA: %s (score %.1f)\n", why, cs);
        else               std::printf("[enxame] ficou 1 IA: %s\n", why);
        ag.erase(ag.begin() + 1, ag.end());
        shown = 0;
        s0.setLearning(false); s0.recent.clear(); s0.sigmaMul = 1.f; s0.resetEpisode();
        best = Brain(); bestScore = NO_SCORE;
        s0.dirty = true; save();
    }

    void setLearning(bool on) {
        if (on == learning()) return;
        if (on) { ag[0]->setLearning(true); spawnClones(); }
        else collapse();
    }

    static void run(Sim* s, int n) {
        auto keep = g_rng;                    // preserva o gerador da thread que chamou
        g_rng.seed(s->rngSeed);
        for (int i = 0; i < n; i++) s->tick();
        s->rngSeed = g_rng();
        g_rng = keep;
    }

    // avanca n passos de fisica em todas as IAs; paralelo tambem em blocos de um frame
    void advance(int n) {
#if MULTITHREAD
        if (ag.size() > 1 && n >= 4) {
            std::vector<std::thread> th;
            for (size_t i = 1; i < ag.size(); i++) th.emplace_back(run, ag[i].get(), n);
            run(ag[0].get(), n);
            for (auto& t : th) t.join();
        } else
#endif
        for (auto& a : ag) run(a.get(), n);
        ticks += (uint64_t)n * ag.size();
        evaluate();
    }

    // fim de geracao: ranking, registra a melhor forma e as piores copiam a melhor
    void evaluate() {
        if (ag.size() < 2) return;
        for (auto& a : ag) if (a->recent.size() < GEN_EPISODES) return;     // todas precisam ter episodios suficientes
        std::vector<int> ord(ag.size()); std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](int p, int q) { return ag[p]->score(1) > ag[q]->score(1); });
        int b = ord[0]; float sb = ag[b]->score(1), sw = ag[ord.back()]->score(1);
        generation++;
        bool improved = sb > bestScore;
        if (improved) { best = ag[b]->brain; bestScore = sb; }
        std::printf("[enxame] geracao %d: melhor IA #%d score %.1f | pior %.1f | recorde %.1f%s\n",
                    generation, b, sb, sw, bestScore, improved ? " (novo)" : "");
        for (int k = (int)ord.size() - N_REPLACE; k < (int)ord.size(); k++) ag[ord[k]]->adopt(ag[b]->brain);
        for (auto& a : ag) a->recent.clear();
        for (auto& a : ag) a->dirty = true;
    }

    // grava no arquivo SEMPRE a melhor forma (nunca uma IA no meio de um tropeco do treino)
    bool save() {
        if (ag.empty()) return false;
        const char* path = cur().memoryFile();
        bool ok = (ag.size() > 1) ? best.save(path) : ag[0]->brain.save(path);
        if (ok) for (auto& a : ag) a->dirty = false;
        return ok;
    }
};

// ================================ DESENHO ====================================
struct View { float yaw = 0.9f, pitch = 0.5f, dist = 7.5f; V3 target; };

// cores de cada IA do enxame
static Color agentColor(int i, int a = 255) {
    static const int P[6][3] = {{70, 130, 220}, {230, 80, 80}, {80, 200, 110}, {200, 90, 220}, {240, 190, 50}, {60, 210, 210}};
    return rgba(P[i % 6][0], P[i % 6][1], P[i % 6][2], a);
}

// desenha UMA IA: recompensas dela, rastro, tronco e patas
static void drawAgent(const Sim& s, int id, bool focus, double tnow) {
    const Creature& c = s.cr;
    Color body = agentColor(id, focus ? 255 : 200);
    for (size_t i = 1; i < s.trail.size(); i++) {                                 // rastro
        V3 a = s.balanceMode ? s.trail[i - 1] : V3{s.trail[i - 1].x, 0.02f, s.trail[i - 1].z};
        V3 b = s.balanceMode ? s.trail[i] : V3{s.trail[i].x, 0.02f, s.trail[i].z};
        DrawLine3D(toRl(a), toRl(b), agentColor(id, focus ? 160 : 70));
    }

    int ni = s.nearestOrb();
    if (!s.balanceMode) for (int i = 0; i < N_ORBS; i++) {
        float bob = 0.65f + 0.1f * std::sin((float)tnow * 3.f + i);
        Vector3 p{s.orbs[i].p.x, bob, s.orbs[i].p.z};
        if (focus) {
            Color col = i == ni ? rgba(255, 215, 0) : rgba(255, 140, 40);
            DrawSphere(p, i == ni ? 0.32f : 0.25f, col);
            DrawLine3D(Vector3{p.x, 0, p.z}, Vector3{p.x, 2.5f, p.z}, rgba(255, 215, 0, 140));
            DrawCircle3D(Vector3{p.x, 0.03f, p.z}, ORB_PICK_R, Vector3{1, 0, 0}, 90.f, rgba(255, 215, 0, 200));
        } else {
            DrawSphere(p, i == ni ? 0.26f : 0.18f, agentColor(id, 230));            // orbe com a cor da dona
            DrawCircle3D(Vector3{p.x, 0.03f, p.z}, ORB_PICK_R, Vector3{1, 0, 0}, 90.f, agentColor(id, 120));
        }
    }
    if (!s.balanceMode) {
        DrawLine3D(Vector3{c.pos.x, 0.05f, c.pos.z}, Vector3{s.orbs[ni].p.x, 0.05f, s.orbs[ni].p.z}, focus ? rgba(255, 215, 0, 120) : agentColor(id, 110));
        DrawCircle3D(Vector3{c.pos.x, 0.015f, c.pos.z}, 0.5f, Vector3{1, 0, 0}, 90.f, rgba(0, 0, 0, 120));
    } else {
        const float dx = c.pos.x - s.spherePos.x, dz = c.pos.z - s.spherePos.z;
        const float radialSq = dx * dx + dz * dz;
        if (radialSq < s.sphereRadius * s.sphereRadius) {
            const float surfaceY = s.spherePos.y + std::sqrt(std::max(0.f, s.sphereRadius * s.sphereRadius - radialSq));
            DrawCircle3D(Vector3{c.pos.x, surfaceY + 0.012f, c.pos.z},
                         0.42f, Vector3{1, 0, 0}, 90.f, rgba(0, 0, 0, 85));
        }
    }
    float m[16] = {c.ax.x, c.ax.y, c.ax.z, 0, c.ay.x, c.ay.y, c.ay.z, 0, c.az.x, c.az.y, c.az.z, 0, c.pos.x, c.pos.y, c.pos.z, 1};
    rlPushMatrix(); rlMultMatrixf(m);                                                                    // tronco
    const V3 bodyDims = c.torsoDimensions();
    DrawCube(Vector3{0, 0, 0}, bodyDims.x, bodyDims.y, bodyDims.z, body);
    DrawCubeWires(Vector3{0, 0, 0}, bodyDims.x, bodyDims.y, bodyDims.z, rgba(230, 235, 245));
    if (c.biped) {
        DrawSphere(Vector3{0.07f, 0.405f, 0.f}, 0.135f, rgba(255, 120, 60));
        DrawSphereWires(Vector3{0.07f, 0.405f, 0.f}, 0.135f, 8, 12, rgba(230, 235, 245));
        DrawCube(Vector3{0.19f, 0.385f, 0.f}, 0.16f, 0.09f, 0.13f, rgba(255, 120, 60));
    } else {
        DrawCube(Vector3{0.52f, 0.06f, 0}, 0.22f, 0.2f, 0.26f, rgba(255, 120, 60));
    }
    if (c.biped) {
        for (int l = 0; l < c.limbCount(); ++l) {
            V3 kneeLocal, ankleLocal;
            c.legPoints(l, kneeLocal, ankleLocal, s.balanceMode);
            V3 heelLocal, midfootLocal, toeLocal;
            c.bipedFootPoints(l, heelLocal, midfootLocal, toeLocal, s.balanceMode);
            const Color footColor = c.contact[l] ? rgba(90, 155, 110) : rgba(85, 100, 125);
            const Color boneColor = rgba(150, 175, 205);
            // Digitigrade profile: a raised calcaneal/hock bump, sloped
            // metatarsus and compact forward toes. There is no flat heel-to-toe shoe.
            DrawCylinderEx(toRl(heelLocal), toRl(ankleLocal), 0.023f, 0.031f, 8, boneColor);
            DrawCylinderEx(toRl(ankleLocal), toRl(midfootLocal), 0.036f, 0.029f, 8, footColor);
            DrawCylinderEx(toRl(midfootLocal), toRl(toeLocal), 0.029f, 0.020f, 8, footColor);
            DrawSphere(toRl(heelLocal), BIPED_HOCK_COLLISION_RADIUS, boneColor);
            DrawSphere(toRl(toeLocal), BIPED_TOE_COLLISION_RADIUS, footColor);
            // Three small toe pads spread across the forefoot; the central pad
            // matches the actual toe contact point used by the physics solver.
            for (int digit = 0; digit < 3; ++digit) {
                const float lateral = ((float)digit - 1.f) * BIPED_FOOT_WIDTH * 0.28f;
                V3 toePad = toeLocal + V3{digit == 1 ? 0.018f : 0.008f, -0.004f, lateral};
                DrawSphere(toRl(toePad), 0.020f, footColor);
            }
        }
    }
    rlPopMatrix();
    for (int l = 0; l < c.limbCount(); l++) {
        V3 kb, fb; c.legPoints(l, kb, fb, s.balanceMode);
        V3 h = c.pos + c.toWorld(c.hipLocal(l)), k = c.pos + c.toWorld(kb), f = c.pos + c.toWorld(fb);
        if (c.biped) {
            // Digitigrade silhouette: pronounced hock and narrow distal section.
            V3 hock = k + (f - k) * 0.72f;
            DrawCylinderEx(toRl(h), toRl(k), 0.062f, 0.048f, 8, rgba(170, 190, 220));
            DrawCylinderEx(toRl(k), toRl(hock), 0.046f, 0.032f, 8, rgba(140, 160, 200));
            DrawCylinderEx(toRl(hock), toRl(f), 0.032f, 0.025f, 8, rgba(140, 160, 200));
            DrawSphere(toRl(k), 0.055f, rgba(230, 230, 240));
            DrawSphere(toRl(hock), 0.035f, rgba(230, 230, 240));
            DrawSphere(toRl(f), 0.045f, rgba(230, 230, 240));
        } else {
            DrawCylinderEx(toRl(h), toRl(k), 0.05f, 0.04f, 8, rgba(170, 190, 220));
            DrawCylinderEx(toRl(k), toRl(f), 0.04f, 0.03f, 8, rgba(140, 160, 200));
            DrawSphere(toRl(k), 0.055f, rgba(230, 230, 240));
            DrawSphere(toRl(f), 0.065f, c.contact[l] ? rgba(80, 255, 120) : rgba(255, 90, 90));
        }
    }
}

static void drawBalanceSphere(const Sim& s, int id, bool focus) {
    Color shell = agentColor(id, focus ? 235 : 145);
    DrawSphere(toRl(s.spherePos), s.sphereRadius, shell);
    DrawSphereWires(toRl(s.spherePos), s.sphereRadius, 18, 28,
                    rgba(220, 235, 255, focus ? 190 : 90));
    V3 velocityArrow = s.sphereVel * 0.75f;
    DrawLine3D(toRl(s.spherePos), toRl(s.spherePos + velocityArrow),
               focus ? rgba(255, 220, 100, 230) : rgba(230, 230, 240, 130));
}

static void drawWorld(const Swarm& w, const View& v, double tnow) {
    Camera3D cam{};
    cam.target = toRl(v.target);
    cam.position = Vector3{v.target.x + std::cos(v.pitch) * std::sin(v.yaw) * v.dist,
                           v.target.y + std::sin(v.pitch) * v.dist,
                           v.target.z + std::cos(v.pitch) * std::cos(v.yaw) * v.dist};
    cam.up = Vector3{0, 1, 0}; cam.fovy = 55.f; cam.projection = CAMERA_PERSPECTIVE;
    BeginMode3D(cam);
    float gx = std::floor(v.target.x), gz = std::floor(v.target.z);
    DrawPlane(Vector3{gx, -0.01f, gz}, Vector2{120, 120}, rgba(28, 34, 44));
    rlPushMatrix(); rlTranslatef(gx, 0, gz); DrawGrid(60, 1.f); rlPopMatrix();
    for (int i = 0; i < (int)w.ag.size(); i++)
        if (w.ag[i]->balanceMode) drawBalanceSphere(*w.ag[i], i, i == w.shown);
    for (int i = 0; i < (int)w.ag.size(); i++) if (i != w.shown) drawAgent(*w.ag[i], i, false, tnow);
    drawAgent(*w.ag[w.shown], w.shown, true, tnow);                               // a exibida por cima
    EndMode3D();
}

static void drawHud(const Swarm& w, double speed, bool paused, double lastSaveAgo, int fps) {
    const Sim& s = w.cur();
    int W = GetScreenWidth(), H = GetScreenHeight();
    DrawRectangle(8, 8, 440, 314, rgba(0, 0, 0, 175));
    hud(16, 14, 18, rgba(255, 255, 255), "CEREBRO ESPARSO 3D - MORFOLOGIA E APRENDIZADO");
    hud(16, 38, 14, rgba(190, 220, 255), "Neuronios totais : %d", N_NEURONS);
    hud(16, 56, 14, rgba(120, 255, 160), "Ativos agora     : %d (%.2f%%) em %d grupos", s.lastActive, 100.f * s.lastActive / N_NEURONS, s.lastGroups);
    hud(16, 74, 14, rgba(190, 220, 255), "Uso medio        : %.2f%% do cerebro", 100.f * s.avgActiveFrac);
    hud(16, 92, 14, rgba(190, 220, 255), "Grupos ja usados : %d / %d", s.groupsUsed(), N_GROUPS);
    hud(16, 114, 14, rgba(255, 255, 255), "Passos: %llu   Episodios: %llu", (unsigned long long)s.brain.steps, (unsigned long long)s.brain.episodes);
    hud(16, 132, 14, rgba(255, 215, 0), "Orbes coletados: %llu  (media/ep: %.2f)", (unsigned long long)s.brain.orbs, s.avgOrbsEp);
    hud(16, 150, 14, rgba(255, 255, 255), "Retorno ult: %.1f   media: %.1f", s.lastReturn, s.brain.emaReturn);
    hud(16, 168, 14, rgba(255, 255, 255), "Episodio atual: %.1fs  orbes: %d", s.epSteps * CTRL_DT, s.epOrbs);
    hud(16, 186, 14, rgba(255, 255, 255), "Exploracao (sigma): %.3f", s.sigma);
    hud(16, 204, 14, s.learning ? rgba(120, 255, 160) : rgba(255, 160, 80), "Aprendizado: %s", s.learning ? "LIGADO" : "DESLIGADO (so exibe)");
    hud(16, 222, 14, s.balanceMode ? rgba(120, 255, 160) : rgba(190, 220, 255),
        "Modo: %s | E alterna", s.balanceMode ? "EQUILIBRIO" : "CAMINHADA");
    if (s.balanceMode) {
        hud(16, 240, 12, rgba(255, 215, 0),
            "Pontos %.1f | atual %.1fs | recorde %.1fs",
            s.balancePoints, s.stableSeconds, s.bestStableSeconds);
    } else {
        hud(16, 240, 12, s.speedRewardMode ? rgba(255, 215, 0) : rgba(190, 220, 255),
            "Recompensa: %s | C alterna",
            s.speedRewardMode ? "CORRIDA (velocidade)" : "CAMINHADA (orbes)");
        hud(16, 278, 12, rgba(120, 255, 160), "Velocidade horizontal: %.2f m/s", 
            std::sqrt(s.cr.vel.x * s.cr.vel.x + s.cr.vel.z * s.cr.vel.z));
        hud(16, 294, 12, rgba(190, 220, 255), "Recompensa de corrida considera ate 4.0 m/s e postura ereta");
    }
    hud(16, 258, 12, rgba(190, 220, 255), "Velocidade x%.0f %s | FPS %d | Memoria salva ha %.0fs",
        speed, paused ? "[PAUSADO]" : "", fps, lastSaveAgo);
    if (s.balanceMode) {
        const int activeLimbs = s.cr.limbCount();
        const int contacts = (int)s.cr.contact[0] + (int)s.cr.contact[1]
                           + (s.cr.biped ? 0 : (int)s.cr.contact[2] + (int)s.cr.contact[3]);
        hud(16, 278, 12, rgba(255, 220, 120), "Esfera R=%.1fm | mov %.2fm/s | apoio %d/%d",
            s.sphereRadius, len(s.sphereVel), contacts, activeLimbs);
        float actionLevel = 0.f;
        for (int i = 0; i < s.cr.activeActionCount(); ++i) actionLevel += std::fabs(s.lastAction[i]);
        actionLevel /= (float)s.cr.activeActionCount();
        hud(16, 294, 12, rgba(190, 220, 255), "Acao media %.3f | sigma %.3f | atualizacoes RL %llu",
            actionLevel, s.sigma, (unsigned long long)s.learningUpdates);
    }
    hud(16, 310, 12, s.cr.biped ? rgba(255, 220, 120) : rgba(190, 220, 255),
        "Corpo: %s | Q alterna morfologia", s.cr.biped ? "BIPEDO DIGITIGRADO" : "QUADRUPEDO");

    // mapa do cerebro: cada quadradinho = 1 grupo de 32 neuronios
    const int cols = 32, cell = 8, rows = N_GROUPS / cols;
    int ox = W - cols * cell - 16, oy = 34;
    DrawRectangle(ox - 8, 8, cols * cell + 16, rows * cell + 120, rgba(0, 0, 0, 150));
    hud(ox, 12, 14, rgba(255, 255, 255), "Mapa do cerebro (grupos acesos)");
    for (int g = 0; g < N_GROUPS; g++) {
        int cx = ox + (g % cols) * cell, cy = oy + (g / cols) * cell;
        float h = s.heat[g];
        Color c = s.brain.useCount[g] ? rgba(30, 70, 110) : rgba(35, 35, 40);
        if (h > 0.02f) c = rgba((int)(60 + 195 * h), (int)(90 + 130 * h), (int)(110 * (1 - h) + 20), 255);
        DrawRectangle(cx, cy, cell - 1, cell - 1, c);
    }
    // grafico de retorno por episodio
    int gy = oy + rows * cell + 8, gw = cols * cell, gh = 60;
    hud(ox, gy - 2, 12, rgba(190, 220, 255), "ganhos (retorno por episodio) - 1 linha por IA");
    DrawRectangleLines(ox, gy + 12, gw, gh, rgba(120, 120, 130));
    {   // uma linha por IA (a exibida em branco/mais grossa), mesma escala para comparar
        float mn = 1e30f, mx = -1e30f; bool any = false;
        for (auto& a : w.ag) for (float v : a->retHist) { mn = std::min(mn, v); mx = std::max(mx, v); any = true; }
        if (any) {
            if (mx - mn < 1e-3f) mx = mn + 1;
            for (int k = 0; k < (int)w.ag.size(); k++) {
                const auto& hh = w.ag[k]->retHist;
                Color lc = k == w.shown ? rgba(255, 255, 255) : agentColor(k);
                for (size_t i = 1; i < hh.size(); i++) {
                    float x0 = ox + (float)(i - 1) / 159.f * gw, x1 = ox + (float)i / 159.f * gw;
                    float y0 = gy + 12 + gh - (hh[i - 1] - mn) / (mx - mn) * (gh - 4) - 2;
                    float y1 = gy + 12 + gh - (hh[i] - mn) / (mx - mn) * (gh - 4) - 2;
                    DrawLine((int)x0, (int)y0, (int)x1, (int)y1, lc);
                }
            }
        }
    }
    hud(10, H - 22, 13, rgba(220, 220, 220), "CIMA/BAIXO velocidade | E equilibrio/caminhada | C recompensa corrida | Q biped digitigrado | L aprendizado | TAB IA | S salvar | ESPACO pausa | mouse/roda camera");
}

static void drawSwarmHud(const Swarm& w) {
    const int n = (int)w.ag.size();
    const int panelH = (n == 1) ? 30 : 46 + 17 * n;
    // Keep the swarm panel below the status panel instead of drawing on top of it.
    const int y0 = std::max(330, std::min(342, GetScreenHeight() - panelH - 30));
    if (n == 1) {
        DrawRectangle(8, y0, 440, 30, rgba(0, 0, 0, 175));
        hud(16, y0 + 8, 13, rgba(255, 160, 80), "Enxame: 1 IA | aprendizado desligado");
        return;
    }
    DrawRectangle(8, y0, 440, panelH, rgba(0, 0, 0, 175));
    hud(16, y0 + 5, 13, rgba(255, 255, 255), "ENXAME: %d IAs | geracao %d", n, w.generation);
    if (w.bestScore > NO_SCORE) hud(16, y0 + 22, 12, rgba(255, 215, 0), "Recorde global: %.1f", w.bestScore);
    else                        hud(16, y0 + 22, 12, rgba(190, 220, 255), "Recorde: aguardando geracao completa");
    int lead = -1; float ls = NO_SCORE;
    for (int i = 0; i < n; i++) { const float sc = w.ag[i]->score(1); if (sc > ls) { ls = sc; lead = i; } }
    for (int i = 0; i < n; i++) {
        const Sim& a = *w.ag[i]; const float sc = a.score(1);
        const Color c = i == w.shown ? rgba(255, 215, 0) : (i == lead ? rgba(120, 255, 160) : rgba(190, 200, 215));
        const int rowY = y0 + 43 + 17 * i;
        if (sc > NO_SCORE) hud(16, rowY, 12, c, "%s#%d  score %7.1f  %2zu/%zu ep  noise x%.2f", i == w.shown ? ">" : " ", i, sc, a.recent.size(), GEN_EPISODES, a.sigmaMul);
        else               hud(16, rowY, 12, c, "%s#%d  score    --  %2zu/%zu ep  noise x%.2f", i == w.shown ? ">" : " ", i, a.recent.size(), GEN_EPISODES, a.sigmaMul);
        DrawRectangle(416, rowY + 2, 9, 9, agentColor(i));
    }
}

// ================================== MAIN =====================================
static double nowSec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static void runHeadless(Swarm& sw, double seconds) {
    std::printf("[headless] treinando por %.0f s...\n", seconds);
    double t0 = nowSec(), lastSave = t0, lastPrint = t0; uint64_t startTicks = sw.ticks;
    while (nowSec() - t0 < seconds) {
        sw.advance(2400);
        double t = nowSec();
        if (t - lastSave > 5.0) { sw.save(); lastSave = t; }
        if (t - lastPrint > 10.0) {
            double ret = 0, orbs = 0, len = 0; int n = 0; float act = 0;
            for (auto& a : sw.ag) { ret += a->accRet; orbs += a->accOrbs; len += a->accLen; n += a->accN; act += a->avgActiveFrac / sw.ag.size();
                                    a->accRet = a->accOrbs = a->accLen = 0; a->accN = 0; }
            if (n > 0)
                std::printf("t=%4.0fs ias=%zu gen=%d (%.0f p/s no total) | retorno=%7.2f orbes/ep=%5.2f dur=%5.1fs | ativos=%.2f%% | recorde=%.1f\n",
                            t - t0, sw.ag.size(), sw.generation, (double)(sw.ticks - startTicks) / SUBSTEPS / (t - t0),
                            ret / n, orbs / n, len / n * CTRL_DT, 100.f * act, sw.bestScore > NO_SCORE ? sw.bestScore : 0.f);
            lastPrint = t;
        }
    }
    sw.save();
    std::printf("[headless] fim. memoria salva em %s\n", sw.cur().memoryFile());
}

int main(int argc, char** argv) {
    g_rng.seed(std::random_device{}());
    Swarm swarm; swarm.init();

    if (argc >= 2 && std::string(argv[1]) == "--headless-biped-balance") {
        // Train the exact biped + moving-sphere task without opening a window.
        swarm.setBalanceMode(true);
        swarm.setBipedMode(true);
        runHeadless(swarm, argc >= 3 ? std::atof(argv[2]) : 60.0);
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--headless") {
        runHeadless(swarm, argc >= 3 ? std::atof(argv[2]) : 60.0);
        return 0;
    }

    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(1280, 720, "IA - aprendizagem por recompensas");
    SetTargetFPS(60);

    View view; double speed = 1.0; bool paused = false; double acc = 0.0;
    double lastSave = nowSec();

    while (!WindowShouldClose()) {
        // --- entrada ---
        if (IsKeyPressed(KEY_UP)) speed = std::min(1024.0, speed * 2.0);
        if (IsKeyPressed(KEY_DOWN)) speed = std::max(1.0, speed / 2.0);
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_L)) swarm.setLearning(!swarm.learning());
        if (IsKeyPressed(KEY_E)) {
            const bool enableBalance = !swarm.balanceMode();
            // E inicia o treino de equilibrio mesmo se o aprendizado tiver sido desligado.
            if (enableBalance && !swarm.learning()) swarm.setLearning(true);
            swarm.setBalanceMode(enableBalance);
        }
        if (IsKeyPressed(KEY_C) && !swarm.balanceMode()) {
            // Como E e Q, C tambem liga o treinamento se ele estiver desligado.
            if (!swarm.learning()) swarm.setLearning(true);
            swarm.setSpeedRewardMode(!swarm.speedRewardMode());
        }
        if (IsKeyPressed(KEY_Q)) {
            const bool enableBiped = !swarm.bipedMode();
            // Q changes only morphology; the current task (walking/balance) stays selected.
            if (!swarm.learning()) swarm.setLearning(true);
            swarm.setBipedMode(enableBiped);
        }
        if (IsKeyPressed(KEY_TAB)) swarm.shown = (swarm.shown + 1) % (int)swarm.ag.size();
        if (IsKeyPressed(KEY_S)) { swarm.save(); lastSave = nowSec(); }
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            Vector2 d = GetMouseDelta();
            view.yaw -= d.x * 0.005f; view.pitch = clampf(view.pitch + d.y * 0.005f, 0.05f, 1.5f);
        }
        view.dist = clampf(view.dist - GetMouseWheelMove() * 0.6f, 2.5f, 25.f);

        // --- simulacao (com orcamento de tempo por quadro) ---
        if (!paused) {
            acc += std::min((double)GetFrameTime(), 0.05) * speed;
            double t0 = nowSec();
            while (acc >= PHYS_DT) {
                int n = std::min((int)(acc / PHYS_DT), 120);
                swarm.advance(n); acc -= n * (double)PHYS_DT;
                if (nowSec() - t0 > 0.012) { acc = 0; break; }
            }
        }
        // --- memoria: regrava o arquivo periodicamente ---
        if (swarm.dirty() && nowSec() - lastSave > 5.0) { swarm.save(); lastSave = nowSec(); }

        Sim& sim = swarm.cur();
        // camera segue a criatura
        view.target.x += (sim.cr.pos.x - view.target.x) * 0.08f;
        const float targetY = sim.balanceMode ? sim.spherePos.y + (sim.cr.biped ? 1.35f : 1.05f)
                                               : (sim.cr.biped ? 0.65f : 0.4f);
        view.target.y += (targetY - view.target.y) * 0.08f;
        view.target.z += (sim.cr.pos.z - view.target.z) * 0.08f;

        BeginDrawing();
        ClearBackground(rgba(14, 18, 26));
        drawWorld(swarm, view, GetTime());
        drawHud(swarm, speed, paused, nowSec() - lastSave, GetFPS());
        drawSwarmHud(swarm);
        EndDrawing();
    }
    swarm.save();          // ultima gravacao da memoria (melhor forma)
    CloseWindow();
    return 0;
}
