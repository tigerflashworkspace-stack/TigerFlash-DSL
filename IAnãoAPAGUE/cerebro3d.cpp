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
//           TAB troca a IA que esta sendo exibida | S salvar agora | ESPACO pausa
//           mouse esquerdo gira a camera | roda = zoom
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

constexpr int   N_GROUPS   = 512;                   // grupos (colunas corticais)
constexpr int   GROUP_SIZE = 32;                    // neuronios por grupo
constexpr int   N_NEURONS  = N_GROUPS * GROUP_SIZE; // total de neuronios
constexpr int   N_IN       = 33;                    // sensores
constexpr int   N_ACT      = 12;                    // atuadores (4 patas x 3 juntas)
constexpr int   K_MIN      = 3;                     // minimo de grupos ativos
constexpr int   K_MAX      = 12;                    // maximo de grupos ativos
constexpr float ROUTER_BETA = 6.0f;                 // "foco" do roteador
constexpr float ROUTER_MASS = 0.60f;                // massa de atencao que basta cobrir

constexpr float PHYS_DT   = 1.0f / 240.0f;          // passo da fisica
constexpr int   SUBSTEPS  = 12;                     // fisica por decisao do cerebro
constexpr float CTRL_DT   = PHYS_DT * SUBSTEPS;     // 20 decisoes por segundo
constexpr int   MAX_EP_STEPS = 600;                 // 30 s por episodio
constexpr int   N_ORBS    = 3;                      // recompensas simultaneas
constexpr float ORB_PICK_R = 0.75f;

constexpr int   ROLLOUT_T = 256;                    // passos por atualizacao
constexpr float GAMMA = 0.98f, LAMBDA = 0.95f;
constexpr float LR_ACTOR = 0.35f, LR_CRITIC = 0.5f, LR_HIDDEN = 0.03f;
constexpr float GCLIP = 5.0f, VCLIP = 5.0f;

// ---- enxame de IAs ----
constexpr int    N_AGENTS      = 10;     // total de IAs treinando juntas (a original + clones)
constexpr int    N_REPLACE     = 2;     // quantas das piores viram copia da melhor a cada geracao
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
    void train(std::vector<Sample>& B, int n, float lastV, float sigma) {
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

        const float la = LR_ACTOR / n, lc = LR_CRITIC / n, lh = LR_HIDDEN / n;
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
// Corpo: X = frente, Y = cima, Z = lateral.   4 patas: quadril(balanco, abducao) + joelho.
static const float MASS = 10.f, INERT_X = 0.26f, INERT_Y = 0.88f, INERT_Z = 0.72f;
static const float L1 = 0.35f, L2 = 0.35f;
static const float TORSO_LX = 0.9f, TORSO_LY = 0.2f, TORSO_LZ = 0.5f;
static const V3 HIP[4] = {{0.38f, -0.05f, 0.2f}, {0.38f, -0.05f, -0.2f}, {-0.38f, -0.05f, 0.2f}, {-0.38f, -0.05f, -0.2f}};
static const float KN = 3000.f, CN = 100.f, KT = 2000.f, CT = 40.f, MU = 1.0f;
static const float SERVO_RATE = 15.f;

struct Creature {
    V3 pos, vel, ax{1, 0, 0}, ay{0, 1, 0}, az{0, 0, 1}, wb;
    float q[N_ACT], cmd[N_ACT];
    V3 fbPrev[4]; bool fbInit = false;
    bool contact[4]; float anchX[4], anchZ[4], fn[4];

    void reset(float yaw) {
        pos = {0, 0.69f, 0}; vel = {}; wb = {};
        ax = {std::cos(yaw), 0, -std::sin(yaw)}; ay = {0, 1, 0}; az = {std::sin(yaw), 0, std::cos(yaw)};
        for (int i = 0; i < N_ACT; i++) q[i] = cmd[i] = 0;
        for (int l = 0; l < 4; l++) { contact[l] = false; fn[l] = 0; anchX[l] = anchZ[l] = 0; }
        fbInit = false;
    }
    V3 toWorld(V3 b) const { return ax * b.x + ay * b.y + az * b.z; }
    V3 toBody(V3 w) const { return {dot(w, ax), dot(w, ay), dot(w, az)}; }

    // posicao (no referencial do corpo) do joelho e do pe da perna l
    void legPoints(int l, V3& knee, V3& foot) const {
        float th = q[l * 3] * 0.6f, ph = 0.1f + q[l * 3 + 1] * 0.35f, kn = 0.7f + q[l * 3 + 2] * 0.6f;
        float o = HIP[l].z > 0 ? 1.f : -1.f;
        float x1 = L1 * std::sin(th), y1 = -L1 * std::cos(th);
        float x2 = x1 + L2 * std::sin(th - kn), y2 = y1 - L2 * std::cos(th - kn);
        float cp = std::cos(ph), sp = std::sin(ph);
        knee = HIP[l] + V3{x1, y1 * cp, -y1 * sp * o};
        foot = HIP[l] + V3{x2, y2 * cp, -y2 * sp * o};
    }

    void physics(float dt) {
        float al = 1.f - std::exp(-SERVO_RATE * dt);               // servos com atraso
        for (int i = 0; i < N_ACT; i++) q[i] += (cmd[i] - q[i]) * al;

        V3 F{0, -9.81f * MASS, 0}, T{};
        V3 wW = toWorld(wb);
        for (int l = 0; l < 4; l++) {
            V3 kneeB, fb; legPoints(l, kneeB, fb);
            V3 fvB = fbInit ? (fb - fbPrev[l]) * (1.f / dt) : V3{};
            fbPrev[l] = fb;
            V3 r = toWorld(fb), pf = pos + r;
            fn[l] = 0;
            if (pf.y < 0.f) {
                V3 vf = vel + cross(wW, r) + toWorld(fvB);
                float f_n = KN * (-pf.y) - CN * vf.y; if (f_n < 0) f_n = 0;
                if (!contact[l]) { contact[l] = true; anchX[l] = pf.x; anchZ[l] = pf.z; }
                float fx = -KT * (pf.x - anchX[l]) - CT * vf.x;
                float fz = -KT * (pf.z - anchZ[l]) - CT * vf.z;
                float fmax = MU * f_n, fm = std::sqrt(fx * fx + fz * fz);
                if (fm > fmax) {                                    // escorregou (atrito de Coulomb)
                    float s = fm > 1e-6f ? fmax / fm : 0.f; fx *= s; fz *= s;
                    anchX[l] = pf.x + fx / KT; anchZ[l] = pf.z + fz / KT;
                }
                V3 f{fx, f_n, fz};
                F = F + f; T = T + cross(r, f); fn[l] = f_n;
            } else contact[l] = false;
        }
        fbInit = true;

        vel = vel + F * (dt / MASS);
        pos = pos + vel * dt;
        V3 Tb = toBody(T);
        wb.x += Tb.x / INERT_X * dt; wb.y += Tb.y / INERT_Y * dt; wb.z += Tb.z / INERT_Z * dt;
        wb = wb * (1.f / (1.f + 0.3f * dt));
        V3 Om = toWorld(wb) * dt; float ang = len(Om);
        if (ang > 1e-9f) {
            V3 k = Om * (1.f / ang);
            ax = rotAxis(ax, k, ang); ay = rotAxis(ay, k, ang);
        }
        ax = norm(ax); az = norm(cross(ax, ay)); ay = cross(az, ax);   // reortonormaliza
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
    Act act; float xobs[N_IN]; float lastEps[N_ACT], lastAction[N_ACT];
    // estatisticas
    float lastReturn = 0, avgOrbsEp = 0, avgLen = 0, avgActiveFrac = 0, sigma = 0.3f;
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

    void init() {
        setupBuffers();
        if (std::filesystem::exists(MEM_FILE) && brain.load(MEM_FILE))
            std::printf("[memoria] carregada: %llu passos, %llu episodios, %llu orbes coletados\n",
                        (unsigned long long)brain.steps, (unsigned long long)brain.episodes, (unsigned long long)brain.orbs);
        else { brain.init(); std::printf("[memoria] cerebro novo criado (%d neuronios)\n", N_NEURONS); }
        resetEpisode();
    }

    void spawnOrb(int i, V3 around) {
        float a = rnd01() * kTwoPi, r = rndRange(2.5f, 6.5f);
        orbs[i].p = {around.x + std::cos(a) * r, 0.6f, around.z + std::sin(a) * r};
    }
    void resetEpisode() {
        cr.reset(rnd01() * kTwoPi);
        for (int i = 0; i < N_ORBS; i++) spawnOrb(i, cr.pos);
        epSteps = 0; epOrbs = 0; epReturn = 0; orbTimer = 0; sub = 0; trail.clear();
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
        V3 vb = c.toBody(c.vel);
        x[0] = vb.x / 1.5f; x[1] = vb.y / 1.5f; x[2] = vb.z / 1.5f;
        x[3] = c.wb.x / 3.f; x[4] = c.wb.y / 3.f; x[5] = c.wb.z / 3.f;
        x[6] = c.ax.y; x[7] = c.ay.y; x[8] = c.az.y;                 // "cima" visto do corpo
        x[9] = (c.pos.y - 0.66f) * 3.f;
        V3 d = orbs[nearestOrb()].p - c.pos; d.y = 0;
        float dist = len(d); V3 db = c.toBody(dist > 1e-6f ? d * (1.f / dist) : V3{});
        x[10] = db.x; x[11] = db.z; x[12] = clampf(dist / 6.f, 0.f, 1.5f);
        for (int i = 0; i < N_ACT; i++) x[13 + i] = c.q[i];
        for (int l = 0; l < 4; l++) x[25 + l] = c.contact[l] ? 1.f : 0.f;
        x[29] = std::sin(kTwoPi * 1.5f * simT); x[30] = std::cos(kTwoPi * 1.5f * simT);
        x[31] = std::sin(kTwoPi * 3.0f * simT); x[32] = std::cos(kTwoPi * 3.0f * simT);
    }

    float currentSigma() const { return sigmaMul * (0.10f + 0.25f * std::exp(-(float)brain.steps / 200000.f)); }

    void decide() {
        buildObs(xobs);
        brain.forward(xobs, act);
        sigma = currentSigma();
        for (int k = 0; k < N_ACT; k++) {
            float e = learning ? randn() : 0.f;
            lastEps[k] = e;
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
        int ni = nearestOrb();
        V3 d = orbs[ni].p - cr.pos; d.y = 0; V3 dir = norm(d);
        float vt = cr.vel.x * dir.x + cr.vel.z * dir.z;
        float energy = 0; for (int k = 0; k < N_ACT; k++) energy += lastAction[k] * lastAction[k];
        float reward = CTRL_DT * (2.5f * clampf(vt, -1.f, 2.f) + 0.4f * cr.ay.y) - 0.002f * energy;
        for (int i = 0; i < N_ORBS; i++) {                        // coleta recompensas
            float dx = orbs[i].p.x - cr.pos.x, dz = orbs[i].p.z - cr.pos.z;
            if (dx * dx + dz * dz < ORB_PICK_R * ORB_PICK_R) {
                reward += 4.f; epOrbs++; brain.orbs++; spawnOrb(i, cr.pos);
            }
        }
        if (++orbTimer >= 240) { orbTimer = 0; spawnOrb((int)(rnd01() * N_ORBS) % N_ORBS, cr.pos); } // muda de lugar
        epSteps++;
        trail.push_back(cr.pos); if (trail.size() > 300) trail.pop_front();

        bool bad = !std::isfinite(cr.pos.x + cr.pos.y + cr.pos.z + cr.vel.x + cr.vel.y + cr.vel.z);
        bool fell = bad || cr.pos.y < 0.30f || cr.ay.y < 0.5f;
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
            brain.train(buf, nbuf, lastV, sigma);
            nbuf = 0;
        }
    }

    void tick() {
        if (sub == 0) decide();
        cr.physics(PHYS_DT);
        simT += PHYS_DT;
        if (++sub == SUBSTEPS) { sub = 0; finish(); }
    }
    void setLearning(bool on) { learning = on; nbuf = 0; sampleValid = false; }
    bool save() { bool ok = brain.save(MEM_FILE); if (ok) dirty = false; return ok; }
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
    bool dirty() const { for (auto& a : ag) if (a->dirty) return true; return false; }

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
        }
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

    // avanca n passos de fisica em TODAS as IAs (em paralelo, quando vale a pena)
    void advance(int n) {
#if MULTITHREAD
        if (ag.size() > 1 && n >= 32) {
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
        bool ok = (ag.size() > 1) ? best.save(MEM_FILE) : ag[0]->brain.save(MEM_FILE);
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
    for (size_t i = 1; i < s.trail.size(); i++)                                   // rastro
        DrawLine3D(toRl(V3{s.trail[i - 1].x, 0.02f, s.trail[i - 1].z}), toRl(V3{s.trail[i].x, 0.02f, s.trail[i].z}), agentColor(id, focus ? 160 : 70));

    int ni = s.nearestOrb();                                                       // recompensas desta IA
    for (int i = 0; i < N_ORBS; i++) {
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
    DrawLine3D(Vector3{c.pos.x, 0.05f, c.pos.z}, Vector3{s.orbs[ni].p.x, 0.05f, s.orbs[ni].p.z}, focus ? rgba(255, 215, 0, 120) : agentColor(id, 110));

    DrawCircle3D(Vector3{c.pos.x, 0.015f, c.pos.z}, 0.5f, Vector3{1, 0, 0}, 90.f, rgba(0, 0, 0, 120));   // sombra
    float m[16] = {c.ax.x, c.ax.y, c.ax.z, 0, c.ay.x, c.ay.y, c.ay.z, 0, c.az.x, c.az.y, c.az.z, 0, c.pos.x, c.pos.y, c.pos.z, 1};
    rlPushMatrix(); rlMultMatrixf(m);                                                                    // tronco
    DrawCube(Vector3{0, 0, 0}, TORSO_LX, TORSO_LY, TORSO_LZ, body);
    DrawCubeWires(Vector3{0, 0, 0}, TORSO_LX, TORSO_LY, TORSO_LZ, rgba(230, 235, 245));
    DrawCube(Vector3{0.52f, 0.06f, 0}, 0.22f, 0.2f, 0.26f, rgba(255, 120, 60));                         // cabeca (frente)
    rlPopMatrix();
    for (int l = 0; l < 4; l++) {                                                                        // patas
        V3 kb, fb; c.legPoints(l, kb, fb);
        V3 h = c.pos + c.toWorld(HIP[l]), k = c.pos + c.toWorld(kb), f = c.pos + c.toWorld(fb);
        DrawCylinderEx(toRl(h), toRl(k), 0.05f, 0.04f, 8, rgba(170, 190, 220));
        DrawCylinderEx(toRl(k), toRl(f), 0.04f, 0.03f, 8, rgba(140, 160, 200));
        DrawSphere(toRl(k), 0.055f, rgba(230, 230, 240));
        DrawSphere(toRl(f), 0.065f, c.contact[l] ? rgba(80, 255, 120) : rgba(255, 90, 90));
    }
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
    for (int i = 0; i < (int)w.ag.size(); i++) if (i != w.shown) drawAgent(*w.ag[i], i, false, tnow);
    drawAgent(*w.ag[w.shown], w.shown, true, tnow);                               // a exibida por cima
    EndMode3D();
}

static void drawHud(const Swarm& w, double speed, bool paused, double lastSaveAgo, int fps) {
    const Sim& s = w.cur();
    int W = GetScreenWidth(), H = GetScreenHeight();
    DrawRectangle(8, 8, 360, 256, rgba(0, 0, 0, 150));
    hud(16, 14, 18, rgba(255, 255, 255), "CEREBRO ESPARSO 3D");
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
    hud(16, 222, 14, rgba(190, 220, 255), "Velocidade: x%.0f %s   FPS: %d", speed, paused ? "[PAUSADO]" : "", fps);
    hud(16, 240, 14, rgba(190, 220, 255), "Memoria: %s (salvo ha %.0fs)", MEM_FILE, lastSaveAgo);

    // mapa do cerebro: cada quadradinho = 1 grupo de 32 neuronios
    const int cols = 32, cell = 8, rows = N_GROUPS / cols;
    int ox = W - cols * cell - 16, oy = 34;
    DrawRectangle(ox - 8, 8, cols * cell + 16, rows * cell + 56, rgba(0, 0, 0, 150));
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
    hud(10, H - 22, 14, rgba(220, 220, 220), "CIMA/BAIXO velocidade | L aprendizado | TAB troca de IA | S salvar | ESPACO pausa | mouse esq. gira | roda zoom");
}

static void drawSwarmHud(const Swarm& w) {
    int n = (int)w.ag.size(), y0 = 272;
    if (n == 1) {
        DrawRectangle(8, y0, 360, 28, rgba(0, 0, 0, 150));
        hud(16, y0 + 7, 14, rgba(255, 160, 80), "Enxame: 1 IA (aprendizado desligado)");
        return;
    }
    DrawRectangle(8, y0, 360, 46 + 17 * n, rgba(0, 0, 0, 150));
    hud(16, y0 + 6, 14, rgba(255, 255, 255), "ENXAME: %d IAs | geracao %d", n, w.generation);
    if (w.bestScore > NO_SCORE) hud(16, y0 + 24, 14, rgba(255, 215, 0), "Recorde (melhor forma): %.1f", w.bestScore);
    else                        hud(16, y0 + 24, 14, rgba(190, 220, 255), "Recorde: aguardando 1a geracao");
    int lead = -1; float ls = NO_SCORE;
    for (int i = 0; i < n; i++) { float sc = w.ag[i]->score(1); if (sc > ls) { ls = sc; lead = i; } }
    for (int i = 0; i < n; i++) {
        const Sim& a = *w.ag[i]; float sc = a.score(1);
        Color c = i == w.shown ? rgba(255, 215, 0) : (i == lead ? rgba(120, 255, 160) : rgba(190, 200, 215));
        DrawRectangle(176 + 188, y0 + 46 + 17 * i, 10, 10, agentColor(i));
        if (sc > NO_SCORE) hud(16, y0 + 44 + 17 * i, 14, c, "%s#%d  score %7.1f  (%2zu/%zu ep)  ruido x%.2f", i == w.shown ? ">" : " ", i, sc, a.recent.size(), GEN_EPISODES, a.sigmaMul);
        else               hud(16, y0 + 44 + 17 * i, 14, c, "%s#%d  score    --   (%2zu/%zu ep)  ruido x%.2f", i == w.shown ? ">" : " ", i, a.recent.size(), GEN_EPISODES, a.sigmaMul);
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
    std::printf("[headless] fim. memoria salva em %s\n", MEM_FILE);
}

int main(int argc, char** argv) {
    g_rng.seed(std::random_device{}());
    Swarm swarm; swarm.init();

    if (argc >= 2 && std::string(argv[1]) == "--headless") {
        runHeadless(swarm, argc >= 3 ? std::atof(argv[2]) : 60.0);
        return 0;
    }

    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(1280, 720, "Cerebro esparso 3D - aprendendo a caminhar");
    SetTargetFPS(60);

    View view; double speed = 1.0; bool paused = false; double acc = 0.0;
    double lastSave = nowSec();

    while (!WindowShouldClose()) {
        // --- entrada ---
        if (IsKeyPressed(KEY_UP)) speed = std::min(1024.0, speed * 2.0);
        if (IsKeyPressed(KEY_DOWN)) speed = std::max(1.0, speed / 2.0);
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_L)) swarm.setLearning(!swarm.learning());
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
        view.target.y += (0.4f - view.target.y) * 0.08f;
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
