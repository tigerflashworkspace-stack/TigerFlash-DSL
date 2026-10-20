// =============================================================================
//  AI Workspace - uma IA que aprende a controlar um corpo 3D (C++17 + raylib, arquivo único)
//
//  VISÃO GERAL
//  * Cérebro esparso: 16384 neurônios (512 grupos de 32). Um "roteador" escolhe, a cada
//    decisão, só os grupos necessários (de 3 a 512) e só eles são calculados. Neurônios
//    com ativação <= 0 também ficam fora da conta e do aprendizado.
//  * Aprendizado: Actor-Critic (A2C + GAE), atualizando apenas os neurônios ativos.
//  * Física 3D própria: tronco com 6 graus de liberdade e pernas de 3 juntas, com contato e
//    atrito. Dois corpos: quadrúpede (4 patas) ou bípede digitígrado (2 pernas).
//    A única dependência externa é a raylib.
//
//  ENXAME
//  * Com o aprendizado LIGADO, o cérebro atual é clonado: N_AGENTS IAs (a original + clones)
//    treinam ao mesmo tempo, cada uma no seu mundo e na sua thread, com um nível de
//    exploração diferente.
//  * A cada geração (GEN_EPISODES episódios por IA) elas são comparadas pelo retorno médio.
//    O cérebro da melhor vira o "recorde" (é ele que vai para o arquivo de memória) e as
//    piores passam a ser cópias da melhor.
//  * Com o aprendizado DESLIGADO (tecla L) sobra uma única IA, com o melhor cérebro.
//
//  MODOS
//  * Caminhada (padrão): coletar orbes.        * Corrida (C): premia a velocidade horizontal.
//  * Equilíbrio (E): ficar em pé sobre uma esfera em movimento.
//  * Modo geral (G): um meta-controlador (Q-learning) escolhe a habilidade - andar, desviar,
//    correr ou empurrar - e as memórias motoras já treinadas movem as juntas. A IA enxerga
//    por duas câmeras de baixa resolução (frente e traseira), sem acesso à posição exata dos objetos.
//  * Parkour (K): plataformas separadas até uma orbe final; premia terminar rápido e pousar estável.
//
//  MEMÓRIA EM ARQUIVO
//  * Cada tarefa/corpo tem o seu arquivo .bin (veja CONFIGURAÇÃO). Eles são regravados a cada
//    5 s e ao sair, e lidos ao abrir o programa: o treino continua de onde parou.
//
//  COMPILAR
//    Linux:   g++ AiWorskspace.cpp -o aiworkspace -O2 -std=c++17 -pthread -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
//    Windows (MinGW):
//             g++ AiWorskspace.cpp -o aiworkspace.exe -O2 -std=c++17 -pthread -lraylib -lopengl32 -lgdi32 -lwinmm
//    macOS:   g++ AiWorskspace.cpp -o aiworkspace -O2 -std=c++17 -pthread -lraylib -framework OpenGL -framework Cocoa -framework IOKit
//    (MinGW: precisa da versão "posix threads"; se não tiver, ponha MULTITHREAD 0 abaixo.)
//
//  EXECUTAR
//    ./aiworkspace                       abre a janela 3D
//    ./aiworkspace --headless 120        treina 120 s sem janela (bem mais rápido)
//    ./aiworkspace --headless-general 120   idem, no modo geral
//    ./aiworkspace --headless-biped-balance 120   idem, bípede em equilíbrio sobre a esfera
//
//  TECLAS
//    Cima/Baixo  velocidade da simulação       G  modo geral          K  parkour
//    E  equilíbrio/caminhada                    C  corrida             Q  quadrúpede/bípede
//    L  liga/desliga o aprendizado              V  câmera traseira     Tab  trocar de IA
//    S  salvar a memória agora                  Espaço  pausa          Mouse/roda  câmera livre
//
//  FONTE DA INTERFACE
//    O texto usa uma fonte TTF suavizada (bordas arredondadas, sem serrilhado). O programa procura
//    "ui_font.ttf" (ou "fonts/ui_font.ttf") ao lado do executável; se não achar, tenta fontes do
//    sistema (Segoe UI, Arial Rounded, Nunito, Poppins, Ubuntu, DejaVu...). Para mudar o visual,
//    basta colocar ali uma fonte de sua preferência (ex.: Nunito ou Quicksand).
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
// Arquivos de memória, um por tarefa e morfologia (sufixo _BIP = bípede).
// São criados/lidos na pasta de execução; apague um deles para recomeçar aquela habilidade do zero.
static const char* MEM_FILE = "Brain_Walk.bin";
static const char* BALANCE_MEM_FILE = "Brain_Physical_Equilibrium.bin";
static const char* BIPED_MEM_FILE = "Brain_Walk_BIP.bin";
static const char* BIPED_BALANCE_MEM_FILE = "Brain_Physical_Equilibrium_BIP.bin";
static const char* RUN_MEM_FILE = "Brain_Run.bin";
static const char* BIPED_RUN_MEM_FILE = "Brain_Run_BIP.bin";
static const char* GENERAL_MEM_FILE = "Brain_General_Mode.bin"; // legado/estatisticas, nao controla as juntas
static const char* BIPED_GENERAL_MEM_FILE = "Brain_General_Mode_BIP.bin";
static const char* GENERAL_META_FILE = "Brain_General_Decisions.bin";
static const char* BIPED_GENERAL_META_FILE = "Brain_General_Decisions_BIP.bin";


// ---- cérebro esparso ----
constexpr int   N_GROUPS   = 512;                   // grupos (colunas corticais)
constexpr int   GROUP_SIZE = 32;                    // neuronios por grupo
constexpr int   N_NEURONS  = N_GROUPS * GROUP_SIZE; // total de neuronios
constexpr int   N_IN       = 33;                    // sensores
constexpr int   N_ACT      = 12;                    // atuadores: 4 patas x 3, ou 2 pernas x 3 no bípede
constexpr int   K_MIN      = 3;                     // minimo de grupos ativos
constexpr int   K_MAX      = 512;                    // maximo de grupos ativos
constexpr float ROUTER_BETA = 6.0f;                 // "foco" do roteador
constexpr float ROUTER_MASS = 0.60f;                // massa de atencao que basta cobrir


// ---- tempo e física ----
constexpr float PHYS_DT   = 1.0f / 240.0f;          // passo da fisica
constexpr int   SUBSTEPS  = 12;                     // fisica por decisao do cerebro
constexpr float CTRL_DT   = PHYS_DT * SUBSTEPS;     // 20 decisoes por segundo
constexpr int   MAX_EP_STEPS = 600;                 // 30 s por episodio

// ---- recompensas e visão da IA ----
constexpr int   N_ORBS    = 1;                      // recompensas simultaneas
constexpr float ORB_PICK_R = 0.75f;
constexpr int   VIS_W = 64, VIS_H = 32;             // imagem da camera da IA (pixels)
constexpr float VIS_HFOV = 2.0944f;                 // 120 graus por camera (frente e traseira)
constexpr float VIS_MAX = 14.f;                     // alcance da visao (m)
constexpr float WORLD_COLLISION_RADIUS = 12.0f;       // nao testa blocos distantes da IA
constexpr float WORLD_RENDER_RADIUS = 28.0f;           // nao desenha blocos fora da regiao visivel
constexpr float MAX_FORCE_N = 10.0f * 5.81f;             // teto absoluto de força: 15 kgf

constexpr float VIS_MEMORY = 3.0f;                  // quanto tempo lembra o que viu e saiu da vista (s)
constexpr float PASSAGE_MIN_WIDTH = 1.2f;           // largura minima de um vao para a IA tentar passar (m)
constexpr float TALL_OBSTACLE_H = 1.0f;             // objetos acima de 1 m (paredes): evitar, nunca empurrar/tocar


// ---- aprendizado Actor-Critic ----
constexpr int   ROLLOUT_T = 128;                    // lote menor: atualiza a politica a cada 6,4 s simulados
constexpr float GAMMA = 0.98f, LAMBDA = 0.95f;
constexpr float LR_ACTOR = 0.35f, LR_CRITIC = 0.5f, LR_HIDDEN = 0.03f;
constexpr float GCLIP = 5.0f, VCLIP = 5.0f;

// ---- enxame de IAs ----
constexpr int    N_AGENTS      = 10;     // total de IAs treinando juntas (a original + clones)
constexpr int    N_REPLACE     = 8;     // quantas das piores viram copia da melhor a cada geracao
constexpr size_t GEN_EPISODES  = 10;    // episodios por IA em cada geracao (janela de comparacao)
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

// Vetor 3D mínimo, com as operações que a física usa: soma, subtração, escala,
// produto escalar (dot) e vetorial (cross), módulo (len) e normalização (norm).
struct V3 { float x = 0, y = 0, z = 0; };
static inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static inline float len(V3 a) { return std::sqrt(dot(a, a)); }
static inline V3 norm(V3 a) { float l = len(a); return l > 1e-9f ? a * (1.f / l) : V3{}; }
// Gira v em torno do eixo unitário k pelo ângulo a (fórmula de Rodrigues).
static inline V3 rotAxis(V3 v, V3 k, float a) {
    float c = std::cos(a), s = std::sin(a);
    return v * c + cross(k, v) * s + k * (dot(k, v) * (1.f - c));
}
// Conversão para os tipos da raylib.
static inline Vector3 toRl(V3 v) { return Vector3{v.x, v.y, v.z}; }
static inline Color rgba(int r, int g, int b, int a = 255) {
    return Color{(unsigned char)r, (unsigned char)g, (unsigned char)b, (unsigned char)a};
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

// Rede neural esparsa com política (ator) e valor (crítico) compartilhando os neurônios.
//  * router   : um vetor de pesos por grupo; sua pontuação com a entrada decide quais grupos "acendem".
//  * Win/bias : camada oculta (entrada -> neurônio) com ReLU; ativação <= 0 não entra na conta.
//  * Wout     : ativações -> média (mu) de cada junta (ator).
//  * V        : ativações -> valor estimado do estado (crítico).
//  * useCount : quantas vezes cada grupo foi usado (alimenta o mapa do cérebro na tela).
struct Brain {
    std::vector<float> router, Win, bias, Wout, V;
    std::vector<uint32_t> useCount;
    uint64_t steps = 0, episodes = 0, orbs = 0;
    float emaReturn = 0;

    std::vector<float> score; std::vector<int> order;

    // Pesos novos: roteador com vetores unitários aleatórios, camada oculta aleatória e saída
    // zerada (a IA começa sem nenhum comportamento e aprende tudo por recompensa).
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
    // Aloca os buffers de trabalho do roteador (pontuação e ordem dos grupos).
    void prepare() { score.assign(N_GROUPS, 0.f); order.resize(N_GROUPS); std::iota(order.begin(), order.end(), 0); }

    // Passagem para frente ESPARSA. Etapas:
    //  1) o roteador pontua os 512 grupos contra a entrada x;
    //  2) escolhe os melhores grupos (de K_MIN a K_MAX) até cobrir ROUTER_MASS da "atenção" softmax;
    //  3) só os neurônios desses grupos são calculados, e só os que passam do ReLU ficam ativos;
    //  4) as ativações ativas são normalizadas e geram a média das juntas (mu) e o valor do estado.
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

    // Actor-Critic (A2C com GAE): atualiza o lote B com n amostras.
    //  1) calcula as vantagens com GAE (GAMMA/LAMBDA) e as normaliza;
    //  2) ator: empurra Wout na direção do ruído que rendeu vantagem positiva;
    //  3) crítico: aproxima o retorno (V);
    //  4) camada oculta: ajuste leve (LR_HIDDEN) pelo gradiente que volta de Wout e V.
    // Só os neurônios que estavam ativos em cada amostra são modificados.
    // lastV = valor do estado logo após o lote; lrScale reduz o passo em tarefas sensíveis.
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

    // ----------------------------- MEMÓRIA EM ARQUIVO --------------------------
    // Formato: assinatura "CBR3D001", dimensões da rede, contadores e depois os vetores de pesos em binário.
    template <class T> static void wr(std::ofstream& f, const std::vector<T>& v) { f.write((const char*)v.data(), v.size() * sizeof(T)); }
    template <class T> static bool rd(std::ifstream& f, std::vector<T>& v) { f.read((char*)v.data(), v.size() * sizeof(T)); return (bool)f; }

    // Grava num arquivo temporário e só então renomeia, para nunca deixar a memória
    // corrompida se o programa for fechado no meio da gravação.
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

    // Lê a memória. Se a assinatura ou as dimensões não baterem, guarda o arquivo antigo
    // como .old e devolve false (o chamador cria um cérebro novo).
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

// Obstaculos estaticos do ambiente geral. Degraus tambem funcionam como um
// terreno elevado para que as patas encontrem fisicamente cada superficie.
struct WorldBlock { V3 center, size; bool step = false; };
// Verdadeiro se o bloco está a até 'radius' de p no plano XZ (usado para ignorar blocos distantes).
static inline bool blockNearXZ(const WorldBlock& b, V3 p, float radius) {
    const float dx = std::fabs(p.x - b.center.x);
    const float dz = std::fabs(p.z - b.center.z);
    const float reachX = radius + b.size.x * 0.5f;
    const float reachZ = radius + b.size.z * 0.5f;
    return dx <= reachX && dz <= reachZ;
}
// Meia-diagonal do bloco no plano XZ.
static inline float blockDiagXZ(const WorldBlock& b) {
    const float hx = b.size.x * 0.5f, hz = b.size.z * 0.5f;
    return std::sqrt(hx * hx + hz * hz);
}
// Limita o módulo da força a maxN (teto físico de 15 kgf).
static inline V3 limitForce(V3 f, float maxN = MAX_FORCE_N) {
    const float m = len(f);
    return (m > maxN && m > 1e-6f) ? f * (maxN / m) : f;
}


// Controlador hierarquico: escolhe intencoes de movimento, nao angulos das juntas.
// A locomocao continua sendo executada por memorias de Brain ja treinadas.
// Dimensões da tabela Q do meta-controlador. Estado = tarefa x ângulo até o alvo x obstáculo
// à frente x lado mais livre x distância ao alvo (3 faixas cada); 5 ações possíveis.
static constexpr int META_TASKS = 4;
static constexpr int META_ANGLE_BINS = 3;
static constexpr int META_FRONT_BINS = 3;
static constexpr int META_SIDE_BINS = 3;
static constexpr int META_DISTANCE_BINS = 3;
static constexpr int META_STATES = META_TASKS * META_ANGLE_BINS * META_FRONT_BINS * META_SIDE_BINS * META_DISTANCE_BINS;
static constexpr int META_ACTIONS = 5;
static const char* META_MAGIC = "METAQ002";

// Controlador hierárquico por Q-learning tabular. Escolhe a ação macro (andar, desviar,
// correr, empurrar) a cada ~0,6-1,8 s; as memórias motoras já treinadas movem as juntas.
struct MetaController {
    // Acoes: caminhar direto, contornar para um lado, contornar para o outro,
    // correr usando a memoria de corrida, ou insistir no empurrao.
    std::vector<float> q;
    uint64_t decisions = 0, updates = 0;
    static constexpr float ALPHA = 0.18f;
    static constexpr float GAMMA = 0.92f;

    void init() { q.assign((size_t)META_STATES * META_ACTIONS, 0.f); decisions = updates = 0; }
    float value(int state, int action) const { return q[(size_t)state * META_ACTIONS + action]; }
    // Quais ações fazem sentido em cada tarefa/situação (ver regras abaixo).
    static bool allowed(int task, int action, bool runningReady, int frontBin) {
        // 0=andar ate objetivo; 1/2=desviar; 3=correr com memoria existente;
        // 4=aproximar e empurrar o bloco. Nao oferecer habilidades sem sentido.
        // Ao detectar parede muito perto no labirinto, insistir em andar reto nao
        // e uma opcao; perto de um bloco, o empurrao passa a ser opcao real.
        if (task == 0 && frontBin == 0) return action == 1 || action == 2;
        if (task == 2 && frontBin == 0) return action == 1 || action == 2 || action == 4;
        if (action >= 0 && action <= 2) return true;
        if (action == 3) return runningReady && (task == 0 || task == 3);
        if (action == 4) return task == 2;
        return false;
    }
    // Escolhe a ação: com probabilidade epsilon explora; senão pega a de maior valor Q (empate sorteado).
    int choose(int state, bool explore, int task, bool runningReady) {
        if (q.size() != (size_t)META_STATES * META_ACTIONS) init();
        ++decisions;
        const float epsilon = explore ? std::max(0.08f, 0.38f * std::exp(-(float)decisions / 24000.f)) : 0.f;
        const int frontBin = (state / (META_SIDE_BINS * META_DISTANCE_BINS)) % META_FRONT_BINS;
        int choices[META_ACTIONS]{}; int nChoices = 0;
        for (int a = 0; a < META_ACTIONS; ++a)
            if (allowed(task, a, runningReady, frontBin)) choices[nChoices++] = a;
        if (nChoices == 0) return 0;
        if (explore && rnd01() < epsilon)
            return choices[std::min(nChoices - 1, (int)(rnd01() * nChoices))];
        // Desempate aleatorio entre as acoes validas; nunca seleciona equilibrio
        // como substituto nem inventa uma memoria de corrida inexistente.
        for (int i = nChoices - 1; i > 0; --i)
            std::swap(choices[i], choices[(int)(rnd01() * (i + 1))]);
        int best = choices[0]; float bv = value(state, best);
        for (int i = 1; i < nChoices; ++i) {
            const int a = choices[i]; const float v = value(state, a);
            if (v > bv) { bv = v; best = a; }
        }
        return best;
    }
    // Atualização Q: Q <- Q + ALPHA * (recompensa + GAMMA * melhor Q do próximo estado - Q).
    void learn(int state, int action, float reward, int nextState, bool terminal,
               int currentTask, bool runningReady) {
        if (q.size() != (size_t)META_STATES * META_ACTIONS) init();
        if (state < 0 || state >= META_STATES || action < 0 || action >= META_ACTIONS) return;
        const int currentFront = (state / (META_SIDE_BINS * META_DISTANCE_BINS)) % META_FRONT_BINS;
        if (!allowed(currentTask, action, runningReady, currentFront)) return;
        float nextBest = 0.f;
        if (!terminal && nextState >= 0 && nextState < META_STATES) {
            const int nextTask = nextState / (META_ANGLE_BINS * META_FRONT_BINS * META_SIDE_BINS * META_DISTANCE_BINS);
            const int nextFront = (nextState / (META_SIDE_BINS * META_DISTANCE_BINS)) % META_FRONT_BINS;
            bool found = false;
            for (int a = 0; a < META_ACTIONS; ++a) if (allowed(nextTask, a, runningReady, nextFront)) {
                const float v = value(nextState, a);
                if (!found || v > nextBest) nextBest = v;
                found = true;
            }
            if (!found) nextBest = 0.f;
        }
        float& old = q[(size_t)state * META_ACTIONS + action];
        old += ALPHA * (reward + (terminal ? 0.f : GAMMA * nextBest) - old);
        ++updates;
    }
    // Grava/lê a tabela Q (assinatura METAQ002 + dimensões + contadores + valores).
    bool save(const char* path) const {
        if (q.size() != (size_t)META_STATES * META_ACTIONS) return false;
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        const int32_t dims[2] = {META_STATES, META_ACTIONS};
        f.write(META_MAGIC, 8); f.write((const char*)dims, sizeof dims);
        f.write((const char*)&decisions, sizeof decisions); f.write((const char*)&updates, sizeof updates);
        f.write((const char*)q.data(), (std::streamsize)(q.size() * sizeof(float)));
        return (bool)f;
    }
    bool load(const char* path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { init(); return false; }
        char magic[8]{}; int32_t dims[2]{}; uint64_t d = 0, u = 0;
        f.read(magic, 8); f.read((char*)dims, sizeof dims);
        if (!f || std::memcmp(magic, META_MAGIC, 8) || dims[0] != META_STATES || dims[1] != META_ACTIONS) { init(); return false; }
        f.read((char*)&d, sizeof d); f.read((char*)&u, sizeof u);
        std::vector<float> loaded((size_t)META_STATES * META_ACTIONS);
        f.read((char*)loaded.data(), (std::streamsize)(loaded.size() * sizeof(float)));
        if (!f) { init(); return false; }
        q.swap(loaded); decisions = d; updates = u; return true;
    }
};

// ================================ CRIATURA ===================================
// Corpo: X = frente, Y = cima, Z = lateral. Quadrupede: 4 patas; bipede: 2 pernas digitigradas.
// Corpo leve: 5 kg. A força física total aplicada por passo fica limitada
// a 15 kgf (~147 N), portanto 10-15 kgf e o teto de força, não o peso mínimo.
static constexpr float MASS = 5.0f;
static constexpr float BODY_SCALE = 1.00f; // tamanho original da IA
static const float INERT_X = 0.26f * BODY_SCALE, INERT_Y = 0.88f * BODY_SCALE, INERT_Z = 0.72f * BODY_SCALE;
static const float L1 = 0.35f * BODY_SCALE, L2 = 0.35f * BODY_SCALE;
static const float TORSO_LX = 0.9f * BODY_SCALE, TORSO_LY = 0.2f * BODY_SCALE, TORSO_LZ = 0.5f * BODY_SCALE;
static const V3 HIP[4] = {{0.38f * BODY_SCALE, -0.05f * BODY_SCALE, 0.2f * BODY_SCALE}, {0.38f * BODY_SCALE, -0.05f * BODY_SCALE, -0.2f * BODY_SCALE}, {-0.38f * BODY_SCALE, -0.05f * BODY_SCALE, 0.2f * BODY_SCALE}, {-0.38f * BODY_SCALE, -0.05f * BODY_SCALE, -0.2f * BODY_SCALE}};
static const float KN = 3000.f, CN = 100.f, KT = 2000.f, CT = 40.f, MU = 1.0f;
static constexpr float SPHERE_STATIC_FRICTION = 1.05f;
static constexpr float SPHERE_TANGENT_STIFFNESS = 240.f;
static constexpr float SPHERE_TANGENT_DAMPING = 30.f;
static const float SERVO_RATE = 15.f;
static const float BALANCE_BODY_CLEARANCE = 0.603f * BODY_SCALE; // places the neutral feet near load-bearing contact, avoiding launch impulses
static constexpr float FOOT_COLLISION_RADIUS = 0.065f * BODY_SCALE; // quadruped toe collider
// Anatomia digitígrada: o calcanhar fica elevado, o metatarso desce inclinado
// para a frente e só a pequena almofada do dedo sustenta o peso normalmente.
static constexpr float BIPED_HOCK_BACK = 0.040f * BODY_SCALE;
static constexpr float BIPED_HOCK_UP = 0.055f * BODY_SCALE;
static constexpr float BIPED_METATARSAL_FORWARD = 0.075f * BODY_SCALE;
static constexpr float BIPED_METATARSAL_DROP = 0.030f * BODY_SCALE;
static constexpr float BIPED_TOE_FORWARD = 0.190f * BODY_SCALE;
static constexpr float BIPED_TOE_DROP = 0.095f * BODY_SCALE;
static constexpr float BIPED_HOCK_COLLISION_RADIUS = 0.023f * BODY_SCALE;
static constexpr float BIPED_METATARSAL_COLLISION_RADIUS = 0.027f * BODY_SCALE;
static constexpr float BIPED_TOE_COLLISION_RADIUS = 0.034f * BODY_SCALE;
static constexpr float BIPED_FOOT_WIDTH = 0.105f * BODY_SCALE;

// Corpo físico da IA: tronco rígido (posição pos, velocidade vel, base ax/ay/az e velocidade
// angular wb) e pernas com 3 juntas cada. q = ângulo atual de cada junta; cmd = alvo vindo da
// política (ambos em [-1, 1]); contact/fn = apoio e força normal de cada perna.
struct Creature {
    bool biped = false; // morphology only; policy still controls every active joint
    V3 pos, vel, ax{1, 0, 0}, ay{0, 1, 0}, az{0, 0, 1}, wb;
    float q[N_ACT], cmd[N_ACT];
    V3 fbPrev[4], kneePrev[4]; bool fbInit = false; bool floorContact = false;
    bool torsoSphereContactPrev = false;
    V3 torsoSphereAnchor{};
    bool contact[4]; float anchX[4], anchZ[4], fn[4];
    // Pontos materiais de contato na esfera em movimento; é atrito estático físico,
    // não um controlador de equilíbrio programado.
    V3 sphereAnchor[4][9]; // biped: hock/metatarsus/toe x outer/centre/inner rows
    bool sphereAnchorActive[4][9]{};
    float soleAnchorX[4][9]{}, soleAnchorZ[4][9]{};
    bool soleGroundContact[4][9]{};

    // Pernas ativas (4 ou 2) e total de saídas de junta em uso (3 por perna).
    int limbCount() const { return biped ? 2 : 4; }
    int activeActionCount() const { return limbCount() * 3; }
    // Posição do quadril da perna l no referencial do corpo.
    V3 hipLocal(int l) const {
        if (biped) return {0.0f, -0.24f * BODY_SCALE, (l == 0) ? 0.14f * BODY_SCALE : -0.14f * BODY_SCALE};
        return HIP[l];
    }
    V3 torsoDimensions() const {
        // O bípede digitígrado tem tronco mais alto e estreito; o quadrúpede mantém
        // as dimensões horizontais originais.
        return biped ? V3{0.40f * BODY_SCALE, 0.68f * BODY_SCALE, 0.32f * BODY_SCALE}
                     : V3{TORSO_LX, TORSO_LY, TORSO_LZ};
    }

    // Recoloca o corpo parado na altura inicial, girado em yaw (em torno de Y) e com as juntas zeradas.
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
    // Conversão entre o referencial do corpo (ax, ay, az) e o do mundo.
    V3 toWorld(V3 b) const { return ax * b.x + ay * b.y + az * b.z; }
    V3 toBody(V3 w) const { return {dot(w, ax), dot(w, ay), dot(w, az)}; }

    // Cinemática dos membros. A caminhada mantém o mapeamento original; o treino na esfera
    // usa toda a faixa de ação da rede, sem limites estreitos só para dentro.
    // q já está em [-1, 1], pois é a saída tanh da política.
    void legPoints(int l, V3& knee, V3& foot, bool sphereMode = false) const {
        const V3 hip = hipLocal(l);
        const float hipAction = q[l * 3];
        const float abductionAction = q[l * 3 + 1];
        const float kneeAction = q[l * 3 + 2];
        const float side = hip.z >= 0.f ? 1.f : -1.f;
        float th, ph, kn, terminalAngle;

        if (biped) {
            // Bípede digitígrado: cada perna tem flexão do quadril, abdução lateral e
            // flexão do joelho independentes. A ponta do pé se estende para a frente a partir
            // do jarrete; os seis alvos de junta continuam sendo saídas da política neural.
            th = 1.35f * hipAction;
            ph = 0.85f * abductionAction;
            kn = 0.30f + 0.95f * kneeAction; // broad travel around a digitigrade hock rest angle
            terminalAngle = th - kn;          // raised heel / backward hock, toes forward
        } else if (sphereMode) {
            // Curso amplo e contínuo das juntas do quadrúpede, com rolagem para dentro/fora.
            th = 1.40f * hipAction;
            ph = 1.00f * abductionAction;
            kn = 0.75f * (kneeAction + 1.0f);
            terminalAngle = th + (hip.x > 0.f ? kn : -kn);
        } else {
            // Preserva o modelo de articulação original da caminhada do quadrúpede.
            th = hipAction * 0.6f;
            ph = 0.1f + abductionAction * 0.35f;
            kn = 0.7f + kneeAction * 0.6f;
            terminalAngle = th - kn;
        }

        // Os segmentos do bípede são um pouco mais curtos para o jarrete ficar acima do
        // chão enquanto os dedos o alcançam. O quadrúpede mantém os comprimentos antigos.
        const float upperLength = biped ? 0.32f * BODY_SCALE : L1;
        const float lowerLength = biped ? 0.32f * BODY_SCALE : L2;
        const float x1 = upperLength * std::sin(th), y1 = -upperLength * std::cos(th);
        const float x2 = x1 + lowerLength * std::sin(terminalAngle);
        const float y2 = y1 - lowerLength * std::cos(terminalAngle);
        const float cp = std::cos(ph), sp = std::sin(ph);
        knee = hip + V3{x1, y1 * cp, -y1 * sp * side};
        foot = hip + V3{x2, y2 * cp, -y2 * sp * side};
        // No modo bípede este ponto é o tornozelo/jarrete. O pé é uma sola separada e finita,
        // que se estende para trás (calcanhar) e para a frente (dedos).
    }

    // Três pontos locais ao longo do pé digitígrado: calcâneo elevado, metatarso
    // inclinado e almofada do dedo. Renderização e física usam os mesmos pontos.
    void bipedFootPoints(int l, V3& heel, V3& middle, V3& toe, bool sphereMode = false) const {
        V3 knee, ankle;
        legPoints(l, knee, ankle, sphereMode);
        // O calcanhar fica atrás E acima do tornozelo; não é um calcanhar plano que
        // suporte carga. O metatarso desce para a frente até uma almofada compacta.
        // Calcanhar elevado + apoio só nos dedos é a geometria digitígrada
        // (e não uma sola plana plantígrada).
        heel   = ankle + V3{-BIPED_HOCK_BACK, BIPED_HOCK_UP, 0.f};
        middle = ankle + V3{ BIPED_METATARSAL_FORWARD, -BIPED_METATARSAL_DROP, 0.f};
        toe    = ankle + V3{ BIPED_TOE_FORWARD, -BIPED_TOE_DROP, 0.f};
    }

    // Ponto do tronco (caixa orientada) mais próximo do centro da esfera. É apenas uma
    // primitiva de colisão: não impõe nenhuma orientação preferida.
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

        // Se o centro da esfera entrou na caixa por erro numérico, escolhe a face mais
        // próxima para a restrição ainda ter uma normal de saída válida.
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

    // Um passo de física (dt = PHYS_DT):
    //  1) os servos levam cada junta q até o alvo cmd da política;
    //  2) calcula as forças de contato dos pés/pernas com o chão, os blocos ou a esfera (atrito com âncoras);
    //  3) limita a força total e integra o tronco (6 graus de liberdade);
    //  4) corrige penetrações profundas por posição.
    // externalForce/externalTorque vêm de paredes e da caixa (modo geral); noBaseFloor = parkour (sem chão nos vãos).
    void physics(float dt, bool sphereMode = false, V3 sphereCenter = {},
                 V3 sphereVelocity = {}, float sphereRadius = 1.7f,
                 V3 sphereAngularVelocity = {},
                 const std::vector<WorldBlock>* worldBlocks = nullptr,
                 V3 externalForce = {}, V3 externalTorque = {}, bool noBaseFloor = false) {
        auto surfaceY = [&](float x, float z) {
            // Nos vãos do parkour não há chão de colisão: só os blocos de plataforma sustentam os pés.
            float h = noBaseFloor ? -1000.f : 0.f;
            if (worldBlocks) for (const auto& b : *worldBlocks) {
                if (!b.step || !blockNearXZ(b, V3{x, 0.f, z}, WORLD_COLLISION_RADIUS)) continue;
                if (std::fabs(x - b.center.x) <= b.size.x * 0.5f &&
                    std::fabs(z - b.center.z) <= b.size.z * 0.5f)
                    h = std::max(h, b.center.y + b.size.y * 0.5f);
            }
            return h;
        };
        // Os servos só seguem os valores de junta pedidos pela política. Não existe
        // pose de equilíbrio programada nem controlador corretivo.
        const float servoRate = sphereMode ? 10.0f : SERVO_RATE;
        const float al = 1.f - std::exp(-servoRate * dt);
        for (int i = 0; i < N_ACT; i++) q[i] += (cmd[i] - q[i]) * al;

        V3 F = V3{0, -9.81f * MASS, 0} + externalForce, T = externalTorque;
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
            constexpr float SPHERE_MAX_NORMAL_FORCE = MAX_FORCE_N;
            const float fnLocal = clampf(SPHERE_KN * penetration - SPHERE_CN * vn,
                                         0.f, SPHERE_MAX_NORMAL_FORCE);
            if (fnLocal <= 0.f) return;

            V3 force = normal * fnLocal;
            const V3 tangentVelocity = relativeVelocity - normal * vn;
            // Todo segmento de membro tem atrito dinâmico tipo Coulomb. A parte distal da
            // perna inferior também mantém um ponto-âncora material, para que um pé apoiado
            // resista ao deslizamento em vez de ser reiniciado a cada quadro.
            const bool useStaticAnchor = lowerSegment && anchorSlot >= 0 && anchorSlot < 9
                                      && (biped || t >= 0.70f);
            if (useStaticAnchor) {
                const V3 footSurfacePoint = point - normal * colliderRadius;
                // Um ponto material persistente para cada fileira calcanhar/centro/dedos.
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
                // Atrito de deslizamento para canelas/coxas que roçam na esfera. Não restringe
                // as juntas nem cria autocolisões.
                V3 tangentForce = tangentVelocity * (-28.f);
                const float maxDynamicFriction = 0.82f * fnLocal;
                const float tangentMagnitude = len(tangentForce);
                if (tangentMagnitude > maxDynamicFriction && tangentMagnitude > 1e-6f)
                    tangentForce = tangentForce * (maxDynamicFriction / tangentMagnitude);
                force = force + tangentForce;
            }
            force = limitForce(force);
            F = F + force;
            T = T + cross(r, force);
        };

        auto applyGroundPoint = [&](V3 localPoint, V3 localPointVelocity, float colliderRadius) {
            const V3 r = toWorld(localPoint);
            const V3 point = pos + r;
            const float floorY = surfaceY(point.x, point.z);
            const float penetration = colliderRadius + floorY - point.y;
            const V3 pointVelocity = vel + cross(wW, r) + toWorld(localPointVelocity);
            if (penetration <= 0.f && !(point.y < floorY + colliderRadius + 0.015f && pointVelocity.y < 0.f)) return;
            if (penetration > 0.f) floorContact = true;
            constexpr float GROUND_KN = 1800.f;
            constexpr float GROUND_CN = 70.f;
            const float fnGround = clampf(GROUND_KN * std::max(0.f, penetration)
                                          - GROUND_CN * pointVelocity.y, 0.f, MAX_FORCE_N);
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
            const float floorY = surfaceY(point.x, point.z);
            const float penetration = colliderRadius + floorY - point.y;
            const V3 pointVelocity = vel + cross(wW, r) + toWorld(localPointVelocity);
            if (penetration <= 0.f && !(point.y < floorY + colliderRadius + 0.015f && pointVelocity.y < 0.f))
                return;
            const float fnSole = clampf(KN * std::max(0.f, penetration) - CN * pointVelocity.y,
                                        0.f, MAX_FORCE_N);
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
                // Coxa e canela são cápsulas. A sola separada do bípede (calcanhar até os dedos)
                // tem três fileiras de contato paralelas (dentro/centro/fora).
                // Não há autocolisão nem colisão entre agentes.
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
                            // Nove almofadas de contato compactas num pé digitígrado inclinado. Não há
                            // cápsula plana do calcanhar aos dedos: normalmente só a fileira dos dedos
                            // alcança a superfície de apoio.
                            applyCapsuleContact(p, p, footVelB[l], footVelB[l],
                                                digitigradeRadii[segment], l, true, slot);
                            applyGroundPoint(p, footVelB[l], digitigradeRadii[segment]);
                        }
                    }
                    applyGroundPoint(kneeB[l], kneeVelB[l], 0.055f);
                    // O jarrete faz parte do colisor da canela e pode bater no chão após uma queda;
                    // não é uma sola de apoio em pé.
                } else {
                    applyCapsuleContact(kneeB[l], footB[l], kneeVelB[l], footVelB[l],
                                        FOOT_COLLISION_RADIUS, l, true, 0);
                    // Um agente caído ainda colide com o plano do chão.
                    applyGroundPoint(kneeB[l], kneeVelB[l], 0.055f);
                    applyGroundPoint(footB[l], footVelB[l], FOOT_COLLISION_RADIUS);
                }
            } else if (biped) {
                // O metatarso inclinado e as almofadas dos dedos tocam o chão separadamente.
                // Em postura neutra, o calcanhar elevado não deve funcionar como apoio plano.
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
                // A largura física total vem de pequenas almofadas separadas ao longo do pé,
                // enquanto o jarrete elevado fica sem contato na postura neutra.
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
                const float floorY = surfaceY(pf.x, pf.z);
                const float penetration = floorY - pf.y;
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
        // Grava o histórico de contato por fileira depois de testar as três fileiras da sola do bípede.
        if (sphereMode) {
            for (int l = 0; l < limbCount(); ++l)
                for (int slot = 0; slot < 9; ++slot)
                    sphereAnchorActive[l][slot] = sphereAnchorUsedNow[l][slot];
        } else {
            for (int l = 0; l < limbCount(); ++l)
                for (int slot = 0; slot < 9; ++slot)
                    soleGroundContact[l][slot] = soleGroundUsedNow[l][slot];
        }

        // O tronco é um colisor de caixa orientada contra a esfera. Além da força normal,
        // recebe atrito tangencial ao deslizar na casca.
        // É só contato com o ambiente: não há colisão entre agentes nem autocolisão.
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
                force = limitForce(force);
                torsoSphereContactPrev = true;
                F = F + force;
                T = T + cross(r, force);
            }
        } else {
            torsoSphereContactPrev = false;
        }
        if (sphereMode) {
            // O contato da caixa orientada com o chão segura o tronco se a política
            // cair da bola. É colisão com o ambiente, não autocolisão.
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
                const float fnFloor = clampf(2200.f * penetration - 100.f * pointVelocity.y,
                                             0.f, MAX_FORCE_N);
                if (fnFloor > 0.f) {
                    const V3 force{0.f, fnFloor, 0.f};
                    F = F + force;
                    T = T + cross(r, force);
                    floorContact = true;
                }
            }
        }

        fbInit = true;
        // Teto absoluto da força resultante: 15 kgf. Isso inclui gravidade,
        // contatos e forças externas neste passo físico.
        F = limitForce(F);
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
            // Escolhe um vetor perpendicular a ax como reserva. Um Z fixo do mundo pode
            // ficar paralelo a ax e zerar toda a base do corpo.
            const V3 seed = std::fabs(ax.y) < 0.90f ? V3{0.f, 1.f, 0.f}
                                                    : V3{0.f, 0.f, 1.f};
            az = norm(cross(ax, seed));
        }
        ay = norm(cross(az, ax));
        az = norm(cross(ax, ay));

        if (sphereMode) {
            // Passo de não-penetração por posição para quedas rápidas. As forças tratam o
            // contato comum; este passo só remove sobreposição profunda/atravessamento e elimina
            // a velocidade direcionada para dentro da esfera, sem adicionar quique.
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
                        // O anti-atravessamento por posição usa três pequenos pontos arredondados.
                        // Propositalmente não há um colisor de sola plana e largo.
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
                        // As pontas das cápsulas também são restritas contra o chão.
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
// Recompensa a coletar: posição da orbe no mundo.
struct Orb { V3 p; };

// Um mundo completo de UMA IA: cérebro, corpo, cenário, recompensas, percepção (câmeras)
// e estatísticas. Cada IA do enxame tem o seu Sim, que roda na própria thread.
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
    bool generalMode = false;     // hierarquia: Q escolhe a intencao; memorias controlam as juntas
    bool parkourMode = false;     // troca apenas ambiente/recompensa; usa o mesmo Brain da caminhada
    V3 parkourGoal{};
    float parkourFallSeconds = 0.f;
    float lastParkourTime = 0.f, bestParkourTime = 0.f;
    int parkourCompletions = 0;
    enum GeneralTask { TASK_MAZE = 0, TASK_STAIRS = 1, TASK_PUSH = 2, TASK_SPEED = 3, TASK_BALANCE = 4 };
    int generalTask = TASK_MAZE;
    std::vector<WorldBlock> generalBlocks;
    V3 generalGoal{}, boxPos{}, boxVel{}, pushTarget{}, generalStart{};
    float prevGoalDistance = 0.f, prevBoxGoalDistance = 0.f, prevAgentBoxDistance = 0.f;
    int generalSuccesses = 0;
    Brain walkingSkill, runningSkill, balanceSkill;
    bool walkingSkillReady = false, runningSkillReady = false, balanceSkillReady = false;
    MetaController meta;
    int generalAction = 0, generalActionAge = 0, generalPrevState = -1, generalPrevAction = -1;
    float generalRewardWindow = 0.f;
    bool generalHasAction = false;
    float generalFallSeconds = 0.f;
    // Busca de passagem / limite de tempo parado
    mutable V3 lastDir{1, 0, 0};                     // ultimo rumo escolhido pelo planejador visual
    mutable float lastDirAge = 99.f;
    mutable V3 commitDir{1, 0, 0}; mutable float commitT = 0.f;   // compromisso com o rumo escolhido
    std::vector<V3> visited; float visitTimer = 0.f; // onde a IA ja esteve (para explorar lugares novos)
    float stallSeconds = 0.f;
    static constexpr float STALL_FREE = 0.6f;  // pode ficar parada ate 0,6 s sem penalidade extra
    // Imagem das duas cameras da IA (frente e traseira) e o que ela percebeu nelas.
    struct VisionImage {
        uint8_t rgb[VIS_W * VIS_H * 3]{};
        float depth[VIS_W * VIS_H]{};
        V3 eye{}, fwd{1, 0, 0}, right{0, 0, 1};
        bool valid = false;
    };
    struct SeenObj { bool seen = false; V3 world{}; float age = 99.f; };
    VisionImage visFront, visRear;
    float colObs[2][VIS_W]{}, colTall[2][VIS_W]{};  // distancia horizontal (m) por coluna da imagem
    SeenObj seenGoal, seenBox;                      // alvo/caixa vistos (e lembrados por VIS_MEMORY s)
    float pFront = 5.f, pL = 5.f, pR = 5.f;         // obstaculo a frente / diagonais, medido na IMAGEM
    V3 lastHeading{1, 0, 0};
    struct WallMemPt { V3 p; float age; };
    std::vector<WallMemPt> wallMem;                 // paredes altas que a IA acabou de ver (memoria visual curta)
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

    // Prepara o lote de amostras (ROLLOUT_T) usado pelo aprendizado.
    void setupBuffers() {
        buf.resize(ROLLOUT_T);
        for (auto& s : buf) { s.idx.reserve(K_MAX * GROUP_SIZE); s.hn.reserve(K_MAX * GROUP_SIZE); }
    }
    // cria esta IA como clone de um cerebro ja existente
    void initClone(const Brain& b, float sMul, uint32_t seed) {
        setupBuffers(); brain = b; sigmaMul = sMul; rngSeed = seed; resetEpisode();
    }
    // troca o cerebro por uma copia de outro (recomeca o episodio e a janela de score)
    void adopt(const Brain& b, const MetaController* m = nullptr) {
        brain = b; if (m) meta = *m; nbuf = 0; recent.clear(); resetEpisode();
    }

    // Altura do tronco acima do chão/esfera na postura neutra.
    float bodyClearance() const { return cr.biped ? 0.988f : BALANCE_BODY_CLEARANCE; }
    // Verdadeiro quando o episódio é de equilíbrio sobre a esfera (modo E ou tarefa de equilíbrio do modo geral).
    bool balanceSurfaceEpisode() const { return balanceMode || (generalMode && generalTask == TASK_BALANCE); }
    const char* metaMemoryFile() const { return cr.biped ? BIPED_GENERAL_META_FILE : GENERAL_META_FILE; }
    // Arquivo .bin de memória da tarefa/morfologia ativa.
    const char* memoryFile() const {
        if (generalMode) return cr.biped ? BIPED_GENERAL_MEM_FILE : GENERAL_MEM_FILE;
        if (balanceMode)
            return cr.biped ? BIPED_BALANCE_MEM_FILE : BALANCE_MEM_FILE;
        if (speedRewardMode)
            return cr.biped ? BIPED_RUN_MEM_FILE : RUN_MEM_FILE;
        return cr.biped ? BIPED_MEM_FILE : MEM_FILE;
    }

    // Carrega a memória de caminhada (se existir) ou cria um cérebro novo, e começa o primeiro episódio.
    void init() {
        setupBuffers();
        if (std::filesystem::exists(MEM_FILE) && brain.load(MEM_FILE))
            std::printf("[memoria] caminhada carregada: %llu passos, %llu episodios, %llu orbes coletados\n",
                        (unsigned long long)brain.steps, (unsigned long long)brain.episodes, (unsigned long long)brain.orbs);
        else { brain.init(); std::printf("[memoria] cerebro novo criado (%d neuronios)\n", N_NEURONS); }
        resetEpisode();
    }

    // Coloca a orbe i num ponto aleatório (de 2,5 a 6,5 m) ao redor de 'around'.
    void spawnOrb(int i, V3 around) {
        float a = rnd01() * kTwoPi, r = rndRange(2.5f, 6.5f);
        orbs[i].p = {around.x + std::cos(a) * r, 0.6f, around.z + std::sin(a) * r};
    }

    // Carrega as memórias motoras (caminhada, corrida, equilíbrio) que o modo geral usa.
    // Sem a de caminhada treinada o modo geral fica bloqueado.
    void loadGeneralSkills(const Brain& fallback) {
        const char* walkPath = cr.biped ? BIPED_MEM_FILE : MEM_FILE;
        const char* runPath = cr.biped ? BIPED_RUN_MEM_FILE : RUN_MEM_FILE;
        const char* balancePath = cr.biped ? BIPED_BALANCE_MEM_FILE : BALANCE_MEM_FILE;
        const bool walkLoaded = walkingSkill.load(walkPath);
        walkingSkillReady = walkLoaded && walkingSkill.steps > 0;
        if (!walkingSkillReady && fallback.steps > 0) {
            walkingSkill = fallback;
            walkingSkillReady = true;
        }
        if (!walkingSkillReady) walkingSkill = fallback; // retained only to keep state valid; not selectable
        runningSkill = walkingSkill;
        const bool runLoaded = runningSkill.load(runPath);
        runningSkillReady = runLoaded && runningSkill.steps > 0;
        if (!runningSkillReady) runningSkill = walkingSkill;
        balanceSkill = walkingSkill;
        const bool balanceLoaded = balanceSkill.load(balancePath);
        balanceSkillReady = balanceLoaded && balanceSkill.steps > 0;
        std::printf("[MODO GERAL] memoria caminhada: %s; corrida: %s; equilibrio: %s\n",
                    walkingSkillReady ? walkPath : "AUSENTE/SEM TREINO (MODO GERAL sera bloqueado)",
                    runningSkillReady ? runPath : "ausente/sem treino (acao de corrida bloqueada)",
                    balanceSkillReady ? balancePath : "ausente/sem treino (equilibrio fica no modo E)");
    }

    // Discretiza o que a IA percebeu (tarefa, ângulo até o alvo, obstáculo à frente, lado livre, distância) em um índice da tabela Q.
    int metaState() const {
        // Tudo vem do que a IA viu: alvo (so se apareceu na imagem ou foi lembrado) e
        // distancias a frente/diagonais lidas na imagem. Alvo desconhecido = "longe, a frente".
        V3 decisionTarget{};
        const bool known = visualTarget(true, decisionTarget);
        int angleBin = 1, distBin = 2;
        if (known) {
            V3 goal = decisionTarget - cr.pos; goal.y = 0.f;
            const float dGoal = len(goal);
            const V3 localGoal = cr.toBody(dGoal > 1e-6f ? goal * (1.f / dGoal) : V3{});
            const float angle = std::atan2(localGoal.z, localGoal.x);
            angleBin = angle < -0.38f ? 0 : (angle > 0.38f ? 2 : 1);
            distBin = dGoal < 1.2f ? 0 : (dGoal < 3.2f ? 1 : 2);
        }
        const int frontBin = pFront < 0.85f ? 0 : (pFront < 1.9f ? 1 : 2);
        const float sideDiff = pL - pR;
        const int sideBin = sideDiff < -0.45f ? 0 : (sideDiff > 0.45f ? 2 : 1);
        const int task = std::clamp(generalTask, 0, META_TASKS - 1);
        return (((task * META_ANGLE_BINS + angleBin) * META_FRONT_BINS + frontBin)
                * META_SIDE_BINS + sideBin) * META_DISTANCE_BINS + distBin;
    }

    // Nome da ação macro escolhida pelo meta-controlador (mostrado na tela).
    const char* generalActionName() const {
        switch (generalAction) {
            case 1: return "Desviar (lado +)";
            case 2: return "Desviar (lado -)";
            case 3: return runningSkillReady ? "Memória de corrida" : "Corrida indisponível";
            case 4: return "Caminhada + empurrão";
            default: return "Memória de caminhada";
        }
    }

    // Monta a entrada (N_IN sensores) das memórias motoras no modo geral, no mesmo formato da caminhada/corrida.
    void buildGeneralSkillObs(float* x) const {
        // Contrato de entrada igual ao da memoria de caminhada/corrida.
        // Os raios de obstaculos alimentam o meta-controlador, nao alteram o cerebro motor.
        const Creature& c = cr;
        V3 vb = c.toBody(c.vel);
        x[0] = vb.x / 1.5f; x[1] = vb.y / 1.5f; x[2] = vb.z / 1.5f;
        x[3] = c.wb.x / 3.f; x[4] = c.wb.y / 3.f; x[5] = c.wb.z / 3.f;
        x[6] = c.ax.y; x[7] = c.ay.y; x[8] = c.az.y;
        x[9] = (c.pos.y - 0.66f) * 3.f;
        // Opcao hierarquica de empurrao: chega perto do bloco primeiro; ao entrar em
        // contato, aponta para a zona-alvo. O alvo vem da IMAGEM (visto/lembrado).
        V3 target{};
        const bool known = visualTarget(generalTask == TASK_PUSH && generalAction == 4, target);
        V3 direction; float distance;
        if (known) {
            direction = target - c.pos; direction.y = 0.f;
            distance = len(direction);
            direction = distance > 1e-6f ? direction * (1.f / distance) : V3{};
        } else {                       // ainda nao viu o alvo: explora seguindo em frente
            direction = lastHeading; distance = 6.f;
        }
        if (generalTask != TASK_BALANCE) {
            // A IA decide o rumo pela imagem: alvo visto -> vai nele por onde ha espaco livre;
            // alvo nao visto -> explora. A intencao de desvio do seletor (acoes 1/2) so inclina a escolha.
            direction = steerByVision(known, target, generalAction == 1 ? 1.f : (generalAction == 2 ? -1.f : 0.f));
        }
        direction = avoidTallObstacles(direction); // so a direcao; a memoria nao muda
        V3 db = c.toBody(direction);
        x[10] = db.x; x[11] = db.z; x[12] = clampf(distance / 6.f, 0.f, 1.5f);
        for (int i = 0; i < N_ACT; ++i) x[13 + i] = c.q[i];
        for (int l = 0; l < 4; ++l) x[25 + l] = c.contact[l] ? 1.f : 0.f;
        x[29] = std::sin(kTwoPi * 1.5f * simT); x[30] = std::cos(kTwoPi * 1.5f * simT);
        x[31] = std::sin(kTwoPi * 3.0f * simT); x[32] = std::cos(kTwoPi * 3.0f * simT);
    }

    // Reinicia a esfera móvel do equilíbrio: fase, direção, velocidade e dificuldade (conforme o desempenho).
    void resetBalanceSphere() {
        sphereClock = 0.f;
        spherePhase = rndRange(0.f, kTwoPi);
        // Começa na altura exata desta fase. Antes, a primeira atualização mudava o Y de
        // uma vez, gerando uma velocidade falsa enorme e renascimentos girando.
        const float initialY = 2.05f + 0.16f * std::sin(spherePhase)
                                      + 0.05f * std::sin(spherePhase * 0.61f);
        spherePos = {arenaOrigin.x, initialY, arenaOrigin.z};
        sphereVel = {};
        sphereAngularVelocity = {};
        sphereTurnTimer = rndRange(3.0f, 5.0f);
        float a = rndRange(0.f, kTwoPi);
        sphereDir = sphereTargetDir = {std::cos(a), 0.f, std::sin(a)};
        const float practice = clampf(bestStableSeconds / 8.f, 0.f, 1.f);
        // A dificuldade segue o desempenho medido de equilíbrio, não o tempo de computação.
        // Uma política com dificuldade ganha tempo para descobrir o apoio antes de a esfera
        // ficar mais difícil; nenhuma ação de junta é prescrita por este currículo.
        sphereMoveSpeed = rndRange(0.30f + 0.30f * practice,
                                   0.45f + 0.55f * practice);
        offSphereSeconds = 0.f;
    }

    // Move a esfera (direção suave, dentro da faixa deste agente) e a faz rolar.
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
        // Direção suave, mas mantendo a esfera deste agente dentro da sua própria faixa.
        // Sem limite, esferas que se movem sozinhas acabariam se cruzando e invadindo os
        // mundos dos outros agentes, pois não há solucionador de colisão entre esferas.
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
        // Segurança numérica: mantém o centro dentro da faixa mesmo num passo grande.
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
        // Rola a esfera ao transladar: ômega = cima x velocidade / raio. As âncoras dos
        // pés existentes giram com a superfície, então ficar parado não equivale a ser
        // carregado numa plataforma que não gira.
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

    // Distância no plano horizontal (ignora a altura).
    static float distXZ(V3 a, V3 b) {
        const float dx = a.x - b.x, dz = a.z - b.z;
        return std::sqrt(dx * dx + dz * dz);
    }

    // Adiciona uma parede alta (obstáculo) em coordenadas locais da arena.
    void addGeneralWall(float localX, float localZ, float sx, float sz, float height = 2.4f) {
        generalBlocks.push_back({{arenaOrigin.x + localX, height * 0.5f, arenaOrigin.z + localZ},
                                 {sx, height, sz}, false});
    }

    // Adiciona um degrau (bloco pisável) em coordenadas locais da arena.
    void addGeneralStep(float localX, float height, float width = 1.0f, float depth = 2.6f) {
        generalBlocks.push_back({{arenaOrigin.x + localX, height * 0.5f, arenaOrigin.z},
                                 {width, height, depth}, true});
    }

    // Adiciona uma plataforma de parkour (bloco pisável) em coordenadas locais da arena.
    void addParkourPlatform(float localX, float localZ, float height,
                            float width = 1.45f, float depth = 1.8f) {
        generalBlocks.push_back({{arenaOrigin.x + localX, height * 0.5f, arenaOrigin.z + localZ},
                                 {width, height, depth}, true});
    }

    // Altura do chão em (x, z): 0, ou o topo do degrau/plataforma que estiver embaixo.
    float groundHeight(float x, float z) const {
        float h = 0.f;
        if (generalMode || parkourMode) for (const auto& b : generalBlocks) {
            if (!b.step) continue;
            if (std::fabs(x - b.center.x) <= b.size.x * 0.5f &&
                std::fabs(z - b.center.z) <= b.size.z * 0.5f)
                h = std::max(h, b.center.y + b.size.y * 0.5f);
        }
        return h;
    }

    // Nome da tarefa atual do modo geral (mostrado na tela).
    const char* generalTaskName() const {
        switch (generalTask) {
            case TASK_MAZE: return "Labirinto";
            case TASK_STAIRS: return "Escadas";
            case TASK_PUSH: return "Empurrar objeto";
            case TASK_SPEED: return "Velocidade";
            default: return "Equilíbrio livre";
        }
    }

    // Monta o cenário do episódio do modo geral conforme a tarefa (labirinto, escadas, empurrar, velocidade ou equilíbrio).
    void setupGeneralEpisode() {
        generalBlocks.clear();
        boxVel = {};
        // MODO GERAL fica sempre em navegacao. Equilibrio e uma memoria separada,
        // acessada por E, nunca um episodio sorteado no meio do labirinto.
        generalTask = std::min(3, (int)(rnd01() * 4.f));
        cr.pos = {arenaOrigin.x - 5.f, cr.biped ? 1.0f : 0.69f, arenaOrigin.z};
        cr.vel = {};
        generalStart = cr.pos;
        generalGoal = {arenaOrigin.x + 6.f, 0.5f, arenaOrigin.z};
        pushTarget = {arenaOrigin.x + 5.4f, 0.42f, arenaOrigin.z};
        boxPos = {arenaOrigin.x + 0.2f, 0.42f, arenaOrigin.z};

        // O desafio de equilibrio acontece na esfera; nao recebe o mundo do labirinto.
        if (generalTask != TASK_BALANCE) {
            addGeneralWall(-8.f, 0.f, 0.30f, 12.4f);
            addGeneralWall( 8.f, 0.f, 0.30f, 12.4f);
            addGeneralWall(0.f, -6.0f, 16.3f, 0.30f);
            addGeneralWall(0.f,  6.0f, 16.3f, 0.30f);
        }

        if (generalTask == TASK_MAZE) {
            // Duas paredes alternadas deixam passagens por lados opostos.
            // A rede precisa perceber as distancias e descobrir por onde contornar.
            addGeneralWall(-1.0f, -0.85f, 0.34f, 7.0f, 2.2f);
            addGeneralWall( 2.1f,  0.85f, 0.34f, 7.0f, 2.2f);
        } else if (generalTask == TASK_STAIRS) {
            // Tres degraus baixos e continuos, com colisao real nas superficies.
            addGeneralStep(-0.65f, 0.18f, 1.05f, 2.8f);
            addGeneralStep( 0.35f, 0.36f, 1.05f, 2.8f);
            addGeneralStep( 1.35f, 0.54f, 1.05f, 2.8f);
        } else if (generalTask == TASK_PUSH) {
            // Corredor largo para o bloco se mover ate a zona-alvo.
            addGeneralWall(0.f, -1.75f, 14.0f, 0.25f, 1.8f);
            addGeneralWall(0.f,  1.75f, 14.0f, 0.25f, 1.8f);
            boxPos = {arenaOrigin.x - 0.3f, 0.42f, arenaOrigin.z};
        } else if (generalTask == TASK_SPEED) {
            addGeneralWall(0.f, -2.15f, 14.0f, 0.25f, 1.8f);
            addGeneralWall(0.f,  2.15f, 14.0f, 0.25f, 1.8f);
        } else if (generalTask == TASK_BALANCE) {
            generalGoal = generalStart; // este episodio usa a memoria de equilibrio e a esfera movel
        }
        prevGoalDistance = distXZ(cr.pos, generalGoal);
        prevBoxGoalDistance = distXZ(boxPos, pushTarget);
        prevAgentBoxDistance = distXZ(cr.pos, boxPos);
    }

    // Monta o parkour: corredor, plataformas separadas e a orbe de chegada.
    void setupParkourEpisode() {
        generalBlocks.clear();
        boxVel = {};
        generalTask = TASK_MAZE; // keeps the shared wall collision path away from the push-object logic

        const float floorTop = 0.18f;
        cr.pos = {arenaOrigin.x - 9.35f, floorTop + (cr.biped ? 1.0f : 0.69f), arenaOrigin.z};
        cr.vel = {};
        generalStart = cr.pos;

        // Duas paredes contínuas formam um corredor. Não há chão entre as plataformas.
        addGeneralWall(0.f, -1.95f, 23.0f, 0.25f, 2.7f);
        addGeneralWall(0.f,  1.95f, 23.0f, 0.25f, 2.7f);

        // Largada, seis plataformas de apoio separadas e uma plataforma larga de chegada na orbe.
        addParkourPlatform(-9.20f,  0.00f, 0.18f, 2.00f, 2.85f);
        addParkourPlatform(-6.50f,  0.38f, 0.27f, 1.40f, 1.80f);
        addParkourPlatform(-3.80f, -0.34f, 0.19f, 1.42f, 1.78f);
        addParkourPlatform(-1.10f,  0.40f, 0.35f, 1.38f, 1.72f);
        addParkourPlatform( 1.60f, -0.38f, 0.24f, 1.42f, 1.78f);
        addParkourPlatform( 4.30f,  0.34f, 0.40f, 1.38f, 1.72f);
        addParkourPlatform( 7.00f, -0.32f, 0.23f, 1.42f, 1.78f);
        addParkourPlatform( 9.55f,  0.00f, 0.18f, 2.10f, 2.85f);

        parkourGoal = {arenaOrigin.x + 9.55f, 0.60f, arenaOrigin.z};
        generalGoal = parkourGoal;
        orbs[0].p = parkourGoal; // one fixed orb per Sim/agent; parkour does not respawn it elsewhere
        prevGoalDistance = distXZ(cr.pos, parkourGoal);
        parkourFallSeconds = 0.f;
    }

    // Raio horizontal contra um retângulo no plano XZ.
    float rayRectDistance(V3 origin, V3 dir, V3 center, float sx, float sz, float maxD = 5.f) const {
        float tMin = 0.f, tMax = maxD;
        const float oo[2] = {origin.x, origin.z};
        const float dd[2] = {dir.x, dir.z};
        const float cc[2] = {center.x, center.z};
        const float hh[2] = {sx * 0.5f, sz * 0.5f};
        for (int axis = 0; axis < 2; ++axis) {
            const float lo = cc[axis] - hh[axis], hi = cc[axis] + hh[axis];
            if (std::fabs(dd[axis]) < 1e-6f) {
                if (oo[axis] < lo || oo[axis] > hi) return maxD;
            } else {
                float a = (lo - oo[axis]) / dd[axis];
                float b = (hi - oo[axis]) / dd[axis];
                if (a > b) std::swap(a, b);
                tMin = std::max(tMin, a);
                tMax = std::min(tMax, b);
                if (tMax < tMin) return maxD;
            }
        }
        // Distancia da PRIMEIRA face atingida (0 se a origem ja esta dentro/encostada).
        return clampf(tMin, 0.f, maxD);
    }

    float parkourSupportDistance(V3 direction) const {
        direction.y = 0.f;
        direction = norm(direction);
        if (len(direction) < 1e-6f) return 6.f;
        float bestDistance = 6.f;
        for (const auto& b : generalBlocks) {
            if (!b.step) continue;
            const bool standingOnThis = std::fabs(cr.pos.x - b.center.x) <= b.size.x * 0.5f + 0.18f
                                     && std::fabs(cr.pos.z - b.center.z) <= b.size.z * 0.5f + 0.18f;
            if (standingOnThis) continue;
            const float d = rayRectDistance(cr.pos, direction, b.center, b.size.x, b.size.z, 6.f);
            if (d > 0.03f) bestDistance = std::min(bestDistance, d);
        }
        return bestDistance;
    }

    // ------------------------------------------------------------------
    //  VISAO DA IA: duas cameras (frente e traseira, 120 graus cada) geram uma
    //  IMAGEM 64x32 (cor + profundidade) do que realmente existe em volta dela.
    //  A IA so age pelo que aparece nessa imagem: o alvo (marcador verde, ou a
    //  caixa laranja), paredes e passagens sao detectados nos pixels. O que esta
    //  atras de uma parede nao aparece na imagem, entao ela nao "sabe" que existe.
    // ------------------------------------------------------------------
    static const V3* camTable() {      // direcao de cada pixel: x = frente, y = cima, z = direita
        static const std::vector<V3> table = [] {
            std::vector<V3> v((size_t)VIS_W * VIS_H);
            const float f = (VIS_W * 0.5f) / std::tan(VIS_HFOV * 0.5f);
            for (int r = 0; r < VIS_H; ++r)
                for (int c = 0; c < VIS_W; ++c)
                    v[(size_t)r * VIS_W + c] = norm(V3{f, VIS_H * 0.5f - (r + 0.5f), c + 0.5f - VIS_W * 0.5f});
            return v;
        }();
        return table.data();
    }
    static float camFocal() { return (VIS_W * 0.5f) / std::tan(VIS_HFOV * 0.5f); }

    V3 visionHeading() const {         // camera nivelada (so gira com o corpo, nao inclina)
        const V3 h{cr.ax.x, 0.f, cr.ax.z};
        return len(h) > 0.2f ? norm(h) : lastHeading;
    }
    V3 markerPos() const { return generalTask == TASK_PUSH ? pushTarget : generalGoal; }

    // Interseção raio x caixa alinhada aos eixos; devolve a distância e o eixo da face atingida.
    static bool rayBox(V3 o, V3 d, V3 c, V3 sz, float& tHit, int& axisHit) {
        const float oo[3] = {o.x, o.y, o.z}, dd[3] = {d.x, d.y, d.z};
        const float cc[3] = {c.x, c.y, c.z}, hh[3] = {sz.x * 0.5f, sz.y * 0.5f, sz.z * 0.5f};
        float tmin = -1e30f, tmax = 1e30f; int ax = -1;
        for (int i = 0; i < 3; ++i) {
            const float lo = cc[i] - hh[i], hi = cc[i] + hh[i];
            if (std::fabs(dd[i]) < 1e-8f) { if (oo[i] < lo || oo[i] > hi) return false; continue; }
            float t1 = (lo - oo[i]) / dd[i], t2 = (hi - oo[i]) / dd[i];
            if (t1 > t2) std::swap(t1, t2);
            if (t1 > tmin) { tmin = t1; ax = i; }
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return false;
        }
        if (tmax < 0.f) return false;
        if (tmin < 0.f) { tHit = 0.f; axisHit = -1; return true; }   // camera dentro do bloco
        tHit = tmin; axisHit = ax; return true;
    }
    // Interseção raio x cilindro vertical (marcador do alvo).
    static bool rayVCyl(V3 o, V3 d, float cx, float cz, float r, float y0, float y1, float& tHit) {
        bool hit = false; float best = 1e30f;
        const float ox = o.x - cx, oz = o.z - cz, a = d.x * d.x + d.z * d.z;
        if (a > 1e-9f) {
            const float b = ox * d.x + oz * d.z, c = ox * ox + oz * oz - r * r, disc = b * b - a * c;
            if (disc >= 0.f) {
                const float sq = std::sqrt(disc);
                float t = (-b - sq) / a; if (t < 0.f) t = (-b + sq) / a;
                const float y = o.y + d.y * t;
                if (t >= 0.f && y >= y0 && y <= y1) { best = t; hit = true; }
            }
        }
        if (std::fabs(d.y) > 1e-6f) {                    // tampa de cima
            const float t = (y1 - o.y) / d.y;
            if (t >= 0.f && t < best) {
                const float px = ox + d.x * t, pz = oz + d.z * t;
                if (px * px + pz * pz <= r * r) { best = t; hit = true; }
            }
        }
        tHit = best; return hit;
    }

    // Lança um raio e define cor e profundidade de um pixel da câmera com o que ele atinge (chão, blocos, caixa, alvo).
    void castPixel(V3 o, V3 d, uint8_t* out, float& depthOut) const {
        float bestT = VIS_MAX, base[3] = {30.f, 36.f, 56.f}, shade = 1.f;
        bool emissive = true;                              // ceu
        auto setC = [&](float r, float g, float b) { base[0] = r; base[1] = g; base[2] = b; };
        if (d.y < -1e-5f) {                                // piso quadriculado
            const float t = -o.y / d.y;
            if (t > 0.f && t < bestT) {
                const V3 p = o + d * t;
                const bool chk = (((int)std::floor(p.x) + (int)std::floor(p.z)) & 1) != 0;
                bestT = t; shade = 1.f; emissive = false;
                if (chk) setC(50, 54, 64); else setC(58, 62, 72);
            }
        }
        auto faceShade = [](int ax) { return ax == 0 ? 0.80f : (ax == 1 ? 1.10f : (ax == 2 ? 0.95f : 0.65f)); };
        for (const auto& b : generalBlocks) {
            const float reach = VIS_MAX + blockDiagXZ(b);
            const float dx = b.center.x - o.x, dz = b.center.z - o.z;
            if (dx * dx + dz * dz > reach * reach) continue;
            float t; int ax;
            if (!rayBox(o, d, b.center, b.size, t, ax) || t >= bestT) continue;
            bestT = t; emissive = false; shade = faceShade(ax);
            if (b.step) setC(165, 125, 82); else setC(92, 104, 123);
        }
        if (generalTask == TASK_PUSH) {
            float t; int ax;
            if (rayBox(o, d, boxPos, V3{0.90f, 0.84f, 0.90f}, t, ax) && t < bestT) {
                bestT = t; emissive = false; shade = faceShade(ax); setC(220, 145, 65);
            }
        }
        if (generalTask != TASK_BALANCE) {                 // marcador do alvo: verde, sem sombra
            const V3 m = markerPos(); float t;
            if (rayVCyl(o, d, m.x, m.z, 0.30f, 0.f, 1.5f, t) && t < bestT) { bestT = t; emissive = true; setC(80, 255, 130); }
            if (rayVCyl(o, d, m.x, m.z, 0.42f, 0.f, 0.08f, t) && t < bestT) { bestT = t; emissive = true; setC(80, 255, 130); }
        }
        const float fogC[3] = {24.f, 28.f, 40.f};
        const float k = emissive ? 0.f : 0.35f * std::min(1.f, bestT / VIS_MAX);
        for (int i = 0; i < 3; ++i) {
            const float v = emissive ? base[i] : (base[i] * shade * (1.f - k) + fogC[i] * k);
            out[i] = (uint8_t)clampf(v, 0.f, 255.f);
        }
        depthOut = bestT;
    }

    // Renderiza a imagem da câmera da IA (VIS_W x VIS_H) por ray casting. rear = câmera de trás.
    void renderVision(VisionImage& im, bool rear) const {
        const V3 Fh = visionHeading();
        const V3 F = rear ? Fh * -1.f : Fh;
        const V3 U{0.f, 1.f, 0.f};
        const V3 R = norm(cross(F, U));
        const V3 dim = cr.torsoDimensions();
        // Camera na "cabeca": acima de 1 m, entao o que aparece na linha do horizonte ja e mais alto que 1 m;
        // a caixa (0,84 m) e os degraus ficam abaixo da linha de visao.
        im.eye = {cr.pos.x + F.x * dim.x * 0.5f, std::max(cr.pos.y + dim.y * 0.5f + 0.02f, TALL_OBSTACLE_H + 0.05f),
                  cr.pos.z + F.z * dim.x * 0.5f};
        im.fwd = F; im.right = R;
        const V3* T = camTable();
        for (int i = 0; i < VIS_W * VIS_H; ++i) {
            const V3 t = T[i];
            castPixel(im.eye, F * t.x + U * t.y + R * t.z, &im.rgb[3 * i], im.depth[i]);
        }
        im.valid = true;
    }

    static bool isGoalPx(const uint8_t* c) { return c[1] >= 200 && c[0] * 10 <= c[1] * 6 && c[2] * 10 <= c[1] * 7; }
    static bool isBoxPx(const uint8_t* c)  { return c[0] > 90 && c[0] * 4 > c[1] * 5 && c[1] * 4 > c[2] * 5 && c[2] < 140; }

    // Le a imagem: por coluna, distancia horizontal ao obstaculo mais proximo (qualquer
    // altura) e ao objeto ALTO (> 1 m; altura medida pela profundidade de cada pixel).
    void analyzeImage(const VisionImage& im, int cam) {
        const V3* T = camTable();
        for (int u = 0; u < VIS_W; ++u) colObs[cam][u] = colTall[cam][u] = 999.f;
        for (int r = 0; r < VIS_H; ++r)
            for (int u = 0; u < VIS_W; ++u) {
                const int i = r * VIS_W + u;
                const float t = im.depth[i];
                if (t >= VIS_MAX - 0.01f) continue;                  // ceu / nada
                const V3 cd = T[i];
                const float h = im.eye.y + t * cd.y;                   // altura do ponto visto
                if (h < 0.05f || isGoalPx(&im.rgb[3 * i])) continue;   // piso e marcador nao sao obstaculos
                const float hr = t * std::sqrt(cd.x * cd.x + cd.z * cd.z);
                colObs[cam][u] = std::min(colObs[cam][u], hr);
                if (h > TALL_OBSTACLE_H) colTall[cam][u] = std::min(colTall[cam][u], hr);
            }
    }
    float obsRangeAt(int cam, float bearing, float half) const {
        const float f = camFocal(); float m = 999.f;
        for (int u = 0; u < VIS_W; ++u) {
            const float a = std::atan((u + 0.5f - VIS_W * 0.5f) / f);
            if (std::fabs(a - bearing) <= half) m = std::min(m, colObs[cam][u]);
        }
        return m;
    }

    // Uma "olhada": gera as duas imagens, le distancias e procura o alvo nos pixels.
    // Percepção: renderiza as duas câmeras, lê as distâncias de obstáculos nos pixels e procura o alvo/caixa;
    // o que foi visto fica numa memória visual curta (VIS_MEMORY).
    void perceive() {
        { const V3 h{cr.ax.x, 0.f, cr.ax.z}; if (len(h) > 0.2f) lastHeading = norm(h); }
        renderVision(visFront, false); renderVision(visRear, true);
        analyzeImage(visFront, 0); analyzeImage(visRear, 1);
        const float hx = cr.torsoDimensions().x * 0.5f;
        pFront = std::min(5.f, obsRangeAt(0, 0.f, 0.105f) + hx);
        pL = std::min(5.f, obsRangeAt(0, 0.611f, 0.12f) + hx);    // diagonal +az
        pR = std::min(5.f, obsRangeAt(0, -0.611f, 0.12f) + hx);   // diagonal -az
        const V3* T = camTable();
        V3 gSum{}, bSum{}; int gN = 0, bN = 0;
        for (int cam = 0; cam < 2; ++cam) {
            const VisionImage& im = cam == 0 ? visFront : visRear;
            for (int i = 0; i < VIS_W * VIS_H; ++i) {
                const uint8_t* c = &im.rgb[3 * i];
                const bool g = isGoalPx(c), bx = generalTask == TASK_PUSH && isBoxPx(c);
                if (!g && !bx) continue;
                const float t = im.depth[i];
                if (t >= VIS_MAX - 0.01f) continue;
                const V3 cd = T[i];
                const V3 p = im.eye + (im.fwd * cd.x + im.right * cd.z) * t;
                if (g) { gSum = gSum + p; ++gN; } else { bSum = bSum + p; ++bN; }
            }
        }
        auto update = [&](SeenObj& o, V3 sum, int n, float centerOffset) {
            if (n > 0) {
                V3 p = sum * (1.f / (float)n);
                V3 dirH = p - cr.pos; dirH.y = 0.f; dirH = norm(dirH);
                o.world = p + dirH * centerOffset;       // so se ve a face da frente: estima o centro
                o.age = 0.f;
            } else o.age += CTRL_DT;
            o.seen = o.age <= VIS_MEMORY;
        };
        update(seenGoal, gSum, gN, 0.30f);
        update(seenBox, bSum, bN, 0.45f);
        // Memoria visual curta (3 s) das paredes altas vistas: as laterais do corpo nao
        // aparecem nas cameras, mas uma parede que acabou de passar continua lembrada.
        visitTimer += CTRL_DT;
        if (visitTimer >= 0.5f) { visitTimer = 0.f; visited.push_back(cr.pos); if (visited.size() > 240) visited.erase(visited.begin()); }
        for (auto& m : wallMem) m.age += CTRL_DT;
        wallMem.erase(std::remove_if(wallMem.begin(), wallMem.end(), [](const WallMemPt& m) { return m.age > 3.f; }), wallMem.end());
        for (int cam = 0; cam < 2; ++cam) {
            const VisionImage& im = cam == 0 ? visFront : visRear;
            for (int u = 0; u < VIS_W; ++u) {
                const float R = colTall[cam][u];
                if (R >= 999.f) continue;
                const V3 t = T[(VIS_H / 2) * VIS_W + u];
                const float hl = std::sqrt(t.x * t.x + t.z * t.z);
                const V3 p = im.eye + (im.fwd * t.x + im.right * t.z) * (R / hl);
                bool merged = false;
                for (auto& m : wallMem) {
                    const float dx = m.p.x - p.x, dz = m.p.z - p.z;
                    if (dx * dx + dz * dz < 0.0144f) { m.age = 0.f; merged = true; break; }
                }
                if (!merged && wallMem.size() < 1500) wallMem.push_back({p, 0.f});
            }
        }
    }
    bool goalKnown() const { return seenGoal.age <= VIS_MEMORY; }
    bool boxKnown() const { return seenBox.age <= VIS_MEMORY; }
    // Alvo conforme o que a IA viu (ou lembra ter visto). false = ainda nao viu o alvo.
    bool visualTarget(bool pushing, V3& target) const {
        if (generalTask == TASK_PUSH && pushing) {
            if (boxKnown() && distXZ(cr.pos, seenBox.world) > 0.98f) { target = seenBox.world; return true; }
            if (goalKnown()) { target = seenGoal.world; return true; }
            if (boxKnown()) { target = seenBox.world; return true; }
            return false;
        }
        if (goalKnown()) { target = seenGoal.world; return true; }
        return false;
    }

    // Distancia livre (m, a partir do olho) ate uma parede ALTA na direcao `bearing`
    // (radianos, 0 = frente da camera), lida na imagem, considerando a largura do corpo.
    float freeRange(int cam, float bearing) const {
        const float f = camFocal();
        const int uc = std::clamp((int)std::floor(VIS_W * 0.5f + f * std::tan(bearing)), 0, VIS_W - 1);
        const float d0 = std::min(colTall[cam][uc], 4.f);
        const float alpha = std::atan(0.45f / std::max(d0, 0.6f));      // meia largura do corpo nessa distancia
        float m = 999.f;
        for (int u = 0; u < VIS_W; ++u)
            if (std::fabs(std::atan((u + 0.5f - VIS_W * 0.5f) / f) - bearing) <= alpha) m = std::min(m, colTall[cam][u]);
        return m;
    }
    // Folga em METROS na direcao `dir`: ate onde a IA anda antes de uma parede alta (vista
    // agora ou lembrada) chegar a menos de `halfW` do seu eixo.
    float freeAlong(V3 dir, float halfW) const {
        float m = VIS_MAX;
        for (const auto& w : wallMem) {
            const float dx = w.p.x - cr.pos.x, dz = w.p.z - cr.pos.z;
            const float proj = dx * dir.x + dz * dir.z;
            if (proj <= 0.f || proj >= m) continue;
            if (std::fabs(dx * dir.z - dz * dir.x) < halfW) m = proj;
        }
        return m;
    }
    float visitedDensity(V3 p) const {
        int n = 0;
        for (const auto& v : visited) { const float dx = v.x - p.x, dz = v.z - p.z; if (dx * dx + dz * dz < 1.44f) ++n; }
        return std::min(2.f, (float)n * 0.25f);
    }

    // PLANEJADOR VISUAL: escolhe o rumo olhando as duas imagens. So considera rumos com
    // espaco livre (sem parede alta) para o corpo passar. Com o alvo a vista, prefere o rumo
    // que mais o aproxima; sem alvo, prefere vao longo/novo (explorar). Mantem a decisao
    // anterior para nao oscilar e nunca aponta contra uma parede.
    V3 steerByVision(bool known, V3 target, float sideHint) const {
        lastDirAge += CTRL_DT;
        if (stallSeconds > STALL_FREE && lastDirAge < 1.5f) return lastDir;     // parada demais: assume o rumo e anda
        if (commitT > 0.f) {                                                    // mantem o rumo escolhido enquanto estiver livre
            commitT -= CTRL_DT;
            if (freeAlong(commitDir, 0.45f) >= 1.0f) { lastDir = commitDir; lastDirAge = 0.f; return commitDir; }
            commitT = 0.f;
        }
        const V3 ref = lastDirAge < 2.f ? lastDir : lastHeading;
        V3 tgtXZ = target; tgtXZ.y = 0.f;
        const float dG = known ? distXZ(cr.pos, target) : 0.f;
        float bestScore = -1e9f; V3 best = lastHeading; bool any = false;
        for (int cam = 0; cam < 2; ++cam) {
            const VisionImage& im = cam == 0 ? visFront : visRear;
            if (!im.valid) continue;
            for (int deg = -56; deg <= 56; deg += 4) {
                const float th = (float)deg * 0.0174533f;
                const V3 dir = norm(im.fwd * std::cos(th) + im.right * std::sin(th));
                const float free = freeAlong(dir, 0.45f);                      // folga p/ o corpo passar
                if (free < 1.0f) continue;                                     // sem passagem
                const float freeWide = freeAlong(dir, 0.90f);                  // folga larga (prefere o meio do vao)
                const float reach = std::min(free - 0.4f, known ? std::min(3.f, dG) : 3.f);
                const V3 pEnd = cr.pos + dir * reach;
                float score;
                if (known) {
                    V3 d = pEnd - tgtXZ; d.y = 0.f;
                    score = -len(d) - 0.5f * (std::min(free, 3.f) - std::min(freeWide, 3.f));
                } else {
                    score = 0.2f * std::min(freeWide, 8.f) + 1.0f * dot(dir, ref);
                }
                score -= 3.0f * visitedDensity(pEnd);
                if (lastDirAge < 2.f) score += 1.0f * dot(dir, lastDir);
                score += 0.25f * sideHint * dot(dir, cr.az);                  // intencao de desvio do seletor
                if (cam == 1) score -= 0.6f;                                  // virar de costas custa
                if (score > bestScore) { bestScore = score; best = dir; any = true; }
            }
        }
        if (!any) {   // encurralada: gira para o lado mais aberto da imagem ate achar passagem
            float l = 0.f, r = 0.f;
            for (int u = 0; u < VIS_W; ++u) (u < VIS_W / 2 ? l : r) += std::min(colTall[0][u], 8.f);
            const float turn = (lastDirAge < 1.f ? dot(lastDir, cr.az) : (r - l)) >= 0.f ? 1.f : -1.f;
            best = norm(lastHeading * 0.35f + cr.az * turn);
        }
        if (dot(best, lastDir) < 0.85f || lastDirAge >= 2.f) { commitDir = best; commitT = 1.0f; }   // rumo novo: compromete ~1 s
        lastDir = best; lastDirAge = 0.f;
        return best;
    }

    float obstacleDistance(V3 dir) const {
        dir.y = 0.f;
        dir = norm(dir);
        if (len(dir) < 1e-6f) return 1.f;
        float bestDist = 5.f;
        for (const auto& b : generalBlocks) {
            if (!blockNearXZ(b, cr.pos, 6.0f)) continue;
            bestDist = std::min(bestDist, rayRectDistance(cr.pos, dir, b.center, b.size.x, b.size.z));
        }
        if (generalTask == TASK_PUSH) {
            bestDist = std::min(bestDist, rayRectDistance(cr.pos, dir, boxPos, 0.90f, 0.90f));
        }
        return bestDist;
    }

    // ------------------------------------------------------------------
    //  COLISAO DO CORPO INTEIRO COM PAREDES
    //  Antes so o tronco (uma caixa sem rotacao) colidia; patas, joelhos e pes
    //  atravessavam as paredes. Agora o tronco (caixa orientada) e cada segmento
    //  das patas viram esferas de contato testadas contra todos os blocos solidos.
    // ------------------------------------------------------------------
    struct BodyPoint { V3 p; float r; };

    static float blockTop(const WorldBlock& b) { return b.center.y + b.size.y * 0.5f; }
    // Objeto alto (> 1 m): nunca e terreno, e parede. Deve ser evitado.
    static bool blockIsTall(const WorldBlock& b) { return blockTop(b) > TALL_OBSTACLE_H; }
    // Degraus baixos sao terreno (as patas sobem); todo o resto e solido.
    static bool blockIsSolidWall(const WorldBlock& b) { return !b.step || blockIsTall(b); }

    int collectBodyPoints(BodyPoint* out) const {
        int n = 0;
        const V3 dim = cr.torsoDimensions();
        const float hx = dim.x * 0.5f, hy = dim.y * 0.5f, hz = dim.z * 0.5f;
        // Tronco: 5 (frente->tras) x 2 (lados) x 2 (cima/baixo) esferas, acompanhando a rotacao do corpo.
        const float fxs[5] = {-1.f, -0.5f, 0.f, 0.5f, 1.f};
        for (int i = 0; i < 5; ++i)
            for (int sz = -1; sz <= 1; sz += 2)
                for (int sy = -1; sy <= 1; sy += 2)
                    out[n++] = {cr.pos + cr.toWorld(V3{fxs[i] * hx, (float)sy * hy * 0.9f, (float)sz * hz}), 0.07f};
        // Patas: meio da coxa, joelho, meio da canela e pe (biped: calcanhar e dedos tambem).
        for (int l = 0; l < cr.limbCount(); ++l) {
            V3 knee, foot; cr.legPoints(l, knee, foot, false);
            const V3 hip = cr.hipLocal(l);
            out[n++] = {cr.pos + cr.toWorld((hip + knee) * 0.5f), 0.06f};
            out[n++] = {cr.pos + cr.toWorld(knee), 0.06f};
            out[n++] = {cr.pos + cr.toWorld((knee + foot) * 0.5f), 0.055f};
            out[n++] = {cr.pos + cr.toWorld(foot), cr.biped ? 0.05f : FOOT_COLLISION_RADIUS};
            if (cr.biped) {
                V3 heel, mid, toe; cr.bipedFootPoints(l, heel, mid, toe, false);
                out[n++] = {cr.pos + cr.toWorld(heel), 0.04f};
                out[n++] = {cr.pos + cr.toWorld(toe), 0.04f};
            }
        }
        return n;
    }

    // Esfera (centro p, raio r) contra a caixa do bloco no plano XZ.
    static bool wallContact(const WorldBlock& b, V3 p, float r, V3& normal, float& pen) {
        const float top = blockTop(b), bottom = b.center.y - b.size.y * 0.5f;
        if (p.y - r >= top || p.y + r <= bottom) return false;
        const float hx = b.size.x * 0.5f, hz = b.size.z * 0.5f;
        const float dx = p.x - b.center.x, dz = p.z - b.center.z;
        const float ex = dx - clampf(dx, -hx, hx), ez = dz - clampf(dz, -hz, hz);
        const float d2 = ex * ex + ez * ez;
        if (d2 > 1e-10f) {
            const float d = std::sqrt(d2);
            if (d >= r) return false;
            normal = {ex / d, 0.f, ez / d};
            pen = r - d;
            return true;
        }
        // centro dentro do bloco: sai pela face mais proxima
        const float px = hx - std::fabs(dx), pz = hz - std::fabs(dz);
        if (px < pz) { normal = {dx >= 0.f ? 1.f : -1.f, 0.f, 0.f}; pen = px + r; }
        else         { normal = {0.f, 0.f, dz >= 0.f ? 1.f : -1.f}; pen = pz + r; }
        return true;
    }

    // Passo de posicao: depois da fisica, nenhuma parte do corpo pode ficar
    // penetrada na parede (alem de ~6 mm). Remove tambem a velocidade que
    // empurraria contra a parede, entao a IA nunca "empurra" um objeto alto.
    // Empurra o corpo para fora de paredes/caixa que ele tenha atravessado (correção de posição).
    void resolveWallPenetration() {
        BodyPoint pts[64];
        for (int iter = 0; iter < 4; ++iter) {
            const int np = collectBodyPoints(pts);
            float worst = 0.006f; V3 wn{}; bool found = false;
            for (const auto& b : generalBlocks) {
                if (!blockIsSolidWall(b) || !blockNearXZ(b, cr.pos, WORLD_COLLISION_RADIUS)) continue;
                for (int i = 0; i < np; ++i) {
                    V3 n; float pen = 0.f;
                    if (wallContact(b, pts[i].p, pts[i].r, n, pen) && pen > worst) { worst = pen; wn = n; found = true; }
                }
            }
            if (!found) break;
            cr.pos = cr.pos + wn * (worst - 0.002f);
            const float vn = dot(cr.vel, wn);
            if (vn < 0.f) cr.vel = cr.vel - wn * vn;
        }
    }

    // ------------------------------------------------------------------
    //  DESVIO DE OBJETOS ALTOS (> 1 m)
    //  Ajusta somente a DIRECAO entregue a memoria motora (x[10], x[11]):
    //  nenhum peso, Q-table, formato de arquivo ou sensor da rede foi alterado.
    //  Longe de paredes devolve exatamente a direcao original.
    // ------------------------------------------------------------------
    V3 avoidTallObstacles(V3 direction) const {
        // Rede de seguranca: se alguma parede alta (vista agora ou lembrada) ficou perto
        // demais do corpo, afasta o rumo dela. O planejador visual ja evita rumos bloqueados.
        direction.y = 0.f;
        if (len(direction) < 1e-6f) return direction;
        // O planejador ja escolhe rumos com folga para o corpo; so age se ha colisao iminente nesse rumo.
        if (freeAlong(norm(direction), 0.40f) >= 0.7f) return direction;
        const float speed = std::sqrt(cr.vel.x * cr.vel.x + cr.vel.z * cr.vel.z);
        const float range = 0.95f + 0.2f * std::min(speed, 3.f);
        const float inner = 0.45f;
        V3 nSum{}; float wMax = 0.f;
        for (const auto& m : wallMem) {
            V3 delta = cr.pos - m.p; delta.y = 0.f;
            const float d = len(delta);
            if (d >= range) continue;
            const V3 n = d > 1e-5f ? delta * (1.f / d) : cr.ax * -1.f;
            const float t = clampf((range - d) / (range - inner), 0.f, 1.f);
            const float w = t * t * (3.f - 2.f * t);
            nSum = nSum + n * w;
            wMax = std::max(wMax, w);
        }
        if (wMax <= 0.f || len(nSum) < 1e-6f) return direction;
        const V3 nAvg = norm(nSum);
        V3 steer = direction;
        const float into = dot(steer, nAvg);
        if (into < 0.f) steer = steer - nAvg * (into * std::min(1.f, 2.f * wMax));     // tira a parte que aponta contra a parede
        const V3 out = steer + nAvg * (0.9f * wMax);
        return len(out) > 1e-6f ? norm(out) : direction;
    }

    V3 generalContactForce(float dt, V3& torque) {
        V3 total{};
        torque = {};
        const V3 dim = cr.torsoDimensions();
        const float hx = dim.x * 0.5f, hz = dim.z * 0.5f, hy = dim.y * 0.5f;
        {   // Paredes: forca em cada ponto do corpo (gera tambem torque).
            const V3 wW = cr.toWorld(cr.wb);
            BodyPoint pts[64];
            const int np = collectBodyPoints(pts);
            for (const auto& b : generalBlocks) {
                if (!blockIsSolidWall(b) || !blockNearXZ(b, cr.pos, WORLD_COLLISION_RADIUS)) continue; // degraus baixos sao terreno para as patas
                for (int i = 0; i < np; ++i) {
                    V3 normal; float pen = 0.f;
                    if (!wallContact(b, pts[i].p, pts[i].r, normal, pen)) continue;
                    const V3 rr = pts[i].p - cr.pos;
                    const float vn = dot(cr.vel + cross(wW, rr), normal);
                    const float magnitude = clampf(1800.f * pen - 100.f * vn, 0.f, MAX_FORCE_N);
                    const V3 f = normal * magnitude;
                    total = total + f;
                    torque = torque + cross(rr, f);
                }
            }
        }

        // Bloco movel: a forca de contacto e aplicada ao corpo e, com sinal
        // contrario, ao bloco. Nao existe uma animacao/script de empurrar.
        if (generalTask == TASK_PUSH) {
            const V3 blockSize{0.90f, 0.84f, 0.90f};
            if (cr.pos.y - hy < boxPos.y + blockSize.y * 0.5f && cr.pos.y + hy > 0.f) {
                const float dx = cr.pos.x - boxPos.x, dz = cr.pos.z - boxPos.z;
                const float ox = blockSize.x * 0.5f + hx - std::fabs(dx);
                const float oz = blockSize.z * 0.5f + hz - std::fabs(dz);
                if (ox > 0.f && oz > 0.f) {
                    V3 normal{}; float penetration = 0.f;
                    if (ox < oz) { normal = {dx >= 0.f ? 1.f : -1.f, 0.f, 0.f}; penetration = ox; }
                    else          { normal = {0.f, 0.f, dz >= 0.f ? 1.f : -1.f}; penetration = oz; }
                    const V3 rel = cr.vel - boxVel;
                    const float vn = dot(rel, normal);
                    const float magnitude = clampf(1400.f * penetration - 70.f * vn, 0.f, MAX_FORCE_N);
                    const V3 reaction = normal * (-magnitude);
                    total = total + normal * magnitude;
                    boxVel = boxVel + reaction * (dt / 2.4f);
                }
            }
            // Atrito e limites da pista para o objeto fisico nao fugir do cenario.
            boxVel = boxVel * (1.f / (1.f + 1.7f * dt));
            boxPos = boxPos + boxVel * dt;
            boxPos.x = clampf(boxPos.x, arenaOrigin.x - 7.2f, arenaOrigin.x + 7.2f);
            boxPos.z = clampf(boxPos.z, arenaOrigin.z - 1.12f, arenaOrigin.z + 1.12f);
            boxPos.y = 0.42f;
        }
        // A força resultante sobre a IA nunca ultrapassa o teto definido.
        const float totalMag = len(total);
        if (totalMag > MAX_FORCE_N && totalMag > 1e-6f) {
            const float scale = MAX_FORCE_N / totalMag;
            total = total * scale;
            torque = torque * scale;
        }
        return total;
    }

    // Começa um episódio novo: reposiciona o corpo, monta o cenário da tarefa e zera os contadores.
    void resetEpisode() {
        if (balanceMode) resetBalanceSphere();
        // No parkour, o corredor e a orbe ficam em +X do mundo; iniciar com yaw=PI aponta a frente
        // real (-X local) para o percurso. Isso muda apenas a orientação inicial, nunca a geometria.
        const float initialYaw = parkourMode ? (0.5f * kTwoPi)
                              : ((balanceMode || generalMode) ? 0.f : rnd01() * kTwoPi);
        cr.reset(initialYaw);
        if (balanceMode) cr.pos = {spherePos.x, spherePos.y + sphereRadius + bodyClearance(), spherePos.z};
        if (parkourMode) {
            setupParkourEpisode();
        } else if (generalMode) {
            setupGeneralEpisode();
            if (generalTask == TASK_BALANCE) {
                resetBalanceSphere();
                cr.reset(0.f);
                cr.pos = {spherePos.x, spherePos.y + sphereRadius + bodyClearance(), spherePos.z};
            }
        } else if (!balanceMode) for (int i = 0; i < N_ORBS; i++) spawnOrb(i, cr.pos);
        epSteps = 0; epOrbs = 0; epReturn = 0; orbTimer = 0; sub = 0; trail.clear();
        parkourFallSeconds = 0.f;
        stableSeconds = 0.f;
        balanceQuality = 0.f;
        offSphereSeconds = 0.f;
        generalFallSeconds = 0.f;
        stallSeconds = 0.f; lastDirAge = 99.f; commitT = 0.f; visited.clear(); visitTimer = 0.f;
        seenGoal = SeenObj{}; seenBox = SeenObj{}; visFront.valid = visRear.valid = false;
        pFront = pL = pR = 5.f; wallMem.clear();
        generalRewardWindow = 0.f;
        generalActionAge = 0; generalHasAction = false;
        generalPrevState = generalPrevAction = -1;
        sampleValid = false;
    }
    // Índice da orbe mais próxima (no plano horizontal).
    int nearestOrb() const {
        int b = 0; float bd = 1e30f;
        for (int i = 0; i < N_ORBS; i++) {
            float dx = orbs[i].p.x - cr.pos.x, dz = orbs[i].p.z - cr.pos.z, d = dx * dx + dz * dz;
            if (d < bd) { bd = d; b = i; }
        }
        return b;
    }

    // Monta o vetor de entrada da rede (N_IN valores): velocidade e rotação no referencial do corpo,
    // inclinação, altura, direção/distância do alvo, juntas atuais, contato dos pés e 4 sinais finais
    // que dependem da tarefa (relógio de marcha, radar de obstáculos ou movimento da esfera).
    void buildObs(float* x) const {
        const Creature& c = cr;
        V3 sensedVelocity = balanceSurfaceEpisode() ? (c.vel - sphereVel) : c.vel;
        V3 vb = c.toBody(sensedVelocity);
        x[0] = vb.x / 1.5f; x[1] = vb.y / 1.5f; x[2] = vb.z / 1.5f;
        x[3] = c.wb.x / 3.f; x[4] = c.wb.y / 3.f; x[5] = c.wb.z / 3.f;
        x[6] = c.ax.y; x[7] = c.ay.y; x[8] = c.az.y;                 // "cima" visto do corpo
        if (balanceSurfaceEpisode()) {
            const V3 fromSphere = c.pos - spherePos;
            const float radialDistance = len(fromSphere);
            x[9] = clampf((radialDistance - (sphereRadius + bodyClearance())) / 0.75f, -1.5f, 1.5f);
            const V3 toCenter = radialDistance > 1e-6f ? (spherePos - c.pos) * (1.f / radialDistance) : V3{};
            const V3 centerInBody = c.toBody(toCenter);
            x[10] = centerInBody.x;
            x[11] = centerInBody.y;
            x[12] = centerInBody.z;
        } else if (parkourMode) {
            x[9] = clampf((c.pos.y - groundHeight(c.pos.x, c.pos.z) - 0.66f) * 3.f, -1.5f, 1.5f);
            V3 d = parkourGoal - c.pos; d.y = 0.f;
            const float dist = len(d);
            const V3 db = c.toBody(dist > 1e-6f ? d * (1.f / dist) : V3{});
            x[10] = db.x; x[11] = db.z; x[12] = clampf(dist / 6.f, 0.f, 1.5f);
        } else if (generalMode) {
            x[9] = clampf((c.pos.y - groundHeight(c.pos.x, c.pos.z) - 0.66f) * 3.f, -1.5f, 1.5f);
            V3 d = generalGoal - c.pos; d.y = 0.f;
            const float dist = len(d);
            const V3 db = c.toBody(dist > 1e-6f ? d * (1.f / dist) : V3{});
            x[10] = db.x; x[11] = db.z; x[12] = clampf(dist / 6.f, 0.f, 1.5f);
        } else {
            x[9] = (c.pos.y - 0.66f) * 3.f;
            V3 d = orbs[nearestOrb()].p - c.pos; d.y = 0;
            float dist = len(d); V3 db = c.toBody(dist > 1e-6f ? d * (1.f / dist) : V3{});
            x[10] = db.x; x[11] = db.z; x[12] = clampf(dist / 6.f, 0.f, 1.5f);
        }
        for (int i = 0; i < N_ACT; i++) x[13 + i] = c.q[i];
        for (int l = 0; l < 4; l++) {
            // No equilíbrio, fornece pressão de contato graduada em vez de um indicador binário.
            // Assim a política distingue um toque leve de um apoio firme.
            x[25 + l] = balanceSurfaceEpisode() ? clampf(c.fn[l] / 40.f, 0.f, 1.5f)
                                    : (c.contact[l] ? 1.f : 0.f);
        }
        if (balanceSurfaceEpisode()) {
            V3 platformMotion = c.toBody(sphereVel);
            x[29] = clampf(platformMotion.x / 1.5f, -1.f, 1.f);
            x[30] = clampf(platformMotion.z / 1.5f, -1.f, 1.f);
            x[31] = clampf(platformMotion.y / 1.0f, -1.f, 1.f);
            x[32] = clampf((len(c.pos - spherePos) - sphereRadius - bodyClearance()) / 0.75f, -1.f, 1.f);
        } else if (generalMode) {
            // Quatro sensores direcionais, tipo distancia visual/radar: frente,
            // lado do corpo, lado oposto e traseira. 1 = longe/sem obstaculo.
            x[29] = obstacleDistance(c.ax) / 5.f;
            x[30] = obstacleDistance(c.az) / 5.f;
            x[31] = obstacleDistance(c.az * -1.f) / 5.f;
            x[32] = obstacleDistance(c.ax * -1.f) / 5.f;
        } else {
            // Preserva as quatro entradas de relógio de marcha aprendidas, na caminhada E no parkour.
            // Trocar esses sinais periódicos por distâncias de plataforma tira da política de
            // caminhada as pistas de cadência e pode deixar a marcha rígida.
            x[29] = std::sin(kTwoPi * 1.5f * simT); x[30] = std::cos(kTwoPi * 1.5f * simT);
            x[31] = std::sin(kTwoPi * 3.0f * simT); x[32] = std::cos(kTwoPi * 3.0f * simT);
        }
    }

    // Intensidade do ruído de exploração: decai com os passos de treino e é multiplicada por sigmaMul (diversidade do enxame).
    float currentSigma() const {
        const float decay = std::exp(-(float)brain.steps / 200000.f);
        // No equilíbrio, mantém o ruído da política menor que a amplitude do antigo ruído OU.
        // O ruído é amostrado de forma independente em decide(), como exige o gradiente do score;
        // os servos das juntas fazem a suavização física. A caminhada mantém o cronograma antigo.
        return sigmaMul * (balanceSurfaceEpisode() ? (0.28f + 0.08f * decay)
                                                    : (0.10f + 0.25f * decay));
    }

    // Uma decisão do cérebro (a cada CTRL_DT): monta a entrada, roda a rede esparsa, aplica o ruído de
    // exploração, define os alvos das juntas (tanh) e guarda a amostra para o aprendizado.
    // No modo geral, antes escolhe a habilidade (meta-controlador) e usa memórias motoras congeladas.
    void decide() {
        Brain* policy = &brain;
        if (generalMode) {
            if (generalTask == TASK_BALANCE) {
                // Usa a memoria de equilibrio somente no episodio fisico de esfera.
                policy = balanceSkillReady ? &balanceSkill : &walkingSkill;
                buildObs(xobs);
            } else {
                // A cada ~0,6 s, o meta-controlador avalia recompensa e escolhe
                // uma intencao; a memoria motora continua comandando as juntas.
                perceive();                       // olha (imagens frente/traseira) antes de decidir
                const float frontNow = pFront;    // distancia a frente lida na imagem
                const bool needsDetour = (generalTask == TASK_MAZE || generalTask == TASK_PUSH)
                                      && frontNow < 0.82f && (generalAction == 0 || generalAction == 3);
                if (!generalHasAction || generalActionAge <= 0 || needsDetour) {
                    const int state = metaState();
                    if (generalHasAction && learning)
                        meta.learn(generalPrevState, generalPrevAction, generalRewardWindow, state, false, generalTask, runningSkillReady);
                    generalRewardWindow = 0.f;
                    generalAction = meta.choose(state, learning, generalTask, runningSkillReady);
                    generalPrevState = state;
                    generalPrevAction = generalAction;
                    generalHasAction = true;
                    generalActionAge = 36; // 1.8 s: uma habilidade macro dura o bastante para mudar a rota
                }
                --generalActionAge;
                policy = (generalAction == 3 && runningSkillReady) ? &runningSkill : &walkingSkill;
                buildGeneralSkillObs(xobs);
            }
        } else {
            buildObs(xobs);
        }

        policy->forward(xobs, act);
        sigma = generalMode ? 0.f : currentSigma();
        for (int k = 0; k < N_ACT; k++) {
            if (k >= cr.activeActionCount()) {
                lastEps[k] = 0.f;
                lastAction[k] = 0.f;
                cr.cmd[k] = 0.f;
                continue;
            }
            // No MODO GERAL, as memorias motoras sao congeladas; exploracao e
            // feita no nivel de escolha de habilidade/rota, nao alterando a marcha.
            const float e = (learning && !generalMode) ? randn() : 0.f;
            lastEps[k] = e;
            lastAction[k] = std::tanh(act.mu[k] + ((learning && !generalMode) ? sigma * e : 0.f));
            cr.cmd[k] = lastAction[k];
        }
        for (int g = 0; g < N_GROUPS; g++) heat[g] *= 0.92f;
        for (int i = 0; i < act.nGroups; i++) {
            heat[act.groups[i]] = 1.f;
            policy->useCount[act.groups[i]]++;
        }
        lastGroups = act.nGroups; lastActive = act.nActive;
        avgActiveFrac += ((float)act.nActive / N_NEURONS - avgActiveFrac) * 0.01f;
        sampleValid = false;
        if (generalMode) return; // nao treina novamente as juntas no MODO GERAL
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

    // Fim de uma decisão: calcula a recompensa da tarefa, trata coleta de orbes, queda e tempo esgotado,
    // encerra o episódio quando preciso e dispara o treino a cada ROLLOUT_T amostras.
    void finish() {
        float energy = 0;
        for (int k = 0; k < N_ACT; k++) energy += lastAction[k] * lastAction[k];

        V3 relativeVelocity = balanceSurfaceEpisode() ? (cr.vel - sphereVel) : cr.vel;
        const float horizontalSpeed = std::sqrt(relativeVelocity.x * relativeVelocity.x + relativeVelocity.z * relativeVelocity.z);
        const float angularSpeed = len(cr.wb);
        // Smooth posture signal in balance mode: a tilted or even inverted body
        // still receives a graded signal that can favour recovery. The old hard
        // cutoff gave zero posture reward as soon as the body tipped too far.
        const float upright = balanceSurfaceEpisode()
            ? clampf((cr.ay.y + 0.50f) / 1.50f, 0.f, 1.f)
            : clampf((cr.ay.y - 0.35f) / 0.65f, 0.f, 1.f);
        const float angularControl = std::exp(-0.8f * angularSpeed);
        const float driftControl = std::exp(-0.55f * horizontalSpeed);
        const float bodyRadialDistance = balanceSurfaceEpisode() ? len(cr.pos - spherePos) : 0.f;
        const float bodyGap = balanceSurfaceEpisode() ? (bodyRadialDistance - sphereRadius) : 0.f;
        // This describes proximity to the sphere surface from any accessible
        // direction; it is not a target position at the crown/top.
        const float bodySupportQuality = balanceSurfaceEpisode()
            ? std::exp(-1.6f * std::fabs(bodyGap - bodyClearance()))
            : clampf((cr.pos.y - 0.25f) / 0.40f, 0.f, 1.f);
        int footContacts = 0;
        for (int l = 0; l < cr.limbCount(); ++l) if (cr.contact[l]) ++footContacts;
        const float contactRatio = (float)footContacts / (float)cr.limbCount();

        // Reward physical support anywhere on the sphere. There is deliberately
        // no reward for being near the crown and no target foot placement.
        float supportForceQuality = 0.f;
        if (balanceSurfaceEpisode()) {
            const float expectedLegLoad = MASS * 9.81f / (float)cr.limbCount();
            for (int l = 0; l < cr.limbCount(); ++l)
                supportForceQuality += clampf(cr.fn[l] / expectedLegLoad, 0.f, 1.f);
            supportForceQuality /= (float)cr.limbCount();
        }
        balanceQuality = upright * angularControl * driftControl * bodySupportQuality
                         * (balanceSurfaceEpisode() ? (0.25f + 0.75f * contactRatio) : 1.f);
        const bool stable = cr.ay.y > 0.90f && angularSpeed < 0.90f &&
                            horizontalSpeed < 0.55f && bodySupportQuality > 0.55f &&
                            (!balanceSurfaceEpisode() || footContacts >= 2);

        float reward = 0.f;
        bool taskSuccess = false;
        if (balanceSurfaceEpisode()) {
            stableSeconds = stable ? stableSeconds + CTRL_DT : 0.f;
            bestStableSeconds = std::max(bestStableSeconds, stableSeconds);
            const float streakBonus = stable ? std::min(stableSeconds, 10.f) : 0.f;
            // Recompensa apoio físico real, postura ereta e pouca deriva relativa. Não há
            // ponto preferido na esfera nem pose corretiva.
            const float rewardRate = 0.65f * upright
                                   + 0.45f * angularControl
                                   + 0.40f * driftControl
                                   + 1.20f * bodySupportQuality
                                   + 1.80f * supportForceQuality
                                   + 2.40f * contactRatio
                                   + 0.80f * balanceQuality
                                   + (stable ? 1.0f + 0.12f * streakBonus : 0.f);
            // Um custo de esforço relevante desencoraja agitar as juntas à toa, sem dizer
            // à rede o que cada junta deve fazer.
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
        } else if (parkourMode) {
            stableSeconds = 0.f;
            const float goalDistance = distXZ(cr.pos, parkourGoal);
            const float goalProgress = prevGoalDistance - goalDistance;
            prevGoalDistance = goalDistance;
            V3 goalDirection = parkourGoal - cr.pos; goalDirection.y = 0.f;
            goalDirection = norm(goalDirection);
            const float speedTowardGoal = cr.vel.x * goalDirection.x + cr.vel.z * goalDirection.z;

            // Recompensa progresso rápido para a frente, aterrissagem ereta, apoio dos pés e pouca rotação.
            // Um custo pequeno por decisão e um bônus de conclusão escalado pelo tempo favorecem chegadas rápidas.
            reward = CTRL_DT * (0.65f * upright + 0.45f * angularControl
                                + 0.55f * contactRatio
                                + 0.62f * clampf(speedTowardGoal, 0.f, 3.5f) - 0.10f)
                     - 0.0015f * energy;
            reward += clampf(goalProgress * 3.2f, -0.20f, 0.32f);
            reward -= 0.006f;

            const float finishPlatformTop = 0.18f;
            if (goalDistance < ORB_PICK_R && cr.pos.y > finishPlatformTop + 0.18f && upright > 0.20f) {
                const float speedBonus = 6.f * clampf(1.f - (float)epSteps / (float)MAX_EP_STEPS, 0.f, 1.f);
                reward += 12.f + speedBonus;
                taskSuccess = true;
                epOrbs++;
                parkourCompletions++;
                lastParkourTime = (epSteps + 1) * CTRL_DT;
                if (bestParkourTime <= 0.f || lastParkourTime < bestParkourTime)
                    bestParkourTime = lastParkourTime;
                // O alvo fica no fim. Ao contrário da caminhada normal, não renasce depois de coletado.
            }
        } else if (generalMode) {
            stableSeconds = 0.f;
            const float contactRatioGeneral = (float)footContacts / (float)cr.limbCount();
            const float boxGoalDistance = distXZ(boxPos, pushTarget);
            const float agentBoxDistance = distXZ(cr.pos, boxPos);
            const float goalDistance = distXZ(cr.pos, generalGoal);
            const float goalProgress = prevGoalDistance - goalDistance;
            prevGoalDistance = goalDistance;

            // A modelagem de recompensa incentiva movimento útil e corpo estável, mas não
            // especifica poses de junta, ciclos de marcha, curvas nem rota.
            // O MODO GERAL premia locomocao util, nao apenas estar em pe.
            // Um custo de inatividade impede que a politica meta aprenda que a
            // melhor opcao e ficar parada com postura estavel.
            reward = CTRL_DT * (0.14f * upright + 0.08f * contactRatioGeneral
                                + 0.06f * angularControl - 0.08f * angularSpeed - 0.26f)
                     - 0.0010f * energy;
            reward += clampf(goalProgress * 4.0f, -0.22f, 0.28f);
            if (goalDistance > 1.0f && horizontalSpeed < 0.08f) reward -= CTRL_DT * 0.25f;
            const float frontDistance = obstacleDistance(cr.ax);
            if (frontDistance < 0.70f) reward -= CTRL_DT * 0.65f;
            // Parar um instante (para olhar/achar a passagem) e permitido; ficar parada
            // alem de STALL_FREE nao pode render ponto: perde os bonus de postura e
            // a penalidade cresce, entao a IA nunca aprende que ficar parada compensa.
            {
                const bool stalled = goalDistance > 1.0f && horizontalSpeed < 0.12f;
                stallSeconds = stalled ? stallSeconds + CTRL_DT : std::max(0.f, stallSeconds - 2.f * CTRL_DT);
                if (stallSeconds > STALL_FREE) {
                    const float over = stallSeconds - STALL_FREE;
                    reward -= CTRL_DT * (0.14f * upright + 0.08f * contactRatioGeneral + 0.06f * angularControl);
                    reward -= CTRL_DT * std::min(1.6f, 0.5f + 0.9f * over);
                }
            }
            if (generalTask == TASK_SPEED) {
                reward += CTRL_DT * 0.70f * clampf(horizontalSpeed, 0.f, 3.5f) * upright;
            }
            if (generalTask == TASK_STAIRS) {
                const float height = groundHeight(cr.pos.x, cr.pos.z);
                reward += CTRL_DT * 0.18f * clampf(height / 0.54f, 0.f, 1.f);
            }
            if (generalTask == TASK_PUSH) {
                const float boxProgress = prevBoxGoalDistance - boxGoalDistance;
                const float approachProgress = prevAgentBoxDistance - agentBoxDistance;
                reward += clampf(boxProgress * 7.0f, -0.10f, 0.22f);
                reward += clampf(approachProgress * 0.65f, -0.04f, 0.07f);
                prevBoxGoalDistance = boxGoalDistance;
                prevAgentBoxDistance = agentBoxDistance;
                if (boxGoalDistance < 0.62f) { reward += 10.f; taskSuccess = true; generalSuccesses++; epOrbs++; }
            } else if (goalDistance < 0.70f) {
                reward += 8.f; taskSuccess = true; generalSuccesses++; epOrbs++;
            }
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
        // Não reinicia só porque o tronco tombou: recuperar-se de uma má pose faz parte da tarefa.
        // No equilíbrio, termina só após perder o apoio por tempo suficiente ou se afastar
        // fisicamente / cair abaixo da esfera.
        const bool outsideSphere = balanceSurfaceEpisode() &&
            (bodyRadialDistance > sphereRadius + 2.0f || cr.pos.y < spherePos.y - 0.35f);
        // Encerrar o episodio depois de permanecer caido/de lado permite que a
        // hierarquia receba a penalidade. Antes, um corpo tombado mas ainda alto
        // podia continuar eternamente com uma memoria motora incapaz de recuperar.
        const bool unsafeGeneralPose = cr.pos.y < groundHeight(cr.pos.x, cr.pos.z) + 0.32f || cr.ay.y < 0.35f;
        generalFallSeconds = (generalMode && generalTask != TASK_BALANCE && unsafeGeneralPose)
            ? generalFallSeconds + CTRL_DT : std::max(0.f, generalFallSeconds - 2.f * CTRL_DT);
        const bool generalFall = generalMode && generalTask != TASK_BALANCE && generalFallSeconds > 0.75f;
        const bool unsafeParkourPose = cr.pos.y < 0.16f || cr.ay.y < 0.10f;
        parkourFallSeconds = (parkourMode && unsafeParkourPose)
            ? parkourFallSeconds + CTRL_DT : std::max(0.f, parkourFallSeconds - 2.f * CTRL_DT);
        const bool parkourFall = parkourMode &&
            (cr.pos.y < -0.65f || cr.pos.x < arenaOrigin.x - 11.2f ||
             cr.pos.x > arenaOrigin.x + 11.4f ||
             std::fabs(cr.pos.z - arenaOrigin.z) > 2.65f || parkourFallSeconds > 0.95f);
        bool fell = bad || (parkourMode && parkourFall)
                    || (!parkourMode && !balanceSurfaceEpisode() && !generalMode && (cr.pos.y < 0.30f || cr.ay.y < 0.5f))
                    || generalFall
                    || (balanceSurfaceEpisode() && (offSphereSeconds > 0.55f || outsideSphere || cr.floorContact));
        bool timeout = epSteps >= MAX_EP_STEPS;
        if (fell) reward -= parkourMode ? 5.f : 2.f;
        if (generalMode && generalTask != TASK_BALANCE) generalRewardWindow += reward;
        epReturn += reward;

        if (sampleValid) { buf[nbuf].reward = reward; buf[nbuf].done = fell || timeout || taskSuccess; nbuf++; brain.steps++; dirty = true; }
        if (fell || timeout || taskSuccess) {
            if (generalMode && generalTask != TASK_BALANCE && generalHasAction) {
                if (learning) meta.learn(generalPrevState, generalPrevAction, generalRewardWindow, metaState(), true, generalTask, runningSkillReady);
                generalRewardWindow = 0.f; generalHasAction = false; generalActionAge = 0;
                generalPrevState = generalPrevAction = -1; dirty = true;
            }
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
            // A esfera é uma tarefa muito sensível; usa passos de gradiente menores para a
            // política melhorar aos poucos em vez de desfazer a postura de uma vez.
            brain.train(buf, nbuf, lastV, sigma, balanceSurfaceEpisode() ? 0.75f : 1.f);
            learningUpdates++;
            nbuf = 0;
        }
    }

    // Um passo de física (PHYS_DT). O cérebro decide no início de cada bloco de SUBSTEPS e a recompensa fecha no fim.
    void tick() {
        if (sub == 0) decide();
        const bool sphereEpisode = balanceSurfaceEpisode();
        if (sphereEpisode) updateBalanceSphere(PHYS_DT);
        V3 contactForce{}, contactTorque{};
        const bool wallWorld = (generalMode && generalTask != TASK_BALANCE) || parkourMode;
        if (wallWorld) contactForce = generalContactForce(PHYS_DT, contactTorque);
        cr.physics(PHYS_DT, sphereEpisode, spherePos, sphereVel, sphereRadius, sphereAngularVelocity,
                   wallWorld ? &generalBlocks : nullptr, contactForce, contactTorque, parkourMode);
        if (wallWorld) resolveWallPenetration();
        simT += PHYS_DT;
        if (++sub == SUBSTEPS) { sub = 0; finish(); }
    }
    // Liga/desliga o aprendizado desta IA e descarta o lote em andamento.
    void setLearning(bool on) { learning = on; nbuf = 0; sampleValid = false; }

    // Liga/desliga o modo geral nesta IA. Valida a memória de caminhada ANTES de mudar qualquer estado;
    // se faltar, nada muda e devolve false.
    bool setGeneralMode(bool on, const Brain& seedBrain) {
        if (generalMode == on) return true;

        // Valida a habilidade motora ANTES de mudar os modos. Na versao anterior,
        // o modo era ligado e depois desligado silenciosamente se a memoria falhasse,
        // por isso a tecla G parecia nao fazer nada e podia deixar o modo anterior confuso.
        if (on) {
            loadGeneralSkills(seedBrain);
            if (!walkingSkillReady) {
                std::printf("[MODO GERAL] NAO ATIVADO: falta uma memoria de caminhada treinada (%s).\n",
                            cr.biped ? BIPED_MEM_FILE : MEM_FILE);
                std::printf("Treine a caminhada e salve com S; depois tente G novamente.\n");
                return false;
            }
        }

        generalMode = on;
        if (on) { balanceMode = false; speedRewardMode = false; }
        nbuf = 0; sampleValid = false; learningUpdates = 0;
        recent.clear(); retHist.clear();
        accRet = accOrbs = accLen = 0.0; accN = 0;
        avgOrbsEp = avgLen = lastReturn = 0.f; lastEpOrbs = epOrbs = 0;
        const char* path = memoryFile();
        if (std::filesystem::exists(path) && brain.load(path)) {
            std::printf("[memoria] modo geral carregado: %s\n", path);
        } else {
            brain = seedBrain;
            if (brain.save(path)) std::printf("[memoria] nova memoria para o modo geral: %s\n", path);
        }
        if (on) {
            if (meta.load(metaMemoryFile()))
                std::printf("[MODO GERAL] Q-learning carregado: %s (%llu atualizacoes)\n", metaMemoryFile(), (unsigned long long)meta.updates);
            else
                std::printf("[MODO GERAL] Q-learning novo; as decisoes comecam a ser aprendidas agora.\n");
        }
        dirty = false;
        resetEpisode();
        return true;
    }

    // Troca para o treino de equilíbrio (ou volta à caminhada), carregando a memória correspondente.
    void setBalanceMode(bool on) {
        if (balanceMode == on) return;
        if (on && generalMode) generalMode = false;
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
                // Pesos de saída aleatórios NÃO são um controlador de equilíbrio: só dão a uma
                // política nova uma faixa mensurável de ações para explorar. O agente ainda
                // precisa aprender quais ações rendem recompensa.
                for (float& w : brain.Wout) w = randn() * 0.06f;
            }
            if (brain.save(path)) std::printf("[memoria] nova memoria da tarefa criada: %s\n", path);
        }
        dirty = false;
        resetEpisode();
    }

    // Alterna caminhada e corrida. Se a corrida ainda não tem memória, parte da política de caminhada.
    void setSpeedRewardMode(bool on, const Brain& seedBrain) {
        if (speedRewardMode == on || balanceMode) return;
        if (on && generalMode) generalMode = false;
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

    // Troca a morfologia (quadrúpede/bípede) e carrega a memória dela; uma morfologia nova começa sem treino.
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
            // Uma morfologia nova recebe política sem treino e exploratória. Nenhuma pose
            // de junta nem reflexo de equilíbrio é inserido aqui.
            for (float& w : brain.Wout) w = randn() * 0.06f;
            if (brain.save(path)) std::printf("[memoria] nova memoria morfologica: %s\n", path);
        }
        if (generalMode) loadGeneralSkills(brain);
        dirty = false;
        resetEpisode();
    }
    // Grava a memória desta IA (e a tabela Q, no modo geral).
    bool save() {
        bool ok = brain.save(memoryFile());
        if (generalMode) ok = meta.save(metaMemoryFile()) && ok;
        if (ok) dirty = false;
        return ok;
    }
    // Cérebro cujo uso aparece no mapa da tela (no modo geral, a habilidade em uso).
    const Brain& displayedPolicy() const {
        if (!generalMode) return brain;
        if (generalTask == TASK_BALANCE) return balanceSkillReady ? balanceSkill : walkingSkill;
        return (generalAction == 3 && runningSkillReady) ? runningSkill : walkingSkill;
    }
    int groupsUsed() const { int n = 0; for (auto c : displayedPolicy().useCount) if (c) n++; return n; }
};

// ================================== ENXAME ===================================
static constexpr float NO_SCORE = -1e30f;

// Gerencia as N_AGENTS IAs: alterna os modos em todas, avança a simulação (em threads), ranqueia
// por geração, guarda o melhor cérebro e é quem grava a memória em arquivo.
struct Swarm {
    std::vector<std::unique_ptr<Sim>> ag;     // ag[0] e a IA original; so tem 1 elemento com o aprendizado desligado
    int shown = 0, generation = 0;
    Brain best;                               // cerebro legado/skills para os modos antigos
    MetaController bestMeta;                  // melhor controlador de decisoes do MODO GERAL
    float bestScore = NO_SCORE;               // NO_SCORE = ainda nao verificada por nenhuma geracao
    uint64_t ticks = 0;                       // passos de fisica somados de todas as IAs

    Sim& cur() { return *ag[shown]; }
    const Sim& cur() const { return *ag[shown]; }
    bool learning() const { return ag.size() > 1; }
    bool balanceMode() const { return !ag.empty() && ag[0]->balanceMode; }
    bool speedRewardMode() const { return !ag.empty() && ag[0]->speedRewardMode; }
    bool generalMode() const { return !ag.empty() && ag[0]->generalMode; }
    bool parkourMode() const { return !ag.empty() && ag[0]->parkourMode; }
    bool bipedMode() const { return !ag.empty() && ag[0]->cr.biped; }
    bool dirty() const { for (auto& a : ag) if (a->dirty) return true; return false; }

    // Espalha as arenas de equilíbrio numa grade, para as esferas de IAs diferentes não se cruzarem.
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

    // Dá a cada IA a sua arena do modo geral, afastada das demais.
    void arrangeGeneralArenas() {
        if (ag.empty()) return;
        const int cols = std::min((int)ag.size(), 5);
        const int rows = ((int)ag.size() + cols - 1) / cols;
        for (int i = 0; i < (int)ag.size(); ++i) {
            const int col = i % cols, row = i / cols;
            ag[i]->arenaOrigin = {(col - (cols - 1) * 0.5f) * 20.0f,
                                  0.f,
                                  (row - (rows - 1) * 0.5f) * 18.0f};
        }
    }

    // Dá a cada IA o seu percurso de parkour, afastado dos demais.
    void arrangeParkourArenas() {
        if (ag.empty()) return;
        const int cols = std::min((int)ag.size(), 5);
        const int rows = ((int)ag.size() + cols - 1) / cols;
        for (int i = 0; i < (int)ag.size(); ++i) {
            const int col = i % cols, row = i / cols;
            ag[i]->arenaOrigin = {(col - (cols - 1) * 0.5f) * 25.0f,
                                  0.f,
                                  (row - (rows - 1) * 0.5f) * 7.5f};
        }
    }

    // Liga/desliga o parkour em todas as IAs. Só vale a partir da Caminhada (devolve false se E, C ou G estiverem ativos).
    bool setParkourMode(bool on) {
        if (ag.empty()) return false;
        if (parkourMode() == on) return true;
        // K is strictly a scene/reward switch; other task modes cannot be silently replaced.
        if (on && (generalMode() || balanceMode() || speedRewardMode())) {
            std::printf("[PARKOUR] nao ativado: primeiro volte para o modo caminhada (sem G, E ou C).\n");
            return false;
        }

        if (on) arrangeParkourArenas();
        else for (auto& a : ag) a->arenaOrigin = {};

        for (auto& a : ag) {
            a->parkourMode = on;
            a->nbuf = 0;                 // do not mix rollout samples from two different scenes
            a->sampleValid = false;
            a->generalBlocks.clear();
            a->recent.clear(); a->retHist.clear();
            a->accRet = a->accOrbs = a->accLen = 0.0; a->accN = 0;
            a->resetEpisode();           // preserves network weights, counters, and learning on/off state
        }
        best = ag[0]->brain;
        bestMeta = ag[0]->meta;
        bestScore = NO_SCORE;
        generation = 0;
        std::printf("[PARKOUR] cenario %s; memoria motora existente mantida, sem carregar/criar arquivo.\n",
                    on ? "ativado" : "desativado");
        return true;
    }

    // Liga/desliga o modo geral em todas as IAs; se alguma não conseguir (falta memória de caminhada), desfaz nas anteriores.
    bool setGeneralMode(bool on) {
        if (ag.empty() || parkourMode()) return false;
        if (generalMode() == on) return true;
        Brain seed;
        if (on) {
            // Aceita tambem memorias com episodios registados mas contador de passos
            // antigo/zerado. Nunca substitui uma memoria motora por uma memoria de equilibrio.
            const char* walkPath = bipedMode() ? BIPED_MEM_FILE : MEM_FILE;
            const bool loadedWalk = seed.load(walkPath) && (seed.steps > 0 || seed.episodes > 0);
            const bool activeIsWalk = !balanceMode() && !speedRewardMode()
                                      && (ag[0]->brain.steps > 0 || ag[0]->brain.episodes > 0);
            if (!loadedWalk && !activeIsWalk) {
                std::printf("[MODO GERAL] BLOQUEADO: nao encontrei memoria de caminhada treinada (%s).\n", walkPath);
                std::printf("O modo de equilibrio atual nao serve como memoria de caminhada. Treine/salve caminhada (S) e tente G novamente.\n");
                return false;
            }
            if (!loadedWalk) seed = (learning() && bestScore > NO_SCORE) ? best : ag[0]->brain;
        } else {
            seed = (learning() && bestScore > NO_SCORE) ? best : ag[0]->brain;
        }
        save();
        if (on) arrangeGeneralArenas();
        else for (auto& a : ag) a->arenaOrigin = {};

        // Verifica o resultado real de cada agente. Antes, o metodo interno podia
        // cancelar a ativacao sem avisar e este metodo ainda dizia que tinha sucesso.
        for (auto& a : ag) {
            if (!a->setGeneralMode(on, seed)) {
                // Reverte uma ativacao parcial se uma memoria nao puder ser usada.
                for (auto& prev : ag) {
                    if (prev->generalMode != on) continue;
                    prev->setGeneralMode(!on, seed);
                }
                for (auto& reset : ag) reset->arenaOrigin = {};
                return false;
            }
            a->recent.clear(); a->retHist.clear();
            a->accRet = a->accOrbs = a->accLen = 0.0; a->accN = 0;
        }
        best = ag[0]->brain; bestMeta = ag[0]->meta; bestScore = NO_SCORE; generation = 0;
        std::printf("[curriculo] MODO GERAL %s: decisoes hierarquicas; memorias motoras separadas; equilibrio nao sera sorteado\n",
                    on ? "LIGADO" : "DESLIGADO");
        return generalMode() == on;
    }

    // Aplica o modo de equilíbrio a todas as IAs.
    void setBalanceMode(bool on) {
        if (ag.empty() || parkourMode() || balanceMode() == on) return;
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
        if (ag.empty() || parkourMode() || balanceMode() || speedRewardMode() == on) return;
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
        if (ag.empty() || parkourMode() || bipedMode() == on) return;
        if (generalMode()) {
            const char* neededWalkPath = on ? BIPED_MEM_FILE : MEM_FILE;
            Brain candidate;
            if (!candidate.load(neededWalkPath) || candidate.steps == 0) {
                std::printf("[MODO GERAL] troca de morfologia bloqueada: falta memoria motora treinada (%s).\n",
                            neededWalkPath);
                return;
            }
        }
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
        else if (generalMode()) { arrangeGeneralArenas(); for (auto& a : ag) a->resetEpisode(); }
    }

    // Cria a IA original (carregando a memória) e, se o aprendizado estiver ligado, os clones.
    void init() {
        ag.clear(); ag.emplace_back(new Sim()); ag[0]->init();
        shown = 0; generation = 0;
        if (ag[0]->learning) spawnClones();
    }

    // aprendizado ligado: a IA atual vira N_AGENTS IAs
    void spawnClones() {
        static const float MUL[5] = {1.0f, 0.7f, 1.4f, 0.85f, 1.2f};   // exploracao de cada IA
        best = ag[0]->brain; bestMeta = ag[0]->meta; bestScore = NO_SCORE; generation = 0;       // ponto de partida guardado
        ag[0]->sigmaMul = MUL[0]; ag[0]->rngSeed = g_rng(); ag[0]->recent.clear();
        while ((int)ag.size() < N_AGENTS) {
            int i = (int)ag.size();
            ag.emplace_back(new Sim());
            ag[i]->initClone(ag[0]->brain, MUL[i % 5], g_rng());
            ag[i]->balanceMode = ag[0]->balanceMode; // clones inherit the current task
            ag[i]->speedRewardMode = ag[0]->speedRewardMode;
            ag[i]->generalMode = ag[0]->generalMode;
            ag[i]->parkourMode = ag[0]->parkourMode;
            ag[i]->cr.biped = ag[0]->cr.biped;       // and the selected morphology
            if (ag[0]->generalMode) {
                ag[i]->walkingSkill = ag[0]->walkingSkill;
                ag[i]->runningSkill = ag[0]->runningSkill;
                ag[i]->balanceSkill = ag[0]->balanceSkill;
                ag[i]->walkingSkillReady = ag[0]->walkingSkillReady;
                ag[i]->runningSkillReady = ag[0]->runningSkillReady;
                ag[i]->balanceSkillReady = ag[0]->balanceSkillReady;
                ag[i]->meta = ag[0]->meta;
            }
        }
        if (ag[0]->balanceMode) arrangeBalanceArenas();
        else if (ag[0]->generalMode) arrangeGeneralArenas();
        for (auto& a : ag) a->resetEpisode();
        std::printf("[enxame] %d IAs treinando (original + %d clones)\n", N_AGENTS, N_AGENTS - 1);
    }

    // aprendizado desligado: fica so a melhor IA
    void collapse() {
        const Brain* chosen = (parkourMode() && bestScore <= NO_SCORE) ? &ag[0]->brain : &best;
        float cs = bestScore;
        const MetaController* chosenMeta = &bestMeta;
        const char* why = bestScore > NO_SCORE ? "melhor forma registrada"
            : (parkourMode() ? "politica atual do agente principal" : "ponto de partida (nenhuma geracao completa ainda)");
        for (auto& a : ag) {
            float sc = a->score(FINAL_MIN_EPS);
            if (sc > cs) {
                cs = sc; chosen = &a->brain; chosenMeta = &a->meta;
                why = "IA em treino com o melhor score";
            }
        }
        Sim& s0 = *ag[0];
        if (chosen != &s0.brain) s0.brain = *chosen;      // (copia antes de apagar os clones)
        if (generalMode() && chosenMeta != &s0.meta) s0.meta = *chosenMeta;
        if (cs > NO_SCORE) std::printf("[enxame] ficou 1 IA: %s (score %.1f)\n", why, cs);
        else               std::printf("[enxame] ficou 1 IA: %s\n", why);
        ag.erase(ag.begin() + 1, ag.end());
        shown = 0;
        s0.setLearning(false); s0.recent.clear(); s0.sigmaMul = 1.f; s0.resetEpisode();
        best = Brain(); bestMeta = s0.meta; bestScore = NO_SCORE;
        s0.dirty = true; save();
    }

    void setLearning(bool on) {
        if (on == learning()) return;
        if (on) { ag[0]->setLearning(true); spawnClones(); }
        else collapse();
    }

    // Roda n passos de uma IA usando o gerador aleatório dela (e restaura o da thread que chamou).
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
        if (improved) {
            best = ag[b]->brain;
            if (generalMode()) bestMeta = ag[b]->meta;
            bestScore = sb;
        }
        std::printf("[enxame] geracao %d: melhor IA #%d score %.1f | pior %.1f | recorde %.1f%s\n",
                    generation, b, sb, sw, bestScore, improved ? " (novo)" : "");
        for (int k = (int)ord.size() - N_REPLACE; k < (int)ord.size(); k++)
            ag[ord[k]]->adopt(ag[b]->brain, generalMode() ? &ag[b]->meta : nullptr);
        for (auto& a : ag) a->recent.clear();
        for (auto& a : ag) a->dirty = true;
    }

    // grava no arquivo SEMPRE a melhor forma (nunca uma IA no meio de um tropeco do treino)
    bool save() {
        if (ag.empty()) return false;
        const char* path = cur().memoryFile();
        // Before the first complete parkour generation, persist the actively trained
        // primary Brain rather than the untouched generation seed. The file path/format stay the same.
        const Brain& brainToSave = (ag.size() > 1 && parkourMode() && bestScore <= NO_SCORE)
            ? ag[0]->brain : ((ag.size() > 1) ? best : ag[0]->brain);
        bool ok = brainToSave.save(path);
        if (generalMode()) {
            const MetaController& policy = (ag.size() > 1 && bestScore > NO_SCORE) ? bestMeta : ag[0]->meta;
            ok = policy.save(cur().metaMemoryFile()) && ok;
        }
        if (ok) for (auto& a : ag) a->dirty = false;
        return ok;
    }
};

// ================================ DESENHO ====================================
// Câmera 3D: yaw/pitch/dist orbitam em torno de 'target'; rearFollow = segue por trás do corpo.
struct View { float yaw = 0.9f, pitch = 0.5f, dist = 7.5f; V3 target; bool rearFollow = true; };

// cores de cada IA do enxame
// Cor de cada IA do enxame (repete a cada 6).
static Color agentColor(int i, int a = 255) {
    static const int P[6][3] = {{70, 130, 220}, {230, 80, 80}, {80, 200, 110}, {200, 90, 220}, {240, 190, 50}, {60, 210, 210}};
    return rgba(P[i % 6][0], P[i % 6][1], P[i % 6][2], a);
}

// desenha UMA IA: recompensas dela, rastro, tronco e patas
static void drawAgent(const Sim& s, int id, bool focus, double tnow) {
    const Creature& c = s.cr;
    const float rs = BODY_SCALE;
    Color body = agentColor(id, focus ? 255 : 200);
    for (size_t i = 1; i < s.trail.size(); i++) {                                 // rastro
        V3 a = (s.balanceSurfaceEpisode() || s.parkourMode) ? s.trail[i - 1] : V3{s.trail[i - 1].x, 0.02f, s.trail[i - 1].z};
        V3 b = (s.balanceSurfaceEpisode() || s.parkourMode) ? s.trail[i] : V3{s.trail[i].x, 0.02f, s.trail[i].z};
        DrawLine3D(toRl(a), toRl(b), agentColor(id, focus ? 160 : 70));
    }

    int ni = s.nearestOrb();
    if (!s.balanceSurfaceEpisode() && !s.generalMode) for (int i = 0; i < N_ORBS; i++) {
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
    if (!s.balanceSurfaceEpisode() && !s.generalMode) {
        DrawLine3D(Vector3{c.pos.x, 0.05f, c.pos.z}, Vector3{s.orbs[ni].p.x, 0.05f, s.orbs[ni].p.z}, focus ? rgba(255, 215, 0, 120) : agentColor(id, 110));
        DrawCircle3D(Vector3{c.pos.x, 0.015f, c.pos.z}, 0.5f, Vector3{1, 0, 0}, 90.f, rgba(0, 0, 0, 120));
    } else if (s.balanceSurfaceEpisode()) {
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
        DrawSphere(Vector3{0.07f * rs, 0.405f * rs, 0.f}, 0.135f * rs, rgba(255, 120, 60));
        DrawSphereWires(Vector3{0.07f * rs, 0.405f * rs, 0.f}, 0.135f * rs, 8, 12, rgba(230, 235, 245));
        DrawCube(Vector3{0.19f * rs, 0.385f * rs, 0.f}, 0.16f * rs, 0.09f * rs, 0.13f * rs, rgba(255, 120, 60));
    }
    if (c.biped) {
        for (int l = 0; l < c.limbCount(); ++l) {
            V3 kneeLocal, ankleLocal;
            c.legPoints(l, kneeLocal, ankleLocal, s.balanceSurfaceEpisode());
            V3 heelLocal, midfootLocal, toeLocal;
            c.bipedFootPoints(l, heelLocal, midfootLocal, toeLocal, s.balanceSurfaceEpisode());
            const Color footColor = c.contact[l] ? rgba(90, 155, 110) : rgba(85, 100, 125);
            const Color boneColor = rgba(150, 175, 205);
            // Digitigrade profile: a raised calcaneal/hock bump, sloped
            // metatarsus and compact forward toes. There is no flat heel-to-toe shoe.
            DrawCylinderEx(toRl(heelLocal), toRl(ankleLocal), 0.023f * rs, 0.031f * rs, 8, boneColor);
            DrawCylinderEx(toRl(ankleLocal), toRl(midfootLocal), 0.036f * rs, 0.029f * rs, 8, footColor);
            DrawCylinderEx(toRl(midfootLocal), toRl(toeLocal), 0.029f * rs, 0.020f * rs, 8, footColor);
            DrawSphere(toRl(heelLocal), BIPED_HOCK_COLLISION_RADIUS, boneColor);
            DrawSphere(toRl(toeLocal), BIPED_TOE_COLLISION_RADIUS, footColor);
            // Three small toe pads spread across the forefoot; the central pad
            // matches the actual toe contact point used by the physics solver.
            for (int digit = 0; digit < 3; ++digit) {
                const float lateral = ((float)digit - 1.f) * BIPED_FOOT_WIDTH * 0.28f;
                V3 toePad = toeLocal + V3{digit == 1 ? 0.018f * rs : 0.008f * rs, -0.004f * rs, lateral};
                DrawSphere(toRl(toePad), 0.020f * rs, footColor);
            }
        }
    }
    rlPopMatrix();
    for (int l = 0; l < c.limbCount(); l++) {
        V3 kb, fb; c.legPoints(l, kb, fb, s.balanceSurfaceEpisode());
        V3 h = c.pos + c.toWorld(c.hipLocal(l)), k = c.pos + c.toWorld(kb), f = c.pos + c.toWorld(fb);
        if (c.biped) {
            // Digitigrade silhouette: pronounced hock and narrow distal section.
            V3 hock = k + (f - k) * 0.72f;
            DrawCylinderEx(toRl(h), toRl(k), 0.062f * rs, 0.048f * rs, 8, rgba(170, 190, 220));
            DrawCylinderEx(toRl(k), toRl(hock), 0.046f * rs, 0.032f * rs, 8, rgba(140, 160, 200));
            DrawCylinderEx(toRl(hock), toRl(f), 0.032f * rs, 0.025f * rs, 8, rgba(140, 160, 200));
            DrawSphere(toRl(k), 0.055f * rs, rgba(230, 230, 240));
            DrawSphere(toRl(hock), 0.035f * rs, rgba(230, 230, 240));
            DrawSphere(toRl(f), 0.045f * rs, rgba(230, 230, 240));
        } else {
            DrawCylinderEx(toRl(h), toRl(k), 0.05f * rs, 0.04f * rs, 8, rgba(170, 190, 220));
            DrawCylinderEx(toRl(k), toRl(f), 0.04f * rs, 0.03f * rs, 8, rgba(140, 160, 200));
            DrawSphere(toRl(k), 0.055f * rs, rgba(230, 230, 240));
            DrawSphere(toRl(f), 0.065f * rs, c.contact[l] ? rgba(80, 255, 120) : rgba(255, 90, 90));
        }
    }
}

// Esfera de equilíbrio: casca colorida, malha e seta da velocidade.
static void drawBalanceSphere(const Sim& s, int id, bool focus) {
    Color shell = agentColor(id, focus ? 235 : 145);
    DrawSphere(toRl(s.spherePos), s.sphereRadius, shell);
    DrawSphereWires(toRl(s.spherePos), s.sphereRadius, 18, 28,
                    rgba(220, 235, 255, focus ? 190 : 90));
    V3 velocityArrow = s.sphereVel * 0.75f;
    DrawLine3D(toRl(s.spherePos), toRl(s.spherePos + velocityArrow),
               focus ? rgba(255, 220, 100, 230) : rgba(230, 230, 240, 130));
}

// Blocos (paredes/degraus), caixa de empurrar e marcador verde do alvo no modo geral.
static void drawGeneralEnvironment(const Sim& s, bool focus) {
    const V3 cameraCenter = s.cr.pos;
    for (const auto& b : s.generalBlocks) {
        if (!blockNearXZ(b, cameraCenter, WORLD_RENDER_RADIUS)) continue;
        const Color c = b.step ? rgba(165, 125, 82, 255) : rgba(92, 104, 123, 255);
        DrawCube(toRl(b.center), b.size.x, b.size.y, b.size.z, c);
        if (focus || blockNearXZ(b, cameraCenter, 16.0f))
            DrawCubeWires(toRl(b.center), b.size.x, b.size.y, b.size.z, rgba(205, 215, 230, 210));
    }
    if (s.generalTask == Sim::TASK_PUSH) {
        DrawCube(toRl(s.boxPos), 0.90f, 0.84f, 0.90f, rgba(220, 145, 65, 255));
        DrawCubeWires(toRl(s.boxPos), 0.90f, 0.84f, 0.90f, rgba(255, 220, 155, 255));
    }
    if (s.generalTask != Sim::TASK_BALANCE) {
        const V3 marker = s.generalTask == Sim::TASK_PUSH ? s.pushTarget : s.generalGoal;
        DrawCylinder(toRl(V3{marker.x, 0.04f, marker.z}), 0.42f, 0.42f, 0.08f, 16,
                     rgba(80, 255, 130, 190));
        DrawSphere(toRl(V3{marker.x, 0.28f, marker.z}), 0.16f, rgba(100, 255, 145, 255));
        DrawCylinder(toRl(V3{marker.x, 0.f, marker.z}), 0.30f, 0.30f, 1.5f, 12, rgba(80, 255, 130, 130)); // coluna que a IA enxerga de longe
    }

}

// Plataformas do parkour (marrons) e paredes.
static void drawParkourEnvironment(const Sim& s, bool focus) {
    const V3 cameraCenter = s.cr.pos;
    for (const auto& b : s.generalBlocks) {
        if (!blockNearXZ(b, cameraCenter, WORLD_RENDER_RADIUS)) continue;
        const Color blockColor = b.step ? rgba(174, 112, 62, 255) : rgba(72, 91, 116, 255);
        DrawCube(toRl(b.center), b.size.x, b.size.y, b.size.z, blockColor);
        if (focus || blockNearXZ(b, cameraCenter, 15.0f))
            DrawCubeWires(toRl(b.center), b.size.x, b.size.y, b.size.z,
                          b.step ? rgba(255, 196, 112, 230) : rgba(190, 210, 235, 210));
    }
}

// Chão de xadrez preto e branco. Só as casas ao redor da câmera são desenhadas, sempre alinhadas
// à grade do mundo (o padrão não "anda" junto com a câmera). Cada casa tem 1 m.
static void drawCheckerFloor(float centerX, float centerZ, float y) {
    constexpr int   HALF = 40;     // casas para cada lado do centro (80 x 80 no total)
    constexpr float TILE = 1.0f;   // lado de cada casa, em metros
    const int i0 = (int)std::floor(centerX / TILE), j0 = (int)std::floor(centerZ / TILE);
    const Color light = rgba(238, 238, 238), dark = rgba(24, 24, 24);

    rlDisableBackfaceCulling();    // a face de cima sempre aparece, de qualquer ângulo
    rlBegin(RL_QUADS);
    for (int j = -HALF; j < HALF; ++j) {
        for (int i = -HALF; i < HALF; ++i) {
            const int gi = i0 + i, gj = j0 + j;
            const Color c = ((gi + gj) & 1) ? dark : light;
            const float x0 = gi * TILE, z0 = gj * TILE;
            rlColor4ub(c.r, c.g, c.b, 255);
            rlVertex3f(x0, y, z0);
            rlVertex3f(x0, y, z0 + TILE);
            rlVertex3f(x0 + TILE, y, z0 + TILE);
            rlVertex3f(x0 + TILE, y, z0);
        }
    }
    rlEnd();
    rlDrawRenderBatchActive();     // envia tudo agora: o chão fica por baixo do resto da cena
    rlEnableBackfaceCulling();
}

// Desenha a cena 3D: câmera, chão, cenários e as IAs (a exibida é desenhada por último, por cima).
static void drawWorld(const Swarm& w, const View& v, double tnow) {
    Camera3D cam{};
    if (v.rearFollow) {
        const Sim& s = w.cur();
        V3 forward = norm(V3{s.cr.ax.x, 0.f, s.cr.ax.z});
        const V3 look = s.cr.pos + V3{0.f, 0.20f, 0.f} + forward * 1.65f;
        const V3 eye = s.cr.pos + V3{0.f, 0.35f * v.dist, 0.f} + forward * (-v.dist);
        cam.target = toRl(look);
        cam.position = toRl(eye);
    } else {
        cam.target = toRl(v.target);
        cam.position = Vector3{v.target.x + std::cos(v.pitch) * std::sin(v.yaw) * v.dist,
                               v.target.y + std::sin(v.pitch) * v.dist,
                               v.target.z + std::cos(v.pitch) * std::cos(v.yaw) * v.dist};
    }
    cam.up = Vector3{0, 1, 0}; cam.fovy = 55.f; cam.projection = CAMERA_PERSPECTIVE;
    BeginMode3D(cam);
    // No parkour o chão visível fica bem abaixo das plataformas (é o vazio onde a IA cai).
    const float visibleFloorY = w.cur().parkourMode ? -2.25f : -0.01f;
    drawCheckerFloor(v.target.x, v.target.z, visibleFloorY);
    for (int i = 0; i < (int)w.ag.size(); i++) {
        if (w.ag[i]->balanceSurfaceEpisode()) drawBalanceSphere(*w.ag[i], i, i == w.shown);
        if (w.ag[i]->generalMode) drawGeneralEnvironment(*w.ag[i], i == w.shown);
        else if (w.ag[i]->parkourMode) drawParkourEnvironment(*w.ag[i], i == w.shown);
    }
    const V3 focusPos = w.cur().cr.pos;
    for (int i = 0; i < (int)w.ag.size(); i++) if (i != w.shown) {
        const V3 d = w.ag[i]->cr.pos - focusPos;
        if (d.x * d.x + d.z * d.z <= 18.0f * 18.0f)
            drawAgent(*w.ag[i], i, false, tnow);
    }
    drawAgent(*w.ag[w.shown], w.shown, true, tnow);                               // a exibida por cima
    EndMode3D();
}

// ================================ INTERFACE ==================================
// Tudo o que aparece sobre a cena 3D: fonte, cartões arredondados, tabelas e legenda de atalhos.
//
// LAYOUT (recalculado a cada quadro a partir do tamanho da janela, então nada se sobrepõe):
//   * coluna esquerda : cartão de estado (cima) + tabela do enxame (embaixo);
//   * coluna direita  : mapa do cérebro + gráfico de retorno (cima) + câmeras da IA (embaixo);
//   * centro (topo)   : faixa de status e aviso temporário; a cena 3D fica livre;
//   * rodapé          : legenda de atalhos, que define até onde as colunas podem descer.
// Se a janela for baixa, as tabelas encolhem (menos linhas / câmeras menores) em vez de sobrepor.

namespace ui {
    // Paleta: superfícies escuras neutras, texto claro e poucos destaques.
    static const Color surface{16, 18, 22, 228};     // fundo dos cartões
    static const Color chip{255, 255, 255, 22};      // fundo de teclas, gráficos e linha selecionada
    static const Color divider{255, 255, 255, 26};   // linhas finas entre seções
    static const Color text{236, 239, 244, 255};     // texto principal
    static const Color muted{150, 158, 172, 255};    // rótulos e textos secundários
    static const Color accent{110, 168, 254, 255};   // títulos de seção e seleção
    static const Color good{98, 214, 146, 255};      // valores positivos / ligado
    static const Color warn{246, 180, 84, 255};      // atenção
    static const Color gold{250, 204, 21, 255};      // pontuação e recordes
}

// ------------------------------- fonte e texto -------------------------------
static Font g_font{};            // fonte da interface (TTF suavizada, ou a padrão como reserva)
static bool g_fontOwned = false; // true se carregamos um TTF (então precisa de UnloadFont)

// Procura uma fonte TTF e a carrega com antialiasing. Ordem de busca: arquivo próprio ao lado do
// programa, depois fontes do sistema (arredondadas primeiro). Sem nenhuma, usa a fonte padrão da raylib.
static void loadUiFont() {
    static const char* candidates[] = {
        "ui_font.ttf", "fonts/ui_font.ttf",                                   // fonte escolhida por você
        "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/calibri.ttf",       // Windows
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",          // macOS
        "/Library/Fonts/Arial Rounded Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/nunito/Nunito-Regular.ttf",                // Linux
        "/usr/share/fonts/truetype/quicksand/Quicksand-Regular.ttf",
        "/usr/share/fonts/truetype/google-fonts/Poppins-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    };
    int codepoints[224];                                   // ASCII + Latin-1 (acentos do português)
    for (int i = 0; i < 224; ++i) codepoints[i] = 32 + i;
    for (const char* path : candidates) {
        if (!FileExists(path)) continue;
        Font f = LoadFontEx(path, 32, codepoints, 224);    // atlas em 32 px: fica nítido ao reduzir
        if (f.texture.id == 0) continue;
        GenTextureMipmaps(&f.texture);
        SetTextureFilter(f.texture, TEXTURE_FILTER_TRILINEAR);
        g_font = f; g_fontOwned = true;
        std::printf("[interface] fonte: %s\n", path);
        return;
    }
    g_font = GetFontDefault();
    SetTextureFilter(g_font.texture, TEXTURE_FILTER_BILINEAR);
    std::printf("[interface] nenhuma fonte TTF encontrada; usando a padrão. Coloque 'ui_font.ttf' ao lado do programa.\n");
}
static void unloadUiFont() { if (g_fontOwned) UnloadFont(g_font); g_fontOwned = false; }

static float uiSpacing(float size) { return size * 0.02f; }
static float uiWidth(const char* t, float size) { return MeasureTextEx(g_font, t, size, uiSpacing(size)).x; }
// Posições arredondadas para pixel inteiro: evita texto "borrado" por meio pixel.
static void uiDraw(float x, float y, float size, Color c, const char* t) {
    DrawTextEx(g_font, t, Vector2{std::floor(x), std::floor(y)}, size, uiSpacing(size), c);
}
static std::string strf(const char* fmt, ...) {
    char buf[256]; va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    return buf;
}
// Quebra um texto em linhas que cabem em maxW pixels.
static std::vector<std::string> uiWrap(const std::string& s, float maxW, float size) {
    std::vector<std::string> lines; std::string cur;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(' ', i); if (j == std::string::npos) j = s.size();
        const std::string word = s.substr(i, j - i); i = j + 1;
        const std::string test = cur.empty() ? word : cur + " " + word;
        if (!cur.empty() && uiWidth(test.c_str(), size) > maxW) { lines.push_back(cur); cur = word; }
        else cur = test;
    }
    if (!cur.empty()) lines.push_back(cur);
    return lines;
}

// ------------------------------ cartões e linhas ------------------------------
// Retângulo de cantos arredondados (radius em pixels).
static void uiPanel(float x, float y, float w, float h, Color fill = ui::surface, float radius = 10.f) {
    const float roundness = std::min(1.f, radius * 2.f / std::min(w, h));
    DrawRectangleRounded(Rectangle{x, y, w, h}, roundness, 10, fill);
}

// Cursor vertical que vai empilhando linhas dentro de um cartão. Com dry = true só mede a altura
// (o cartão precisa saber o próprio tamanho antes de desenhar o fundo).
struct UiCursor { float x, y, w; bool dry; };

static void uiHeading(UiCursor& c, const char* title) {              // título de seção, com linha fina acima
    if (!c.dry) {
        DrawRectangle((int)c.x, (int)(c.y + 2.f), (int)c.w, 1, ui::divider);
        uiDraw(c.x, c.y + 7.f, 12.5f, ui::accent, title);
    }
    c.y += 26.f;
}
static void uiRow(UiCursor& c, const char* label, Color value, const char* fmt, ...) {   // rótulo à esquerda, valor à direita
    if (!c.dry) {
        char buf[256]; va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
        uiDraw(c.x, c.y, 14.f, ui::muted, label);
        uiDraw(c.x + c.w - uiWidth(buf, 14.f), c.y, 14.f, value, buf);
    }
    c.y += 19.f;
}
static void uiNote(UiCursor& c, Color col, const char* t) {          // observação pequena, de largura total
    if (!c.dry) uiDraw(c.x, c.y + 1.f, 13.f, col, t);
    c.y += 18.f;
}

// Cartão que se ajusta ao conteúdo: mede o corpo (dry), desenha o fundo e então desenha o corpo.
// Devolve a altura, para o próximo cartão saber onde começar.
template <class Body>
static float uiCard(float x, float y, float w, Body body) {
    UiCursor probe{x + 14.f, y + 12.f, w - 28.f, true};
    body(probe);
    const float h = (probe.y - y) + 10.f;
    uiPanel(x, y, w, h);
    UiCursor c{x + 14.f, y + 12.f, w - 28.f, false};
    body(c);
    return h;
}

// ------------------------------- cartão de estado -----------------------------
static float drawStatsCard(const Sim& s, float x, float y, float width) {
    const char* modeName = s.parkourMode ? "Parkour (K)" : (s.generalMode ? "Modo geral (G)"
                         : (s.balanceMode ? "Equilíbrio (E)" : (s.speedRewardMode ? "Corrida (C)" : "Caminhada")));
    const float speedNow = std::sqrt(s.cr.vel.x * s.cr.vel.x + s.cr.vel.z * s.cr.vel.z);

    return uiCard(x, y, width, [&](UiCursor& c) {
        if (!c.dry) {
            uiDraw(c.x, c.y, 18.f, ui::text, "Cérebro esparso 3D");
            uiDraw(c.x, c.y + 23.f, 13.f, ui::muted, "Morfologia e aprendizado");
        }
        c.y += 42.f;

        uiHeading(c, "Cérebro");
        uiRow(c, "Neurônios ativos", ui::good, "%d de %d (%.2f%%)", s.lastActive, N_NEURONS, 100.f * s.lastActive / N_NEURONS);
        uiRow(c, "Grupos", ui::text, "%d agora · %d/%d já usados", s.lastGroups, s.groupsUsed(), N_GROUPS);
        uiRow(c, "Uso médio", ui::text, "%.2f%% do cérebro", 100.f * s.avgActiveFrac);

        uiHeading(c, "Treino");
        uiRow(c, "Passos · episódios", ui::text, "%llu · %llu", (unsigned long long)s.brain.steps, (unsigned long long)s.brain.episodes);
        if (s.generalMode)
            uiRow(c, "Tarefas concluídas", ui::gold, "%d · %.2f objetivos/ep", s.generalSuccesses, s.avgOrbsEp);
        else
            uiRow(c, "Orbes coletados", ui::gold, "%llu · média %.2f/ep", (unsigned long long)s.brain.orbs, s.avgOrbsEp);
        uiRow(c, "Retorno", ui::text, "último %.1f · média %.1f", s.lastReturn, s.brain.emaReturn);
        uiRow(c, "Episódio atual", ui::text, "%.1f s · %d orbes", s.epSteps * CTRL_DT, s.epOrbs);
        uiRow(c, "Aprendizado", s.learning ? ui::good : ui::warn, "%s · sigma %.3f",
              s.learning ? "Ligado" : "Desligado", s.sigma);

        const std::string modeTitle = strf("Modo · %s", modeName);
        uiHeading(c, modeTitle.c_str());
        if (s.parkourMode) {
            uiRow(c, "Concluídos", ui::gold, "%d · tempo atual %.1f s", s.parkourCompletions, s.epSteps * CTRL_DT);
            uiRow(c, "Distância da orbe", ui::good, "%.2f m", Sim::distXZ(s.cr.pos, s.parkourGoal));
            uiRow(c, "Velocidade", ui::good, "%.2f m/s", speedNow);
            uiRow(c, "Melhor tempo", ui::text, "%s", s.bestParkourTime > 0.f ? TextFormat("%.2f s", s.bestParkourTime) : "--");
            uiNote(c, ui::muted, "Recompensa: terminar rápido e pousar estável");
        } else if (s.generalMode) {
            uiRow(c, "Tarefa", ui::gold, "%s", s.generalTaskName());
            uiRow(c, "Decisão", ui::good, "%s", s.generalActionName());
            uiRow(c, "Velocidade · distância", ui::good, "%.2f m/s · %.2f m", speedNow, Sim::distXZ(s.cr.pos, s.generalGoal));
            uiRow(c, "Sucessos · atualizações Q", ui::text, "%d · %llu", s.generalSuccesses, (unsigned long long)s.meta.updates);
            uiNote(c, ui::muted, "Raios: Q-learning · juntas fixas");
        } else if (s.balanceMode) {
            const int limbs = s.cr.limbCount();
            const int contacts = (int)s.cr.contact[0] + (int)s.cr.contact[1]
                               + (s.cr.biped ? 0 : (int)s.cr.contact[2] + (int)s.cr.contact[3]);
            float actionLevel = 0.f;
            for (int i = 0; i < s.cr.activeActionCount(); ++i) actionLevel += std::fabs(s.lastAction[i]);
            actionLevel /= (float)s.cr.activeActionCount();
            uiRow(c, "Pontos", ui::gold, "%.1f", s.balancePoints);
            uiRow(c, "Em equilíbrio", ui::text, "atual %.1f s · recorde %.1f s", s.stableSeconds, s.bestStableSeconds);
            uiRow(c, "Esfera", ui::text, "R=%.1f m · mov %.2f m/s", s.sphereRadius, len(s.sphereVel));
            uiRow(c, "Apoio · ação média", ui::text, "%d/%d patas · %.3f", contacts, limbs, actionLevel);
            uiRow(c, "Atualizações RL", ui::text, "%llu", (unsigned long long)s.learningUpdates);
        } else {
            uiRow(c, "Recompensa", s.speedRewardMode ? ui::gold : ui::text, "%s",
                  s.speedRewardMode ? "Corrida (velocidade)" : "Caminhada (orbes)");
            uiRow(c, "Velocidade horizontal", ui::good, "%.2f m/s", speedNow);
            uiNote(c, ui::muted, s.speedRewardMode ? "Corrida premia até 4 m/s com postura ereta"
                                                   : "Caminhada premia chegar às orbes");
        }
    });
}

// ------------------------------- tabela do enxame -----------------------------
// availH = altura livre até o rodapé; se não couber uma tabela útil, mostra só uma linha-resumo.
static void drawSwarmCard(const Swarm& w, float x, float y, float width, float availH) {
    const int n = (int)w.ag.size();
    const float rowH = 18.f, overhead = 82.f;                       // cabeçalho (72) + respiro inferior (10)
    const int fit = (int)((availH - overhead) / rowH);
    const int rows = std::min(n, fit);
    if (n == 1 || rows < 3) {                                       // forma compacta
        if (availH < 34.f) return;
        uiPanel(x, y, width, 34.f);
        if (n == 1) uiDraw(x + 14.f, y + 9.f, 14.f, ui::warn, "Enxame: 1 IA · aprendizado desligado");
        else        uiDraw(x + 14.f, y + 9.f, 14.f, ui::text, strf("Enxame: %d IAs · geração %d", n, w.generation).c_str());
        return;
    }

    uiPanel(x, y, width, overhead + rows * rowH);
    const float cx = x + 14.f, inner = width - 28.f;
    uiDraw(cx, y + 10.f, 15.f, ui::text, "Enxame");
    uiDraw(cx + inner - uiWidth(strf("Geração %d", w.generation).c_str(), 14.f), y + 11.f, 14.f, ui::muted,
           strf("Geração %d", w.generation).c_str());
    if (w.bestScore > NO_SCORE) uiDraw(cx, y + 31.f, 13.f, ui::gold, strf("Recorde global: %.1f", w.bestScore).c_str());
    else                        uiDraw(cx, y + 31.f, 13.f, ui::muted, "Recorde: aguardando geração completa");

    // colunas (bordas direitas, para alinhar os números)
    const float colScore = cx + 170.f, colEps = cx + 252.f, colNoise = cx + inner;
    const float headY = y + 52.f;
    uiDraw(cx + 16.f, headY, 12.f, ui::muted, "IA");
    uiDraw(colScore - uiWidth("Score", 12.f), headY, 12.f, ui::muted, "Score");
    uiDraw(colEps - uiWidth("Episódios", 12.f), headY, 12.f, ui::muted, "Episódios");
    uiDraw(colNoise - uiWidth("Ruído", 12.f), headY, 12.f, ui::muted, "Ruído");

    int lead = -1; float leadScore = NO_SCORE;                      // quem está na frente no ranking
    for (int i = 0; i < n; i++) { const float sc = w.ag[i]->score(1); if (sc > leadScore) { leadScore = sc; lead = i; } }
    const int first = std::clamp(w.shown - (rows - 1) / 2, 0, n - rows);   // mantém a IA exibida visível

    for (int r = 0; r < rows; ++r) {
        const int i = first + r;
        const Sim& a = *w.ag[i];
        const float rowY = y + 72.f + r * rowH;
        if (i == w.shown)
            DrawRectangleRounded(Rectangle{cx - 6.f, rowY - 1.f, inner + 12.f, rowH}, 0.4f, 6, ui::chip);
        DrawCircle((int)(cx + 4.f), (int)(rowY + 8.f), 4.5f, agentColor(i));
        uiDraw(cx + 16.f, rowY, 14.f, i == w.shown ? ui::text : ui::muted, strf("IA %d", i).c_str());
        const float sc = a.score(1);
        const std::string scoreTxt = sc > NO_SCORE ? strf("%.1f", sc) : std::string("--");
        uiDraw(colScore - uiWidth(scoreTxt.c_str(), 14.f), rowY, 14.f, i == lead ? ui::good : ui::text, scoreTxt.c_str());
        const std::string epsTxt = strf("%zu/%zu", a.recent.size(), GEN_EPISODES);
        uiDraw(colEps - uiWidth(epsTxt.c_str(), 14.f), rowY, 14.f, ui::muted, epsTxt.c_str());
        const std::string noiseTxt = strf("×%.2f", a.sigmaMul);
        uiDraw(colNoise - uiWidth(noiseTxt.c_str(), 14.f), rowY, 14.f, ui::muted, noiseTxt.c_str());
    }
}

// ------------------------------ mapa do cérebro -------------------------------
// Cada quadradinho é um grupo de 32 neurônios: apagado = nunca usado, azul-acinzentado = já usado,
// do laranja ao amarelo = aceso agora. Abaixo, o retorno por episódio de cada IA (uma linha por IA).
static float drawBrainCard(const Swarm& w, float x, float y, float width) {
    const Sim& s = w.cur();
    const Brain& policy = s.displayedPolicy();
    constexpr int   cols = 32, cell = 8, rows = N_GROUPS / cols;
    constexpr float gridW = cols * cell, gridH = rows * cell, graphH = 64.f;
    const float h = 54.f + gridH + 10.f + 20.f + graphH + 12.f;
    uiPanel(x, y, width, h);

    const float gx = x + (width - gridW) * 0.5f, gy = y + 54.f;
    uiDraw(gx, y + 10.f, 15.f, ui::text, "Mapa do cérebro");
    uiDraw(gx, y + 31.f, 12.f, ui::muted, "1 quadrado = 1 grupo (32 neurônios)");
    for (int g = 0; g < N_GROUPS; g++) {
        const int cx = (int)gx + (g % cols) * cell, cy = (int)gy + (g / cols) * cell;
        const float heat = s.heat[g];
        Color c = policy.useCount[g] ? rgba(52, 66, 92) : rgba(38, 41, 48);
        if (heat > 0.02f) c = rgba((int)(60 + 195 * heat), (int)(90 + 130 * heat), (int)(110 * (1 - heat) + 20));
        DrawRectangle(cx, cy, cell - 1, cell - 1, c);
    }

    const float labelY = gy + gridH + 10.f, plotY = labelY + 20.f;
    uiDraw(gx, labelY, 12.5f, ui::muted, "Retorno por episódio · 1 linha por IA");
    DrawRectangleRounded(Rectangle{gx, plotY, gridW, graphH}, 0.12f, 6, ui::chip);
    float mn = 1e30f, mx = -1e30f; bool any = false;                 // mesma escala para todas as IAs
    for (auto& a : w.ag) for (float v : a->retHist) { mn = std::min(mn, v); mx = std::max(mx, v); any = true; }
    if (any) {
        if (mx - mn < 1e-3f) mx = mn + 1.f;
        const float px = gx + 4.f, pw = gridW - 8.f, py = plotY + 4.f, ph = graphH - 8.f;
        for (int k = 0; k < (int)w.ag.size(); k++) {
            const auto& hist = w.ag[k]->retHist;
            const bool shown = k == w.shown;
            const Color lc = shown ? rgba(255, 255, 255) : agentColor(k, 210);
            for (size_t i = 1; i < hist.size(); i++) {
                const float x0 = px + (float)(i - 1) / 159.f * pw, x1 = px + (float)i / 159.f * pw;
                const float y0 = py + ph - (hist[i - 1] - mn) / (mx - mn) * ph;
                const float y1 = py + ph - (hist[i] - mn) / (mx - mn) * ph;
                DrawLineEx(Vector2{x0, y0}, Vector2{x1, y1}, shown ? 2.f : 1.2f, lc);
            }
        }
    }
    return h;
}

// ------------------------------- câmeras da IA --------------------------------
// Imagem REAL que a IA recebe de cada câmera (a mesma que ela usa para decidir).
static void drawVisionImage(const Sim::VisionImage& im, float x, float y, int scale) {
    for (int r = 0; r < VIS_H; ++r)
        for (int c = 0; c < VIS_W; ++c) {
            const uint8_t* p = &im.rgb[3 * (r * VIS_W + c)];
            DrawRectangle((int)x + c * scale, (int)y + r * scale, scale, scale, rgba(p[0], p[1], p[2]));
        }
    DrawRectangleLines((int)x - 1, (int)y - 1, VIS_W * scale + 2, VIS_H * scale + 2, rgba(255, 255, 255, 60));
}
// Só aparece no modo geral (menos na tarefa de equilíbrio). A escala (4, 3 ou 2) se ajusta ao espaço livre.
static void drawVisionCard(const Sim& s, float x, float y, float width, float availH) {
    if (!s.generalMode || s.generalTask == Sim::TASK_BALANCE || !s.visFront.valid || !s.visRear.valid) return;
    const float titleH = 34.f, labelH = 20.f, gap = 10.f;
    int scale = 4;
    while (scale >= 2 && titleH + 2.f * (labelH + VIS_H * scale + gap) + 2.f > availH) --scale;
    if (scale < 2) return;                                          // sem espaço: não desenha
    const float imgW = (float)(VIS_W * scale), imgH = (float)(VIS_H * scale);
    const float h = titleH + 2.f * (labelH + imgH + gap) + 2.f;
    uiPanel(x, y, width, h);
    const float ix = x + (width - imgW) * 0.5f;
    uiDraw(ix, y + 10.f, 15.f, ui::text, "Visão da IA");
    float cy = y + titleH;
    uiDraw(ix, cy, 13.f, ui::muted, "Câmera traseira");
    drawVisionImage(s.visFront, ix, cy + labelH, scale);
    cy += labelH + imgH + gap;
    uiDraw(ix, cy, 13.f, ui::muted, "Câmera frontal");
    drawVisionImage(s.visRear, ix, cy + labelH, scale);
}

// ------------------------------- legenda de atalhos ---------------------------
struct KeyHint { const char* key; const char* label; };
static const KeyHint kHints[] = {
    {"Cima/Baixo", "Velocidade"}, {"G", "Modo geral"}, {"K", "Parkour"}, {"E", "Equilíbrio"},
    {"C", "Corrida"}, {"Q", "Corpo"}, {"L", "Aprendizado"}, {"V", "Câmera fronteira"},
    {"Tab", "Trocar IA"}, {"S", "Salvar"}, {"Espaço", "Pausa"}, {"Mouse", "Câmera livre"},
};
// Barra no rodapé, com as teclas em "chips". Quebra em mais de uma linha se a janela for estreita.
// Devolve a altura ocupada (as colunas usam isso para saber até onde podem descer).
static float drawShortcutLegend(int W, int H) {
    const float M = 12.f, padX = 12.f, padY = 8.f, chipH = 20.f, rowGap = 6.f, itemGap = 16.f;
    const float left = M + padX, right = (float)W - M - padX;
    auto pass = [&](bool draw, float top) {                         // 1ª passada mede, 2ª desenha
        float cx = left, cy = top; int rows = 1;
        for (const KeyHint& k : kHints) {
            const float kw = uiWidth(k.key, 12.5f) + 14.f, lw = uiWidth(k.label, 13.f);
            const float total = kw + 6.f + lw;
            if (cx + total > right && cx > left) { cx = left; cy += chipH + rowGap; ++rows; }
            if (draw) {
                DrawRectangleRounded(Rectangle{cx, cy, kw, chipH}, 0.4f, 6, ui::chip);
                uiDraw(cx + 7.f, cy + 3.f, 12.5f, ui::text, k.key);
                uiDraw(cx + kw + 6.f, cy + 3.f, 13.f, ui::muted, k.label);
            }
            cx += total + itemGap;
        }
        return rows;
    };
    const int rows = pass(false, 0.f);
    const float h = padY * 2.f + rows * chipH + (rows - 1) * rowGap;
    const float top = (float)H - M - h;
    uiPanel(M, top, (float)W - 2.f * M, h);
    pass(true, top + padY);
    return h;
}

// ------------------------------ status no topo central ------------------------
// Faixa com corpo, velocidade da simulação, FPS e último salvamento (itens finais somem se faltar
// espaço) e, abaixo dela, o aviso temporário da última ação (modo ativado, bloqueado etc.).
static void drawCenterStatus(const Sim& s, bool paused, double speed, double saveAgo, int fps,
                             const std::string& status, float left, float right) {
    const float avail = right - left;
    if (avail < 220.f) return;
    std::vector<std::string> parts;
    if (paused) parts.push_back("Pausado");
    parts.push_back(s.cr.biped ? "Bípede digitígrado" : "Quadrúpede");
    parts.push_back(strf("Simulação ×%.0f", speed));
    parts.push_back(strf("%d FPS", fps));
    parts.push_back(strf("Salvo há %.0f s", saveAgo));
    auto join = [&]() { std::string t; for (size_t i = 0; i < parts.size(); ++i) { if (i) t += " · "; t += parts[i]; } return t; };
    std::string line = join();
    while (parts.size() > 1 && uiWidth(line.c_str(), 13.5f) > avail - 28.f) { parts.pop_back(); line = join(); }
    const float pw = std::min(avail, uiWidth(line.c_str(), 13.5f) + 28.f);
    const float px = left + (avail - pw) * 0.5f;
    uiPanel(px, 12.f, pw, 30.f, ui::surface, 15.f);
    uiDraw(px + 14.f, 19.f, 13.5f, paused ? ui::warn : ui::muted, line.c_str());

    if (status.empty()) return;
    const bool alert = status.find("bloqueado") != std::string::npos || status.find("falta") != std::string::npos
                    || status.find("requer") != std::string::npos;
    std::vector<std::string> lines = uiWrap(status, avail - 32.f, 14.f);
    if (lines.size() > 3) lines.resize(3);
    float textW = 0.f;
    for (const auto& l : lines) textW = std::max(textW, uiWidth(l.c_str(), 14.f));
    const float bw = std::min(avail, textW + 32.f), bh = lines.size() * 19.f + 16.f;
    const float bx = left + (avail - bw) * 0.5f, by = 12.f + 30.f + 8.f;
    uiPanel(bx, by, bw, bh);
    DrawRectangleRounded(Rectangle{bx, by, 4.f, bh}, 1.f, 4, alert ? ui::warn : ui::good);   // barra de cor à esquerda
    for (size_t i = 0; i < lines.size(); ++i)
        uiDraw(bx + 16.f, by + 8.f + i * 19.f, 14.f, ui::text, lines[i].c_str());
}

// ------------------------------- montagem final -------------------------------
static void drawInterface(const Swarm& w, double speed, bool paused, double lastSaveAgo, int fps,
                          const std::string& status) {
    const int W = GetScreenWidth(), H = GetScreenHeight();
    const float M = 12.f, GAP = 10.f, LW = 360.f, RW = 280.f;       // margem, espaço entre cartões, larguras das colunas

    const float legendH = drawShortcutLegend(W, H);
    const float bottomLimit = (float)H - M - legendH - GAP;         // as colunas nunca passam daqui

    float yL = M;                                                    // coluna esquerda
    yL += drawStatsCard(w.cur(), M, yL, LW) + GAP;
    drawSwarmCard(w, M, yL, LW, bottomLimit - yL);

    const float xR = (float)W - M - RW;                              // coluna direita
    float yR = M;
    yR += drawBrainCard(w, xR, yR, RW) + GAP;
    drawVisionCard(w.cur(), xR, yR, RW, bottomLimit - yR);

    drawCenterStatus(w.cur(), paused, speed, lastSaveAgo, fps, status, M + LW + GAP, xR - GAP);
}

// ================================== MAIN =====================================
// Relógio monotônico em segundos.
static double nowSec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Treina sem janela por 'seconds' segundos: imprime o progresso a cada 10 s e salva a cada 5 s.
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
                std::printf("t=%4.0fs ias=%zu gen=%d (%.0f p/s no total) | retorno=%7.2f %s=%5.2f dur=%5.1fs | ativos=%.2f%% | recorde=%.1f\n",
                            t - t0, sw.ag.size(), sw.generation, (double)(sw.ticks - startTicks) / SUBSTEPS / (t - t0),
                            ret / n, sw.generalMode() ? "objetivos/ep" : "orbes/ep", orbs / n,
                            len / n * CTRL_DT, 100.f * act, sw.bestScore > NO_SCORE ? sw.bestScore : 0.f);
            lastPrint = t;
        }
    }
    sw.save();
    std::printf("[headless] fim. memoria salva em %s\n", sw.cur().memoryFile());
}

// Ponto de entrada: modos sem janela (--headless...) ou a interface gráfica.
int main(int argc, char** argv) {
    g_rng.seed(std::random_device{}());
    Swarm swarm; swarm.init();

    if (argc >= 2 && std::string(argv[1]) == "--headless-general") {
        if (!swarm.setGeneralMode(true)) return 2;
        runHeadless(swarm, argc >= 3 ? std::atof(argv[2]) : 60.0);
        return 0;
    }
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

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(1280, 720, "AI Workspace");
    SetWindowMinSize(1100, 620);   // abaixo disso as colunas laterais não caberiam sem encostar
    SetTargetFPS(60);
    loadUiFont();                  // precisa vir depois de InitWindow (cria texturas)

    View view; double speed = 1.0; bool paused = false; double acc = 0.0;
    double lastSave = nowSec();
    std::string modeStatus = "K ativa o parkour (cenário e recompensa); L liga/desliga o aprendizado.";
    double modeStatusUntil = nowSec() + 7.0;

    while (!WindowShouldClose()) {
        // --- entrada ---
        if (IsKeyPressed(KEY_UP)) speed = std::min(1024.0, speed * 2.0);
        if (IsKeyPressed(KEY_DOWN)) speed = std::max(1.0, speed / 2.0);
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_L)) swarm.setLearning(!swarm.learning());
        if (IsKeyPressed(KEY_V)) view.rearFollow = !view.rearFollow;
        if (IsKeyPressed(KEY_K)) {
            const bool enableParkour = !swarm.parkourMode();
            if (swarm.setParkourMode(enableParkour)) {
                modeStatus = enableParkour
                    ? "Parkour ativado: mesma memória motora; o estado do aprendizado não mudou."
                    : "Parkour desativado: cenário normal; memória e aprendizado preservados.";
            } else {
                modeStatus = "O parkour requer o modo Caminhada: desative G, E ou C antes. Nenhuma memória foi trocada.";
            }
            modeStatusUntil = nowSec() + 10.0;
        }
        if (IsKeyPressed(KEY_G) && !swarm.parkourMode()) {
            const bool enableGeneral = !swarm.generalMode();
            const bool switched = swarm.setGeneralMode(enableGeneral);
            if (enableGeneral && switched && swarm.generalMode()) {
                if (!swarm.learning()) swarm.setLearning(true);
                modeStatus = "Modo geral ativado: o controlador Q escolhe a habilidade e a memória motora move as juntas.";
            } else if (!enableGeneral && switched && !swarm.generalMode()) {
                modeStatus = "Modo geral desativado. A tarefa anterior foi retomada.";
            } else {
                modeStatus = "Modo geral bloqueado: falta memória de caminhada treinada. Veja o terminal.";
            }
            modeStatusUntil = nowSec() + 10.0;
        }
        if (IsKeyPressed(KEY_E) && !swarm.parkourMode()) {
            if (swarm.generalMode()) swarm.setGeneralMode(false);
            const bool enableBalance = !swarm.balanceMode();
            // E inicia o treino de equilibrio mesmo se o aprendizado tiver sido desligado.
            if (enableBalance && !swarm.learning()) swarm.setLearning(true);
            swarm.setBalanceMode(enableBalance);
        }
        if (IsKeyPressed(KEY_C) && !swarm.balanceMode() && !swarm.parkourMode()) {
            if (swarm.generalMode()) swarm.setGeneralMode(false);
            // Como E e Q, C tambem liga o treinamento se ele estiver desligado.
            if (!swarm.learning()) swarm.setLearning(true);
            swarm.setSpeedRewardMode(!swarm.speedRewardMode());
        }
        if (IsKeyPressed(KEY_Q) && !swarm.parkourMode()) {
            const bool enableBiped = !swarm.bipedMode();
            // Q changes only morphology; the current task (walking/balance) stays selected.
            if (!swarm.learning()) swarm.setLearning(true);
            swarm.setBipedMode(enableBiped);
        }
        if (IsKeyPressed(KEY_TAB)) swarm.shown = (swarm.shown + 1) % (int)swarm.ag.size();
        if (IsKeyPressed(KEY_S)) { swarm.save(); lastSave = nowSec(); }
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !view.rearFollow) {
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
        view.target.x += (sim.cr.pos.x - view.target.x) * 0.12f;
        const float targetY = sim.balanceMode ? sim.spherePos.y + (sim.cr.biped ? 1.35f : 1.05f)
                                               : (sim.cr.biped ? 0.65f : 0.4f);
        view.target.y += (targetY - view.target.y) * 0.12f;
        view.target.z += (sim.cr.pos.z - view.target.z) * 0.12f;

        BeginDrawing();
        ClearBackground(rgba(30, 32, 36));
        drawWorld(swarm, view, GetTime());
        const std::string visibleStatus = nowSec() < modeStatusUntil ? modeStatus : std::string{};
        drawInterface(swarm, speed, paused, nowSec() - lastSave, GetFPS(), visibleStatus);
        EndDrawing();
    }
    swarm.save();          // última gravação da memória (melhor forma)
    unloadUiFont();
    CloseWindow();
    return 0;
}
