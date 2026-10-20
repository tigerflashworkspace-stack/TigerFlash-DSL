# AI Workspace — pacote de compilação para Windows

Este pacote parte do `AiWorskspace.cpp` enviado pelo usuário. O objetivo é permitir a compilação para Windows **sem reduzir a qualidade gráfica, a resolução, a taxa de quadros, o tamanho do cérebro, a física ou os recursos de aprendizado**.

## O que foi alterado

- Foi mantido o código original da simulação.
- A única alteração de código que afeta compilação foi tornar `MULTITHREAD` configurável via `-DMULTITHREAD=0/1`; o padrão continua sendo `1`, igual ao original.
- Foram adicionados scripts de compilação e execução para Windows.

As configurações originais são preservadas: 10 IAs no enxame, janela 1280×720, alvo de 60 FPS, MSAA 4×, mesma rede neural, mesma física, mesmos modos e arquivos de memória. O alvo de 60 FPS continua sendo um limite solicitado ao raylib, não uma garantia para todo hardware/driver.

## Estrutura esperada

Coloque uma distribuição de desenvolvimento da raylib **compilada para o mesmo compilador e arquitetura** dentro de `raylib`, de forma que existam:

- `raylib\src\raylib.h`
- `raylib\src\libraylib.a` (ou `raylib\lib\libraylib.a`)

Também é possível apontar a variável `RAYLIB_DIR` para a pasta onde sua raylib está instalada.

## Compilar com MinGW-w64

1. Instale MinGW-w64 com GCC e suporte a `std::thread` (modelo POSIX recomendado) e adicione a pasta `bin` ao `PATH`.
2. Instale/prepare uma raylib compilada para **esse mesmo** MinGW-w64. Não misture bibliotecas de 32-bit com compilador de 64-bit, nem uma biblioteca MSVC (`.lib`) com MinGW (`.a`).
3. Execute `BUILD_WINDOWS_MINGW.bat`.
4. Para iniciar, use `RUN_WINDOWS.bat` ou abra `AIWorkspace.exe`.

Se a instalação do MinGW não suportar o modelo POSIX de threads, `BUILD_WINDOWS_SINGLE_THREAD.bat` compila com `MULTITHREAD=0`. Isso mantém qualidade gráfica, resolução, física e os modos; apenas o enxame passa a avançar numa thread em vez de distribuir agentes entre threads.

## Versões do Windows

O fonte usa C++17 e raylib. O suporte real depende da versão do Windows, do compilador, da versão da raylib/GLFW e dos drivers OpenGL instalados. Windows 7/8.1/10/11 são os alvos modernos mais plausíveis, mas ainda dependem de uma combinação compatível e precisam ser testados na máquina final.

Não é correto prometer que o mesmo executável roda em **todas** as versões do Windows. XP e Vista precisam de uma toolchain e de versões antigas compatíveis das dependências; releases recentes do GLFW retiraram suporte a esses sistemas. Para referência, veja as [releases oficiais da raylib](https://github.com/raysan5/raylib/releases) e as [notas oficiais do GLFW](https://github.com/glfw/glfw/releases).

## Notas importantes

- Este pacote contém fonte e scripts, não um `.exe` testado. O ambiente onde este pacote foi preparado não tinha MinGW-w64 nem raylib instalados, então não foi possível executar a compilação Windows aqui.
- A aplicação salva os cérebros `Brain_*.bin` na pasta de trabalho atual. Faça cópia de segurança das memórias se trocar ou mover a pasta do programa.
- Se aparecer um erro ao compilar, envie o texto integral do terminal. Não é necessário reduzir a qualidade para resolver dependências ou erros de compilação.
