#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-$BASE}"
[[ -d "$TARGET" ]] || { echo "Uso: $0 /caminho/da/TigerFlash" >&2; exit 2; }
ROOT="$(cd "$TARGET" && pwd)"
if [[ ! -d "$ROOT/android_runtime" ]]; then
  echo "TigerFlash: nenhuma android_runtime encontrada em $ROOT" >&2; exit 3
fi
cp -f "$BASE/android_runtime/tools/prepare_portable_apk_tools.sh" "$ROOT/android_runtime/tools/prepare_portable_apk_tools.sh"
cp -f "$BASE/android_runtime/tools/package_apk.sh" "$ROOT/android_runtime/tools/package_apk.sh"
cp -f "$BASE/android_runtime/tools/build_android_runtime.sh" "$ROOT/android_runtime/tools/build_android_runtime.sh"
cp -f "$BASE/android_runtime/runtime_source/TigerFlash_Android_Runtime.cpp" "$ROOT/android_runtime/runtime_source/TigerFlash_Android_Runtime.cpp"
chmod +x "$ROOT/android_runtime/tools"/*.sh
"$ROOT/android_runtime/tools/prepare_portable_apk_tools.sh"
if [[ ! -f "$ROOT/android_runtime/runtime_template/lib/arm64-v8a/libmain.so" ]]; then
  "$ROOT/android_runtime/tools/build_android_runtime.sh"
fi
echo "TigerFlash: Android runtime portátil pronto."
