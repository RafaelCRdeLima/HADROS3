#!/usr/bin/env bash
# Build bin/hadros3_geodesic_preview_cuda, the self-contained CUDA camera preview.
#
# Usage: build_cuda_preview.sh <required 0|1>
#
# A CUDA toolkit installed by conda and one installed by the distribution need
# different flags, and so do their GLFW/OpenGL counterparts:
#
#   * conda's nvcc defaults to conda's gcc, which is frequently newer than the
#     CUDA release accepts, so a system g++ is preferred as the host compiler;
#   * conda's linker does not resolve the X11 symbols of a system libglfw, while
#     a system linker does not see conda's headers.
#
# Rather than guessing, the script tries the sensible combinations in order and
# keeps the first that produces a binary. The interactive (GLFW) configurations
# come first; a headless build is the last resort and is reported as such.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQUIRED="${1:-0}"
OUTPUT="${ROOT}/bin/hadros3_geodesic_preview_cuda"
SOURCE="${ROOT}/cpp/cuda/hadros3_geodesic_preview_cuda.cu"
LOG="${ROOT}/bin/.cuda_preview_build.log"

NVCC="${NVCC:-nvcc}"
NVCCFLAGS="${NVCCFLAGS:--O3 -std=c++17}"
NVCC_ARCH_FLAGS="${NVCC_ARCH_FLAGS:--arch=all-major}"

say() { echo "[hadros3_geodesic_preview_cuda] $*"; }

if ! command -v "${NVCC}" >/dev/null 2>&1; then
  say "nvcc not found: ${NVCC}"
  say "the interactive camera preview needs the CUDA toolkit and an NVIDIA GPU."
  say "run 'make install' (it installs the CUDA compiler from conda-forge when a GPU is present)"
  say "or install it yourself: sudo apt install nvidia-cuda-toolkit libglfw3-dev pkg-config"
  [ "${REQUIRED}" = "1" ] && exit 1
  exit 0
fi

NVCC_PATH="$(command -v "${NVCC}")"
CUDA_ENV_PREFIX="$(dirname "$(dirname "${NVCC_PATH}")")"

# Host compiler: a system g++ is the safer default for conda CUDA toolkits.
CCBIN=""
for candidate in "${NVCC_CCBIN:-}" /usr/bin/g++ "$(command -v g++ 2>/dev/null || true)"; do
  if [ -n "${candidate}" ] && [ -x "${candidate}" ]; then
    CCBIN="${candidate}"
    break
  fi
done

system_glfw_flags() {
  command -v pkg-config >/dev/null 2>&1 || return 1
  pkg-config --exists glfw3 || return 1
  echo "$(pkg-config --cflags --libs glfw3) -lGL"
}

conda_glfw_flags() {
  local pc="${CUDA_ENV_PREFIX}/lib/pkgconfig/glfw3.pc"
  [ -f "${pc}" ] || return 1
  local flags
  flags="$(PKG_CONFIG_PATH="${CUDA_ENV_PREFIX}/lib/pkgconfig" pkg-config --cflags --libs glfw3)" || return 1
  echo "${flags} -L${CUDA_ENV_PREFIX}/lib -lGL -Xlinker -rpath -Xlinker ${CUDA_ENV_PREFIX}/lib"
}

mkdir -p "${ROOT}/bin"

attempt() {
  local label="$1" ccbin="$2" glfw_flags="$3" define="$4"
  local command=("${NVCC}" ${NVCCFLAGS} ${NVCC_ARCH_FLAGS})
  [ -n "${ccbin}" ] && command+=(-ccbin "${ccbin}")
  [ -n "${define}" ] && command+=("${define}")
  command+=("${SOURCE}" -o "${OUTPUT}" ${glfw_flags})

  say "trying: ${label}"
  if "${command[@]}" >"${LOG}" 2>&1; then
    say "built with: ${label}"
    return 0
  fi
  return 1
}

SYSTEM_GLFW="$(system_glfw_flags || true)"
CONDA_GLFW="$(conda_glfw_flags || true)"

if [ -n "${SYSTEM_GLFW}" ] && attempt "system GLFW, host compiler ${CCBIN:-default}" "${CCBIN}" "${SYSTEM_GLFW}" "-DHADROS_CUDA_PREVIEW_GLFW"; then
  exit 0
fi
if [ -n "${CONDA_GLFW}" ] && attempt "conda GLFW, host compiler ${CCBIN:-default}" "${CCBIN}" "${CONDA_GLFW}" "-DHADROS_CUDA_PREVIEW_GLFW"; then
  exit 0
fi
if [ -n "${CONDA_GLFW}" ] && attempt "conda GLFW, default host compiler" "" "${CONDA_GLFW}" "-DHADROS_CUDA_PREVIEW_GLFW"; then
  exit 0
fi
if [ -n "${SYSTEM_GLFW}" ] && attempt "system GLFW, default host compiler" "" "${SYSTEM_GLFW}" "-DHADROS_CUDA_PREVIEW_GLFW"; then
  exit 0
fi

say "no GLFW configuration linked; falling back to a HEADLESS renderer (no interactive window)"
say "install libglfw3-dev (or add 'glfw' to the conda environment) and rebuild for the live camera"
if attempt "headless, host compiler ${CCBIN:-default}" "${CCBIN}" "" ""; then
  exit 0
fi
if attempt "headless, default host compiler" "" "" ""; then
  exit 0
fi

say "the CUDA preview failed to build; last compiler output:"
tail -n 30 "${LOG}"
[ "${REQUIRED}" = "1" ] && exit 1
exit 0
