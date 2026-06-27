#!/usr/bin/env bash
# Build third-party simulators: SCALE-Sim, Ramulator 2.0, BookSim 2.0, DSENT
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
PIP="${PIP:-python3 -m pip}"

mkdir -p "$TP"
cd "$TP"

build_scalesim() {
  if [ ! -d SCALE-Sim ]; then
    git clone --depth 1 https://github.com/scalesim-project/SCALE-Sim.git
  fi
  cd SCALE-Sim
  patch -p1 -N -i "$ROOT/scripts/patches/scalesim-numpy-max.patch" || true
  cd "$TP"
  $PIP install -e "$TP/SCALE-Sim"
}

build_ramulator() {
  if [ ! -d ramulator2 ]; then
    git clone --depth 1 https://github.com/CMU-SAFARI/ramulator2.git
  fi
  cd ramulator2
  patch -p1 -N -i "$ROOT/scripts/patches/ramulator2-drain.patch" || true
  mkdir -p build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  cmake --build . -j"$(nproc)"
  cd "$TP"
}

build_booksim() {
  if [ ! -d booksim2 ]; then
    git clone --depth 1 https://github.com/booksim/booksim2.git
  fi
  cd booksim2/src
  make -j"$(nproc)"
  cd "$TP"
}

build_dsent() {
  if [ ! -d dsent_standalone ]; then
    git clone --depth 1 https://github.com/gyb1325/Desent_modification.git dsent_standalone
  fi
  cd dsent_standalone
  make -j"$(nproc)" LDFLAGS="-no-pie" || true
  cd "$TP"
}

echo "Building SCALE-Sim..."
build_scalesim || echo "SCALE-Sim install failed; analytic core fallback will be used."

echo "Building Ramulator 2.0..."
build_ramulator || echo "Ramulator build failed; analytic DRAM fallback will be used."

echo "Building BookSim 2.0..."
build_booksim || echo "BookSim build failed; analytic NoC fallback will be used."

echo "Building DSENT..."
build_dsent || echo "DSENT build failed; analytic power fallback will be used."

echo "Done. Expected paths:"
echo "  $TP/SCALE-Sim/  (pip install -e)"
echo "  $TP/ramulator2/build/ramulator2"
echo "  $TP/booksim2/src/booksim"
echo "  $TP/dsent_standalone/dsent"
