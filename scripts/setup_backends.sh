#!/usr/bin/env bash
# Build third-party simulators: Ramulator 2.0, BookSim 2.0, DSENT
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"

mkdir -p "$TP"
cd "$TP"

build_ramulator() {
  if [ ! -d ramulator2 ]; then
    git clone --depth 1 https://github.com/CMU-SAFARI/ramulator2.git
  fi
  cd ramulator2
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
  if [ ! -d dsent ]; then
    git clone --depth 1 https://github.com/dsent-group/dsent.git
  fi
  cd dsent
  make -j"$(nproc)" || true
  cd "$TP"
}

echo "Building Ramulator 2.0..."
build_ramulator || echo "Ramulator build failed; analytic DRAM fallback will be used."

echo "Building BookSim 2.0..."
build_booksim || echo "BookSim build failed; analytic NoC fallback will be used."

echo "Building DSENT..."
build_dsent || echo "DSENT build failed; analytic power fallback will be used."

echo "Done. Binaries expected at:"
echo "  $TP/ramulator2/build/ramulator2"
echo "  $TP/booksim2/src/booksim"
echo "  $TP/dsent/dsent"
