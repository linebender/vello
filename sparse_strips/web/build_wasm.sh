#!/usr/bin/env bash

set -euo pipefail

if (( $# != 2 )); then
    printf 'Usage: %s <module> <non-simd|simd>\n' "$0" >&2
    exit 2
fi

module="$1"
variant="$2"

case "$module" in
    vello_cpu)
        package=wasm_cpu
        binary=wasm_cpu
        ;;
    vello_hybrid_webgl)
        package=native_webgl
        binary=native_webgl
        ;;
    vello_hybrid_wgpu)
        package=wgpu_webgl
        binary=wgpu_webgl
        ;;
    *)
        printf 'Unknown module: %s\n' "$module" >&2
        exit 2
        ;;
esac

case "$variant" in
    non-simd) target_feature=-simd128 ;;
    simd) target_feature=+simd128 ;;
    *)
        printf 'Unknown variant: %s (expected non-simd or simd)\n' "$variant" >&2
        exit 2
        ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-$repo_root/target}"
if [[ "$target_dir" != /* ]]; then
    target_dir="$repo_root/$target_dir"
fi
out_dir="$target_dir/sparse-strips-wasm/$module/$variant"
wasm_bindgen="${WASM_BINDGEN:-wasm-bindgen}"
rustflags="${RUSTFLAGS:-}"
rustflags="${rustflags:+$rustflags }-Ctarget-feature=$target_feature"

cd "$repo_root"
RUSTFLAGS="$rustflags" cargo build \
    --target-dir "$target_dir" \
    --locked \
    --package "$package" \
    --bin "$binary" \
    --profile wasm-size \
    --target wasm32-unknown-unknown

mkdir -p "$out_dir"
"$wasm_bindgen" \
    --target web \
    --no-typescript \
    --out-dir "$out_dir" \
    --out-name "$module" \
    "$target_dir/wasm32-unknown-unknown/wasm-size/$binary.wasm"
