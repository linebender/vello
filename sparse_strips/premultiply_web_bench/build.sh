#!/usr/bin/env bash

set -euo pipefail

variant="${1:-simd}"
case "$variant" in
    simd) target_feature=+simd128 ;;
    non-simd) target_feature=-simd128 ;;
    *)
        printf 'Usage: %s [simd|non-simd]\n' "$0" >&2
        exit 2
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-$repo_root/target}"
if [[ "$target_dir" != /* ]]; then
    target_dir="$repo_root/$target_dir"
fi
out_dir="$script_dir/pkg"
rustflags="${RUSTFLAGS:-}"
rustflags="${rustflags:+$rustflags }-Ctarget-feature=$target_feature"

cd "$repo_root"
RUSTFLAGS="$rustflags" cargo build \
    --target-dir "$target_dir" \
    --locked \
    --package premultiply_web_bench \
    --profile wasm-size \
    --target wasm32-unknown-unknown

mkdir -p "$out_dir"
cp \
    "$target_dir/wasm32-unknown-unknown/wasm-size/premultiply_web_bench.wasm" \
    "$out_dir/premultiply_web_bench.wasm"

printf 'Built %s benchmark. Serve it with:\n\n' "$variant"
printf '  python3 -m http.server 8000 --directory %q\n\n' "$script_dir"
printf 'Then open http://localhost:8000\n'
