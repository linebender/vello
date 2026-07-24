#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-$repo_root/target}"
if [[ "$target_dir" != /* ]]; then
    target_dir="$repo_root/$target_dir"
fi
out_dir="$target_dir/sparse-strips-wasm/vello_hybrid_webgl"
wasm_bindgen="${WASM_BINDGEN:-wasm-bindgen}"

cd "$repo_root"
cargo build \
    --target-dir "$target_dir" \
    --locked \
    --package native_webgl \
    --bin native_webgl \
    --release \
    --target wasm32-unknown-unknown

mkdir -p "$out_dir"
"$wasm_bindgen" \
    --target web \
    --no-typescript \
    --out-dir "$out_dir" \
    --out-name vello_hybrid_webgl \
    "$target_dir/wasm32-unknown-unknown/release/native_webgl.wasm"
