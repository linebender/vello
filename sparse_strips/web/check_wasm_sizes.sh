#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-$repo_root/target}"
if [[ "$target_dir" != /* ]]; then
    target_dir="$repo_root/$target_dir"
fi

"$script_dir/build_vello_cpu.sh"
"$script_dir/build_vello_hybrid_webgl.sh"
"$script_dir/build_vello_hybrid_wgpu.sh"

failed=0

check_size() {
    local name="$1"
    local file="$2"
    local limit="$3"
    local size
    local actual_mib
    local limit_mib

    if [[ ! -f "$file" ]]; then
        printf '%s: expected module was not produced: %s\n' "$name" "$file" >&2
        failed=1
        return
    fi

    size="$(wc -c < "$file" | tr -d '[:space:]')"
    actual_mib="$(awk -v bytes="$size" 'BEGIN { printf "%.2f MiB", bytes / 1048576 }')"
    limit_mib="$(awk -v bytes="$limit" 'BEGIN { printf "%.2f MiB", bytes / 1048576 }')"
    printf '%-22s %10s %10s\n' "$name" "$actual_mib" "$limit_mib"

    if (( size > limit )); then
        printf '  Size limit exceeded by %d bytes.\n' "$((size - limit))" >&2
        failed=1
    fi
}

printf '\n%-22s %10s %10s\n' "WebAssembly module" "Actual" "Limit"
check_size \
    "vello_cpu" \
    "$target_dir/sparse-strips-wasm/vello_cpu/vello_cpu_bg.wasm" \
    4456448
check_size \
    "vello_hybrid_webgl" \
    "$target_dir/sparse-strips-wasm/vello_hybrid_webgl/vello_hybrid_webgl_bg.wasm" \
    4456448
check_size \
    "vello_hybrid_wgpu" \
    "$target_dir/sparse-strips-wasm/vello_hybrid_wgpu/vello_hybrid_wgpu_bg.wasm" \
    8126464

exit "$failed"
