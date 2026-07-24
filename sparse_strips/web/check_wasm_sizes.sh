#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-$repo_root/target}"
if [[ "$target_dir" != /* ]]; then
    target_dir="$repo_root/$target_dir"
fi

for module in vello_cpu vello_hybrid_webgl vello_hybrid_wgpu; do
    "$script_dir/build_wasm.sh" "$module" simd
done

failed=0

check_size() {
    local name="$1"
    local file="$2"
    local raw_limit="$3"
    local gzip_file="$file.gz"
    local size
    local gzip_size
    local actual_mib
    local raw_limit_mib
    local gzip_actual_mib

    if [[ ! -f "$file" ]]; then
        printf '%s: expected module was not produced: %s\n' "$name" "$file" >&2
        failed=1
        return
    fi

    gzip --force --keep "$file"
    size="$(wc -c < "$file" | tr -d '[:space:]')"
    gzip_size="$(wc -c < "$gzip_file" | tr -d '[:space:]')"
    actual_mib="$(awk -v bytes="$size" 'BEGIN { printf "%.2f MiB", bytes / 1048576 }')"
    raw_limit_mib="$(awk -v bytes="$raw_limit" 'BEGIN { printf "%.2f MiB", bytes / 1048576 }')"
    gzip_actual_mib="$(awk -v bytes="$gzip_size" 'BEGIN { printf "%.2f MiB", bytes / 1048576 }')"
    printf '%-22s %10s %10s %10s\n' \
        "$name" "$actual_mib" "$raw_limit_mib" "$gzip_actual_mib"

    if (( size > raw_limit )); then
        printf '  Raw size limit exceeded by %d bytes.\n' "$((size - raw_limit))" >&2
        failed=1
    fi
}

printf '\nBuild profile: SIMD128, opt-level=3, fat LTO, codegen-units=1\n'
printf '%-22s %10s %10s %10s\n' \
    "WebAssembly module" "Raw" "Raw limit" "Gzip"
check_size \
    "vello_cpu" \
    "$target_dir/sparse-strips-wasm/vello_cpu/simd/vello_cpu_bg.wasm" \
    3774874 # 3.6 MiB
check_size \
    "vello_hybrid_webgl" \
    "$target_dir/sparse-strips-wasm/vello_hybrid_webgl/simd/vello_hybrid_webgl_bg.wasm" \
    3774874 # 3.6 MiB
check_size \
    "vello_hybrid_wgpu" \
    "$target_dir/sparse-strips-wasm/vello_hybrid_wgpu/simd/vello_hybrid_wgpu_bg.wasm" \
    7340032 # 7 MiB

exit "$failed"
