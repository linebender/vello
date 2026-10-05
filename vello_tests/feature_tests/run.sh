#!/usr/bin/env bash
set -euo pipefail

crate_dir=$(cd "$(dirname "$0")" && pwd)
backend=""

usage() {
    echo "usage: $0 --features [wgpu|webgl] [wasm-pack options] [cargo test arguments]" >&2
}

command_args=(test)
cargo_args=(--locked --no-default-features)
while [[ $# -gt 0 ]]; do
    case $1 in
        --features)
            if [[ $# -lt 2 || -n $backend ]]; then
                usage
                exit 2
            fi
            backend=$2
            shift
            ;;
        --headless|--chrome|--firefox|--safari|--release)
            command_args+=("$1")
            ;;
        --mode)
            if [[ $# -lt 2 || $2 == -* ]]; then
                usage
                exit 2
            fi
            command_args+=("$1" "$2")
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            cargo_args+=("$@")
            break
            ;;
        *) cargo_args+=("$1") ;;
    esac
    shift
done

case $backend in
    wgpu) test_command=(cargo "${command_args[@]}" --manifest-path "$crate_dir/Cargo.toml" -p vello_gpu_feature_tests) ;;
    webgl) test_command=(wasm-pack "${command_args[@]}" "$crate_dir") ;;
    *) usage; exit 2 ;;
esac

features=(blurred_rounded_rect image_bicubic gradient_sweep)
all_features=$(IFS=,; echo "${features[*]}")
for combination in "" "${features[@]}" "$all_features"; do
    echo "Testing $backend: ${combination:-no shader features}"
    "${test_command[@]}" --test features --features "$backend${combination:+,$combination}" \
        "${cargo_args[@]}"
done
