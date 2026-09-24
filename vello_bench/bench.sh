#!/usr/bin/env bash
set -euo pipefail

crate_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(git -C "$crate_dir" rev-parse --show-toplevel)"
target_dir="$repo_dir/target"
artifact_dir="$target_dir/vello-bench-artifacts"
worktree="$target_dir/vello-bench-worktree"
generated_dir="$crate_dir/web/generated"
mode="${1:-help}"
shift || true
cd "$repo_dir"

usage() {
  echo "usage:"
  echo "  ./bench.sh cli [--ab REVISION_A REVISION_B] [FILTER] [--extended] [--non-simd] [--f32] [--warmup-ms MILLIS] [--measurement-ms MILLIS] [--samples COUNT]"
  echo "  ./bench.sh web [--ab REVISION_A REVISION_B]"
}

cleanup_comparison() {
  local status="$?"
  trap - EXIT
  git worktree remove --force "$worktree" || status=1
  exit "$status"
}

begin_comparison() {
  if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "comparison requires a clean Git checkout" >&2
    exit 1
  fi
  mkdir -p "$target_dir" "$artifact_dir"
  git worktree add --quiet --detach "$worktree" HEAD
  trap cleanup_comparison EXIT
}

select_revision() {
  local revision="$1"
  local benchmark_dir macros_dir
  git -C "$worktree" clean -q -fd -- .
  git -C "$worktree" switch --quiet --discard-changes --detach "$revision"
  if [[ -d "$worktree/vello_bench" ]]; then
    benchmark_dir="$worktree/vello_bench"
    macros_dir="$worktree/vello_tests/vello_dev_macros"
  else
    benchmark_dir="$worktree/sparse_strips/vello_bench"
    macros_dir="$worktree/sparse_strips/vello_dev_macros"
  fi
  cp -pR "$crate_dir/." "$benchmark_dir/"
  cp -pR "$repo_dir/vello_tests/vello_dev_macros/." "$macros_dir/"
}

build_native_revision() {
  local revision="$1"
  local label="$2"
  local targets=(--lib)
  if [[ "$label" == a ]]; then
    targets+=(--bin vello-bench)
  fi
  select_revision "$revision"
  cargo build --manifest-path "$worktree/Cargo.toml" --release \
    -p vello_bench "${targets[@]}" --target-dir "$target_dir"
  cp "$target_dir/release/$native_library" "$artifact_dir/vello-bench-$label.$library_extension"
  if [[ "$label" == a ]]; then
    cp "$target_dir/release/vello-bench" "$artifact_dir/vello-bench"
  fi
}

build_wasm() {
  local source_dir="$1"
  local output="$2"
  RUSTFLAGS="-Ctarget-feature=+simd128" cargo build \
    --manifest-path "$source_dir/Cargo.toml" --release -p vello_bench --lib \
    --target wasm32-unknown-unknown --target-dir "$target_dir"
  cp "$target_dir/wasm32-unknown-unknown/release/vello_bench.wasm" "$output"
}

case "$mode" in
  cli)
    if [[ "${1:-}" != "--ab" ]]; then
      cargo run --release -p vello_bench --bin vello-bench -- run "$@"
      exit
    fi
    revision_a="$(git rev-parse --verify "${2:?missing revision A}^{commit}")"
    revision_b="$(git rev-parse --verify "${3:?missing revision B}^{commit}")"
    shift 3
    case "$(uname -s)" in
      Darwin) library_extension=dylib ;;
      Linux) library_extension=so ;;
      *) echo "native A/B requires macOS or Linux" >&2; exit 2 ;;
    esac
    native_library="libvello_bench.$library_extension"
    begin_comparison
    build_native_revision "$revision_a" a
    build_native_revision "$revision_b" b
    "$artifact_dir/vello-bench" compare \
      "$artifact_dir/vello-bench-a.$library_extension" \
      "$artifact_dir/vello-bench-b.$library_extension" "$@"
    ;;
  web)
    if [[ "${1:-}" == "--ab" ]]; then
      revision_a="$(git rev-parse --verify "${2:?missing revision A}^{commit}")"
      revision_b="$(git rev-parse --verify "${3:?missing revision B}^{commit}")"
      shift 3
      if (($#)); then echo "unknown web option: $1" >&2; exit 2; fi
      begin_comparison
      mkdir -p "$generated_dir"
      select_revision "$revision_a"
      build_wasm "$worktree" "$generated_dir/vello_bench_a.wasm"
      select_revision "$revision_b"
      build_wasm "$worktree" "$generated_dir/vello_bench_b.wasm"
    else
      if (($#)); then echo "unknown web option: $1" >&2; exit 2; fi
      mkdir -p "$generated_dir"
      rm -f "$generated_dir/vello_bench_b.wasm"
      build_wasm "$repo_dir" "$generated_dir/vello_bench_a.wasm"
    fi
    python3 -m http.server --directory "$crate_dir/web" 8000
    ;;
  *)
    usage
    ;;
esac
