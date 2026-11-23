#!/usr/bin/env bash
# Builds the Glifo web harness into $CARGO_TARGET_DIR/glifo_web_harness and optionally serves it.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: build.sh [--profile <release|dev|wasm-size>] [--no-simd] [--serve] [--port <port>]

  --profile   Cargo profile (default: release). wasm-size enables fat LTO.
  --no-simd   Build without the simd128 target feature.
  --serve     Serve the output directory after building, with cross-origin isolation headers.
  --port      Port used by --serve (default: 8000).

Requires the wasm-bindgen CLI version from Cargo.lock. Set WASM_BINDGEN to use a CLI that is not
on PATH.
EOF
}

profile=release
simd=1
serve=0
port=8000
while (( $# )); do
    case "$1" in
        --profile) profile="$2"; shift ;;
        --no-simd) simd=0 ;;
        --serve) serve=1 ;;
        --port) port="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
    shift
done

harness_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$harness_dir/../../.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-$repo_root/target}"
if [[ "$target_dir" != /* ]]; then
    target_dir="$repo_root/$target_dir"
fi
out_dir="$target_dir/glifo_web_harness"

case "$profile" in
    dev) profile_dir=debug ;;
    *) profile_dir="$profile" ;;
esac

cd "$repo_root"

# `--config` arrays merge with .cargo/config.toml, unlike RUSTFLAGS, which would drop its
# `getrandom_backend` cfg.
cargo_args=(
    build
    --locked
    --lib
    --package glifo_web_harness
    --profile "$profile"
    --target wasm32-unknown-unknown
    --target-dir "$target_dir"
)
if (( simd )); then
    cargo_args+=(--config "target.wasm32-unknown-unknown.rustflags=['-Ctarget-feature=+simd128']")
fi
cargo "${cargo_args[@]}"

# A CLI that does not match the `wasm-bindgen` crate fails with an opaque schema error, so check first.
locked_version="$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/version = |"/, ""); print; exit }' Cargo.lock)"
wasm_bindgen="${WASM_BINDGEN:-wasm-bindgen}"
found_version="$("$wasm_bindgen" --version 2>/dev/null || true)"
if [[ "$found_version" != "wasm-bindgen $locked_version" ]]; then
    printf 'error: need wasm-bindgen %s, found %s\n' "$locked_version" "${found_version:-no $wasm_bindgen}" >&2
    printf 'Install it with: cargo install wasm-bindgen-cli --version %s --locked --force\n' "$locked_version" >&2
    printf 'or set WASM_BINDGEN to a matching CLI.\n' >&2
    exit 1
fi

rm -rf "$out_dir"
mkdir -p "$out_dir/pkg"
"$wasm_bindgen" \
    --target web \
    --no-typescript \
    --out-dir "$out_dir/pkg" \
    --out-name glifo_web_harness \
    "$target_dir/wasm32-unknown-unknown/$profile_dir/glifo_web_harness.wasm"
cp -R "$harness_dir/web/." "$out_dir/"

printf 'Built %s (profile: %s, simd128: %s)\n' "$out_dir" "$profile" "$( (( simd )) && echo on || echo off )"

if (( serve )); then
    printf 'Serving on http://localhost:%s/\n' "$port"
    # Cross-origin isolation raises the `performance.now()` resolution from 100 µs to 5 µs, which
    # matters for sub-millisecond frame timings.
    exec python3 - "$port" "$out_dir" <<'EOF'
import functools
import http.server
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


port, directory = int(sys.argv[1]), sys.argv[2]
handler = functools.partial(Handler, directory=directory)
http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
EOF
fi
