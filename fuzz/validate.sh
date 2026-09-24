#!/usr/bin/env bash
# Adds a cpu_gpu_differential finding to the vello_tests snapshot suite and runs it.
#
# The finding's test function is added to the scratch module vello_tests/tests/fuzz_regression.rs,
# a reference image is created from the CPU f32 scalar pipeline, and every CPU and GPU variant is
# compared against it. Everything stays in place, so the test can be rerun with
# `cargo test -p vello_tests --test tests fuzz_regression::<test_name>` while investigating.
# Rerunning this script for the same artifact replaces the function.
#
# Usage: fuzz/validate.sh <artifact | artifact.rs> [target flags such as --channel-tolerance=0]
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <artifact | artifact.rs> [target flags]" >&2
    exit 2
fi
artifact=$1
shift

root=$(cd "$(dirname "$0")/.." && pwd)
module_file=$root/vello_tests/tests/fuzz_regression.rs

cd "$root"
# An existing .rs carries the tolerances of the run that found the input; only regenerate it
# when other tolerances are requested explicitly.
if [[ $artifact == *.rs ]]; then
    test_source=$artifact
elif [ $# -eq 0 ] && [ -f "$artifact.rs" ]; then
    test_source=$artifact.rs
else
    cargo +nightly fuzz run cpu_gpu_differential "$artifact" -- --decode-to-rs "$@" > /dev/null
    test_source=$artifact.rs
fi
echo "Using $test_source"

test_name=$(sed -n 's/^fn \([a-z0-9_]*\)(ctx: .*/\1/p' "$test_source")
if [ -z "$test_name" ]; then
    echo "could not find the generated test in $test_source" >&2
    exit 1
fi
reference=$root/vello_tests/snapshots/$test_name.png

# The generated file is a complete module: imports, then one test function. Imports the scratch
# module does not have yet are appended; `use` items may appear anywhere at module level.
grep '^use ' "$test_source" | while IFS= read -r import; do
    if ! grep -Fxq "$import" "$module_file"; then
        echo "$import" >> "$module_file"
    fi
done
# Drop a previous version of the same function: its attribute, signature, and body.
awk -v name="$test_name" '
    /^#\[vello_test\(/ { attribute = $0 "\n"; in_attribute = 1; next }
    in_attribute {
        attribute = attribute $0 "\n"
        if ($0 ~ /^fn /) {
            in_attribute = 0
            if (index($0, "fn " name "(") == 1) { skipping = 1 } else { printf "%s", attribute }
        }
        next
    }
    skipping { if ($0 == "}") { skipping = 0 }; next }
    { print }
' "$module_file" > "$module_file.tmp"
mv "$module_file.tmp" "$module_file"
{
    echo
    sed -n '/^#\[vello_test(/,$p' "$test_source"
} >> "$module_file"
rustfmt --edition 2024 "$module_file" 2> /dev/null || true
# A stale reference would compare against a previous version of the test.
rm -f "$reference"

echo "Creating reference image from cpu_f32_scalar..."
cargo test -p vello_tests --test tests "fuzz_regression::${test_name}_cpu_f32_scalar" > /dev/null 2>&1 || true

echo "Comparing all variants against the reference..."
status=0
cargo test -p vello_tests --test tests "fuzz_regression::$test_name" || status=$?

cat <<EOF

Added fuzz_regression::$test_name to ${module_file#"$root/"}
  reference: ${reference#"$root/"}
Rerun:  cargo test -p vello_tests --test tests fuzz_regression::$test_name
Done:   move the test into its topic module under a descriptive name, rename the reference to
        match, and reset the scratch module with \`git checkout ${module_file#"$root/"}\`.
EOF
exit $status
