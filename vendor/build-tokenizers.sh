#!/usr/bin/env bash
# Build a free-threaded `tokenizers` wheel, which HuggingFace does not publish.
#
# Nothing here is a port. tokenizers 0.23.1 already declares `#[pymodule(gil_used = false)]`
# in every one of its Rust modules — it supports free-threading and says so. What it does
# not have is a wheel: its `[tool.maturin] features` hardcodes `abi3`, and there is no
# stable ABI for a free-threaded build, so every published wheel is unusable on `cp314t`
# and the sdist cannot link. Dropping that one word is the entire patch.
#
# Rebuild after changing VERSION, then update the path in `[tool.uv.sources]`.
# See docs/adr/0006-free-threaded-python-only.md.
#
#   Requires: rust (brew install rust), uv, and a cp314t interpreter.
#   Usage:    vendor/build-tokenizers.sh [path-to-cp314t-python]

set -euo pipefail

VERSION="0.23.1"
SDIST_SHA256="1feeeadf865a7915adc25445dea30e9933e593c31bb96c277cee36de227c8bfa"
SDIST_URL="https://files.pythonhosted.org/packages/c1/60/21f715d9faba5f5407ff759472ade058ec4a507ad62bcea47cb847239a73/tokenizers-${VERSION}.tar.gz"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${1:-$here/../.venv/bin/python}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "==> fetching tokenizers ${VERSION}"
curl -sSL -o "$work/sdist.tar.gz" "$SDIST_URL"
echo "${SDIST_SHA256}  $work/sdist.tar.gz" | shasum -a 256 -c -
tar xzf "$work/sdist.tar.gz" -C "$work"
src="$work/tokenizers-${VERSION}"

echo "==> dropping abi3, which cannot coexist with a free-threaded build"
before="$(grep -c 'features = \["pyo3/extension-module", "abi3"\]' "$src/pyproject.toml")"
[ "$before" = "1" ] || { echo "maturin features line not found — upstream changed it"; exit 1; }
sed -i '' 's|features = \["pyo3/extension-module", "abi3"\]|features = ["pyo3/extension-module"]|' \
    "$src/pyproject.toml"

echo "==> building against $python_bin"
# Pins the zip mtimes, which removes one source of variance but not all of them: rustc
# embeds absolute build paths, and this builds in a fresh temp directory each time, so two
# builds of identical source still differ by a few bytes. `uv.lock` records this wheel's
# SHA-256, so a rebuild always needs the lock refreshed — see the note at the end.
export SOURCE_DATE_EPOCH=315532800  # 1980-01-01, the earliest a zip can represent
# macOS extension modules resolve CPython symbols at load time; without this the link
# fails on `_PyBaseObject_Type` and friends.
cd "$src"
RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" \
    uvx maturin build --release --interpreter "$python_bin" --out "$here"

echo "==> built into $here"
ls -1 "$here"/*.whl
shasum -a 256 "$here"/*.whl
echo
echo "The bytes differ on every build even though the source does not, so refresh the lock:"
echo "  uv lock --upgrade-package tokenizers"
