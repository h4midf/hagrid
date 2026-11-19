#!/usr/bin/env bash
# Sets up everything the reproduction scripts run, without root:
#   .venv    the pinned Python packages of requirements.txt
#   .tools   Yosys 0.68 (with ABC) and ngspice-47, built from their release sources
#            (and the bison, flex, cmake and ninja their builds need)
# reproduce/ puts .tools/bin first on PATH.
# Usage: ./setup.sh
#
# Needs a C/C++ toolchain (make, g++ for NVSim, clang/clang++ for CACTI), m4, curl
# and tar. The pins in requirements.txt need Python >= 3.11. If no such interpreter
# is on PATH, a user-local Python 3.14 is fetched with uv (uv and the interpreter
# live under ~/.local).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

missing=()
for tool in make cc c++ g++ clang clang++ m4 curl tar; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [ "${#missing[@]}" -ne 0 ]; then
    echo "error: missing system tools: ${missing[*]}" >&2
    exit 1
fi

# ---- Python

TARGET_VERSION=3.14

py_version() {
    "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])'
}

py_ok() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null
}

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
    elif [ -x "$HOME/.local/bin/uv" ]; then
        echo "$HOME/.local/bin/uv"
    else
        return 1
    fi
}

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && py_ok "$candidate"; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "no Python >= 3.11 on PATH; fetching a user-local Python $TARGET_VERSION via uv"
    if ! UV="$(find_uv)"; then
        echo "installing uv into ~/.local/bin"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
        UV="$HOME/.local/bin/uv"
    fi
    "$UV" python install "$TARGET_VERSION"
    PYTHON_BIN="$("$UV" python find "$TARGET_VERSION")"
fi

VERSION="$(py_version "$PYTHON_BIN")"
echo "using $PYTHON_BIN (Python $VERSION)"
if ! py_ok "$PYTHON_BIN"; then
    echo "error: Python $VERSION is too old for requirements.txt (need >= 3.11)" >&2
    exit 1
fi

# Rebuild a .venv left over from a different interpreter version.
if [ -x .venv/bin/python ] && [ "$(py_version .venv/bin/python)" != "$VERSION" ]; then
    echo "existing .venv uses Python $(py_version .venv/bin/python); recreating"
    rm -rf .venv
fi

if [ ! -d .venv ]; then
    echo "creating .venv"
    "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# ---- Yosys and ngspice, into .tools

TOOLS="$PWD/.tools"
SRC="$TOOLS/src"
JOBS="$(getconf _NPROCESSORS_ONLN)"
mkdir -p "$SRC"
export PATH="$TOOLS/bin:$PATH"

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

# fetch <url> <file> <sha256>: downloads into .tools/src and checks the checksum
fetch() {
    local out="$SRC/$2"
    [ -f "$out" ] || curl -fL --retry 3 -o "$out" "$1"
    if [ "$(sha256 "$out")" != "$3" ]; then
        echo "error: $2 does not match its sha256 (delete it and rerun)" >&2
        exit 1
    fi
}

# has <command> <expected text in its version output> <version flag>
has() {
    [ -x "$TOOLS/bin/$1" ] || return 1
    local out
    out="$("$TOOLS/bin/$1" "$3" 2>&1 || true)"
    [[ "$out" == *"$2"* ]]
}

# autotools <dir name> <tarball> <configure args...>: unpacks, configures, builds, installs
autotools() {
    local dir="$1" tarball="$2"
    shift 2
    rm -rf "$SRC/$dir"
    tar xzf "$SRC/$tarball" -C "$SRC"
    (cd "$SRC/$dir" && ./configure --prefix="$TOOLS" "$@" && make -j"$JOBS" && make install)
    rm -rf "$SRC/$dir"
}

if ! has bison "bison (GNU Bison) 3.8.2" --version; then
    echo "building bison 3.8.2"
    fetch https://ftp.gnu.org/gnu/bison/bison-3.8.2.tar.gz bison-3.8.2.tar.gz \
        06c9e13bdf7eb24d4ceb6b59205a4f67c2c7e7213119644430fe82fbd14a0abb
    autotools bison-3.8.2 bison-3.8.2.tar.gz
fi

if ! has flex "flex 2.6.4" --version; then
    echo "building flex 2.6.4"
    fetch https://github.com/westes/flex/releases/download/v2.6.4/flex-2.6.4.tar.gz flex-2.6.4.tar.gz \
        e87aae032bf07c26f85ac0ed3250998c37621d95f8bd748b31f15b33c45ee995
    autotools flex-2.6.4 flex-2.6.4.tar.gz --disable-nls
fi

# The flags of Homebrew's ngspice formula, without the interactive front end
# (readline, X11).
if ! has ngspice "ngspice-47 " -v; then
    echo "building ngspice-47"
    fetch https://sourceforge.net/projects/ngspice/files/ng-spice-rework/47/ngspice-47.tar.gz \
        ngspice-47.tar.gz 894e649651f1838a14095e5a5439e7d3aa63e87ede14d283173fda4fcdef675f
    autotools ngspice-47 ngspice-47.tar.gz --enable-cider --enable-xspice --disable-openmp \
        --enable-pss --with-readline=no --with-x=no
fi

# The release tarball (the commit Homebrew's yosys 0.68 reports), with its bundled
# ABC and libraries; cmake and ninja come from a build-only venv.
if ! has yosys "Yosys 0.68+post (git sha1 c12172fbae8af5e20f6fb52e3d4e92d56ed587b6" -V; then
    echo "building Yosys 0.68"
    fetch https://github.com/YosysHQ/yosys/releases/download/v0.68/yosys.tar.gz yosys-0.68.tar.gz \
        ad8d2198e1a486e9089cc51a3158ecc764669267879518723fb98acc6fb24787
    [ -x "$TOOLS/build-venv/bin/cmake" ] || {
        "$PYTHON_BIN" -m venv "$TOOLS/build-venv"
        "$TOOLS/build-venv/bin/pip" install cmake==4.4.5 ninja==1.13.2
    }
    rm -rf "$SRC/yosys-0.68"
    mkdir -p "$SRC/yosys-0.68"
    tar xzf "$SRC/yosys-0.68.tar.gz" -C "$SRC/yosys-0.68"
    (
        export PATH="$TOOLS/build-venv/bin:$PATH"
        cd "$SRC/yosys-0.68"
        # no git: the version comes from the tarball's .gitcommit, not from the git
        # checkout this repository may be
        cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$TOOLS" \
            -DCMAKE_DISABLE_FIND_PACKAGE_Git=ON \
            -DPython3_EXECUTABLE="$(command -v "$PYTHON_BIN")" \
            -DFLEX_EXECUTABLE="$TOOLS/bin/flex" -DBISON_EXECUTABLE="$TOOLS/bin/bison" \
            -DYOSYS_USE_BUNDLED_LIBS=ON -DYOSYS_WITHOUT_SLANG=ON -DYOSYS_WITHOUT_TCL=ON \
            -DYOSYS_WITHOUT_READLINE=ON -DYOSYS_WITHOUT_EDITLINE=ON -DYOSYS_WITHOUT_LIBFFI=ON
        cmake --build build -j "$JOBS"
        cmake --install build
    )
    rm -rf "$SRC/yosys-0.68"
fi

for check in "yosys|Yosys 0.68+post (git sha1 c12172fbae8af5e20f6fb52e3d4e92d56ed587b6|-V" \
             "yosys-abc|[-c cmd]|-h" "ngspice|ngspice-47 |-v"; do
    IFS='|' read -r tool text flag <<< "$check"
    if ! has "$tool" "$text" "$flag"; then
        echo "error: .tools/bin/$tool is not the expected build" >&2
        exit 1
    fi
done

echo
echo "done. activate with: source .venv/bin/activate"
