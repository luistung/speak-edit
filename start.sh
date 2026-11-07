#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
venv_dir="$project_dir/.venv"
venv_python="$venv_dir/bin/python"

if ! command -v python3 >/dev/null 2>&1; then
    echo "请先安装 Python 3.11 或更新版本。" >&2
    exit 1
fi

if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    echo "需要 Python 3.11 或更新版本，请检查 python3 的版本。" >&2
    exit 1
fi

if [ ! -x "$venv_python" ]; then
    echo "正在创建虚拟环境…"
    python3 -m venv "$venv_dir"
fi

"$venv_python" -m pip install --quiet --disable-pip-version-check -r "$project_dir/requirements.txt"
exec "$venv_python" "$project_dir/whisper_recorder.py" "$@"
