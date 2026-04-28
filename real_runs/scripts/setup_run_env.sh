#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "usage: source real_runs/scripts/setup_run_env.sh <profile>" >&2
  exit 1
fi

scenario="${1:-small_medium}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$script_dir/../.." && pwd)"
profile_config="${REAL_RUN_PROFILE_CONFIG:-$root/real_runs/config/run_profiles.json}"

profile_exports="$(
  python3 - "$root" "$scenario" "$profile_config" <<'PY'
import json
import shlex
import sys
from pathlib import Path

root, scenario, config_path = sys.argv[1:]
config = Path(config_path).expanduser()
if not config.is_absolute():
    config = Path(root) / config
payload = json.loads(config.read_text())
profiles = payload.get("profiles", {})
if scenario not in profiles:
    names = "|".join(sorted(profiles))
    print(f"echo 'usage: source real_runs/scripts/setup_run_env.sh {{{names}}}' >&2")
    print("return 2")
    raise SystemExit(0)

values = {}
values.update(payload.get("common", {}))
values.update(profiles[scenario])
for key, value in values.items():
    if value is None:
        print(f"unset {shlex.quote(key)}")
    else:
        expanded = str(value).replace("{root}", root)
        print(f"export {shlex.quote(key)}={shlex.quote(expanded)}")
PY
)" || return

eval "$profile_exports"
mkdir -p "$REAL_RUN_DIR"
echo "REAL_RUN_DIR=$REAL_RUN_DIR"
