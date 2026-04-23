#!/usr/bin/env bash
set -euxo pipefail
$(dirname "$0")/create_local_env_for_cil_development.sh -t "$@"
