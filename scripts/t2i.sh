#!/usr/bin/env bash
# Thin shim kept for backward compatibility; all logic lives in t2i.py.
#   t2i.sh doctor | login | generate -p "..." [--json] | stop
#   t2i.sh -p "..."        (same as: generate -p "...")
#   t2i.sh --stop-only     (same as: stop)
exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/t2i.py" "$@"
