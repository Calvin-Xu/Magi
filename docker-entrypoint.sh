#!/usr/bin/env sh
set -e

# Optional: Debug / verbose mode
# set -x

PATH=/app/bin:$PATH

echo "Running basic Python checks..."
python -V
python -Ic 'import magi'

if [ $# -gt 0 ]; then
  echo "Executing user-provided command: $*"
  exec "$@"
else
  echo "No command provided; running python -m magi.main"
  exec python -m magi.main
fi
