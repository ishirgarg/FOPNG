#!/usr/bin/env bash
set -euo pipefail

BASE="experiments"

for outer in "$BASE"/*; do
    [ -d "$outer" ] || continue   # skip non-dirs
    name=$(basename "$outer")
    inner="$outer/$name"

    # Check for redundant subfolder
    if [ -d "$inner" ]; then
        echo "Flattening: $outer"

        # Move inner contents to outer
        mv "$inner"/* "$outer"/ 2>/dev/null || true

        # Remove now-empty subfolder
        rmdir "$inner"
    fi
done
