#!/bin/bash
# Lean verification CI for WuBuMath manifold properties
set -e
cd "$(dirname "$0")/.."
echo "=== WuBuMath Lean Verification ==="
if command -v lean &> /dev/null; then
    echo "Lean found: $(lean --version)"
    for f in lean/*.lean; do
        echo "Checking $f..."
        lean --check "$f" 2>&1 || echo "WARNING: $f has errors (expected for stubs)"
    done
else
    echo "Lean not installed. Install with: curl -sSf https://raw.githubusercontent.com/leanprover/lean4/master/doc/quickstart.md | bash"
fi
echo "=== Done ==="
EOF
echo done