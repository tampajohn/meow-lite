#!/bin/bash
# Build MeowLite.app — no Xcode project, just swiftc + a bundle assembly.
# Usage: cd macos/MeowLite && ./build.sh && open build/MeowLite.app
set -euo pipefail
cd "$(dirname "$0")"

APP="build/MeowLite.app"
REPO_ROOT="$(cd ../.. && pwd)"

rm -rf build
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

echo "== swiftc =="
# -parse-as-library lets a single main.swift carry the SwiftUI @main attribute.
swiftc -O -parse-as-library -target arm64-apple-macos14 \
  -o "$APP/Contents/MacOS/MeowLite" main.swift

echo "== bundle =="
cp Info.plist "$APP/Contents/Info.plist"
# App icon + splash logo (same asset, per spec — included, not skipped).
if [[ -f "$REPO_ROOT/assets/logo.png" ]]; then
  cp "$REPO_ROOT/assets/logo.png" "$APP/Contents/Resources/logo.png"
else
  echo "note: assets/logo.png not found; app icon/splash logo omitted" >&2
fi

echo "== codesign (ad-hoc) =="
codesign --sign - --force "$APP"

cat <<EOF

built: $PWD/$APP
open:  open $PWD/$APP
EOF
