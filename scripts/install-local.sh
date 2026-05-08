#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="DevVoice"
SOURCE_APP="$ROOT_DIR/src-tauri/target/release/bundle/macos/${APP_NAME}.app"
INSTALL_ROOT="${1:-$HOME/Applications}"
INSTALL_ROOT="${INSTALL_ROOT%/}"
DEST_APP="$INSTALL_ROOT/${APP_NAME}.app"

"$ROOT_DIR/scripts/package-macos-app.sh"
"$ROOT_DIR/scripts/sign-macos-app.sh" - "$SOURCE_APP"

mkdir -p "$INSTALL_ROOT"
rm -rf "$DEST_APP"
ditto "$SOURCE_APP" "$DEST_APP"
xattr -dr com.apple.quarantine "$DEST_APP" 2>/dev/null || true

echo "Installed $DEST_APP"
