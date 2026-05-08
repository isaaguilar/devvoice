#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="DevVoice"
APP_DIR="$ROOT_DIR/src-tauri/target/release/bundle/macos/${APP_NAME}.app"
DMG_DIR="$ROOT_DIR/src-tauri/target/release/bundle/dmg"
DMG_PATH="$(find "$DMG_DIR" -maxdepth 1 -type f -name "${APP_NAME}_*.dmg" | head -n 1)"
IDENTITY="${1:-${SIGNING_IDENTITY:-}}"

"$ROOT_DIR/scripts/package-macos-app.sh"

if [[ -n "$IDENTITY" ]]; then
  "$ROOT_DIR/scripts/sign-macos-app.sh" "$IDENTITY" "$APP_DIR"
fi

if [[ -z "$DMG_PATH" || ! -f "$DMG_PATH" ]]; then
  echo "DMG not found in $DMG_DIR after the Tauri build." >&2
  exit 1
fi

echo "Created installer at $DMG_PATH"
