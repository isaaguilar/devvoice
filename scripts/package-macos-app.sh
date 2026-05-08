#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="DevVoice"
APP_DIR="$ROOT_DIR/src-tauri/target/release/bundle/macos/${APP_NAME}.app"

if [[ ! -d "$ROOT_DIR/node_modules" ]]; then
  (cd "$ROOT_DIR" && npm install)
fi

(cd "$ROOT_DIR" && npm run tauri:build)

echo "Created app bundle at $APP_DIR"
