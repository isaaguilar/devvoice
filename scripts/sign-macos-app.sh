#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IDENTITY="${1:-}"
APP_DIR="${2:-$ROOT_DIR/src-tauri/target/release/bundle/macos/DevVoice.app}"

if [[ -z "$IDENTITY" ]]; then
  echo "Usage: $0 <codesign-identity-or--> [app-path]" >&2
  exit 1
fi

if [[ ! -d "$APP_DIR" ]]; then
  echo "Bundle not found at $APP_DIR. Run scripts/package-macos-app.sh first." >&2
  exit 1
fi

if [[ "$IDENTITY" == "-" ]]; then
  codesign --force --deep --sign - "$APP_DIR"
else
  codesign --force --deep --options runtime --sign "$IDENTITY" "$APP_DIR"
fi

codesign --verify --deep --strict --verbose=2 "$APP_DIR"

echo "Signed $APP_DIR"
