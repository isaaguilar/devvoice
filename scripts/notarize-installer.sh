#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="DevVoice"
VERSION="$(awk -F ' = ' '/^version = / { gsub(/"/, "", $2); print $2; exit }' "$ROOT_DIR/src-tauri/Cargo.toml")"
RAW_ARCH="$(uname -m)"
case "$RAW_ARCH" in
  arm64) ARCH="aarch64" ;;
  *) ARCH="$RAW_ARCH" ;;
esac

PROFILE_NAME="${1:-}"
DMG_PATH="${2:-$ROOT_DIR/src-tauri/target/release/bundle/dmg/${APP_NAME}_${VERSION}_${ARCH}.dmg}"

if [[ -z "$PROFILE_NAME" ]]; then
  echo "Usage: $0 <notarytool-keychain-profile> [dmg-path]" >&2
  exit 1
fi

if [[ ! -f "$DMG_PATH" ]]; then
  echo "Installer not found at $DMG_PATH. Run scripts/build-installer.sh first." >&2
  exit 1
fi

xcrun notarytool submit "$DMG_PATH" --keychain-profile "$PROFILE_NAME" --wait
xcrun stapler staple "$DMG_PATH"

echo "Notarized $DMG_PATH"
