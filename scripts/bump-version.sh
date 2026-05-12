#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEXT_VERSION="${1:-}"

if [[ -z "$NEXT_VERSION" ]]; then
  echo "Usage: $0 <next-version>" >&2
  exit 1
fi

NEXT_VERSION="$NEXT_VERSION" node - "$ROOT_DIR" <<'NODE'
const fs = require("fs");
const path = require("path");

const root = process.argv[2];
const nextVersion = process.env.NEXT_VERSION;

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function replaceTomlVersion(contents, newVersion) {
  return contents.replace(/^version = ".*"$/m, `version = "${newVersion}"`);
}

function replacePlistValue(contents, key, value) {
  const pattern = new RegExp(`(<key>${key}<\\/key>\\s*<string>)([^<]+)(<\\/string>)`);
  if (!pattern.test(contents)) {
    throw new Error(`Could not find plist key ${key}`);
  }
  return contents.replace(pattern, `$1${value}$3`);
}

const packageJsonPath = path.join(root, "package.json");
const packageLockPath = path.join(root, "package-lock.json");
const cargoTomlPath = path.join(root, "src-tauri", "Cargo.toml");
const tauriConfigPath = path.join(root, "src-tauri", "tauri.conf.json");
const infoPlistPath = path.join(root, "macos", "Info.plist");

if (!nextVersion) {
  throw new Error("NEXT_VERSION is required");
}

const packageJson = readJson(packageJsonPath);
const currentVersion = packageJson.version;
if (currentVersion === nextVersion) {
  console.log(`Version already set to ${nextVersion}; no changes made.`);
  process.exit(0);
}

packageJson.version = nextVersion;
writeJson(packageJsonPath, packageJson);

if (fs.existsSync(packageLockPath)) {
  const packageLock = readJson(packageLockPath);
  packageLock.version = nextVersion;
  if (packageLock.packages && packageLock.packages[""]) {
    packageLock.packages[""].version = nextVersion;
  }
  writeJson(packageLockPath, packageLock);
}

const cargoToml = fs.readFileSync(cargoTomlPath, "utf8");
fs.writeFileSync(cargoTomlPath, replaceTomlVersion(cargoToml, nextVersion));

const tauriConfig = readJson(tauriConfigPath);
tauriConfig.version = nextVersion;
writeJson(tauriConfigPath, tauriConfig);

const infoPlist = fs.readFileSync(infoPlistPath, "utf8");
const buildMatch = infoPlist.match(/<key>CFBundleVersion<\/key>\s*<string>([^<]+)<\/string>/);
if (!buildMatch) {
  throw new Error("Could not find CFBundleVersion in Info.plist");
}
const nextBuild = String(Number.parseInt(nextVersion.split(".").pop(), 10));
let nextInfoPlist = replacePlistValue(infoPlist, "CFBundleShortVersionString", nextVersion);
nextInfoPlist = replacePlistValue(nextInfoPlist, "CFBundleVersion", nextBuild);
fs.writeFileSync(infoPlistPath, nextInfoPlist);

console.log(`Set DevVoice to ${nextVersion} (${nextBuild})`);
NODE
