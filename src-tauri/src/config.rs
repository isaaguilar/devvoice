use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

const SETTINGS_FILE: &str = "settings.json";
const KEY_SERVICE: &str = "devvoice";
const KEY_ACCOUNT: &str = "gemini-api-key";
const CONFIG_VERSION: u8 = 1;

pub const DEFAULT_GEMINI_MODEL: &str = "gemini-2.5-flash";
pub const DEFAULT_GEMINI_PROMPT: &str = "\
Rewrite the selected text for speech.\n\
\n\
1. Preserve the meaning.\n\
2. Make the wording sound natural when spoken aloud.\n\
3. Keep technical terms precise.\n\
4. Expand abbreviations only when that improves pronunciation.\n\
5. Keep code-adjacent phrases readable for a senior software engineer.\n\
\n\
Return only the rewritten text.";
pub const DEFAULT_SHORTCUT: &str = "Command+Control+S";
pub const DEFAULT_HTTP_PORT: u16 = 9876;

pub fn normalize_shortcut(shortcut: &str) -> String {
    shortcut
        .trim()
        .replace("CmdOrControl", "CommandOrControl")
        .replace("CmdOrCtrl", "CommandOrControl")
        .replace("Cmd", "Command")
        .replace("Ctrl", "Control")
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoiceGender {
    #[default]
    Woman,
    Man,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppConfig {
    #[serde(default)]
    pub config_version: u8,
    #[serde(default)]
    pub gemini_enabled: bool,
    #[serde(default = "default_gemini_model")]
    pub gemini_model: String,
    #[serde(default = "default_gemini_prompt")]
    pub gemini_prompt: String,
    #[serde(default = "default_shortcut")]
    pub shortcut: String,
    #[serde(default)]
    pub voice_gender: VoiceGender,
    /// Port for the local HTTP API. Set to 0 to disable.
    #[serde(default = "default_http_port")]
    pub http_port: u16,
}

fn default_gemini_model() -> String {
    DEFAULT_GEMINI_MODEL.to_owned()
}

fn default_gemini_prompt() -> String {
    DEFAULT_GEMINI_PROMPT.to_owned()
}

fn default_shortcut() -> String {
    DEFAULT_SHORTCUT.to_owned()
}

fn default_http_port() -> u16 {
    DEFAULT_HTTP_PORT
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            config_version: CONFIG_VERSION,
            gemini_enabled: false,
            gemini_model: DEFAULT_GEMINI_MODEL.to_owned(),
            gemini_prompt: DEFAULT_GEMINI_PROMPT.to_owned(),
            shortcut: DEFAULT_SHORTCUT.to_owned(),
            voice_gender: VoiceGender::Woman,
            http_port: DEFAULT_HTTP_PORT,
        }
    }
}

fn migrate_config(config: &mut AppConfig) {
    if config.config_version != CONFIG_VERSION {
        config.config_version = CONFIG_VERSION;
    }
    config.shortcut = normalize_shortcut(&config.shortcut);
    if config.shortcut.trim().is_empty() {
        config.shortcut = DEFAULT_SHORTCUT.to_owned();
    }
    if config.gemini_model.trim().is_empty() {
        config.gemini_model = DEFAULT_GEMINI_MODEL.to_owned();
    }
    if config.gemini_prompt.trim().is_empty() {
        config.gemini_prompt = DEFAULT_GEMINI_PROMPT.to_owned();
    }
}

#[derive(Debug)]
pub struct ConfigStore {
    config_path: PathBuf,
    data_dir: PathBuf,
    keyring: keyring_core::Entry,
}

impl ConfigStore {
    pub fn load() -> Result<(Self, AppConfig)> {
        let dirs = ProjectDirs::from("com", "isa", "DevVoice")
            .context("resolve DevVoice configuration directories")?;
        let config_dir = dirs.config_dir().to_path_buf();
        let data_dir = dirs.data_dir().to_path_buf();
        fs::create_dir_all(&config_dir)
            .with_context(|| format!("create config directory {}", config_dir.display()))?;
        fs::create_dir_all(&data_dir)
            .with_context(|| format!("create data directory {}", data_dir.display()))?;

        #[cfg(target_os = "macos")]
        keyring_core::set_default_store(apple_native_keyring_store::keychain::Store::new()?);

        let config_path = config_dir.join(SETTINGS_FILE);
        let keyring = keyring_core::Entry::new(KEY_SERVICE, KEY_ACCOUNT)?;
        let config = if config_path.exists() {
            let mut config = Self::read_config(&config_path)?;
            migrate_config(&mut config);
            config
        } else {
            AppConfig::default()
        };

        let store = Self {
            config_path,
            data_dir,
            keyring,
        };
        store.save_config(&config)?;

        Ok((store, config))
    }

    fn read_config(path: &Path) -> Result<AppConfig> {
        let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Ok(serde_json::from_slice(&bytes).context("parse settings.json")?)
    }

    pub fn save_config(&self, config: &AppConfig) -> Result<()> {
        let text = serde_json::to_string_pretty(config).context("serialize settings")?;
        fs::write(&self.config_path, text)
            .with_context(|| format!("write {}", self.config_path.display()))?;
        Ok(())
    }

    pub fn read_api_key(&self) -> Result<Option<String>> {
        match self.keyring.get_password() {
            Ok(password) => Ok(Some(password)),
            Err(keyring_core::Error::NoEntry) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    pub fn set_api_key(&self, api_key: Option<&str>) -> Result<()> {
        match api_key.map(str::trim).filter(|value| !value.is_empty()) {
            Some(api_key) => self.keyring.set_password(api_key)?,
            None => match self.keyring.delete_credential() {
                Ok(()) | Err(keyring_core::Error::NoEntry) => {}
                Err(error) => return Err(error.into()),
            },
        }

        Ok(())
    }

    pub fn config_path(&self) -> &Path {
        &self.config_path
    }
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }
}
