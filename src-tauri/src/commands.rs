use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::State;
use tauri_plugin_shell::ShellExt;

/// Tracks the state of the Python sidecar detection process.
pub struct DetectionState {
    pub running: Mutex<bool>,
    pub mode: Mutex<String>,
    pub sidecar_child: Mutex<Option<tauri_plugin_shell::process::CommandChild>>,
}

impl Default for DetectionState {
    fn default() -> Self {
        Self {
            running: Mutex::new(false),
            mode: Mutex::new("live".into()),
            sidecar_child: Mutex::new(None),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct StatusResponse {
    pub running: bool,
    pub mode: String,
}

/// Starts the Python sidecar detection process.
/// Spawns `python/main.py` as a child process managed by Tauri.
#[tauri::command]
pub async fn start_detection(
    app: tauri::AppHandle,
    state: State<'_, DetectionState>,
) -> Result<String, String> {
    let mut running = state.running.lock().map_err(|e| e.to_string())?;
    if *running {
        return Err("Detection is already running".into());
    }

    // Spawn the Python sidecar process
    let shell = app.shell();
    let command = shell
        .sidecar("silentwitness-backend")
        .map_err(|e| format!("Failed to create sidecar command: {}", e))?;

    let (mut _rx, child) = command
        .spawn()
        .map_err(|e| format!("Failed to spawn sidecar: {}", e))?;

    // Store the child process handle for later cleanup
    let mut sidecar = state.sidecar_child.lock().map_err(|e| e.to_string())?;
    *sidecar = Some(child);

    *running = true;
    Ok("Detection started".into())
}

/// Stops the Python sidecar detection process.
#[tauri::command]
pub async fn stop_detection(state: State<'_, DetectionState>) -> Result<String, String> {
    let mut running = state.running.lock().map_err(|e| e.to_string())?;
    if !*running {
        return Err("Detection is not running".into());
    }

    // Kill the sidecar process
    let mut sidecar = state.sidecar_child.lock().map_err(|e| e.to_string())?;
    if let Some(child) = sidecar.take() {
        child.kill().map_err(|e| format!("Failed to kill sidecar: {}", e))?;
    }

    *running = false;
    Ok("Detection stopped".into())
}

/// Switches between live webcam and deepfake file input.
#[tauri::command]
pub async fn switch_mode(
    mode: String,
    state: State<'_, DetectionState>,
) -> Result<String, String> {
    let mut current_mode = state.mode.lock().map_err(|e| e.to_string())?;
    *current_mode = mode.clone();
    Ok(format!("Mode switched to: {}", mode))
}

/// Toggles Grad-CAM heatmap overlay on or off.
#[tauri::command]
pub async fn toggle_gradcam(enabled: bool) -> Result<String, String> {
    Ok(format!(
        "Grad-CAM {}",
        if enabled { "enabled" } else { "disabled" }
    ))
}

/// Returns the current detection status.
#[tauri::command]
pub async fn get_detection_status(
    state: State<'_, DetectionState>,
) -> Result<StatusResponse, String> {
    let running = *state.running.lock().map_err(|e| e.to_string())?;
    let mode = state.mode.lock().map_err(|e| e.to_string())?.clone();
    Ok(StatusResponse { running, mode })
}
