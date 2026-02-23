use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::State;

/// Tracks the state of the Python sidecar detection process.
#[derive(Default)]
pub struct DetectionState {
    pub running: Mutex<bool>,
    pub mode: Mutex<String>,
}

#[derive(Serialize, Deserialize)]
pub struct StatusResponse {
    pub running: bool,
    pub mode: String,
}

/// Starts the Python sidecar detection process.
#[tauri::command]
pub async fn start_detection(state: State<'_, DetectionState>) -> Result<String, String> {
    let mut running = state.running.lock().map_err(|e| e.to_string())?;
    if *running {
        return Err("Detection is already running".into());
    }
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
    Ok(format!("Grad-CAM {}", if enabled { "enabled" } else { "disabled" }))
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
