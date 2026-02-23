mod commands;

use commands::{
    get_detection_status, start_detection, stop_detection, switch_mode, toggle_gradcam,
    DetectionState,
};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(DetectionState::default())
        .invoke_handler(tauri::generate_handler![
            start_detection,
            stop_detection,
            switch_mode,
            toggle_gradcam,
            get_detection_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
