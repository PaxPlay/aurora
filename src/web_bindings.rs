use crate::AuroraEvent;
use std::sync::{Mutex, OnceLock, RwLock};
use wasm_bindgen::prelude::*;
use web_sys::js_sys;
use winit::event_loop::EventLoopProxy;

#[derive(Debug)]
struct BindingState {
    available_scenes: Vec<String>,
    current_scene: Option<String>,
    event_loop: EventLoopProxy<AuroraEvent>,
}

static BINDING_STATE: OnceLock<RwLock<BindingState>> = OnceLock::new();

pub fn send_js_window_event(name: &str) {
    let window = web_sys::window().expect("no global `window` exists");
    let event = web_sys::CustomEvent::new(name).unwrap();
    window
        .dispatch_event(&event)
        .expect("Failed to dispatch event");
}

pub fn init_binding_state(scenes: Vec<String>, event_loop: EventLoopProxy<AuroraEvent>) {
    BINDING_STATE
        .set(RwLock::new(BindingState {
            available_scenes: scenes,
            current_scene: None,
            event_loop,
        }))
        .expect("Failed to set binding state");
}

pub fn set_binding_state_scene(scene: &str) {
    if let Some(state) = BINDING_STATE.get() {
        let mut state = state.write().unwrap();
        if state.available_scenes.contains(&scene.to_string()) {
            state.current_scene = Some(scene.to_string());
        } else {
            log::warn!("Scene '{}' not found in avaliable scenes", scene);
        }
    } else {
        log::error!("Binding state not initialized");
    }
}

#[wasm_bindgen]
pub fn set_scene(scene: &str) {
    if let Some(state) = BINDING_STATE.get() {
        let state = state.read().unwrap();
        if state.available_scenes.contains(&scene.to_string()) {
            state
                .event_loop
                .send_event(AuroraEvent::ChangeScene(scene.to_string()))
                .expect("Failed to send SetScene event");
        } else {
            log::warn!("Scene '{}' not found in avaliable scenes", scene);
        }
    } else {
        log::error!("Binding state not initialized");
    }
}

#[wasm_bindgen]
pub fn get_current_scene() -> Option<String> {
    if let Some(state) = BINDING_STATE.get() {
        let state = state.read().unwrap();
        state.current_scene.clone()
    } else {
        log::error!("Binding state not initialized");
        None
    }
}

#[wasm_bindgen]
pub fn get_available_scenes() -> js_sys::Array {
    let array = js_sys::Array::new();
    if let Some(state) = BINDING_STATE.get() {
        let state = state.read().unwrap();
        for scene in &state.available_scenes {
            array.push(&JsValue::from_str(scene));
        }
    } else {
        log::error!("Binding state not initialized");
    }
    array
}
