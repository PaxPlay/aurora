use aurora::vis::ScalarFieldScene;
use aurora::Aurora;
use egui::Scene;

fn main() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            console_log::init_with_level(log::Level::Debug).unwrap();
            wasm_bindgen_futures::spawn_local(main_async());
        } else {
            env_logger::init();
            pollster::block_on(main_async());
        }
    }
}

async fn main_async() {
    let mut aurora = Aurora::new().await.unwrap();
    let scene = ScalarFieldScene::new(aurora.get_gpu(), "models/tooth.nhdr")
        .await
        .expect("Failed initializing scalar field scene");
    aurora.add_scene("scalar_field", Box::new(scene));
    aurora.run().unwrap();
}
