mod nbodysim_impl;
mod pathtracer_impl;

use crate::pathtracer_impl::PathTracerView;
use aurora::scenes::BasicScene3d;
use aurora::Aurora;
use nbodysim_impl::NBodySim;

fn main() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            console_log::init_with_level(log::Level::Debug).unwrap();
            wasm_bindgen_futures::spawn_local(run());
        } else {
            env_logger::init();
            pollster::block_on(run());
        }
    }
}

async fn run() {
    let mut aurora = Aurora::new().await.unwrap();
    let scene = NBodySim::new(aurora.get_gpu(), aurora.get_target()).await;
    aurora.add_scene("nbodysim", Box::new(scene));

    let mut scene =
        BasicScene3d::new("cornell_box.toml", aurora.get_gpu(), aurora.get_target()).await;
    scene.add_view(
        "pathtracer",
        Box::new(PathTracerView::new(aurora.get_gpu(), &scene)),
    );
    aurora.add_scene("pathtracer", Box::new(scene));

    aurora.run().unwrap();
}
