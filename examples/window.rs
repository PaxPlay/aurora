use std::cell::RefCell;
use std::sync::Arc;
use aurora::Aurora;

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let mut aurora = Aurora::new().await.unwrap();
//    let scene = aurora::scenes::BasicScene3d::new(&window);
//    window.set_scene(Arc::new(RefCell::new(scene)));
    aurora.run().unwrap();
}

