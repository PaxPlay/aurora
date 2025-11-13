use aurora::{scenes::BasicScene3d, Aurora};

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let mut aurora = Aurora::new().await.unwrap();
    let scene = BasicScene3d::new("cornell_box.toml", aurora.get_gpu(), aurora.get_target()).await;
    aurora.add_scene("basic_3d", Box::new(scene));
    aurora.run().unwrap();
}
