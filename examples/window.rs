use aurora::Aurora;

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let mut aurora = Aurora::new().await.unwrap();
    let scene = aurora::scenes::BasicScene3d::new();
    aurora.add_scene("basic_3d", Box::new(scene));
    aurora.run().unwrap();
}

