use std::cell::RefCell;
use std::sync::Arc;
use aurora::AuroraWindow;

fn main() {
    let mut window = AuroraWindow::new();
    let scene = aurora::scenes::BasicScene3d::new(&window);
    window.set_scene(Arc::new(RefCell::new(scene)));
    window.run();
}