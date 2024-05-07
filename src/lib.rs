pub mod scenes;
mod shader;

use std::cell::{Ref, RefCell};
use winit::{
    window::{Window, WindowBuilder},
    event_loop::EventLoop,
    event::{Event, WindowEvent},
};
use std::default::Default;
use std::sync::Arc;
use crate::scenes::Scene;

pub struct AuroraWindow {
    event_loop: EventLoop<()>,
    application: Application
}

impl AuroraWindow {
    pub fn new() -> Self {
        let event_loop = EventLoop::new().unwrap();
        let window = WindowBuilder::new().with_title("Aurora Window").build(&event_loop).unwrap();
        let application = Application::new(window);
        Self { event_loop, application }
    }

    pub fn run(mut self) {
        self.event_loop.run(move | event, elwt | match event {
            Event::WindowEvent {
                event,
                window_id,
                ..
            } => if window_id == self.application.gpu_state.window.id() {
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(new_size) =>
                        self.application.gpu_state.resize(new_size),
                    WindowEvent::ScaleFactorChanged { .. } =>
                        self.application.gpu_state.resize(self.application.gpu_state.window.inner_size()),
                    _ => ()
                }
            },
            Event::AboutToWait => {
                if !self.application.think_and_draw() {
                    elwt.exit();
                }
            },
            _ => ()
        }).unwrap();
    }

    pub fn set_scene(&mut self, scene: Arc<RefCell<dyn scenes::Scene>>) {
        self.application.scene = Some(scene);
    }
}

pub struct Application {
    gpu_state: GpuState,
    scene: Option<Arc<RefCell<dyn scenes::Scene>>>
}

impl Application {
    pub fn new(window: Window) -> Self {
        let gpu_state = pollster::block_on(GpuState::new(window));
        Application { gpu_state, scene: None }
    }
    fn think_and_draw(&mut self) -> bool {
        match self.render() {
            Ok(_) => {},
            Err(wgpu::SurfaceError::Lost) => self.gpu_state.resize(self.gpu_state.size),
            Err(wgpu::SurfaceError::OutOfMemory) => return false,
            Err(e) => log::error!("{:?}", e)
        }

        true
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.gpu_state.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.gpu_state.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Application Render Command Encoder")
            });

        {
            let mut _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 1.0,
                            b: 0.0,
                            a: 1.0
                        }),
                        store: wgpu::StoreOp::Store
                    }
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None
            });

            match &self.scene {
                None => {}
                Some(s) => {
                    let mut scene = s.borrow_mut();
                    {
                        let mut _render_pass = _render_pass;
                        scene.render(&mut _render_pass);
                    }
                }
            }
        }

        self.gpu_state.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

struct GpuState {
    window: Window,
    instance: wgpu::Instance,
    surface: wgpu::Surface,
    surface_format: wgpu::TextureFormat,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    surface_config: wgpu::SurfaceConfiguration
}

impl GpuState {
    pub async fn new(window: Window) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = unsafe { instance.create_surface(&window) }.unwrap();
        let adapter = instance.enumerate_adapters(wgpu::Backends::all())
            .filter(| adapter | adapter.is_surface_supported(&surface)).next().unwrap();
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default()
            },
            None
        ).await.unwrap();

        let capabilities = surface.get_capabilities(&adapter);
        let surface_format = capabilities.formats
            .iter().copied().filter(|f| f.is_srgb()).next().unwrap_or(capabilities.formats[0]);

        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: capabilities.present_modes[0],
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![]
        };
        surface.configure(&device, &surface_config);

        Self { window, instance, surface, surface_format, device: Arc::new(device), queue, size,
            surface_config }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(self.device.as_ref(), &self.surface_config);
        }
    }

    pub fn get_surface_format(&self) -> wgpu::TextureFormat {
        self.surface_format
    }
}
