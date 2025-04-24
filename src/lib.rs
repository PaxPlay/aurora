pub mod scenes;
mod shader;

use scenes::Scene;
use winit::{
    event::WindowEvent, event_loop::ActiveEventLoop, window::Window
};
use std::cell::RefCell;
use std::{collections::BTreeMap, default::Default};
use std::sync::Arc;

use log::{debug, info, error};

/// Central Aurora context
pub struct Aurora {
    gpu: Arc<GpuContext>,
    window: Option<AuroraWindow>,
    scenes: BTreeMap<String, Box<dyn Scene>>,
    current_scene: Option<String>,
}

impl Aurora {
    pub async fn new() -> Result<Self, ()> {
        let gpu = Arc::new(GpuContext::new().await?);

        Ok(Self {
            gpu,
            window: None,
            scenes: BTreeMap::new(),
            current_scene: None,
        })
    }
    
    pub fn run(&mut self) -> Result<(), winit::error::EventLoopError> {
        info!(target: "aurora", "Running Aurora in windowed mode!");
        
        let event_loop = winit::event_loop::EventLoop::new()?;
        event_loop.run_app(self)?;
        Ok(())
    }

    pub fn run_headless(&mut self) {
        info!(target: "aurora", "Running Aurora in headless mode!");
        todo!()
    }

    pub fn add_scene(&mut self, name: &str, scene: Box<dyn Scene>) {
        self.scenes.insert(String::from(name), scene);
    }

    fn get_window_mut(&mut self) -> &mut AuroraWindow {
        self.window.as_mut().unwrap()
    }

    pub fn get_window(&self) -> &AuroraWindow {
        self.window.as_ref().unwrap()
    }

    pub fn get_gpu(&self) -> Arc<GpuContext> {
        self.gpu.clone()
    }

    fn render(&mut self) {
        let gpu = self.gpu.clone();
        let out_surface_texture = self.get_window().surface.get_current_texture().unwrap();
        let view = out_surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let window_size: [u32; 2] = [self.get_window().surface_config.width, self.get_window().surface_config.height];

        let render_cb = if let Some(mut entry) = self.scenes.first_entry() {
            Some(entry.get_mut().render(self.gpu.clone(), &view))
        } else {
            None
        };

        let ui_context = &mut self.get_window_mut().ui_context;
        let ui_cb = ui_context.render(&gpu,  &view, window_size);

        match ui_cb {
            Some(ui_cb) => { gpu.queue.submit([render_cb.unwrap(), ui_cb]); },
            None => (),
        }
        out_surface_texture.present();
    }

}

impl winit::application::ApplicationHandler for Aurora {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.window = AuroraWindow::new(&self.gpu, event_loop).ok();

        let surface_format = self.get_window().surface_format;
        if let Some(mut entry) = self.scenes.first_entry() {
            entry.get_mut().build_pipeline(self.gpu.clone(), surface_format);
        }
    }

    fn window_event(
            &mut self,
            event_loop: &winit::event_loop::ActiveEventLoop,
            _window_id: winit::window::WindowId,
            event: WindowEvent,
        ) {
        self.get_window_mut().ui_context.on_window_event(&event);

        use winit::event::WindowEvent;
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            },
            WindowEvent::RedrawRequested => {
                self.render();
            },
            WindowEvent::Resized(physical_size) => {
                {
                    let sc = &mut self.get_window_mut().surface_config;
                    sc.width = physical_size.width;
                    sc.height = physical_size.height;
                }

                let window = self.get_window();
                window.surface.configure(&self.gpu.device, &window.surface_config);
            },
            _ => (),
        }
        
    }
}

pub struct GpuContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuContext {
    fn select_adapter(instance: &wgpu::Instance) -> Result<wgpu::Adapter, ()> {
        const ADAPTER_PRIORITIES: [wgpu::Backend; 6] = [
            wgpu::Backend::Vulkan,
            wgpu::Backend::Metal,
            wgpu::Backend::Dx12,
            wgpu::Backend::BrowserWebGpu,
            wgpu::Backend::Gl,
            wgpu::Backend::Empty,
        ];

        let mut adapters = instance.enumerate_adapters(wgpu::Backends::all());
        adapters.sort_by_key(|a| {
            for (i, b) in ADAPTER_PRIORITIES.iter().enumerate() {
                if *b == a.get_info().backend { return i as u32; }
            }
            u32::MAX
        });

        if adapters.is_empty() {
            error!(target: "aurora", "Couldn't find any suitable adapters!");
            Err(())
        } else {
            let info = adapters[0].get_info();
            info!(target: "aurora", "Chose WGPU adapter --- Backend: {}, Adapter: {}", info.backend.to_str(), info.name);

            Ok(adapters[0].clone())
        }
    }

    pub async fn new() -> Result<Self, ()> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = Self::select_adapter(&instance)?;

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
            None
        ).await.map_err(|_| ())?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}

pub struct AuroraWindow {
    pub window: Arc<Window>,
    pub surface: wgpu::Surface<'static>,
    pub surface_format: wgpu::TextureFormat,
    surface_config: wgpu::SurfaceConfiguration,
    ui_context: UiContext,
}

impl AuroraWindow {
    fn new(gpu: &GpuContext, event_loop: &ActiveEventLoop) -> Result<Self, ()> {
        let attributes = winit::window::WindowAttributes::default()
            .with_title("Aurora");
        let window = Arc::new(event_loop.create_window(attributes).map_err(|_| ())?);

        let surface = gpu.instance.create_surface(window.clone()).map_err(|_| ())?;
        let capabilities = surface.get_capabilities(&gpu.adapter);
        let surface_format = capabilities.formats
            .iter().copied().filter(|c| c.is_srgb()).next().unwrap_or(capabilities.formats[0]);

        debug!(target: "aurora", "Selected surface format for main window: {:?}", surface_format);

        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: capabilities.present_modes[0],
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&gpu.device, &surface_config);
        let ui_context = UiContext::new(gpu, window.clone(), surface_format);

        Ok(Self {
            window,
            surface,
            surface_format,
            surface_config,
            ui_context,
        })
    }
}

struct UiContext {
    window: Arc<Window>,
    renderer: egui_wgpu::Renderer,
    state: egui_winit::State,
}

impl UiContext {
    pub fn new(gpu: &GpuContext, window: Arc<Window>, surface_format: wgpu::TextureFormat) -> Self {
        let renderer = egui_wgpu::Renderer::new(&gpu.device, surface_format, None, 1, false);

        let context = egui::Context::default();
        let viewport_id = context.viewport_id();
        let state = egui_winit::State::new(context, viewport_id, &window, None, None, None);

        Self {
            window,
            renderer,
            state,
        }
    }

    pub fn build_ui(ctx: &egui::Context) {
        egui::Window::new("Aurora Configuration").default_open(true).show(ctx, |ui| {
            ui.label("Hi there :)");
            if ui.button("Click Me :)").clicked() {
                info!(target: "aurora", "Click me button clicked :)");
            }
        });
    }

    pub fn render(&mut self, gpu: &GpuContext, view: &wgpu::TextureView, window_size: [u32; 2]) -> Option<wgpu::CommandBuffer> {
        let mut ce = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let platform_output = {
            let render_pass = ce.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rp_aurora_ui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let mut rp_static = render_pass.forget_lifetime();
            let input = self.state.take_egui_input(&self.window); 
            let egui_ctx = self.state.egui_ctx();
            let ui_out = egui_ctx.run(input, |ctx| { Self::build_ui(ctx); });

            let clipped_primitives = egui_ctx.tessellate(ui_out.shapes, ui_out.pixels_per_point);

            let screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: window_size,
                pixels_per_point: ui_out.pixels_per_point,
            };

            for (id, delta) in ui_out.textures_delta.set {
                self.renderer.update_texture(&gpu.device, &gpu.queue, id, &delta);
            }

            self.renderer.update_buffers(&gpu.device, &gpu.queue, &mut ce, &clipped_primitives, &screen_descriptor);
            self.renderer.render(&mut rp_static, &clipped_primitives, &screen_descriptor);

            ui_out.platform_output
        };
        self.state.handle_platform_output(&self.window, platform_output);

        Some(ce.finish())
    }

    pub fn on_window_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        let result = self.state.on_window_event(&self.window, event);
        if result.repaint {
            self.window.request_redraw();
        }

        result.consumed
    }
}

