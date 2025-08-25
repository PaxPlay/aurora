pub mod buffers;
pub mod files;
mod internal;
pub mod scenes;
pub mod shader;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
extern crate console_error_panic_hook;
use core::time;
#[cfg(target_arch = "wasm32")]
use std::panic;

use buffers::{Buffer, BufferCopyUtil};
use internal::TargetViewPipeline;
use scenes::Scene;
use shader::ShaderManager;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use std::{collections::BTreeMap, default::Default};
use thiserror::Error;
use wgpu::Extent3d;
#[cfg(target_os = "linux")]
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::{event::WindowEvent, event_loop::ActiveEventLoop, window::Window};

use log::{error, info, warn};

use clap::Parser;

/// Aurora CLI
#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    list_formats: bool,

    #[cfg(target_os = "linux")]
    #[arg(long)]
    force_x11: bool,

    #[arg(short, long)]
    format: Option<String>,
}

/// Central Aurora context
pub struct Aurora {
    gpu: Arc<GpuContext>,
    target: Arc<RenderTarget>,
    window: Option<AuroraWindow>,
    scenes: BTreeMap<String, SceneHandle>,
    current_scene: Option<String>,
    timestamp_queries: ManagedTimestampQuerySet,
    args: Args,
}

type SceneHandle = Arc<RefCell<Box<dyn Scene>>>; // this language might not be real

#[derive(Error, Debug)]
pub enum NewAuroraError {
    #[error("failed creating gpu context")]
    NewGpuContextError(#[from] NewGpuContextError),
}

impl Aurora {
    /// Prioritized list of texture formats supported by the framework
    pub const OUT_FORMATS: [wgpu::TextureFormat; 3] = [
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba32Float,
    ];

    #[cfg(target_arch = "wasm32")]
    fn replace_aurora_div(text: String) -> Result<(), &'static str> {
        use wasm_bindgen::JsCast;
        use web_sys::{Document, HtmlElement, Window};

        let window: Window = web_sys::window().ok_or("window not found")?;
        let document: Document = window.document().ok_or("document not found")?;
        let aurora_div = document
            .get_element_by_id("aurora-container")
            .ok_or("aurora div not found")?
            .dyn_into::<HtmlElement>()
            .map_err(|_| "dyn_into cast failed")?;

        aurora_div.set_inner_html(&format!("<p>{text}</p>"));
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    fn panic_hook(panic_info: &std::panic::PanicInfo) {
        // #[wasm_bindgen]
        // extern "C" {
        //     #[wasm_bindgen(js_namespace = console)]
        //     fn error(msg: String);

        //     type Error;

        //     #[wasm_bindgen(constructor)]
        //     fn new() -> Error;

        //     #[wasm_bindgen(structural, method, getter)]
        //     fn stack(error: &Error) -> String;
        // }
        // let stack = Error::new().stack();

        error!(target: "aurora", "Panic occurred: {}", panic_info);
        if let Err(text) = Self::replace_aurora_div(format!("Panic occurred: {}", panic_info)) {
            error!(target: "aurora", "Failed to replace aurora div: {text}");
        }
        console_error_panic_hook::hook(panic_info);
    }

    pub async fn new() -> Result<Self, NewAuroraError> {
        #[cfg(target_arch = "wasm32")]
        panic::set_hook(Box::new(|panic_info| Self::panic_hook(panic_info)));

        let args = Args::parse();
        let gpu = Arc::new(GpuContext::new().await?);

        let usage = RenderTarget::usage();
        let formats = Self::list_texture_formats(&gpu, usage);
        if args.list_formats {
            println!("Supported texture formats:");
            for format in formats {
                println!("- {format:?}");
            }
            std::process::exit(0);
        }

        let format = Self::select_format(&formats, args.format.clone());
        let target = Arc::new(RenderTarget::new(&gpu, format));
        let timestamp_queries = ManagedTimestampQuerySet::new(&gpu, 64);

        Ok(Self {
            gpu,
            target,
            window: None,
            scenes: BTreeMap::new(),
            current_scene: None,
            timestamp_queries,
            args,
        })
    }

    fn list_texture_formats(
        gpu: &GpuContext,
        usage: wgpu::TextureUsages,
    ) -> Vec<wgpu::TextureFormat> {
        let formats: Vec<_> = Self::OUT_FORMATS
            .iter()
            .filter(|f| {
                let features = gpu.adapter.get_texture_format_features(**f);
                const FEATURES: wgpu::TextureFormatFeatureFlags =
                    wgpu::TextureFormatFeatureFlags::BLENDABLE;

                features.allowed_usages.contains(usage) && features.flags.contains(FEATURES)
            })
            .copied()
            .collect();

        formats
    }

    fn select_format(
        supported_formats: &Vec<wgpu::TextureFormat>,
        selection: Option<String>,
    ) -> wgpu::TextureFormat {
        if let Some(selected) = selection {
            for format in supported_formats {
                let format_str = format!("{format:?}");
                if selected == format_str {
                    return *format;
                }
            }

            error!(target: "aurora", "Selected format {selected} is not supported");
        }

        let format = *supported_formats
            .first()
            .expect("None of the supported formats are abailable on this platform.");
        info!(target: "aurora", "Automatically selected format \"{format:?}\"");
        format
    }

    pub fn run(&mut self) -> Result<(), winit::error::EventLoopError> {
        info!(target: "aurora", "Running Aurora in windowed mode!");

        let mut event_loop_builder = winit::event_loop::EventLoop::builder();
        #[cfg(target_os = "linux")]
        {
            if self.args.force_x11 {
                event_loop_builder.with_x11();
            }
        }
        let event_loop = event_loop_builder.build()?;

        event_loop.run_app(self)?;
        Ok(())
    }

    pub fn run_headless(&mut self) {
        info!(target: "aurora", "Running Aurora in headless mode!");
        todo!()
    }

    pub fn add_scene(&mut self, name: &str, scene: Box<dyn Scene>) {
        self.scenes
            .insert(String::from(name), Arc::new(RefCell::new(scene)));
    }

    fn get_window_mut(&mut self) -> &mut AuroraWindow {
        self.window.as_mut().unwrap()
    }

    pub fn get_window(&self) -> &AuroraWindow {
        self.window.as_ref().unwrap()
    }

    fn create_surface_texture(&self) -> (wgpu::SurfaceTexture, wgpu::TextureView) {
        let window = self.get_window();
        let texture = window.surface.get_current_texture().unwrap();
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    pub fn get_gpu(&self) -> Arc<GpuContext> {
        self.gpu.clone()
    }

    pub fn get_target(&self) -> Arc<RenderTarget> {
        self.target.clone()
    }

    fn get_current_scene(&mut self) -> Option<SceneHandle> {
        let scene_name = match self.current_scene.as_ref() {
            Some(name) => Some(name.clone()),
            None => {
                if let Some(entry) = self.scenes.first_entry() {
                    let name = Some(entry.key().clone());
                    self.current_scene = name.clone();
                    name
                } else {
                    None
                }
            }
        };

        scene_name.map(|name| (self.scenes.get(&name).unwrap()).clone())
    }

    fn render(&mut self) {
        let gpu = self.gpu.clone();

        let render_gpu = self.gpu.clone();
        let render_target = self.target.clone();
        let scene_handle = self.get_current_scene().unwrap();

        let mut queries = self.timestamp_queries.begin(&gpu);

        let mut command_buffers = {
            let mut scene = scene_handle.try_borrow_mut().unwrap();
            scene.render(render_gpu, render_target, &mut queries)
        };

        let (surface_texture, view) = self.create_surface_texture();
        command_buffers.append(&mut self.get_window_mut().render(
            &view,
            scene_handle,
            &mut queries,
        ));

        gpu.queue.submit(command_buffers);
        surface_texture.present();

        self.timestamp_queries
            .resolve_query_set(queries, gpu.clone());
        gpu.device.poll(wgpu::Maintain::Wait);
    }
}

impl winit::application::ApplicationHandler for Aurora {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.window = AuroraWindow::new(self.gpu.clone(), self.target.clone(), event_loop).ok();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if self.get_window_mut().ui_context.on_window_event(&event) {
            return;
        }

        use winit::event::WindowEvent;
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.render();
            }
            WindowEvent::Resized(physical_size) => {
                let gpu = self.gpu.clone();
                let target_size = self.target.size;
                self.get_window_mut().on_resize(
                    &gpu,
                    [physical_size.width, physical_size.height],
                    target_size,
                );
            }
            _ => (),
        }
    }
}

pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub shaders: ShaderManager,
    pub filesystem: files::Filesystem,
    pub timing_results: Arc<Mutex<TimingResults>>,
}

#[derive(Error, Debug)]
pub enum NewGpuContextError {
    #[error("no suitable adapter found")]
    NoAdapterFound,

    #[error("failed requesting device")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),

    #[cfg(target_arch = "wasm32")]
    #[error("failed creating adapter")]
    RequestAdapterError,
}

impl GpuContext {
    #[cfg(not(target_arch = "wasm32"))]
    fn select_adapter(instance: &wgpu::Instance) -> Result<wgpu::Adapter, NewGpuContextError> {
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
                if *b == a.get_info().backend {
                    return i as u32;
                }
            }
            u32::MAX
        });

        if adapters.is_empty() {
            error!(target: "aurora", "Couldn't find any suitable adapters!");
            Err(NewGpuContextError::NoAdapterFound)
        } else {
            let info = adapters[0].get_info();
            info!(target: "aurora", "Chose WGPU adapter --- Backend: {}, Adapter: {}",
                info.backend.to_str(), info.name);

            Ok(adapters[0].clone())
        }
    }

    pub async fn new() -> Result<Self, NewGpuContextError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                let adapter = instance
                    .request_adapter(&wgpu::RequestAdapterOptions::default())
                    .await
                    .ok_or(NewGpuContextError::RequestAdapterError)?;
            } else {
                let adapter = Self::select_adapter(&instance)?;
            }
        }

        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 10,
            max_compute_workgroup_storage_size: 32768,
            ..Default::default()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                        | wgpu::Features::TIMESTAMP_QUERY,
                    required_limits: limits,
                    ..Default::default()
                },
                None,
            )
            .await?;

        let shaders = ShaderManager::new(device.clone());

        let filesystem = files::Filesystem::new(None);

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            shaders,
            filesystem,
            timing_results: Arc::new(Mutex::new(TimingResults::new())),
        })
    }

    pub fn create_buffer_init_padded<T: bytemuck::Pod>(
        &self,
        label: &str,
        data: &[T],
        size: usize,
        value: T,
        usage: wgpu::BufferUsages,
    ) -> Buffer<T> {
        Buffer::from_data_padded(self, label, data, size, value, usage)
    }

    pub fn create_buffer_init<T: bytemuck::Pod>(
        &self,
        label: &str,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> Buffer<T> {
        Buffer::from_data(self, label, data, usage)
    }

    pub fn create_buffer<T: bytemuck::Pod>(
        &self,
        label: &str,
        size: usize,
        usage: wgpu::BufferUsages,
    ) -> Buffer<T> {
        Buffer::new(self, label, size, usage)
    }
}

pub struct RenderTarget {
    pub format: wgpu::TextureFormat,
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub size: [u32; 2],
}

impl RenderTarget {
    fn usage() -> wgpu::TextureUsages {
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::TEXTURE_BINDING
    }

    pub fn new(gpu: &GpuContext, format: wgpu::TextureFormat) -> Self {
        let width: u32 = 1024;
        let height: u32 = 1024;
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("aurora_target"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: Self::usage(),
            view_formats: &[format],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("aurora_target_view"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });

        Self {
            format,
            texture,
            view,
            size: [width, height],
        }
    }
}

pub struct AuroraWindow {
    gpu: Arc<GpuContext>,
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    surface_config: wgpu::SurfaceConfiguration,
    ui_context: UiContext,
    target_view_pipeline: TargetViewPipeline,
    buffer_copy_util: BufferCopyUtil,
    copy_command: Option<wgpu::CommandBuffer>,
}

#[derive(Error, Debug)]
pub enum NewWindowResult {
    #[error("create window failed")]
    CreateWindowError(#[from] winit::error::OsError),

    #[error("create surface failed")]
    CreateSurfaceError(#[from] wgpu::CreateSurfaceError),

    #[cfg(target_arch = "wasm32")]
    #[error("html element `{0}` not found")]
    HTMLElementNotFound(String),
}

impl AuroraWindow {
    fn new(
        gpu: Arc<GpuContext>,
        render_target: Arc<RenderTarget>,
        event_loop: &ActiveEventLoop,
    ) -> Result<Self, NewWindowResult> {
        let mut attributes = winit::window::WindowAttributes::default();
        attributes = attributes.with_title("Aurora");

        #[cfg(target_arch = "wasm32")]
        const WINDOW_SIZE: winit::dpi::PhysicalSize<u32> =
            winit::dpi::PhysicalSize::new(1024, 1024);

        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowAttributesExtWebSys;
            use NewWindowResult::HTMLElementNotFound as HTMLNF;

            const CANVAS_ID: &str = "aurora";
            let window = web_sys::window().ok_or(HTMLNF("window".to_string()))?;
            let document = window.document().ok_or(HTMLNF("document".to_string()))?;
            let canvas = document
                .get_element_by_id(CANVAS_ID)
                .ok_or(HTMLNF(format!("canvas - {CANVAS_ID}")))?;
            let html_canvas_element = canvas.unchecked_into();

            attributes = attributes
                .with_canvas(Some(html_canvas_element))
                .with_inner_size(WINDOW_SIZE);
        }

        let window = Arc::new(event_loop.create_window(attributes)?);

        let surface = gpu.instance.create_surface(window.clone())?;
        let capabilities = surface.get_capabilities(&gpu.adapter);
        let surface_format = capabilities
            .formats
            .iter()
            .copied()
            .find(|c| c.is_srgb())
            .unwrap_or(capabilities.formats[0]);

        #[cfg(not(target_arch = "wasm32"))]
        let size = window.inner_size();
        #[cfg(target_arch = "wasm32")]
        let size = WINDOW_SIZE;

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
        let ui_context = UiContext::new(&gpu, window.clone(), surface_format);

        let target_view_pipeline = TargetViewPipeline::new(
            gpu.clone(),
            render_target,
            surface_format,
            [size.width, size.height],
        );
        let buffer_copy_util = BufferCopyUtil::new(2048);

        Ok(Self {
            gpu,
            window,
            surface,
            surface_format,
            surface_config,
            ui_context,
            target_view_pipeline,
            buffer_copy_util,
            copy_command: None,
        })
    }

    fn on_resize(&mut self, gpu: &GpuContext, size: [u32; 2], target_size: [u32; 2]) {
        self.surface_config.width = size[0];
        self.surface_config.height = size[1];

        self.surface.configure(&gpu.device, &self.surface_config);
        let cb = self.buffer_copy_util.create_copy_command(&self.gpu, |ctx| {
            self.target_view_pipeline
                .update_size_buffer(ctx, target_size, size);
        });
        self.copy_command = Some(cb);
    }

    fn render(
        &mut self,
        view: &wgpu::TextureView,
        scene_handle: SceneHandle,
        queries: &mut TimestampQueries,
    ) -> Vec<wgpu::CommandBuffer> {
        let window_size: [u32; 2] = [self.surface_config.width, self.surface_config.height];

        let mut result = Vec::with_capacity(3);
        if let Some(cb) = self.copy_command.take() {
            result.push(cb);
        }

        result.push(
            self.target_view_pipeline
                .build_command_buffer(&self.gpu, view),
        );

        if let Some(cb) =
            self.ui_context
                .render(&self.gpu, view, window_size, Some(scene_handle), queries)
        {
            result.push(cb);
        }

        result
    }
}
struct UiContext {
    window: Arc<Window>,
    renderer: egui_wgpu::Renderer,
    state: egui_winit::State,
    show_performance_window: bool,
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
            show_performance_window: false,
        }
    }

    pub fn render(
        &mut self,
        gpu: &GpuContext,
        view: &wgpu::TextureView,
        window_size: [u32; 2],
        scene: Option<SceneHandle>,
        queries: &mut TimestampQueries,
    ) -> Option<wgpu::CommandBuffer> {
        let mut ce = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let platform_output = {
            let render_pass = ce.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rp_aurora_ui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: queries.render_pass_writes("rp_aurora_ui"),
                occlusion_query_set: None,
            });
            let mut rp_static = render_pass.forget_lifetime();
            let input = self.state.take_egui_input(&self.window);
            let egui_ctx = self.state.egui_ctx();
            let ui_out = egui_ctx.run(input, |ctx| {
                if let Some(scene) = scene.clone() {
                    let mut scene = scene.try_borrow_mut().unwrap();
                    egui::Window::new("Scene Configuration")
                        .default_open(true)
                        .resizable(true)
                        .show(ctx, |ui| {
                            egui::ScrollArea::both().show(ui, |ui| {
                                ui.checkbox(
                                    &mut self.show_performance_window,
                                    "Show Performance Window",
                                );
                                scene.draw_ui(ui);
                            });
                        });

                    if self.show_performance_window {
                        let timing_results = gpu.timing_results.lock().unwrap();
                        egui::Window::new("Performance")
                            .default_open(true)
                            .resizable(true)
                            .show(ctx, |ui| {
                                egui::ScrollArea::vertical().show(ui, |ui| {
                                    timing_results.plot_stacked_bars(ui);
                                    for (i, label) in timing_results.labels.iter().enumerate() {
                                        ui.label(format!(
                                            "{}: {:.3} ms",
                                            label,
                                            timing_results.series[i].last().unwrap_or(&0.0)
                                        ));
                                    }
                                });
                            });
                    }
                }
            });

            let clipped_primitives = egui_ctx.tessellate(ui_out.shapes, ui_out.pixels_per_point);

            let screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: window_size,
                pixels_per_point: ui_out.pixels_per_point,
            };

            for (id, delta) in ui_out.textures_delta.set {
                self.renderer
                    .update_texture(&gpu.device, &gpu.queue, id, &delta);
            }

            self.renderer.update_buffers(
                &gpu.device,
                &gpu.queue,
                &mut ce,
                &clipped_primitives,
                &screen_descriptor,
            );
            self.renderer
                .render(&mut rp_static, &clipped_primitives, &screen_descriptor);

            ui_out.platform_output
        };
        self.state
            .handle_platform_output(&self.window, platform_output);

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

trait DebugUi {
    fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool;
}

pub fn dispatch_size(total_threads: (u32, u32, u32), wg_size: (u32, u32, u32)) -> (u32, u32, u32) {
    (
        total_threads.0.div_ceil(wg_size.0),
        total_threads.1.div_ceil(wg_size.1),
        total_threads.2.div_ceil(wg_size.2),
    )
}

pub struct TimestampQueries {
    query_set: wgpu::QuerySet,
    preallocated: usize,
    queries: Vec<String>,
}

impl TimestampQueries {
    pub fn compute_pass_writes(&mut self, label: &str) -> Option<wgpu::ComputePassTimestampWrites> {
        let index = self.queries.len();
        self.queries.push(format!("bgn_{}", label.to_string()));
        self.queries.push(format!("end_{}", label.to_string()));
        if index + 2 >= self.preallocated {
            warn!(
                target: "aurora",
                "Not enough preallocated timestamp queries for compute pass {label}! Used {index}, preallocated {}",
                self.preallocated
            );

            None
        } else {
            Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.query_set,
                beginning_of_pass_write_index: Some(index as u32),
                end_of_pass_write_index: Some((index + 1) as u32),
            })
        }
    }

    pub fn render_pass_writes(&mut self, label: &str) -> Option<wgpu::RenderPassTimestampWrites> {
        let index = self.queries.len();
        self.queries.push(format!("bgn_{}", label.to_string()));
        self.queries.push(format!("end_{}", label.to_string()));
        if index + 2 >= self.preallocated {
            warn!(
                target: "aurora",
                "Not enough preallocated timestamp queries for render pass {label}! Used {index}, preallocated {}",
                self.preallocated
            );

            None
        } else {
            Some(wgpu::RenderPassTimestampWrites {
                query_set: &self.query_set,
                beginning_of_pass_write_index: Some(index as u32),
                end_of_pass_write_index: Some((index + 1) as u32),
            })
        }
    }
}

pub trait CommandEncoderTimestampExt {
    fn begin_compute_pass_timestamped(
        &mut self,
        label: &str,
        queries: &mut TimestampQueries,
    ) -> wgpu::ComputePass<'_>;
}

impl CommandEncoderTimestampExt for wgpu::CommandEncoder {
    fn begin_compute_pass_timestamped(
        &mut self,
        label: &str,
        queries: &mut TimestampQueries,
    ) -> wgpu::ComputePass<'_> {
        self.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: queries.compute_pass_writes(label),
        })
    }
}

struct ManagedTimestampQuerySet {
    query_set: wgpu::QuerySet,
    buffer: Buffer<u64>,
    allocated: usize,
    queries_last_frame: Vec<String>,
    results_last_frame: Vec<(String, u64)>,
}

impl ManagedTimestampQuerySet {
    fn new(gpu: &GpuContext, count: usize) -> Self {
        let (query_set, buffer) = Self::allocate(gpu, count);

        Self {
            query_set,
            buffer,
            allocated: count,
            queries_last_frame: Vec::new(),
            results_last_frame: Vec::new(),
        }
    }

    fn allocate(gpu: &GpuContext, count: usize) -> (wgpu::QuerySet, Buffer<u64>) {
        let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("aurora_managed_query_set"),
            ty: wgpu::QueryType::Timestamp,
            count: count as u32,
        });

        let buffer = Buffer::new(
            gpu,
            "aurora_managed_query_buffer",
            count * 8,
            wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        );

        (query_set, buffer)
    }

    fn begin(&mut self, gpu: &GpuContext) -> TimestampQueries {
        if self.results_last_frame.len() > self.allocated {
            (self.query_set, self.buffer) = Self::allocate(gpu, self.results_last_frame.len());
            self.allocated = self.results_last_frame.len();

            info!(
                target: "aurora",
                "Increased timestamp query allocation to {}",
                self.allocated
            );
        }

        TimestampQueries {
            query_set: self.query_set.clone(),
            preallocated: self.allocated,
            queries: Vec::new(),
        }
    }

    fn resolve_query_set(&mut self, queries: TimestampQueries, gpu: Arc<GpuContext>) {
        self.queries_last_frame = queries.queries;
        let count = self.queries_last_frame.len();
        let copy_size = (count * std::mem::size_of::<u64>()) as wgpu::BufferAddress;

        let download = Arc::new(gpu.device.create_buffer(&wgpu::BufferDescriptor {
            size: copy_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
            label: None,
        }));

        let mut ce = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora_timestamp_query_copy"),
            });
        ce.resolve_query_set(&self.query_set, 0..count as u32, &self.buffer, 0);
        ce.copy_buffer_to_buffer(&self.buffer, 0, &download, 0, copy_size);

        gpu.queue.submit([ce.finish()]);

        let queries = self.queries_last_frame.clone();
        let timestamp_period = gpu.queue.get_timestamp_period() as f64;
        let timing_results = gpu.timing_results.clone();

        download
            .clone()
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                if let Err(e) = r {
                    error!(target: "aurora", "Failed mapping timestamp query buffer: {e}");
                } else {
                    let data = download.slice(..).get_mapped_range();
                    let result: &[u64] = bytemuck::cast_slice(&data);

                    let min = result.iter().min().copied().unwrap_or(0);
                    let mut durations = Vec::with_capacity(result.len());
                    for timestamp in result {
                        let time_ns: f32 = if *timestamp == 0 || min == 0 {
                            0.0
                        } else {
                            ((timestamp - min) as f64 * timestamp_period) as f32 / 1_000_000.0
                        };
                        durations.push(time_ns);
                    }

                    let mut timing_results = timing_results.lock().unwrap();
                    timing_results.process(queries, durations);
                }

                download.unmap();
            });
    }
}

struct TimingResults {
    labels: Vec<String>,
    series: Vec<CircularBuffer<f32>>,
}

impl TimingResults {
    const INNER_CAPACITY: usize = 100;

    fn new() -> Self {
        Self {
            labels: Vec::new(),
            series: Vec::new(),
        }
    }

    fn process(&mut self, labels_in: Vec<String>, timestamps: Vec<f32>) {
        let mut i: usize = 0;
        let mut labels = Vec::new();
        let mut durations = Vec::new();

        while i < labels_in.len() {
            let label: &String = &labels_in[i];
            let time = timestamps[i];
            if label.starts_with("bgn_") {
                let end_label = label.replacen("bgn_", "end_", 1);
                if let Some(end_index) = labels_in[i + 1..].iter().position(|l| l == &end_label) {
                    let end_time = timestamps[i + 1 + end_index];
                    let duration = end_time - time;
                    labels.push(label.replacen("bgn_", "", 1));
                    durations.push(duration);
                }
                i += 1;
            }
            i += 1;
        }

        if labels != self.labels {
            self.labels = labels;
            self.series.clear();
            for _ in 0..self.labels.len() {
                self.series.push(CircularBuffer::new(Self::INNER_CAPACITY));
            }
        }

        for (i, &duration) in durations.iter().enumerate() {
            self.series[i].push(duration);
        }
    }

    fn plot_stacked_bars(&self, ui: &mut egui::Ui) -> egui::Response {
        use egui_plot::{Bar, BarChart, Legend, Plot};

        const BAR_WIDTH: f64 = 1.0;

        let mut charts = Vec::new();
        for (i, label) in self.labels.iter().enumerate() {
            let series = &self.series[i];
            let mut values = Vec::new();
            for j in 0..series.len() {
                if let Some(v) = series.get(j) {
                    values.push(Bar::new(j as f64 * BAR_WIDTH, *v as f64));
                }
            }

            let others: Vec<&BarChart> = charts.iter().collect();
            let bar = BarChart::new(label, values)
                .name(label)
                .width(BAR_WIDTH)
                .stack_on(&others);
            charts.push(bar);
        }

        Plot::new("timing_results")
            .legend(Legend::default())
            .view_aspect(2.0)
            .show_grid([false, true])
            .show(ui, |plot_ui| {
                for chart in charts {
                    plot_ui.bar_chart(chart);
                }
            })
            .response
    }
}

struct CircularBuffer<T> {
    data: Vec<T>,
    capacity: usize,
    start: usize,
    len: usize,
}

impl<T: Default + Clone> CircularBuffer<T> {
    fn new(capacity: usize) -> Self {
        Self {
            data: vec![T::default(); capacity],
            capacity,
            start: 0,
            len: 0,
        }
    }

    fn push(&mut self, value: T) {
        if self.len < self.capacity {
            self.len += 1;
        } else {
            self.start = (self.start + 1) % self.capacity;
        }
        let end = (self.start + self.len - 1) % self.capacity;
        self.data[end] = value;
    }

    fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            Some(&self.data[(self.start + index) % self.capacity])
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn last(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            self.get(self.len - 1)
        }
    }
}
