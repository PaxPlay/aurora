use aurora::{
    buffers::{Buffer, MirroredBuffer},
    compute_pipeline, dispatch_size, register_default,
    scenes::{BasicScene3d, Scene3dView, SceneGeometry, SceneRenderError},
    shader::{BindGroupLayout, BindGroupLayoutBuilder, ComputePipeline},
    Aurora, CommandEncoderTimestampExt, GpuContext, TimestampQueries,
};
use log::{info, warn};
use rand::Rng;
use std::{num::NonZero, ops::DerefMut, sync::Arc};

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
    let scene_geometry = SceneGeometry::new("cornell_box.toml", aurora.get_gpu()).await;
    let mut scene = BasicScene3d::new(scene_geometry, aurora.get_gpu(), aurora.get_target());
    scene.add_view(
        "path_tracer",
        Box::new(PathTracerView::new(aurora.get_gpu(), &scene)),
    );
    aurora.add_scene("basic_3d", Box::new(scene));
    aurora.run().unwrap();
}

struct ImageStorageBuffer {
    resolution: [u32; 2],
    pixel_buffer: Option<Buffer<f32>>,
}

impl ImageStorageBuffer {
    fn new() -> Self {
        Self {
            resolution: [0, 0],
            pixel_buffer: None,
        }
    }

    fn resize(&mut self, gpu: &GpuContext, type_size: usize, width: u32, height: u32) -> bool {
        if width == self.resolution[0] && height == self.resolution[1] {
            return false;
        }

        self.resolution = [width, height];
        self.pixel_buffer = Some(gpu.create_buffer(
            "image_storage_buffer",
            width as usize * height as usize * 4 * type_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        ));
        true
    }

    fn get(&self) -> Option<Buffer<f32>> {
        self.pixel_buffer.clone()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PathTracerSettings {
    max_iterations: u32,
    output_buffer: u32,
    accumulate: u32,
    nee: u32,
    rr_alpha: f32,
}

impl Default for PathTracerSettings {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            output_buffer: 0,
            accumulate: 1,
            nee: 1,
            rr_alpha: 0.8,
        }
    }
}

impl PathTracerSettings {
    fn checkbox(ui: &mut egui::Ui, value: &mut u32, label: &str) -> bool {
        let mut checked = *value != 0;
        let changed = ui.checkbox(&mut checked, label).changed();
        *value = if checked { 1 } else { 0 };
        changed
    }

    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= ui
            .add(
                egui::widgets::Slider::new(&mut self.max_iterations, 0..=16).text("Max Iterations"),
            )
            .changed();
        changed |= ui
            .add(egui::widgets::Slider::new(&mut self.rr_alpha, 0.0..=1.0).text("RR Alpha"))
            .changed();

        const BUFFERS: [&str; 6] = ["out", "w_i", "n", "w_o", "weight", "t"];
        let mut output_buffer = self.output_buffer;
        egui::ComboBox::from_label("Output Buffer")
            .selected_text(BUFFERS[output_buffer as usize])
            .show_ui(ui, |ui| {
                for i in 0..BUFFERS.len() as u32 {
                    ui.selectable_value(&mut output_buffer, i, BUFFERS[i as usize].to_string());
                }
            });
        changed |= output_buffer != self.output_buffer;
        self.output_buffer = output_buffer;

        changed |= Self::checkbox(ui, &mut self.accumulate, "Accumulate");
        changed |= Self::checkbox(ui, &mut self.nee, "Next Event Estimation");
        changed
    }
}

struct PathTracerView {
    gpu: Arc<GpuContext>,
    schedule_pipeline: ComputePipeline,
    ray_generation_pipeline: ComputePipeline,
    handle_primary_pipeline: ComputePipeline,
    ray_intersection_pipeline: ComputePipeline,
    reorder_intersection_pipeline: ComputePipeline,
    handle_intersections_pipeline: ComputePipeline,
    target_pipeline: ComputePipeline,
    nee_miss_pipeline: ComputePipeline,
    schedule_buffer: Buffer<u32>,
    seed_buffer: Buffer<u32>,
    primary_ray_buffer: Buffer<f32>,
    ray_intersection_buffers: [Buffer<f32>; 2],
    bg_camera: wgpu::BindGroup,
    bg_rays: [wgpu::BindGroup; 2],
    bg_reorder_intersect: wgpu::BindGroup,
    bg_schedule_invocations: wgpu::BindGroup,
    bg_schedule_intersect: wgpu::BindGroup,
    bg_schedule_shade: wgpu::BindGroup,
    bg_schedule_nee_miss: wgpu::BindGroup,
    bg_schedule_copy: wgpu::BindGroup,
    image_f32: ImageStorageBuffer,
    image_target_format: ImageStorageBuffer,
    bgl_image: BindGroupLayout,
    bg_image: Option<wgpu::BindGroup>,
    buffer_copy_util: aurora::buffers::BufferCopyUtil,
    rng: rand::rngs::ThreadRng,
    should_clear: bool,
    settings: MirroredBuffer<PathTracerSettings>,
    buffer_schedule_reorder: Buffer<u32>,
}

impl PathTracerView {
    fn new(gpu: Arc<GpuContext>, scene: &BasicScene3d) -> Self {
        // this fucntion is way too long, but I don't have a reasonable way to make it shorter?
        let total_pixels: usize =
            scene.camera.resolution[0] as usize * scene.camera.resolution[1] as usize;

        // Twice the amount to accomodate rays from the previous frame
        let primary_ray_buffer: Buffer<f32> = gpu.create_buffer(
            "primary_rays",
            size_of::<f32>() * 4 * 4 * total_pixels * 2,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        // Allow 4 times the number of rays as pixels
        // This is done to include NEE rays and accomodate rays from the previous frame
        let ray_buffer: Buffer<f32> =
            gpu.create_buffer("rays", 64 * total_pixels * 2, wgpu::BufferUsages::STORAGE);

        // Twice the amount to accomodate rays from the previous frame
        let ray_intersection_buffers: [Buffer<f32>; 2] = [
            gpu.create_buffer(
                "ray_intersections_0",
                80 * 2 * total_pixels,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            gpu.create_buffer(
                "ray_intersections_1",
                80 * 2 * total_pixels,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
        ];

        let schedule_buffer: Buffer<u32> = gpu.create_buffer(
            "schedule",
            64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        );

        let seed_buffer: Buffer<u32> = gpu.create_buffer(
            "pt_seeds",
            64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let settings = MirroredBuffer::new(
            &gpu,
            "pt_settings",
            1,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let bgl_camera = BindGroupLayoutBuilder::new(gpu.clone())
            .label("bgl_camera")
            .add_buffer(
                0,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Uniform,
            )
            .add_buffer(
                1,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Uniform,
            )
            .build();
        let bg_camera = bgl_camera
            .bind_group_builder()
            .label("bg_camera")
            .buffer(0, &scene.camera.buffer)
            .buffer(1, &settings)
            .build()
            .unwrap();

        let bgl_rays = BindGroupLayoutBuilder::new(gpu.clone())
            .label("bgl_rays")
            .add_buffer(
                0,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
            )
            .add_buffer(
                1,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
            )
            .add_buffer(
                2,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
            )
            .add_buffer(
                3,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: true },
            )
            .build();

        let bg_rays: [wgpu::BindGroup; 2] = (0..2)
            .map(|i| {
                bgl_rays
                    .bind_group_builder()
                    .label(&format!("bg_rays_{i}"))
                    .buffer(0, &primary_ray_buffer)
                    .buffer(1, &ray_buffer)
                    .buffer(2, &ray_intersection_buffers[i])
                    .buffer(3, &seed_buffer)
                    .build()
                    .unwrap()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let bgl_image = BindGroupLayoutBuilder::new(gpu.clone())
            .label("bgl_image")
            .add_buffer(
                0,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
            )
            .add_buffer(
                1,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
            )
            .build();

        let image_f32 = ImageStorageBuffer::new();
        let image_target_format = ImageStorageBuffer::new();

        let bgl_schedule = BindGroupLayoutBuilder::new(gpu.clone())
            .label("bgl_schedule")
            .add_buffer(
                0,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
            )
            .build();

        let buffer_schedule_intersect: Buffer<u32> =
            gpu.create_buffer("pt_schedule_intersect", 256, wgpu::BufferUsages::STORAGE);
        let buffer_schedule_reorder: Buffer<u32> = gpu.create_buffer(
            "pt_schedule_reorder",
            256 * 17, // 16 iterations + primary
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let buffer_schedule_shade: Buffer<u32> =
            gpu.create_buffer("pt_schedule_shade", 32, wgpu::BufferUsages::STORAGE);
        let buffer_schedule_nee_miss: Buffer<u32> =
            gpu.create_buffer("pt_schedule_nee_miss", 32, wgpu::BufferUsages::STORAGE);

        let bg_schedule_intersect = bgl_schedule
            .bind_group_builder()
            .label("bg_schedule_intersect")
            .buffer(0, &buffer_schedule_intersect)
            .build()
            .expect("failed creating intersect schedule bind group");
        let bg_schedule_shade = bgl_schedule
            .bind_group_builder()
            .label("bg_schedule_shade")
            .buffer(0, &buffer_schedule_shade)
            .build()
            .expect("failed creating shade schedule bind group");
        let bg_schedule_nee_miss = bgl_schedule
            .bind_group_builder()
            .label("bg_schedule_nee_miss")
            .buffer(0, &buffer_schedule_nee_miss)
            .build()
            .expect("failed creating nee miss schedule bind group");

        let bgl_schedule_invocations = BindGroupLayoutBuilder::new(gpu.clone())
            .label("bgl_schedule_invocations")
            .add_buffer(
                0,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
            )
            .add_buffer(
                1,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
            )
            .add_buffer(
                2,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
            )
            .add_buffer_dynamic(
                3,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
                NonZero::new(256),
            )
            .add_buffer(
                4,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
            )
            .build();

        let bg_schedule_invocations = bgl_schedule_invocations
            .bind_group_builder()
            .label("bg_schedule_invocations")
            .buffer(0, &schedule_buffer)
            .buffer(1, &buffer_schedule_intersect)
            .buffer(2, &buffer_schedule_shade)
            .buffer_sized(3, &buffer_schedule_reorder, NonZero::new(256).unwrap())
            .buffer(4, &buffer_schedule_nee_miss)
            .build()
            .expect("failed creating invocations schedule bind group");

        let bgl_schedule_copy = BindGroupLayoutBuilder::new(gpu.clone())
            .label("bgl_schedule_copy")
            .add_buffer(
                0,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
            )
            .add_buffer_dynamic(
                1,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
                NonZero::new(256),
            )
            .build();

        let bg_schedule_copy = bgl_schedule_copy
            .bind_group_builder()
            .label("bg_schedule_copy")
            .buffer(0, &schedule_buffer)
            .buffer_sized(1, &buffer_schedule_reorder, NonZero::new(256).unwrap())
            .build()
            .expect("failed creating copy schedule bind group");

        let bgl_reorder_intersect = BindGroupLayoutBuilder::new(gpu.clone())
            .label("bgl_reorder_intersect")
            .add_buffer_dynamic(
                0,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
                NonZero::new(256),
            )
            .add_buffer(
                1,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
            )
            .add_buffer(
                2,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
            )
            .build();

        let bg_reorder_intersect = bgl_reorder_intersect
            .bind_group_builder()
            .label(&format!("bg_reorder_intersect"))
            .buffer_sized(0, &buffer_schedule_reorder, NonZero::new(256).unwrap())
            .buffer(1, &ray_intersection_buffers[0])
            .buffer(2, &ray_intersection_buffers[1])
            .build()
            .unwrap();

        let pipeline_layout_handle_intersect =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pt_pipeline_layout_handle_intersections"),
                    bind_group_layouts: &[
                        &bgl_camera.get(),
                        &bgl_rays.get(),
                        &bgl_schedule.get(),
                        &scene.gpu_scene_geometry.bind_group_layout.get(),
                    ],
                    push_constant_ranges: &[],
                });

        let pipeline_layout_primary =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pt_pipeline_layout_primary"),
                    bind_group_layouts: &[
                        &bgl_camera.get(),
                        &bgl_rays.get(),
                        &bgl_schedule_invocations.get(),
                    ],
                    push_constant_ranges: &[],
                });

        let pipeline_layout_intersect =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pt_pipeline_layout_intersect"),
                    bind_group_layouts: &[
                        &bgl_rays.get(),
                        &bgl_schedule.get(),
                        &scene.gpu_scene_geometry.bind_group_layout.get(),
                    ],
                    push_constant_ranges: &[],
                });

        let pipeline_layout_reorder_intersect =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pt_pipeline_layout_reorder_intersect"),
                    bind_group_layouts: &[&bgl_reorder_intersect.get()],
                    push_constant_ranges: &[],
                });

        let pipeline_layout_image =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pt_pipeline_layout_image"),
                    bind_group_layouts: &[
                        &bgl_camera.get(),
                        &bgl_rays.get(),
                        &bgl_image.get(),
                        &bgl_schedule_copy.get(),
                    ],
                    push_constant_ranges: &[],
                });

        let pipeline_layout_schedule =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pt_pipeline_layout_schedule"),
                    bind_group_layouts: &[&bgl_schedule_invocations.get()],
                    push_constant_ranges: &[],
                });
        let pipeline_layout_nee_miss =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pt_pipeline_layout_nee_miss"),
                    bind_group_layouts: &[&bgl_rays.get(), &bgl_schedule.get()],
                    push_constant_ranges: &[],
                });

        register_default!(gpu.shaders, "schedule", "schedule.wgsl");
        register_default!(gpu.shaders, "primary", "primary.wgsl");
        register_default!(gpu.shaders, "handle_primary", "handle_primary.wgsl");
        register_default!(gpu.shaders, "copy", "copy.wgsl");
        register_default!(gpu.shaders, "path_tracer", "pathtracer.wgsl");
        register_default!(gpu.shaders, "intersect", "intersect.wgsl");
        register_default!(
            gpu.shaders,
            "reorder_intersections",
            "reorder_intersections.wgsl"
        );
        register_default!(gpu.shaders, "nee_miss", "nee_miss.wgsl");

        let schedule_pipeline = compute_pipeline!(gpu, schedule; &wgpu::ComputePipelineDescriptor {
            label: Some("pt_pipeline_schedule"),
            layout: Some(&pipeline_layout_schedule),
            module: &schedule,
            entry_point: Some("schedule_invocations"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let ray_generation_pipeline = compute_pipeline!(gpu, primary; &wgpu::ComputePipelineDescriptor {
            label: Some("pt_pipeline_ray_generation"),
            layout: Some(&pipeline_layout_primary),
            module: &primary,
            entry_point: Some("generate_rays"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let ray_intersection_pipeline = compute_pipeline!(gpu, intersect; &wgpu::ComputePipelineDescriptor {
            label: Some("pt_pipeline_ray_intersection"),
            layout: Some(&pipeline_layout_intersect),
            module: &intersect,
            entry_point: Some("intersect_rays"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let reorder_intersection_pipeline = compute_pipeline!(gpu, reorder_intersections; &wgpu::ComputePipelineDescriptor {
            label: Some("pt_pipeline_reorder_intersections"),
            layout: Some(&pipeline_layout_reorder_intersect),
            module: &reorder_intersections,
            entry_point: Some("reorder_intersections"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pl = pipeline_layout_handle_intersect.clone();
        let handle_primary_pipeline = compute_pipeline!(gpu, handle_primary; &wgpu::ComputePipelineDescriptor {
            label: Some("pt_pipeline_handle_primary"),
            layout: Some(&pl),
            module: &handle_primary,
            entry_point: Some("handle_primary"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let handle_intersections_pipeline = compute_pipeline!(gpu, path_tracer; &wgpu::ComputePipelineDescriptor {
            label: Some("pt_pipeline_handle_intersections"),
            layout: Some(&pipeline_layout_handle_intersect),
            module: &path_tracer,
            entry_point: Some("handle_intersections"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let target_pipeline = compute_pipeline!(gpu, copy; &wgpu::ComputePipelineDescriptor {
            label: Some("pt_pipeline_target"),
            layout: Some(&pipeline_layout_image),
            module: &copy,
            entry_point: Some("copy_target"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let nee_miss_pipeline = compute_pipeline!(gpu, nee_miss; &wgpu::ComputePipelineDescriptor {
            label: Some("pt_pipeline_nee_miss"),
            layout: Some(&pipeline_layout_nee_miss),
            module: &nee_miss,
            entry_point: Some("nee_miss"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            gpu,
            schedule_pipeline,
            ray_generation_pipeline,
            ray_intersection_pipeline,
            handle_primary_pipeline,
            handle_intersections_pipeline,
            reorder_intersection_pipeline,
            target_pipeline,
            nee_miss_pipeline,
            schedule_buffer,
            seed_buffer,
            primary_ray_buffer,
            ray_intersection_buffers,
            bg_camera,
            bg_rays,
            bg_reorder_intersect,
            bg_schedule_invocations,
            bg_schedule_intersect,
            bg_schedule_shade,
            bg_schedule_nee_miss,
            bg_schedule_copy,
            image_f32,
            image_target_format,
            bgl_image,
            bg_image: None,
            buffer_copy_util: aurora::buffers::BufferCopyUtil::new(2048),
            rng: rand::rng(),
            should_clear: true,
            settings,
            buffer_schedule_reorder,
        }
    }

    /// Check images against resolution and rebuild bind group if neccesary
    fn check_image_bg(&mut self, gpu: Arc<GpuContext>, target: &aurora::RenderTarget) {
        let mut resized = self
            .image_f32
            .resize(&gpu, 4, target.size[0], target.size[1]);

        use wgpu::TextureFormat as TF;
        let target_type_size = match target.format {
            TF::Rgba8Unorm => 1,
            TF::Rgba16Float => 2,
            TF::Rgba32Float => 4,
            _ => panic!("Target texture format not supported by PathTracerView::check_image_bg"),
        };

        resized |=
            self.image_target_format
                .resize(&gpu, target_type_size, target.size[0], target.size[1]);

        if resized || self.bg_image.is_none() {
            let builder = self.bgl_image.bind_group_builder();
            let buffer_f32 = self.image_f32.get().expect(
                "Failed getting image f32 buffer after resize, this should not be possible",
            );
            let buffer_target_format = self.image_target_format.get().expect(
                "Failed getting image target type buffer after resize, this should not be possible",
            );
            self.bg_image = Some(
                builder
                    .buffer(0, &buffer_f32)
                    .buffer(1, &buffer_target_format)
                    .build()
                    .unwrap(),
            );
        }
    }

    fn screenshot(&self) {
        let image = &self.image_f32;

        use exr::prelude::*;
        let resolution: (usize, usize) =
            (image.resolution[0] as usize, image.resolution[1] as usize);

        let layer_attributes = LayerAttributes::named("rgba main layer");

        let buffer = image.pixel_buffer.as_ref().unwrap();
        let mut file = self.gpu.filesystem.create_writer("image.exr");

        wgpu::util::DownloadBuffer::read_buffer(
            &self.gpu.device,
            &self.gpu.queue,
            &buffer.buffer.slice(..),
            move |res| {
                if let Ok(download_buffer) = res {
                    let data: &[u8] = &download_buffer;
                    let data_f32: &[f32] = bytemuck::try_cast_slice(data)
                        .expect("Could not cast image data from u8 to f32");
                    let data_vec: Vec<f32> = data_f32.to_vec();

                    {
                        let layer = Layer::new(
                            resolution,
                            layer_attributes,
                            Encoding::FAST_LOSSLESS,
                            SpecificChannels::rgba(move |pos: Vec2<usize>| {
                                let index = pos.x() + pos.y() * resolution.0;
                                let color: (f32, f32, f32, f32) = (
                                    data_vec[4 * index],
                                    data_vec[4 * index + 1],
                                    data_vec[4 * index + 2],
                                    data_vec[4 * index + 3],
                                );
                                color
                            }),
                        );

                        let image = Image::from_layer(layer);
                        image
                            .write()
                            .to_buffered(file.deref_mut())
                            .expect("Failed to write image to file");
                    }
                }
            },
        );
    }
}

impl PathTracerView {
    fn do_primary(
        &mut self,
        compute_pass: &mut wgpu::ComputePass<'_>,
        scene: &BasicScene3d,
    ) -> Result<(), SceneRenderError> {
        let resolution = scene.camera.resolution;
        compute_pass.set_bind_group(0, &self.bg_camera, &[]);
        compute_pass.set_bind_group(1, &self.bg_rays[0], &[]);
        compute_pass.set_bind_group(2, &self.bg_schedule_invocations, &[0]);
        compute_pass.set_pipeline(&self.ray_generation_pipeline.get()?);
        let (x, y, _) = dispatch_size((resolution[0], resolution[1], 1), (16, 16, 1));
        compute_pass.dispatch_workgroups(x, y, 1);

        self.do_intersect(compute_pass, scene)?;

        compute_pass.set_bind_group(0, &self.bg_camera, &[]);
        compute_pass.set_bind_group(1, &self.bg_rays[0], &[]);
        compute_pass.set_bind_group(2, &self.bg_schedule_intersect, &[]);
        compute_pass.set_bind_group(3, &scene.gpu_scene_geometry.bind_group, &[]);
        compute_pass.set_pipeline(&self.handle_primary_pipeline.get()?);
        let (x, y, _) = dispatch_size(
            (scene.camera.resolution[0], scene.camera.resolution[1], 1),
            (16, 16, 1),
        );
        compute_pass.dispatch_workgroups(x, y, 1);

        Ok(())
    }

    fn do_intersect(
        &mut self,
        compute_pass: &mut wgpu::ComputePass<'_>,
        scene: &BasicScene3d,
    ) -> Result<(), SceneRenderError> {
        compute_pass.set_bind_group(0, &self.bg_rays[0], &[]);
        compute_pass.set_bind_group(1, &self.bg_schedule_intersect, &[]);
        compute_pass.set_bind_group(2, &scene.gpu_scene_geometry.bind_group, &[]);
        compute_pass.set_pipeline(&self.ray_intersection_pipeline.get()?);
        compute_pass.dispatch_workgroups_indirect(&self.schedule_buffer.buffer, 0);

        Ok(())
    }

    fn do_schedule(
        &mut self,
        compute_pass: &mut wgpu::ComputePass<'_>,
        reorder_offset: u32,
    ) -> Result<(), SceneRenderError> {
        compute_pass.set_pipeline(&self.schedule_pipeline.get()?);
        compute_pass.set_bind_group(0, &self.bg_schedule_invocations, &[reorder_offset * 256]);
        compute_pass.dispatch_workgroups(1, 1, 1);

        Ok(())
    }

    fn do_reorder(
        &mut self,
        compute_pass: &mut wgpu::ComputePass<'_>,
        reorder_offset: u32,
    ) -> Result<(), SceneRenderError> {
        compute_pass.set_pipeline(&self.reorder_intersection_pipeline.get()?);
        compute_pass.set_bind_group(0, &self.bg_reorder_intersect, &[reorder_offset * 256]);
        compute_pass.dispatch_workgroups_indirect(&self.schedule_buffer.buffer, 32);

        Ok(())
    }

    fn do_shade(
        &mut self,
        compute_pass: &mut wgpu::ComputePass<'_>,
        scene: &BasicScene3d,
    ) -> Result<(), SceneRenderError> {
        compute_pass.set_bind_group(0, &self.bg_camera, &[]);
        compute_pass.set_bind_group(1, &self.bg_rays[1], &[]);
        compute_pass.set_bind_group(2, &self.bg_schedule_shade, &[]);
        compute_pass.set_bind_group(3, &scene.gpu_scene_geometry.bind_group, &[]);
        compute_pass.set_pipeline(&self.handle_intersections_pipeline.get()?);
        compute_pass.dispatch_workgroups_indirect(&self.schedule_buffer.buffer, 16);

        Ok(())
    }

    fn do_nee(&mut self, compute_pass: &mut wgpu::ComputePass<'_>) -> Result<(), SceneRenderError> {
        compute_pass.set_bind_group(0, &self.bg_rays[1], &[]);
        compute_pass.set_bind_group(1, &self.bg_schedule_nee_miss, &[]);
        compute_pass.set_pipeline(&self.nee_miss_pipeline.get()?);
        compute_pass.dispatch_workgroups_indirect(&self.schedule_buffer.buffer, 48);

        Ok(())
    }

    fn buffer_check(&mut self) {
        let gpu = self.gpu.clone();
        wgpu::util::DownloadBuffer::read_buffer(
            &gpu.device,
            &gpu.queue,
            &self.ray_intersection_buffers[1].slice(..),
            move |res| {
                if let Ok(download_buffer) = res {
                    let data: &[u8] = &download_buffer;
                    let data_u32: &[u32] = bytemuck::try_cast_slice(data)
                        .expect("Could not cast intersection data from u8 to f32");
                    let chunks = data_u32.chunks_exact(20);
                    let event_types = chunks.map(|chunk| chunk[19]);
                    let chunks = data_u32.chunks_exact(20);
                    info!(target: "aurora", "Downloaded {} intersections", chunks.len());
                    info!(target: "aurora", "First 5: {:?}", chunks.take(5).collect::<Vec<_>>());

                    info!(target: "aurora", "--- Intersection Buffer Check ---");
                    let mut count = [0u32; 16];
                    let mut oob_errors: usize = 0;
                    let mut order_errors: usize = 0;
                    let mut last_et: i32 = -1;
                    for et in event_types {
                        if (et as usize) < count.len() {
                            count[et as usize] += 1;
                            if last_et > et as i32 {
                                order_errors += 1;
                            }
                            last_et = et as i32;
                        } else {
                            oob_errors += 1;
                        }
                    }

                    info!(target: "aurora", "Event Type Counts: {:?}", &count);
                    if oob_errors > 0 {
                        warn!(target: "aurora", "There were {oob_errors} out of bounds event types in the intersection buffer");
                    }
                    if order_errors > 0 {
                        warn!(target: "aurora", "There were {order_errors} out of order event types in the intersection buffer");
                    }
                    info!(target: "aurora", "---------------------------------");
                }
            },
        );
    }
}

impl Scene3dView for PathTracerView {
    fn copy(&mut self, gpu: Arc<GpuContext>) -> Vec<wgpu::CommandBuffer> {
        // rust is fun some times
        let seeds: Vec<_> = self.rng.clone().random_iter::<u32>().take(16).collect();
        // let mut seeds: Vec<u32> = Vec::with_capacity(16);
        // seeds.resize(16, 0);

        let mut ce = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pt_ce_clear"),
            });

        if self.should_clear {
            ce.clear_buffer(
                &self.primary_ray_buffer.buffer,
                0,
                Some(self.primary_ray_buffer.size.get() as u64),
            );
        }

        let iterations = self.settings[0].max_iterations;
        wgpu::util::DownloadBuffer::read_buffer(
            &gpu.device,
            &gpu.queue,
            &self.buffer_schedule_reorder.slice(..),
            move |res| {
                if let Ok(download_buffer) = res {
                    let data: &[u8] = &download_buffer;
                    let data_u32: &[u32] = bytemuck::try_cast_slice(data)
                        .expect("Could not cast schedule reorder data from u8 to u32");
                    for i in 0..(iterations + 1) as usize {
                        let num_events = &data_u32[64 * i..64 * i + 16];
                        //let event_type_start = &data_u32[64 * i + 32..64 * i + 48];
                        let intersect_invocations = data_u32[64 * i + 48];
                        //log::info!(
                        //    target: "aurora",
                        //    "Iteration {i}: invocations: {:>8}, miss: {:>8}, nee_hit: {:>8}, nee_miss: {:>8}, primary_hit: {:>8}, shade: {:?}",
                        //    intersect_invocations, num_events[7], num_events[1], num_events[2], num_events[0],
                        //    &num_events[8..]
                        //);
                    }
                }
            },
        );

        vec![
            ce.finish(),
            self.buffer_copy_util.create_copy_command(&gpu, |ctx| {
                self.seed_buffer.write(ctx, &seeds);

                if self.should_clear {
                    self.settings.write(ctx);
                    self.should_clear = false;
                }
            }),
        ]
    }

    fn render(
        &mut self,
        gpu: Arc<aurora::GpuContext>,
        target: Arc<aurora::RenderTarget>,
        scene: &BasicScene3d,
        queries: &mut TimestampQueries,
    ) -> Result<wgpu::CommandBuffer, SceneRenderError> {
        self.check_image_bg(gpu.clone(), &target);

        let mut ce = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pt_ce"),
            });

        ce.clear_buffer(&self.buffer_schedule_reorder, 0, None);
        let resolution = scene.camera.resolution;
        {
            let mut compute_pass = ce.begin_compute_pass_timestamped("pt_cp_ray", queries);

            // primary rays
            self.do_primary(&mut compute_pass, scene)?;
            self.do_schedule(&mut compute_pass, 0)?;
            self.do_reorder(&mut compute_pass, 0)?;

            for i in 0..self.settings[0].max_iterations {
                compute_pass.push_debug_group(&format!("Path Tracer Iteration {i}"));
                // handle intersections
                self.do_shade(&mut compute_pass, scene)?;
                if self.settings[0].nee != 0 {
                    self.do_nee(&mut compute_pass)?;
                }
                self.do_schedule(&mut compute_pass, i + 1)?;

                self.do_intersect(&mut compute_pass, scene)?;
                self.do_schedule(&mut compute_pass, i + 1)?;
                self.do_reorder(&mut compute_pass, i + 1)?;

                // shade_invocations gets set to 0 by schedule, if we want to export
                // buffers depending on the number of shade invocations, we need to
                // skip scheduling
                // if !([1, 2, 5].contains(&self.settings[0].output_buffer)
                //     && i == self.settings[0].max_iterations - 1)
                // {
                //     // schedule
                //     self.do_schedule(&mut compute_pass)?;
                // }
                compute_pass.pop_debug_group();
            }
        }
        {
            let mut compute_pass = ce.begin_compute_pass_timestamped("pt_cp_target", queries);

            compute_pass.set_bind_group(0, &self.bg_camera, &[]);
            compute_pass.set_bind_group(1, &self.bg_rays[1], &[]);
            compute_pass.set_bind_group(2, self.bg_image.as_ref().unwrap(), &[]);
            compute_pass.set_bind_group(
                3,
                &self.bg_schedule_copy,
                &[self.settings[0].max_iterations * 256],
            );
            compute_pass.set_pipeline(&self.target_pipeline.get()?);
            let (x, y, _) = dispatch_size((resolution[0], resolution[1], 1), (16, 16, 1));
            compute_pass.dispatch_workgroups(x, y, 1);
        }

        ce.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &self.image_target_format.get().unwrap().buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(target.size[0] * 4 * 2),
                    rows_per_image: Some(target.size[1] * 4 * 2),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: target.size[0],
                height: target.size[1],
                depth_or_array_layers: 1,
            },
        );

        Ok(ce.finish())
    }

    fn draw_ui(&mut self, ui: &mut egui::Ui) {
        self.should_clear |= ui.button("Clear Buffer").clicked();
        if ui.button("Get Screenshot").clicked() {
            self.screenshot();
        }

        if ui.button("buffer_check").clicked() {
            self.buffer_check();
        }

        self.should_clear |= self.settings[0].ui(ui);
    }
}
