use aurora::{
    buffers::Buffer,
    compute_pipeline, dispatch_size, register_default,
    scenes::{BasicScene3d, Scene3dView, SceneGeometry},
    shader::{BindGroupLayout, BindGroupLayoutBuilder, ComputePipeline},
    Aurora, GpuContext,
};
use rand::Rng;
use std::sync::Arc;

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
    let scene_geometry = SceneGeometry::new("cornell_box.obj").await;
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

struct PathTracerView {
    schedule_pipeline: ComputePipeline,
    ray_generation_pipeline: ComputePipeline,
    ray_intersection_pipeline: ComputePipeline,
    handle_intersections_pipeline: ComputePipeline,
    target_pipeline: ComputePipeline,
    schedule_buffer: Buffer<u32>,
    seed_buffer: Buffer<u32>,
    primary_ray_buffer: Buffer<f32>,
    bg_camera: wgpu::BindGroup,
    bg_rays: wgpu::BindGroup,
    bg_schedule_invocations: wgpu::BindGroup,
    bg_schedule_intersect: wgpu::BindGroup,
    bg_schedule_shade: wgpu::BindGroup,
    image_f32: ImageStorageBuffer,
    image_target_format: ImageStorageBuffer,
    bgl_image: BindGroupLayout,
    bg_image: Option<wgpu::BindGroup>,
    buffer_copy_util: aurora::buffers::BufferCopyUtil,
    rng: rand::rngs::ThreadRng,
    should_clear: bool,
}

impl PathTracerView {
    fn new(gpu: Arc<GpuContext>, scene: &BasicScene3d) -> Self {
        // this fucntion is way too long, but I don't have a reasonable way to make it shorter?
        let total_pixels: usize =
            scene.camera.resolution[0] as usize * scene.camera.resolution[1] as usize;

        let primary_ray_buffer: Buffer<f32> = gpu.create_buffer(
            "primary_rays",
            size_of::<f32>() * 4 * 4 * total_pixels,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let ray_buffer: Buffer<f32> =
            gpu.create_buffer("rays", 64 * total_pixels, wgpu::BufferUsages::STORAGE);

        let ray_intersection_buffer: Buffer<f32> = gpu.create_buffer(
            "ray_intersections",
            80 * total_pixels,
            wgpu::BufferUsages::STORAGE,
        );

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

        let bgl_camera = BindGroupLayoutBuilder::new(gpu.clone())
            .label("bgl_camera")
            .add_buffer(
                0,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Uniform,
            )
            .build();
        let bg_camera = bgl_camera
            .bind_group_builder()
            .label("bg_camera")
            .buffer(0, &scene.camera.buffer)
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

        let bg_rays = bgl_rays
            .bind_group_builder()
            .label("bg_rays")
            .buffer(0, &primary_ray_buffer)
            .buffer(1, &ray_buffer)
            .buffer(2, &ray_intersection_buffer)
            .buffer(3, &seed_buffer)
            .build()
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
            .add_buffer(
                0,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
            )
            .build();

        let buffer_schedule_intersect: Buffer<u32> =
            gpu.create_buffer("pt_schedule_intersect", 16, wgpu::BufferUsages::STORAGE);
        let buffer_schedule_shade: Buffer<u32> =
            gpu.create_buffer("pt_schedule_shade", 16, wgpu::BufferUsages::STORAGE);

        let bg_schedule_intersect = bgl_schedule
            .bind_group_builder()
            .buffer(0, &buffer_schedule_intersect)
            .build()
            .expect("failed creating intersect schedule bind group");
        let bg_schedule_shade = bgl_schedule
            .bind_group_builder()
            .buffer(0, &buffer_schedule_shade)
            .build()
            .expect("failed creating shade schedule bind group");

        let bgl_schedule_invocations = BindGroupLayoutBuilder::new(gpu.clone())
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
            .build();

        let bg_schedule_invocations = bgl_schedule_invocations
            .bind_group_builder()
            .buffer(0, &schedule_buffer)
            .buffer(1, &buffer_schedule_intersect)
            .buffer(2, &buffer_schedule_shade)
            .build()
            .expect("failed creating invocations schedule bind group");

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pt_pipeline_layout"),
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
                    label: Some("pt_pipeline_layout"),
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
                    label: Some("pt_pipeline_layout"),
                    bind_group_layouts: &[
                        &bgl_rays.get(),
                        &bgl_schedule.get(),
                        &scene.gpu_scene_geometry.bind_group_layout.get(),
                    ],
                    push_constant_ranges: &[],
                });

        let pipeline_layout_image =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pt_pipeline_layout"),
                    bind_group_layouts: &[&bgl_camera.get(), &bgl_rays.get(), &bgl_image.get()],
                    push_constant_ranges: &[],
                });

        let pipeline_layout_schedule =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pt_pipeline_layout_schedule"),
                    bind_group_layouts: &[&bgl_schedule_invocations.get()],
                    push_constant_ranges: &[],
                });

        register_default!(gpu.shaders, "schedule", "pathtracer/shader/schedule.wgsl");
        register_default!(gpu.shaders, "primary", "pathtracer/shader/primary.wgsl");
        register_default!(gpu.shaders, "copy", "pathtracer/shader/copy.wgsl");
        register_default!(
            gpu.shaders,
            "path_tracer",
            "pathtracer/shader/pathtracer.wgsl"
        );
        register_default!(gpu.shaders, "intersect", "pathtracer/shader/intersect.wgsl");

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
            cache: None
        });

        let handle_intersections_pipeline = compute_pipeline!(gpu, path_tracer; &wgpu::ComputePipelineDescriptor {
            label: Some("pt_pipeline_handle_intersections"),
            layout: Some(&pipeline_layout),
            module: &path_tracer,
            entry_point: Some("handle_intersections"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        let target_pipeline = compute_pipeline!(gpu, copy; &wgpu::ComputePipelineDescriptor {
            label: Some("pt_pipeline_target"),
            layout: Some(&pipeline_layout_image),
            module: &copy,
            entry_point: Some("copy_target"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        Self {
            schedule_pipeline,
            ray_generation_pipeline,
            ray_intersection_pipeline,
            handle_intersections_pipeline,
            target_pipeline,
            schedule_buffer,
            seed_buffer,
            primary_ray_buffer,
            bg_camera,
            bg_rays,
            bg_schedule_invocations,
            bg_schedule_intersect,
            bg_schedule_shade,
            image_f32,
            image_target_format,
            bgl_image,
            bg_image: None,
            buffer_copy_util: aurora::buffers::BufferCopyUtil::new(2048),
            rng: rand::rng(),
            should_clear: true,
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
}

impl Scene3dView for PathTracerView {
    fn copy(&mut self, gpu: Arc<GpuContext>) -> Option<wgpu::CommandBuffer> {
        // rust is fun some times
        let seeds: Vec<_> = self.rng.clone().random_iter::<u32>().take(16).collect();

        Some(self.buffer_copy_util.create_copy_command(&gpu, |ctx| {
            self.seed_buffer.write(ctx, &seeds);
        }))
    }

    fn render(
        &mut self,
        gpu: Arc<aurora::GpuContext>,
        target: Arc<aurora::RenderTarget>,
        scene: &BasicScene3d,
    ) -> wgpu::CommandBuffer {
        self.check_image_bg(gpu.clone(), &target);

        let mut ce = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pt_ce"),
            });

        if self.should_clear {
            self.should_clear = false;
            ce.clear_buffer(
                &self.primary_ray_buffer.buffer,
                0,
                Some(self.primary_ray_buffer.size.get() as u64),
            );
        }

        let resolution = scene.camera.resolution;
        {
            let mut compute_pass = ce.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pt_cp_ray"),
                timestamp_writes: None,
            });

            // primary rays
            compute_pass.set_bind_group(0, &self.bg_camera, &[]);
            compute_pass.set_bind_group(1, &self.bg_rays, &[]);
            compute_pass.set_bind_group(2, &self.bg_schedule_invocations, &[]);
            compute_pass.set_pipeline(&self.ray_generation_pipeline.get());
            let (x, y, _) = dispatch_size((resolution[0], resolution[1], 1), (16, 16, 1));
            compute_pass.dispatch_workgroups(x, y, 1);

            for _ in 0..15 {
                // ray triangle intersections
                compute_pass.set_bind_group(0, &self.bg_rays, &[]);
                compute_pass.set_bind_group(1, &self.bg_schedule_intersect, &[]);
                compute_pass.set_bind_group(2, &scene.gpu_scene_geometry.bind_group, &[]);
                compute_pass.set_pipeline(&self.ray_intersection_pipeline.get());
                compute_pass.dispatch_workgroups_indirect(&self.schedule_buffer.buffer, 0);

                // schedule
                compute_pass.set_pipeline(&self.schedule_pipeline.get());
                compute_pass.set_bind_group(0, &self.bg_schedule_invocations, &[]);
                compute_pass.dispatch_workgroups(1, 1, 1);

                // handle intersections
                compute_pass.set_bind_group(0, &self.bg_camera, &[]);
                compute_pass.set_bind_group(1, &self.bg_rays, &[]);
                compute_pass.set_bind_group(2, &self.bg_schedule_shade, &[]);
                compute_pass.set_bind_group(3, &scene.gpu_scene_geometry.bind_group, &[]);
                compute_pass.set_pipeline(&self.handle_intersections_pipeline.get());
                compute_pass.dispatch_workgroups_indirect(&self.schedule_buffer.buffer, 16);

                // schedule
                compute_pass.set_pipeline(&self.schedule_pipeline.get());
                compute_pass.set_bind_group(0, &self.bg_schedule_invocations, &[]);
                compute_pass.dispatch_workgroups(1, 1, 1);
            }
        }
        {
            let mut compute_pass = ce.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pt_cp_target"),
                timestamp_writes: None,
            });

            compute_pass.set_bind_group(0, &self.bg_camera, &[]);
            compute_pass.set_bind_group(1, &self.bg_rays, &[]);
            compute_pass.set_bind_group(2, self.bg_image.as_ref().unwrap(), &[]);
            compute_pass.set_pipeline(&self.target_pipeline.get());
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

        ce.finish()
    }
}
