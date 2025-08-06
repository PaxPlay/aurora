use aurora::{
    buffers::Buffer,
    compute_pipeline, dispatch_size, register_default,
    scenes::{BasicScene3d, Scene3dView},
    shader::{BindGroupLayout, BindGroupLayoutBuilder, ComputePipeline},
    Aurora, GpuContext,
};
use rand::Rng;
use std::sync::Arc;

fn main() -> std::process::ExitCode {
    env_logger::init();

    let mut aurora = pollster::block_on(Aurora::new()).unwrap();

    let mut scene = BasicScene3d::new(
        "models/cornell_box.obj",
        aurora.get_gpu(),
        aurora.get_target(),
    );
    scene.add_view(
        "path_tracer",
        Box::new(PathTracerView::new(aurora.get_gpu(), &scene)),
    );
    aurora.add_scene("basic_3d", Box::new(scene));
    aurora.run().unwrap();
    std::process::ExitCode::SUCCESS
}

struct PathTracerView {
    schedule_pipeline: ComputePipeline,
    ray_generation_pipeline: ComputePipeline,
    ray_intersection_pipeline: ComputePipeline,
    handle_intersections_pipeline: ComputePipeline,
    target_pipeline: ComputePipeline,
    schedule_buffer: Buffer<u32>,
    seed_buffer: Buffer<u32>,
    bg_camera: wgpu::BindGroup,
    bg_rays: wgpu::BindGroup,
    bgl_image: BindGroupLayout,
    bg_image: Option<wgpu::BindGroup>,
    buffer_copy_util: aurora::buffers::BufferCopyUtil,
    rng: rand::rngs::ThreadRng,
}

impl PathTracerView {
    fn new(gpu: Arc<GpuContext>, scene: &BasicScene3d) -> Self {
        let total_pixels: usize =
            scene.camera.resolution[0] as usize * scene.camera.resolution[1] as usize;

        let primary_ray_buffer: Buffer<f32> = gpu.create_buffer(
            "primary_rays",
            size_of::<f32>() * 4 * 3 * total_pixels,
            wgpu::BufferUsages::STORAGE,
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
                wgpu::BufferBindingType::Storage { read_only: false },
            )
            .add_buffer(
                4,
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
            .buffer(3, &schedule_buffer)
            .buffer(4, &seed_buffer)
            .build()
            .unwrap();

        let bgl_image = BindGroupLayoutBuilder::new(gpu.clone())
            .add_storage_texture(
                0,
                wgpu::ShaderStages::COMPUTE,
                wgpu::StorageTextureAccess::ReadWrite,
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureViewDimension::D2,
            )
            .build();

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pt_pipeline_layout"),
                bind_group_layouts: &[
                    &bgl_camera.get(),
                    &bgl_rays.get(),
                    &bgl_image.get(),
                    &scene.gpu_scene_geometry.bind_group_layout.get(),
                ],
                push_constant_ranges: &[],
            });

        register_default!(gpu.shaders, "path_tracer", "shader/pathtracer.wgsl");
        register_default!(gpu.shaders, "intersect", "shader/intersect.wgsl");

        let pl = pipeline_layout.clone();
        let schedule_pipeline = compute_pipeline!(gpu, path_tracer; &wgpu::ComputePipelineDescriptor {
            label: Some("pt_pipeline_schedule"),
            layout: Some(&pl),
            module: &path_tracer,
            entry_point: Some("schedule_invocations"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pl = pipeline_layout.clone();
        let ray_generation_pipeline = compute_pipeline!(gpu, path_tracer; &wgpu::ComputePipelineDescriptor {
            label: Some("pt_pipeline_ray_generation"),
            layout: Some(&pl),
            module: &path_tracer,
            entry_point: Some("generate_rays"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pl = pipeline_layout.clone();
        let ray_intersection_pipeline = compute_pipeline!(gpu, intersect; &wgpu::ComputePipelineDescriptor {
            label: Some("pt_pipeline_ray_intersection"),
            layout: Some(&pl),
            module: &intersect,
            entry_point: Some("intersect_rays"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        let pl = pipeline_layout.clone();
        let handle_intersections_pipeline = compute_pipeline!(gpu, path_tracer; &wgpu::ComputePipelineDescriptor {
            label: Some("pt_pipeline_handle_intersections"),
            layout: Some(&pl),
            module: &path_tracer,
            entry_point: Some("handle_intersections"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        let target_pipeline = compute_pipeline!(gpu, path_tracer; &wgpu::ComputePipelineDescriptor {
            label: Some("pt_pipeline_target"),
            layout: Some(&pipeline_layout),
            module: &path_tracer,
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
            bg_camera,
            bg_rays,
            bgl_image,
            bg_image: None,
            buffer_copy_util: aurora::buffers::BufferCopyUtil::new(2048),
            rng: rand::rng(),
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
        if self.bg_image.is_none() {
            let builder = self.bgl_image.bind_group_builder();
            self.bg_image = Some(builder.texture(0, target.view.clone()).build().unwrap());
        }

        let mut ce = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pt_ce"),
            });

        let resolution = scene.camera.resolution;
        {
            let mut compute_pass = ce.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pt_cp_ray"),
                timestamp_writes: None,
            });

            compute_pass.set_bind_group(0, &self.bg_camera, &[]);
            compute_pass.set_bind_group(1, &self.bg_rays, &[]);
            compute_pass.set_bind_group(2, self.bg_image.as_ref().unwrap(), &[]);
            compute_pass.set_bind_group(3, &scene.gpu_scene_geometry.bind_group, &[]);
            // primary rays
            compute_pass.set_pipeline(&self.ray_generation_pipeline.get());
            let (x, y, _) = dispatch_size((resolution[0], resolution[1], 1), (16, 16, 1));
            compute_pass.dispatch_workgroups(x, y, 1);

            for _ in 0..15 {
                // ray triangle intersections
                compute_pass.set_pipeline(&self.ray_intersection_pipeline.get());
                compute_pass.dispatch_workgroups_indirect(&self.schedule_buffer.buffer, 0);
                // schedule
                compute_pass.set_pipeline(&self.schedule_pipeline.get());
                compute_pass.dispatch_workgroups(1, 1, 1);

                // handle intersections
                compute_pass.set_pipeline(&self.handle_intersections_pipeline.get());
                compute_pass.dispatch_workgroups_indirect(&self.schedule_buffer.buffer, 16);
                // schedule
                compute_pass.set_pipeline(&self.schedule_pipeline.get());
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
            compute_pass.set_bind_group(3, &scene.gpu_scene_geometry.bind_group, &[]);
            compute_pass.set_pipeline(&self.target_pipeline.get());
            let (x, y, _) = dispatch_size((resolution[0], resolution[1], 1), (16, 16, 1));
            compute_pass.dispatch_workgroups(x, y, 1);
        }

        ce.finish()
    }
}
