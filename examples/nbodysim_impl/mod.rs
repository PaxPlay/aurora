use aurora::buffers::BufferCopyUtil;
use aurora::{
    buffers::{Buffer, MirroredBuffer},
    compute_pipeline, dispatch_size, register_default, render_pipeline,
    scenes::{Scene, SceneRenderError},
    shader::{BindGroupLayout, BindGroupLayoutBuilder, ComputePipeline, RenderPipeline},
    CommandEncoderTimestampExt, GpuContext, RenderTarget, TimestampQueries,
};
use std::f32;
use std::sync::Arc;
use wgpu::CommandBuffer;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct NBodySimSettings {
    pub num_bodies: u32,
    pub time_step: f32,
    pub seed: u32,
    pub gravity: f32,
}

impl NBodySimSettings {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.add(egui::Slider::new(&mut self.num_bodies, 2..=300_000).text("Number of Bodies"));
        ui.add(
            egui::Slider::new(&mut self.time_step, 0.000001..=0.01)
                .text("Time Step")
                .logarithmic(true),
        );
        ui.add(
            egui::Slider::new(&mut self.gravity, 0.000001..=0.01)
                .text("Gravity")
                .logarithmic(true),
        );
    }
}

impl Default for NBodySimSettings {
    fn default() -> Self {
        Self {
            num_bodies: 1024 * 64,
            time_step: 0.001,
            seed: 42,
            gravity: 1e-6,
        }
    }
}

pub struct NBodySim {
    settings: NBodySimSettings,
    settings_current: MirroredBuffer<NBodySimSettings>,
    sync_settings: bool,
    re_initialize: bool,
    copy_util: BufferCopyUtil,

    // particle_positions: Buffer<f32>,
    // particle_velocities: Buffer<f32>,
    // particle_forces: Buffer<f32>,
    layout_ro: BindGroupLayout,
    layout_forces: BindGroupLayout,
    layout_update: BindGroupLayout,
    bind_group_ro: wgpu::BindGroup,
    bind_group_forces: wgpu::BindGroup,
    bind_group_update: wgpu::BindGroup,
    bind_group_settings: wgpu::BindGroup,
    pipeline_render: RenderPipeline,
    pipeline_calculate_forces: ComputePipeline,
    pipeline_update: ComputePipeline,
    pipeline_initialize: ComputePipeline,
}

impl NBodySim {
    pub async fn new(gpu: Arc<GpuContext>, target: Arc<RenderTarget>) -> Self {
        let settings = NBodySimSettings::default();
        let settings_buffer = MirroredBuffer::from_data(
            &gpu,
            "NBodySim Settings",
            Box::new([settings.clone()]),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let (particle_positions, particle_velocities, particle_forces) =
            Self::allocate_particles(&gpu, settings.num_bodies as usize);

        let copy_util = BufferCopyUtil::new(2048);

        let bind_group_layout_settings = BindGroupLayoutBuilder::new(gpu.clone())
            .label("nbodysim_settings_bgl")
            .add_buffer(
                0,
                wgpu::ShaderStages::VERTEX
                    | wgpu::ShaderStages::FRAGMENT
                    | wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Uniform,
            )
            .build();
        let bind_group_settings = bind_group_layout_settings
            .bind_group_builder()
            .label("nbodysim_settings_bg")
            .buffer(0, &settings_buffer.buffer)
            .build()
            .expect("Failed creating bind group for NBodySim settings");

        let bind_group_layout_ro = BindGroupLayoutBuilder::new(gpu.clone())
            .label("nbodysim_bgl")
            .add_buffer(
                0,
                wgpu::ShaderStages::VERTEX
                    | wgpu::ShaderStages::FRAGMENT
                    | wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: true },
            )
            .add_buffer(
                1,
                wgpu::ShaderStages::VERTEX
                    | wgpu::ShaderStages::FRAGMENT
                    | wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: true },
            )
            .add_buffer(
                2,
                wgpu::ShaderStages::VERTEX
                    | wgpu::ShaderStages::FRAGMENT
                    | wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: true },
            )
            .build();

        let bind_group_layout_forces = BindGroupLayoutBuilder::new(gpu.clone())
            .label("nbodysim_forces_bgl")
            .add_buffer(
                0,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: true },
            )
            .add_buffer(
                1,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
            )
            .build();

        let bind_group_layout_update = BindGroupLayoutBuilder::new(gpu.clone())
            .label("nbodysim_update_bgl")
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
                wgpu::BufferBindingType::Storage { read_only: true },
            )
            .build();

        let (bind_group_ro, bind_group_update, bind_group_forces) = Self::create_bind_groups(
            particle_positions,
            particle_velocities,
            particle_forces,
            &bind_group_layout_ro,
            &bind_group_layout_forces,
            &bind_group_layout_update,
        );

        register_default!(gpu.shaders, "nbodysim_forces", "forces.wgsl");
        let pipeline_layout_forces =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("nbodysim_pipeline_layout_forces"),
                    bind_group_layouts: &[
                        &bind_group_layout_forces.get(),
                        &bind_group_layout_settings.get(),
                    ],
                    push_constant_ranges: &[],
                });
        let pipeline_calculate_forces = compute_pipeline!(gpu, nbodysim_forces; &wgpu::ComputePipelineDescriptor {
            label: Some("nbodysim_pipeline_calculate_forces"),
            layout: Some(&pipeline_layout_forces),
            module: &nbodysim_forces,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        register_default!(gpu.shaders, "nbodysim_update", "update.wgsl");
        let pipeline_layout_update =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("nbodysim_pipeline_layout_update"),
                    bind_group_layouts: &[
                        &bind_group_layout_update.get(),
                        &bind_group_layout_settings.get(),
                    ],
                    push_constant_ranges: &[],
                });
        let pl = pipeline_layout_update.clone();
        let pipeline_update = compute_pipeline!(gpu, nbodysim_update; &wgpu::ComputePipelineDescriptor {
            label: Some("nbodysim_pipeline_update"),
            layout: Some(&pl),
            module: &nbodysim_update,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        register_default!(gpu.shaders, "nbodysim_initialize", "initialize.wgsl");
        let pipeline_initialize = compute_pipeline!(gpu, nbodysim_initialize; &wgpu::ComputePipelineDescriptor {
            label: Some("nbodysim_pipeline_initialize"),
            layout: Some(&pipeline_layout_update),
            module: &nbodysim_initialize,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        register_default!(gpu.shaders, "nbodysim_render", "render.wgsl");
        let pipeline_layout_ro =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("nbodysim_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout_ro.get()],
                    push_constant_ranges: &[],
                });
        let pipeline_render = render_pipeline!(gpu, nbodysim_render; &wgpu::RenderPipelineDescriptor {
            label: Some("nbodysim_pipeline_render"),
            layout: Some(&pipeline_layout_ro),
            vertex: wgpu::VertexState {
                module: &nbodysim_render,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &nbodysim_render,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        Self {
            settings,
            settings_current: settings_buffer,
            sync_settings: false,
            re_initialize: true,
            copy_util,
            // particle_positions,
            // particle_velocities,
            // particle_forces,
            layout_ro: bind_group_layout_ro,
            layout_forces: bind_group_layout_forces,
            layout_update: bind_group_layout_update,
            bind_group_ro,
            bind_group_forces,
            bind_group_update,
            bind_group_settings,
            pipeline_render,
            pipeline_calculate_forces,
            pipeline_update,
            pipeline_initialize,
        }
    }

    fn allocate_particles(
        gpu: &GpuContext,
        num_bodies: usize,
    ) -> (Buffer<f32>, Buffer<f32>, Buffer<f32>) {
        let particle_positions = gpu.create_buffer(
            "Particle Positions",
            num_bodies * 4 * std::mem::size_of::<f32>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let particle_velocities = gpu.create_buffer(
            "Particle Velocities",
            num_bodies * 4 * std::mem::size_of::<f32>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let particle_forces = gpu.create_buffer(
            "Particle Forces",
            num_bodies * 4 * std::mem::size_of::<f32>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        // let mut rng = SmallRng::seed_from_u64(seed);
        // let mut position_vector: Vec<f32> = Vec::with_capacity(4 * num_bodies as usize);
        // for _ in 0..num_bodies as usize {
        //     position_vector.push(rng.random::<f32>());
        //     position_vector.push(rng.random::<f32>());
        //     position_vector.push(0.0);
        //     position_vector.push(0.0);
        // }
        //
        // gpu.queue.write_buffer(
        //     &particle_positions,
        //     0,
        //     bytemuck::cast_slice(&position_vector),
        // );
        //
        // let mut velocity_vector: Vec<f32> = Vec::with_capacity(4 * num_bodies as usize);
        // for _ in 0..num_bodies as usize {
        //     let a = rng.random::<f32>() * f32::consts::PI * 2.0f32;
        //
        //     velocity_vector.push(a.cos() * 0.001);
        //     velocity_vector.push(a.sin() * 0.001);
        //     velocity_vector.push(0.0);
        //     velocity_vector.push(0.0);
        // }
        // gpu.queue.write_buffer(
        //     &particle_velocities,
        //     0,
        //     bytemuck::cast_slice(&velocity_vector),
        // );
        //
        // gpu.device
        //     .poll(wgpu::PollType::wait_indefinitely())
        //     .expect("Polling failed when initializing NBodySim buffers");

        (particle_positions, particle_velocities, particle_forces)
    }

    fn create_bind_groups(
        positions: Buffer<f32>,
        velocities: Buffer<f32>,
        forces: Buffer<f32>,
        layout_ro: &BindGroupLayout,
        layout_forces: &BindGroupLayout,
        layout_update: &BindGroupLayout,
    ) -> (wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup) {
        let bind_group_ro = layout_ro
            .bind_group_builder()
            .label("nbodysim_bg")
            .buffer(0, &positions)
            .buffer(1, &velocities)
            .buffer(2, &forces)
            .build()
            .expect("Failed creating bind group for NBodySim");

        let bind_group_update = layout_update
            .bind_group_builder()
            .label("nbodysim_update_bg")
            .buffer(0, &positions)
            .buffer(1, &velocities)
            .buffer(2, &forces)
            .build()
            .expect("Failed creating bind group for NBodySim update");

        let bind_group_forces = layout_forces
            .bind_group_builder()
            .label("nbodysim_forces_bg")
            .buffer(0, &positions)
            .buffer(1, &forces)
            .build()
            .expect("Failed creating bind group for NBodySim forces");

        (bind_group_ro, bind_group_update, bind_group_forces)
    }
    fn reallocate_particles(&mut self, gpu: &GpuContext) {
        let (particle_positions, particle_velocities, particle_forces) =
            Self::allocate_particles(gpu, self.settings_current.data[0].num_bodies as usize);

        let (bind_group_ro, bind_group_update, bind_group_forces) = Self::create_bind_groups(
            particle_positions,
            particle_velocities,
            particle_forces,
            &self.layout_ro,
            &self.layout_forces,
            &self.layout_update,
        );

        self.bind_group_ro = bind_group_ro;
        self.bind_group_update = bind_group_update;
        self.bind_group_forces = bind_group_forces;
    }
}

impl Scene for NBodySim {
    fn render(
        &mut self,
        gpu: Arc<GpuContext>,
        target: Arc<RenderTarget>,
        queries: &mut TimestampQueries,
    ) -> Result<Vec<CommandBuffer>, SceneRenderError> {
        let mut cbs = Vec::with_capacity(2);
        let mut ce = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nbodysim_ce"),
            });

        if self.sync_settings {
            self.settings_current[0] = self.settings.clone();
            cbs.push(self.copy_util.create_copy_command(&gpu, |ctx| {
                self.settings_current.write(ctx);
            }));

            self.reallocate_particles(&gpu);

            self.re_initialize = true;
            self.sync_settings = false;
        }

        if self.re_initialize {
            let mut init_cp = ce.begin_compute_pass_timestamped("nbodysim_init_cp", queries);
            init_cp.set_bind_group(0, &self.bind_group_update, &[]);
            init_cp.set_bind_group(1, &self.bind_group_settings, &[]);
            init_cp.set_pipeline(&self.pipeline_initialize.get()?);
            let (x, y, z) = dispatch_size((self.settings_current[0].num_bodies, 1, 1), (64, 1, 1));
            init_cp.dispatch_workgroups(x, y, z);

            self.re_initialize = false;
        }

        {
            let mut cp = ce.begin_compute_pass_timestamped("nbodysim_cp", queries);
            cp.set_bind_group(0, &self.bind_group_forces, &[]);
            cp.set_bind_group(1, &self.bind_group_settings, &[]);
            cp.set_pipeline(&self.pipeline_calculate_forces.get()?);
            let (x, y, z) = dispatch_size((self.settings_current[0].num_bodies, 1, 1), (128, 1, 1));
            cp.dispatch_workgroups(x, y, z);

            cp.set_bind_group(0, &self.bind_group_update, &[]);
            cp.set_bind_group(1, &self.bind_group_settings, &[]);
            cp.set_pipeline(&self.pipeline_update.get()?);
            let (x, y, z) = dispatch_size((self.settings_current[0].num_bodies, 1, 1), (64, 1, 1));
            cp.dispatch_workgroups(x, y, z);
        }
        {
            let mut rp = ce.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("nbodysim_rp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: queries.render_pass_writes("nbodysim_rp"),
                occlusion_query_set: None,
            });

            rp.set_bind_group(0, &self.bind_group_ro, &[]);
            rp.set_pipeline(&self.pipeline_render.get()?);
            rp.draw(0..(self.settings_current[0].num_bodies * 6), 0..1);
        }

        cbs.push(ce.finish());

        Ok(cbs)
    }

    fn draw_ui(&mut self, ui: &mut egui::Ui) {
        self.settings.ui(ui);

        ui.horizontal(|ui| {
            if ui
                .button("Reset")
                .on_hover_text("Reset simulation without applying settings")
                .clicked()
            {
                self.re_initialize = true;
            }

            if ui
                .button("Apply Settings")
                .on_hover_text("Apply settings and reset simulation")
                .clicked()
            {
                self.sync_settings = true;
            }
        });
    }
}
