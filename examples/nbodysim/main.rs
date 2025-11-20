use aurora::scenes::SceneRenderError;
use aurora::{
    buffers::{Buffer, MirroredBuffer},
    register_default, render_pipeline,
    scenes::Scene,
    shader::{BindGroupLayoutBuilder, RenderPipeline},
    Aurora, CommandEncoderTimestampExt, GpuContext, RenderTarget, TimestampQueries,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::Arc;
use wgpu::CommandBuffer;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let mut aurora = Aurora::new().await.unwrap();
    let scene = NBodySim::new(aurora.get_gpu(), aurora.get_target()).await;
    aurora.add_scene("nbodysim", Box::new(scene));
    aurora.run().unwrap();
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct NBodySimSettings {
    pub num_bodies: u32,
    pub time_step: f32,
    pub seed: u32,
}

impl NBodySimSettings {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.add(egui::Slider::new(&mut self.num_bodies, 1..=100_000).text("Number of Bodies"));
        ui.add(egui::Slider::new(&mut self.time_step, 0.001..=1.0).text("Time Step"));
    }
}

impl Default for NBodySimSettings {
    fn default() -> Self {
        Self {
            num_bodies: 1024,
            time_step: 0.01,
            seed: 42,
        }
    }
}

struct NBodySim {
    settings: MirroredBuffer<NBodySimSettings>,
    particle_positions: Buffer<f32>,
    particle_velocities: Buffer<f32>,
    bind_group: wgpu::BindGroup,
    pipeline: RenderPipeline,
}

impl NBodySim {
    pub async fn new(gpu: Arc<GpuContext>, target: Arc<RenderTarget>) -> Self {
        let settings = MirroredBuffer::from_data(
            &gpu,
            "NBodySim Settings",
            Box::new([NBodySimSettings::default()]),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let (particle_positions, particle_velocities) =
            Self::allocate_particles(&gpu, settings.data[0].num_bodies as usize);

        let rng = SmallRng::seed_from_u64(settings.data[0].seed as u64);
        let position_vector: Vec<f32> = rng
            .random_iter()
            .take(settings.data[0].num_bodies as usize * 2)
            .collect();
        gpu.queue.write_buffer(
            &particle_positions,
            0,
            bytemuck::cast_slice(&position_vector),
        );
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("Polling failed when initializing NBodySim buffers");

        let bind_group_layout = BindGroupLayoutBuilder::new(gpu.clone())
            .label("nbodysim_bgl")
            .add_buffer(
                0,
                wgpu::ShaderStages::all(),
                wgpu::BufferBindingType::Storage { read_only: true },
            )
            .add_buffer(
                1,
                wgpu::ShaderStages::all(),
                wgpu::BufferBindingType::Storage { read_only: true },
            )
            .build();
        let bind_group = bind_group_layout
            .bind_group_builder()
            .label("nbodysim_bg")
            .buffer(0, &particle_positions)
            .buffer(1, &particle_velocities)
            .build()
            .expect("Failed creating bind group for NBodySim");

        register_default!(gpu.shaders, "nbodysim_render", "render.wgsl");
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("nbodysim_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout.get()],
                push_constant_ranges: &[],
            });
        let pipeline = render_pipeline!(gpu, nbodysim_render; &wgpu::RenderPipelineDescriptor {
            label: Some("nbodysim_pipeline_render"),
            layout: Some(&pipeline_layout),
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
            particle_positions,
            particle_velocities,
            bind_group,
            pipeline,
        }
    }

    fn allocate_particles(gpu: &Arc<GpuContext>, num_bodies: usize) -> (Buffer<f32>, Buffer<f32>) {
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

        (particle_positions, particle_velocities)
    }
}

impl Scene for NBodySim {
    fn render(
        &mut self,
        gpu: Arc<GpuContext>,
        _target: Arc<RenderTarget>,
        queries: &mut TimestampQueries,
    ) -> Result<Vec<CommandBuffer>, SceneRenderError> {
        let mut ce = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nbodysim_ce"),
            });

        {
            let mut rp = ce.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("nbodysim_rp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &_target.view,
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

            rp.set_bind_group(0, &self.bind_group, &[]);
            rp.set_pipeline(&self.pipeline.get()?);
            rp.draw(0..(self.settings[0].num_bodies * 6), 0..1);
        }

        Ok(vec![ce.finish()])
    }

    fn draw_ui(&mut self, ui: &mut egui::Ui) {
        self.settings.data[0].ui(ui);
    }
}
