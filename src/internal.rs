use crate::{
    buffers::{Buffer, BufferCopyContext},
    register_default, render_pipeline,
    shader::{BindGroupLayoutBuilder, RenderPipeline},
    GpuContext, RenderTarget,
};
use glam::UVec2;

use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct TargetViewUniformBuffer {
    target: UVec2,
    window: UVec2,
    srgb: u32,
    _padding: u32,
}

unsafe impl bytemuck::Zeroable for TargetViewUniformBuffer {}
unsafe impl bytemuck::Pod for TargetViewUniformBuffer {}

pub struct TargetViewPipeline {
    pipeline: RenderPipeline,
    bind_group: wgpu::BindGroup,
    size_buffer: Buffer<TargetViewUniformBuffer>,
    srgb: bool,
}

impl TargetViewPipeline {
    pub fn new(
        gpu: Arc<GpuContext>,
        render: Arc<RenderTarget>,
        surface_format: wgpu::TextureFormat,
        window_size: [u32; 2],
        srgb: bool,
    ) -> Self {
        let bind_group_layout = BindGroupLayoutBuilder::new(gpu.clone())
            .add_texture_2d(0, wgpu::ShaderStages::FRAGMENT)
            .add_sampler(
                1,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            )
            .add_buffer(
                2,
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                wgpu::BufferBindingType::Uniform,
            )
            .build();

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("aurora_sampler_target_view"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0f32,
            lod_max_clamp: 0f32,
            ..Default::default()
        });

        let buffer_data = TargetViewUniformBuffer {
            target: UVec2::from_array(render.size),
            window: UVec2::from_array(window_size),
            srgb: if srgb { 1 } else { 0 },
            _padding: 0,
        };

        let buffer = gpu.create_buffer_init(
            "aurora_buffer_target_view",
            &[buffer_data],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let bind_group = bind_group_layout
            .bind_group_builder()
            .texture(0, render.view.clone())
            .sampler(1, sampler)
            .buffer(2, &buffer)
            .build()
            .expect("Couldn't build bind group.");

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora_pipeline_layout_target_view"),
                bind_group_layouts: &[&bind_group_layout.get()],
                push_constant_ranges: &[],
            });

        register_default!(gpu.shaders, "target_view", "shader/target_view.wgsl");

        let pipeline = render_pipeline!(gpu, target_view; &wgpu::RenderPipelineDescriptor {
            label: Some("aurora_pipeline_target_view"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &target_view,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[]
            },
            fragment: Some(wgpu::FragmentState {
                module: &target_view,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group,
            size_buffer: buffer,
            srgb,
        }
    }

    pub fn update_size_buffer(
        &mut self,
        ctx: &mut BufferCopyContext,
        target_size: [u32; 2],
        window_size: [u32; 2],
    ) {
        let contents = TargetViewUniformBuffer {
            target: UVec2::from_array(target_size),
            window: UVec2::from_array(window_size),
            srgb: if self.srgb { 1 } else { 0 },
            _padding: 0,
        };

        self.size_buffer.write(ctx, &[contents]);
    }

    pub fn build_command_buffer(
        &mut self,
        gpu: &GpuContext,
        target_view: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        let mut ce = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut render_pass = ce.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("aurora_rp_target_view"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });

            render_pass.set_pipeline(
                &self
                    .pipeline
                    .get()
                    .expect("TargetViewPipeline not available"),
            );
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }
        ce.finish()
    }
}
