use crate::shader::ComputePipeline;
use crate::GpuContext;
use crate::{compute_pipeline, register_default};
use std::marker::PhantomData;
use std::num::NonZero;
use wgpu::util::DeviceExt;

#[derive(Clone)]
pub struct Buffer<T: bytemuck::Pod> {
    pub buffer: wgpu::Buffer,
    pub size: NonZero<usize>,
    phantom: PhantomData<T>,
}

impl<T: bytemuck::Pod> Buffer<T> {
    pub fn from_data(gpu: &GpuContext, label: &str, data: &[T], usage: wgpu::BufferUsages) -> Self {
        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage,
            });

        Self {
            buffer,
            size: NonZero::new(size_of_val(data)).unwrap(),
            phantom: PhantomData,
        }
    }

    pub fn new(gpu: &GpuContext, label: &str, size: usize, usage: wgpu::BufferUsages) -> Self {
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            size: NonZero::new(size).unwrap(),
            phantom: PhantomData,
        }
    }

    pub fn write(&self, ctx: &mut BufferCopyContext, data: &[T]) {
        let mut buffer_view = ctx.staging_belt.write_buffer(
            &mut ctx.command_encoder,
            &self.buffer,
            0,
            NonZero::new(size_of_val(data) as u64).unwrap(),
            ctx.device,
        );

        let content: &mut [T] = bytemuck::cast_slice_mut(&mut buffer_view);
        content.copy_from_slice(data);
    }
}

pub struct BufferCopyUtil {
    staging_belt: wgpu::util::StagingBelt,
}

impl BufferCopyUtil {
    pub fn new(chunk_size: u64) -> Self {
        let staging_belt = wgpu::util::StagingBelt::new(chunk_size);

        Self { staging_belt }
    }

    pub fn create_copy_command<F>(
        &mut self,
        gpu: &GpuContext,
        mut copy_commands: F,
    ) -> wgpu::CommandBuffer
    where
        F: FnMut(&mut BufferCopyContext),
    {
        self.staging_belt.recall();
        let command_encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut ctx = BufferCopyContext {
            command_encoder,
            device: &gpu.device,
            staging_belt: &mut self.staging_belt,
        };

        copy_commands(&mut ctx);
        ctx.finish()
    }
}

pub struct BufferCopyContext<'a> {
    command_encoder: wgpu::CommandEncoder,
    device: &'a wgpu::Device,
    staging_belt: &'a mut wgpu::util::StagingBelt,
}

impl BufferCopyContext<'_> {
    fn finish(self) -> wgpu::CommandBuffer {
        self.staging_belt.finish();
        self.command_encoder.finish()
    }
}

pub struct BufferConvertCopy {
    pipeline: ComputePipeline,
    bind_group: wgpu::BindGroup,
    size_buffer: Buffer<u32>,
    size: Option<NonZero<u32>>,
}

impl BufferConvertCopy {
    pub fn new<T: bytemuck::Pod, U: bytemuck::Pod>(
        gpu: std::sync::Arc<GpuContext>,
        src_buffer: &Buffer<T>,
        dst_buffer: &Buffer<U>,
        size: u32,
    ) -> Self {
        let bgl = crate::shader::BindGroupLayoutBuilder::new(gpu.clone())
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
            .add_buffer(
                2,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Uniform,
            )
            .build();

        let size_buffer = gpu.create_buffer_init(
            "buffer_convert_size",
            &[size, 0, 0, 0],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let bind_group = bgl
            .bind_group_builder()
            .buffer(0, src_buffer)
            .buffer(1, dst_buffer)
            .buffer(2, &size_buffer)
            .build()
            .expect("Failed creating bind group for BufferConvertCopy");

        register_default!(
            gpu.shaders,
            "convert_f32_f16",
            "shader/buffer_convert_f32_f16.wgsl"
        );

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pt_pipeline_layout"),
                bind_group_layouts: &[&bgl.get()],
                push_constant_ranges: &[],
            });

        let pipeline = compute_pipeline!(gpu, convert_f32_f16; &wgpu::ComputePipelineDescriptor {
            label: Some("pt_pipeline_schedule"),
            layout: Some(&pipeline_layout),
            module: &convert_f32_f16,
            entry_point: Some("copy_buffer"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group,
            size_buffer,
            size: NonZero::new(size),
        }
    }

    pub fn copy(&mut self, compute_pass: &mut wgpu::ComputePass<'_>) {
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.set_pipeline(&self.pipeline.get());
        let x = self.size.unwrap().get().div_ceil(256);
        compute_pass.dispatch_workgroups(x, 1, 1);
    }
}
