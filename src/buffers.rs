use crate::GpuContext;
use std::marker::PhantomData;
use std::num::NonZero;
use wgpu::util::DeviceExt;

pub struct Buffer<T: bytemuck::Pod> {
    pub buffer: wgpu::Buffer,
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
