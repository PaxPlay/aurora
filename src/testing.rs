use crate::shader::BindGroupLayout;
use crate::{
    buffers::Buffer,
    compute_pipeline,
    shader::{BindGroupLayoutBuilder, ComputePipeline},
    GpuContext,
};
use std::sync::Arc;

pub fn create_gpu_context() -> Arc<GpuContext> {
    Arc::new(
        pollster::block_on(GpuContext::new())
            .expect("Couldn't initialize GpuContext for testing environment"),
    )
}

pub struct ComputeTestEnvironment<A: bytemuck::Pod, B: bytemuck::Pod> {
    pub gpu: Arc<GpuContext>,
    pub data_a: Vec<A>,
    pub data_b: Vec<B>,
    buffer_a: Buffer<A>,
    buffer_b: Buffer<B>,
    upload_staging_buffer: Buffer<u8>,
    download_staging_buffer: Buffer<u8>,
    bind_group: wgpu::BindGroup,
    bind_group_layout: BindGroupLayout,
    pipeline: Option<ComputePipeline>,
}

impl<A: bytemuck::Pod, B: bytemuck::Pod> ComputeTestEnvironment<A, B> {
    pub fn new(a_size: usize, b_size: usize) -> Self {
        let gpu = create_gpu_context();
        let data_a = vec![unsafe { std::mem::zeroed() }; a_size];
        let data_b = vec![unsafe { std::mem::zeroed() }; b_size];

        let buffer_a = gpu.create_buffer_init(
            "aurora_test_buffer_a",
            &data_a,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let buffer_b = gpu.create_buffer_init(
            "aurora_test_buffer_b",
            &data_b,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let upload_staging_buffer = gpu.create_buffer(
            "aurora_test_upload_staging_buffer",
            std::mem::size_of::<A>() * a_size + std::mem::size_of::<B>() * b_size,
            wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        );

        let download_staging_buffer = gpu.create_buffer(
            "aurora_test_download_staging_buffer",
            std::mem::size_of::<A>() * a_size + std::mem::size_of::<B>() * b_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let bind_group_layout = BindGroupLayoutBuilder::new(gpu.clone())
            .label("aurora_cbt_bgl")
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
        let bind_group = bind_group_layout
            .bind_group_builder()
            .buffer(0, &buffer_a)
            .buffer(1, &buffer_b)
            .build()
            .expect("Failed creating bind group for ComputeTestEnvironment");

        Self {
            gpu,
            data_a,
            data_b,
            buffer_a,
            buffer_b,
            upload_staging_buffer,
            download_staging_buffer,
            bind_group,
            bind_group_layout,
            pipeline: None,
        }
    }

    fn generate_include_dir_candidates() -> Vec<std::path::PathBuf> {
        let base_path = std::path::PathBuf::from(file!());
        let crate_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        vec![
            base_path.clone(),
            base_path.join("shader"),
            base_path.join("shaders"),
            crate_dir.join("shader"),
            crate_dir.join("shaders"),
            crate_dir.join("src"),
            crate_dir.join("src").join("shader"),
            crate_dir.join("src").join("shaders"),
        ]
    }

    pub fn set_shader_inline_single(&mut self, includes: &[&str], code: &str) {
        let combined_includes: String = includes
            .iter()
            .map(|include| format!("#include \"{}\"\n", include))
            .collect();

        let final_code = format!(
            "{combined_includes}

            @group(0) @binding(0) var<storage, read_write> buffer_a: array<u32>;
            @group(0) @binding(1) var<storage, read_write> buffer_b: array<u32>;

            @compute @workgroup_size(1)
            fn main() {{
                {code}
            }}
            "
        );

        let include_dirs = Self::generate_include_dir_candidates();

        let mut preprocessor = wgsl_preprocessor::WgslPreprocessor::new();
        for dir in &include_dirs {
            preprocessor.include_path(dir);
        }
        let processed_shader = preprocessor
            .process_string(&final_code, None)
            .expect("Failed to preprocess inline shader code");

        self.gpu.shaders.register_wgsl_static(
            "test_main",
            wgpu::ShaderSource::Wgsl(processed_shader.content.into()),
        );

        let pipeline_layout =
            self.gpu
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("aurora_test_main_pipeline_layout"),
                    bind_group_layouts: &[&self.bind_group_layout.get()],
                    push_constant_ranges: &[],
                });

        self.pipeline = Some(compute_pipeline!(
            self.gpu,
            test_main;
            &wgpu::ComputePipelineDescriptor {
                label: Some("aurora_test_main_pipeline"),
                layout: Some(&pipeline_layout),
                module: &test_main,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }
        ));
    }

    fn poll(&self) {
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("Failed polling device");
    }

    pub fn execute(&mut self) {
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora_test_command_encoder"),
            });

        // Upload data_a and data_b to GPU
        {
            let buffer = self.upload_staging_buffer.buffer.clone();
            let upload_a = bytemuck::cast_slice::<A, u8>(&self.data_a).to_vec();
            let upload_b = bytemuck::cast_slice::<B, u8>(&self.data_b).to_vec();
            self.upload_staging_buffer
                .map_async(wgpu::MapMode::Write, .., move |res| {
                    res.expect("Failed to map upload staging buffer for writing");
                    let mut view = buffer.get_mapped_range_mut(..);
                    let (a_bytes, b_bytes) = view.split_at_mut(upload_a.len());
                    a_bytes.copy_from_slice(&upload_a);
                    b_bytes.copy_from_slice(&upload_b);

                    drop(view);
                    buffer.unmap();
                });

            encoder.copy_buffer_to_buffer(
                &self.upload_staging_buffer,
                0,
                &self.buffer_a,
                0,
                (std::mem::size_of::<A>() * self.data_a.len()) as u64,
            );
            encoder.copy_buffer_to_buffer(
                &self.upload_staging_buffer,
                (std::mem::size_of::<A>() * self.data_a.len()) as u64,
                &self.buffer_b,
                0,
                (std::mem::size_of::<B>() * self.data_b.len()) as u64,
            );
        }

        self.poll();

        // Execute compute shader
        {
            let pipeline = self.pipeline.as_mut().expect("Compute pipeline not set");
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora_test_compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.get().expect("Failed getting compute pipeline"));
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        // Download results back to CPU
        {
            encoder.copy_buffer_to_buffer(
                &self.buffer_a,
                0,
                &self.download_staging_buffer,
                0,
                (std::mem::size_of::<A>() * self.data_a.len()) as u64,
            );
            encoder.copy_buffer_to_buffer(
                &self.buffer_b,
                0,
                &self.download_staging_buffer,
                (std::mem::size_of::<A>() * self.data_a.len()) as u64,
                (std::mem::size_of::<B>() * self.data_b.len()) as u64,
            );

            encoder.map_buffer_on_submit(
                &self.download_staging_buffer,
                wgpu::MapMode::Read,
                ..,
                |res| {
                    res.expect("Failed to map download staging buffer for reading");
                    // Data is read after polling
                },
            );
        }

        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        self.poll();

        {
            let view = self.download_staging_buffer.get_mapped_range(..);
            let (a_bytes, b_bytes) = view.split_at(std::mem::size_of::<A>() * self.data_a.len());
            self.data_a
                .copy_from_slice(bytemuck::cast_slice::<u8, A>(a_bytes));
            self.data_b
                .copy_from_slice(bytemuck::cast_slice::<u8, B>(b_bytes));

            drop(view);
            self.download_staging_buffer.unmap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_shader_single_simple() {
        let mut env: ComputeTestEnvironment<u32, u32> = ComputeTestEnvironment::new(4, 4);

        env.set_shader_inline_single(
            &[],
            "
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                buffer_b[i] = buffer_a[i] * 2u;
            }
            ",
        );

        env.data_a = vec![1, 2, 3, 4];
        env.execute();

        assert_eq!(env.data_b, vec![2, 4, 6, 8]);
    }
}
