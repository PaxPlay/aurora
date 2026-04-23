use crate::buffers::BufferCopyUtil;
use crate::scenes::SceneRenderError;
use crate::shader::BindGroupLayoutBuilder;
use crate::{
    register_default, render_pipeline,
    scenes::{Camera3d, CameraWithBuffer},
    shader::RenderPipeline,
    Buffer, CommandEncoderTimestampExt, DebugUi, GpuContext, RenderTarget, Scene, TimestampQueries,
};
use futures_lite::AsyncReadExt;
use glam::vec3;
use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;
use winit::event::{DeviceEvent, WindowEvent};

pub mod nrrd;

#[derive(Error, Debug)]
pub enum CreateScalarFieldSceneError {
    #[error("Failed to read NRRD header")]
    ReadNRRDHeaderError(#[from] nrrd::ParseHeaderError),

    #[error("Failed to read data file: \"{0}\"")]
    LoadDataFileError(#[from] std::io::Error),

    #[error("Invalid data file contents: \"{0}\"")]
    InvalidDataFileError(String),

    #[error("Error allocating GPU resources: {0}")]
    GPUResourcesError(String),
}

pub struct ScalarFieldScene {
    data: Vec<u8>,
    scalar_field_texture: wgpu::Texture,
    pipeline: RenderPipeline,
    camera: CameraWithBuffer,
    vertex_buffer: Buffer<f32>,
    index_buffer: Buffer<u32>,
    bind_group: wgpu::BindGroup,
    buffer_copy_util: BufferCopyUtil,
}

impl ScalarFieldScene {
    pub async fn new(
        gpu: Arc<GpuContext>,
        file: &str,
        target_format: wgpu::TextureFormat,
    ) -> Result<ScalarFieldScene, CreateScalarFieldSceneError> {
        let header =
            nrrd::NRRDHeader::from_buf_async(gpu.filesystem.create_reader(file).await).await?;

        let base_path = Path::new(file)
            .parent()
            .expect("Failed to get parent directory of NRRD file");
        let data_file_path = base_path.join(
            header
                .data_file
                .as_ref()
                .expect("NRRD header must specify data file"),
        );
        let mut data_file = gpu
            .filesystem
            .create_reader(data_file_path.to_str().unwrap())
            .await;

        let mut data = Vec::new();
        data_file.read_to_end(&mut data).await?;

        let total_size = header.sizes.iter().map(|s| s.get()).product();
        if data.len() != total_size {
            return Err(CreateScalarFieldSceneError::InvalidDataFileError(format!(
                "Data file size ({}) does not match expected size from header {} ({:?})",
                data.len(),
                total_size,
                header.sizes,
            )));
        }
        log::info!(
            "Loaded data file {}, total size {}, axis sizes {:?}",
            data_file_path.display(),
            total_size,
            header.sizes,
        );

        if header.dimension.get() != 3 || header.sizes.len() != 3 {
            return Err(CreateScalarFieldSceneError::InvalidDataFileError(
                "Header dimension is not 3 or axis sizes are invalid".to_string(),
            ));
        }

        let texture_size = wgpu::Extent3d {
            width: header.sizes[0].get() as u32,
            height: header.sizes[1].get() as u32,
            depth_or_array_layers: header.sizes[2].get() as u32,
        };

        use nrrd::NRRDType as NT;
        let texture_format = match header.data_type {
            NT::U8 => wgpu::TextureFormat::R8Unorm,
            _ => {
                return Err(CreateScalarFieldSceneError::InvalidDataFileError(format!(
                    "The NRRD files's data type is not supported: {:?}",
                    header.data_type,
                )));
            }
        };

        let scalar_field_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("aur_scalarfield_texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: texture_format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[texture_format],
        });

        let bind_group_layout = BindGroupLayoutBuilder::new(gpu.clone())
            .add_buffer(
                0,
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                wgpu::BufferBindingType::Uniform,
            )
            .add_texture(
                1,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::TextureSampleType::Float { filterable: true },
                wgpu::TextureViewDimension::D3,
                false,
            )
            .add_sampler(
                2,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            )
            .build();

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("aurora_sampler_scalar_field"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0f32,
            lod_max_clamp: 0f32,
            ..Default::default()
        });

        let camera = CameraWithBuffer::new(
            Camera3d::Centered {
                position: vec3(0.5, 0.5, 0.5),
                angle: vec3(-PI / 4.0, PI / 4.0, 0.0).into(),
                distance: 3.0,
                fov: 60.0,
                near: 0.1,
                far: 20.0,
                aspect_ratio: 1.0,
            },
            gpu.clone(),
        );

        let bind_group = bind_group_layout
            .bind_group_builder()
            .label("aur_bg_scalar_field")
            .buffer(0, &camera.buffer)
            .texture(
                1,
                scalar_field_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            )
            .sampler(2, sampler)
            .build()
            .map_err(|e| {
                CreateScalarFieldSceneError::GPUResourcesError(format!(
                    "Failed to create bind group: {e}"
                ))
            })?;

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aur_pipeline_layout_scalar_field"),
                bind_group_layouts: &[&bind_group_layout.get()],
                push_constant_ranges: &[],
            });

        let vertices: [f32; 24] = [
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            1.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            1.0, 0.0, 1.0, //
            0.0, 1.0, 1.0, //
            1.0, 1.0, 1.0, //
        ];

        let indices = [
            0, 1, 3, 0, 3, 2, //
            4, 6, 7, 4, 7, 5, //
            0, 4, 5, 0, 5, 1, //
            2, 7, 6, 2, 3, 7, //
            1, 5, 7, 1, 7, 3, //
            0, 6, 4, 0, 2, 6, //
        ];

        let vertex_buffer =
            gpu.create_buffer_init("aur_vb_sf_unit_cube", &vertices, wgpu::BufferUsages::VERTEX);

        let index_buffer =
            gpu.create_buffer_init("aur_ib_sf_unit_cube", &indices, wgpu::BufferUsages::INDEX);

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<f32>() as u64 * 3u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0u64,
                shader_location: 0,
            }],
        };

        register_default!(gpu.shaders, "scalar_field", "scalar_field.wgsl");
        let pipeline = render_pipeline!(gpu, scalar_field; &wgpu::RenderPipelineDescriptor {
            label: Some("aur_pipeline_scalar_field"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &scalar_field,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[vertex_buffer_layout.clone()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &scalar_field,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                ]
            }),
            multiview: None,
            cache: None,
        });

        let buffer_copy_util = BufferCopyUtil::new(2048);

        Ok(ScalarFieldScene {
            data,
            scalar_field_texture,
            pipeline,
            camera,
            vertex_buffer,
            index_buffer,
            bind_group,
            buffer_copy_util,
        })
    }
}
impl Scene for ScalarFieldScene {
    fn render(
        &mut self,
        gpu: Arc<GpuContext>,
        target: Arc<RenderTarget>,
        queries: &mut TimestampQueries,
    ) -> Result<Vec<wgpu::CommandBuffer>, SceneRenderError> {
        let mut result = Vec::with_capacity(2);

        self.camera.process_controller_update();
        if !self.camera.is_buffer_current() {
            result.push(
                self.buffer_copy_util
                    .create_copy_command(&gpu, |ctx| self.camera.update_buffer(ctx)),
            );
        }

        let mut ce = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aur_ce_scalar_field"),
            });

        {
            let mut rp = ce.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("aur_rp_scalar_field"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: queries.render_pass_writes("rp_scalar_field"),
                occlusion_query_set: None,
            });

            rp.set_pipeline(&self.pipeline.get()?);
            rp.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rp.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rp.set_bind_group(0, &self.bind_group, &[]);
            rp.draw_indexed(0..36, 0, 0..1);
        }

        result.push(ce.finish());

        Ok(result)
    }

    fn draw_ui(&mut self, ui: &mut egui::Ui) {
        self.camera.draw_ui(ui);
    }

    fn update_target_parameters(&mut self, _gpu: Arc<GpuContext>, _target: Arc<RenderTarget>) {}

    fn on_window_event(&mut self, event: &WindowEvent) -> bool {
        self.camera.controller.on_window_event(event)
    }
    fn on_device_event(&mut self, event: &DeviceEvent) -> bool {
        self.camera.controller.on_device_event(event)
    }
}
