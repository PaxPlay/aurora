use crate::buffers::{BufferCopyContext, BufferCopyUtil, MirroredBuffer};
use crate::scenes::SceneRenderError;
use crate::shader::BindGroupLayoutBuilder;
use crate::{
    register_default, render_pipeline,
    scenes::{Camera3d, CameraWithBuffer},
    shader::RenderPipeline,
    Buffer, CommandEncoderTimestampExt, DebugUi, GpuContext, RenderTarget, Scene, TimestampQueries,
};
use futures_lite::AsyncReadExt;
use glam::{vec3, vec4, Mat4, Vec3, Vec4, Vec4Swizzles};
use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;
use wgpu::util::DeviceExt;
use wgpu::Origin3d;
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

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalarFieldParameters {
    model_matrix: Mat4,
    world_to_field: Mat4,
    origin: Vec4,
    bb_size: Vec4,
}

impl TryFrom<&nrrd::NRRDHeader> for ScalarFieldParameters {
    type Error = CreateScalarFieldSceneError;

    fn try_from(header: &nrrd::NRRDHeader) -> Result<Self, Self::Error> {
        if header.dimension.get() != 3 {
            return Err(CreateScalarFieldSceneError::InvalidDataFileError(
                "NRRD header dimension is not 3".to_string(),
            ));
        }

        let origin = header.space_origin.as_ref().ok_or(
            CreateScalarFieldSceneError::InvalidDataFileError(
                "NRRD header must specify space origin".to_string(),
            ),
        )?;
        if origin.len() != 3 {
            return Err(CreateScalarFieldSceneError::InvalidDataFileError(
                "NRRD header origin length is not 3".to_string(),
            ));
        }
        let origin = vec3(origin[0] as f32, origin[1] as f32, origin[2] as f32);

        let directions = header.space_directions.as_ref().ok_or(
            CreateScalarFieldSceneError::InvalidDataFileError(
                "NRRD header must specify space directions".to_string(),
            ),
        )?;
        if directions.len() != 3 {
            return Err(CreateScalarFieldSceneError::InvalidDataFileError(
                "NRRD header directions length is not 3".to_string(),
            ));
        }
        let directions: Vec<Vec3> = directions
            .iter()
            .map(|d| -> Result<Vec3, CreateScalarFieldSceneError> {
                if let Some(v) = d {
                    if v.len() != 3 {
                        return Err(CreateScalarFieldSceneError::InvalidDataFileError(
                            "NRRD header space direction length is not 3".to_string(),
                        ));
                    }
                    Ok(vec3(v[0] as f32, v[1] as f32, v[2] as f32))
                } else {
                    Err(CreateScalarFieldSceneError::InvalidDataFileError(
                        "Space directions contain none direction".to_string(),
                    ))
                }
            })
            .collect::<Result<Vec<Vec3>, CreateScalarFieldSceneError>>()?;

        let sizes: [f32; 3] = [
            header.sizes[0].get() as f32,
            header.sizes[1].get() as f32,
            header.sizes[2].get() as f32,
        ];

        let model_matrix = Mat4::from_cols(
            directions[0].extend(0.0) * sizes[0],
            directions[1].extend(0.0) * sizes[1],
            directions[2].extend(0.0) * sizes[2],
            origin.extend(1.0),
        );

        let bb_size =
            directions[0] * sizes[0] + directions[1] * sizes[1] + directions[2] * sizes[2];

        Ok(ScalarFieldParameters {
            model_matrix,
            world_to_field: model_matrix.inverse(),
            origin: origin.extend(0.0),
            bb_size: bb_size.extend(0.0),
        })
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderParameters {
    isosurface_value: f32,
    isosurface_stddev: f32,
    step_size: f32,
}

impl Default for RenderParameters {
    fn default() -> Self {
        RenderParameters {
            isosurface_value: 0.5,
            isosurface_stddev: 50.0,
            step_size: 1.0,
        }
    }
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
    render_parameters: MirroredBuffer<RenderParameters>,
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

        let total_size = header.sizes.iter().map(|s| s.get()).product::<usize>();
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

        let parameters = ScalarFieldParameters::try_from(&header)?;
        let parameter_buffer = gpu.create_buffer_init(
            "aur_buf_sf_parameters",
            &[parameters],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

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

        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &scalar_field_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(header.sizes[0].get() as u32),
                rows_per_image: Some(header.sizes[1].get() as u32),
            },
            texture_size,
        );

        let bind_group_layout = BindGroupLayoutBuilder::new(gpu.clone())
            .add_buffer(
                0,
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                wgpu::BufferBindingType::Uniform,
            )
            .add_buffer(
                1,
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                wgpu::BufferBindingType::Uniform,
            )
            .add_buffer(
                2,
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                wgpu::BufferBindingType::Uniform,
            )
            .add_texture(
                3,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::TextureSampleType::Float { filterable: true },
                wgpu::TextureViewDimension::D3,
                false,
            )
            .add_sampler(
                4,
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
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            lod_min_clamp: 0f32,
            lod_max_clamp: 0f32,
            ..Default::default()
        });

        let camera = CameraWithBuffer::new(
            Camera3d::Centered {
                position: parameters.origin.xyz() + parameters.bb_size.xyz() / 2.0,
                angle: vec3(-PI / 4.0, PI / 4.0, 0.0).into(),
                distance: parameters.bb_size.xyz().length(),
                fov: 60.0,
                near: 0.1,
                far: parameters.bb_size.xyz().length() * 10.0,
                aspect_ratio: 1.0,
            },
            gpu.clone(),
        );

        let render_parameters = MirroredBuffer::new(
            &gpu,
            "aur_buf_sf_render_parameters",
            1,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        );

        let bind_group = bind_group_layout
            .bind_group_builder()
            .label("aur_bg_scalar_field")
            .buffer(0, &camera.buffer)
            .buffer(1, &parameter_buffer)
            .buffer(2, &render_parameters)
            .texture(
                3,
                scalar_field_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            )
            .sampler(4, sampler)
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
                bind_group_layouts: &[bind_group_layout.get_ref()],
                immediate_size: 0,
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
            multiview_mask: None,
            cache: None,
        });

        let buffer_copy_util = BufferCopyUtil::new(gpu.device.clone(), 2048);

        Ok(ScalarFieldScene {
            data,
            scalar_field_texture,
            pipeline,
            camera,
            vertex_buffer,
            index_buffer,
            bind_group,
            buffer_copy_util,
            render_parameters,
        })
    }

    fn do_copy_pass(&mut self, gpu: &GpuContext) -> wgpu::CommandBuffer {
        self.camera.process_controller_update();
        self.buffer_copy_util.create_copy_command(gpu, |ctx| {
            if !self.camera.is_buffer_current() {
                self.camera.update_buffer(ctx);
            }

            self.render_parameters.write(ctx);
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
        result.push(self.do_copy_pass(&gpu));

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
                multiview_mask: None,
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

        ui.separator();

        if ui.button("Reset Parameters").clicked() {
            self.render_parameters[0] = Default::default();
        }

        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(
                &mut self.render_parameters[0].isosurface_value,
                0.0..=1.0,
            ));
            ui.label("Isosurface Value");
        });
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(
                    &mut self.render_parameters[0].isosurface_stddev,
                    10.0..=2000.0,
                )
                .logarithmic(true),
            );
            ui.label("Isosurface Stddev");
        });
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut self.render_parameters[0].step_size, 0.001..=10.0)
                    .logarithmic(true),
            );
            ui.label("Step Size");
        });
    }

    fn update_target_parameters(&mut self, _gpu: Arc<GpuContext>, _target: Arc<RenderTarget>) {}

    fn on_window_event(&mut self, event: &WindowEvent) -> bool {
        self.camera.controller.on_window_event(event)
    }
    fn on_device_event(&mut self, event: &DeviceEvent) -> bool {
        self.camera.controller.on_device_event(event)
    }
}
