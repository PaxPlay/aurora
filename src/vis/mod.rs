use crate::scenes::SceneRenderError;
use crate::{GpuContext, RenderTarget, Scene, TimestampQueries};
use futures_lite::AsyncReadExt;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

pub mod nrrd;

#[derive(Error, Debug)]
pub enum CreateScalarFieldSceneError {
    #[error("Failed to read NRRD header")]
    ReadNRRDHeaderError(#[from] nrrd::ParseHeaderError),

    #[error("Failed to read data file: \"{0}\"")]
    LoadDataFileError(#[from] std::io::Error),

    #[error("Invalid data file contents: \"{0}\"")]
    InvalidDataFileError(String),
}

pub struct ScalarFieldScene {
    data: Vec<u8>,
    scalar_field_texture: wgpu::Texture,
}

impl ScalarFieldScene {
    pub async fn new(
        gpu: Arc<GpuContext>,
        file: &str,
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

        // let pipeline_layout = gpu
        //     .device
        //     .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {});
        //
        // let pipeline = gpu
        //     .device
        //     .create_render_pipeline(&wgpu::RenderPipelineDescriptor {});

        Ok(ScalarFieldScene {
            data,
            scalar_field_texture,
        })
    }
}
impl Scene for ScalarFieldScene {
    fn render(
        &mut self,
        _gpu: Arc<GpuContext>,
        _target: Arc<RenderTarget>,
        _queries: &mut TimestampQueries,
    ) -> Result<Vec<wgpu::CommandBuffer>, SceneRenderError> {
        Ok(vec![])
    }

    fn draw_ui(&mut self, _ui: &mut egui::Ui) {}

    fn update_target_parameters(&mut self, _gpu: Arc<GpuContext>, _target: Arc<RenderTarget>) {}
}
