use crate::scenes::SceneRenderError;
use crate::{GpuContext, RenderTarget, Scene, TimestampQueries};
use std::sync::Arc;
use thiserror::Error;

pub mod nrrd;

#[derive(Error, Debug)]
pub enum CreateScalarFieldSceneError {
    #[error("Failed to read NRRD header")]
    ReadNRRDHeaderError(#[from] nrrd::ParseHeaderError),
}

pub struct ScalarFieldScene {}

impl ScalarFieldScene {
    pub async fn new(
        gpu: Arc<GpuContext>,
        file: &str,
    ) -> Result<ScalarFieldScene, CreateScalarFieldSceneError> {
        let header =
            nrrd::NRRDHeader::from_buf_async(gpu.filesystem.create_reader(file).await).await?;

        Ok(ScalarFieldScene {})
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
