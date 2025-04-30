use glam::{mat3, mat4, vec3, vec4};
use glam::{Mat3, Mat4, Vec3, Vec4};

use log::info;

use crate::shader::{BindGroupLayoutBuilder, RenderPipeline};
use crate::{register_default, render_pipeline, DebugUi, GpuContext, RenderTarget};
use std::f32::consts::*;
use std::num::NonZero;
use std::pin;
use std::sync::Arc;

pub trait Scene {
    fn render<'a>(
        &'a mut self,
        gpu: Arc<GpuContext>,
        target: Arc<RenderTarget>,
    ) -> wgpu::CommandBuffer;

    fn draw_ui(&mut self, _ui: &mut egui::Ui) {}
}

#[derive(Copy, Clone)]
pub struct Angle {
    pitch: f32,
    yaw: f32,
    roll: f32,
}

impl From<Vec3> for Angle {
    fn from(value: Vec3) -> Self {
        Self {
            pitch: value.x,
            yaw: value.y,
            roll: value.z,
        }
    }
}

impl Into<Vec3> for Angle {
    fn into(self) -> Vec3 {
        Vec3 {
            x: self.pitch,
            y: self.yaw,
            z: self.roll,
        }
    }
}

impl Angle {
    fn direction(self) -> Vec3 {
        let sin_phi = self.yaw.sin();
        let cos_phi = self.yaw.cos();
        let sin_theta = self.pitch.sin();
        let cos_theta = self.pitch.cos();

        vec3(cos_phi * cos_theta, sin_phi * cos_theta, sin_theta)
    }

    fn normalize(self) -> Self {
        let mut res = self.clone();
        while res.pitch > PI {
            res.pitch -= 2.0f32 * PI;
        }
        while res.pitch < -PI {
            res.pitch += 2.0f32 * PI;
        }

        if res.pitch > FRAC_PI_2 {
            res.pitch = PI - res.pitch;
            res.yaw += PI;
        }
        if res.pitch < -FRAC_PI_2 {
            res.pitch = -PI - res.pitch;
            res.yaw += PI;
        }

        while res.yaw > PI {
            res.yaw -= 2.0f32 * PI;
        }
        while res.yaw < -PI {
            res.yaw += 2.0f32 * PI;
        }

        res
    }

    fn up(self) -> Vec3 {
        let mut res = self;
        res.pitch += FRAC_PI_2;
        res.normalize().direction()
    }
}

pub enum Camera3d {
    Centered {
        position: Vec3,
        angle: Angle,
        distance: f32,
        fov: f32,
        near: f32,
        far: f32,
        aspect_ratio: f32,
    },
    Perspective {
        position: Vec3,
        angle: Angle,
        fov: f32,
        near: f32,
        far: f32,
        aspect_ratio: f32,
    },
    Orthographic {
        position: Vec3,
        angle: Angle,
        zoom: f32,
        near: f32,
        far: f32,
        aspect_ratio: f32,
    },
}

impl Camera3d {
    fn default() -> Self {
        Self::Centered {
            position: vec3(0.0f32, 0.0f32, 0.0f32),
            angle: Angle::from(vec3(0.0f32, 0.0f32, 0.0f32)),
            distance: 10.0f32,
            fov: 60.0f32,
            near: 0.1f32,
            far: 50.0f32,
            aspect_ratio: 16.0f32 / 9.0f32,
        }
    }
}

impl Camera3d {
    fn view_projection_matrix(&self) -> Mat4 {
        match self {
            Camera3d::Centered {
                position,
                angle,
                distance,
                fov,
                near,
                far,
                aspect_ratio,
            } => {
                Mat4::perspective_lh(fov.to_radians(), *aspect_ratio, *near, *far)
                    * Mat4::look_at_lh(
                        position - angle.direction() * distance,
                        position.clone(),
                        angle.up(),
                    )
            }
            Camera3d::Perspective {
                position,
                angle,
                fov,
                near,
                far,
                aspect_ratio,
            } => {
                Mat4::perspective_lh(fov.to_radians(), *aspect_ratio, *near, *far)
                    * Mat4::look_at_lh(position.clone(), position + angle.direction(), angle.up())
            }
            Camera3d::Orthographic {
                position,
                angle,
                zoom,
                near,
                far,
                aspect_ratio,
            } => {
                let right = zoom * aspect_ratio;

                Mat4::orthographic_lh(-right, right, -zoom, *zoom, *near, *far)
                    * Mat4::look_at_lh(position.clone(), position + angle.direction(), angle.up())
            }
        }
    }

    //fn handle_keyboard_input() {}

    //fn handle_mouse_movement() {}

    fn to_centered(&mut self) {
        match *self {
            Self::Centered { .. } => (),
            Self::Perspective {
                position,
                angle,
                fov,
                near,
                far,
                aspect_ratio,
            } => {
                let distance = 500.0f32;
                *self = Self::Centered {
                    position: position - angle.direction() * distance,
                    angle,
                    distance,
                    fov,
                    near,
                    far,
                    aspect_ratio,
                }
            }
            Self::Orthographic {
                position,
                angle,
                zoom,
                near,
                far,
                aspect_ratio,
            } => {
                let distance = 500.0f32;
                *self = Self::Centered {
                    position: position - angle.direction() * distance,
                    angle,
                    distance,
                    fov: zoom / 8.0,
                    near,
                    far,
                    aspect_ratio,
                }
            }
        }
    }

    fn to_perspective(&mut self) {
        match *self {
            Self::Centered {
                position,
                angle,
                distance,
                fov,
                near,
                far,
                aspect_ratio,
            } => {
                *self = Self::Perspective {
                    position: position + angle.direction() * distance,
                    angle,
                    fov,
                    near,
                    far,
                    aspect_ratio,
                }
            }
            Self::Perspective { .. } => (),
            Self::Orthographic {
                position,
                angle,
                zoom,
                near,
                far,
                aspect_ratio,
            } => {
                *self = Self::Perspective {
                    position,
                    angle,
                    fov: zoom / 8.0,
                    near,
                    far,
                    aspect_ratio,
                }
            }
        }
    }

    fn to_orthographic(&mut self) {
        match *self {
            Self::Centered {
                position,
                angle,
                distance,
                fov,
                near,
                far,
                aspect_ratio,
            } => {
                *self = Self::Orthographic {
                    position: position + angle.direction() * distance,
                    angle,
                    zoom: fov * 8.0,
                    near,
                    far,
                    aspect_ratio,
                }
            }
            Self::Perspective {
                position,
                angle,
                fov,
                near,
                far,
                aspect_ratio,
            } => {
                *self = Self::Orthographic {
                    position,
                    angle,
                    zoom: fov * 8.0,
                    near,
                    far,
                    aspect_ratio,
                }
            }
            Self::Orthographic { .. } => (),
        }
    }
}

impl DebugUi for Camera3d {
    fn draw_ui(&mut self, ui: &mut egui::Ui) {
        let current_type = match self {
            &mut Self::Centered { .. } => 0u32,
            &mut Self::Perspective { .. } => 1u32,
            &mut Self::Orthographic { .. } => 2u32,
        };

        let mut new_type = current_type;

        let value_text = |i: u32| -> &'static str {
            match i {
                0 => "Centered",
                1 => "Perspective",
                2 => "Orthographic",
                _ => "Invalid",
            }
        };

        egui::ComboBox::from_label("Camera Type")
            .selected_text(value_text(current_type))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut new_type, 0, value_text(0));
                ui.selectable_value(&mut new_type, 1, value_text(1));
                ui.selectable_value(&mut new_type, 2, value_text(2));
            });

        if current_type != new_type {
            match new_type {
                0 => {
                    self.to_centered();
                }
                1 => {
                    self.to_perspective();
                }
                2 => {
                    self.to_orthographic();
                }
                _ => (),
            }
        }

        match self {
            Self::Centered {
                position,
                angle,
                distance,
                fov,
                near,
                far,
                ..
            } => {
                position.draw_ui(ui);
                angle.draw_ui(ui);
                ui.horizontal(|ui| {
                    ui.label("distance");
                    ui.add(egui::DragValue::new(distance));
                });
                ui.horizontal(|ui| {
                    ui.label("near");
                    ui.add(egui::DragValue::new(near));
                    ui.label("far");
                    ui.add(egui::DragValue::new(far));
                    ui.label("fov");
                    ui.add(egui::DragValue::new(fov));
                });
            }
            Self::Perspective {
                position,
                angle,
                fov,
                near,
                far,
                ..
            } => {
                position.draw_ui(ui);
                angle.draw_ui(ui);
                ui.horizontal(|ui| {
                    ui.label("near");
                    ui.add(egui::DragValue::new(near));
                    ui.label("far");
                    ui.add(egui::DragValue::new(far));
                    ui.label("fov");
                    ui.add(egui::DragValue::new(fov));
                });
            }
            Self::Orthographic {
                position,
                angle,
                zoom,
                near,
                far,
                ..
            } => {
                position.draw_ui(ui);
                angle.draw_ui(ui);
                ui.horizontal(|ui| {
                    ui.label("near");
                    ui.add(egui::DragValue::new(near));
                    ui.label("far");
                    ui.add(egui::DragValue::new(far));
                    ui.label("zoom");
                    ui.add(egui::DragValue::new(zoom));
                });
            }
        }
    }
}

impl DebugUi for Vec3 {
    fn draw_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("x");
            ui.add(egui::DragValue::new(&mut self.x).speed(1.0));
            ui.label("y");
            ui.add(egui::DragValue::new(&mut self.y).speed(1.0));
            ui.label("z");
            ui.add(egui::DragValue::new(&mut self.z).speed(1.0));
        });
    }
}

impl DebugUi for Angle {
    fn draw_ui(&mut self, ui: &mut egui::Ui) {
        let mut pitch = self.pitch.to_degrees();
        let mut yaw = self.yaw.to_degrees();
        let mut roll = self.roll.to_degrees();

        ui.horizontal(|ui| {
            ui.label("pitch");
            ui.add(
                egui::DragValue::new(&mut pitch)
                    .range(-90.0..=90.0)
                    .speed(1.0),
            );
            ui.label("yaw");
            ui.add(egui::DragValue::new(&mut yaw).range(-180..=180).speed(1.0));
            ui.label("roll");
            ui.add(egui::DragValue::new(&mut roll).range(-180..=180).speed(1.0));
        });

        self.pitch = pitch.to_radians();
        self.yaw = yaw.to_radians();
        self.roll = roll.to_radians();
    }
}

struct SceneGeometry {
    vertices: Vec<f32>,
    indices: Vec<u32>,
    model_start_indices: Vec<u32>,
    material_indices: Vec<u32>,
    ambient: Vec<f32>,
    diffuse: Vec<f32>,
    specular: Vec<f32>,
}

impl SceneGeometry {
    pub fn new(file: &str) -> Self {
        let (models, materials) = tobj::load_obj(file, &tobj::GPU_LOAD_OPTIONS)
            .expect(format!("Unable to load obj file for {}", file).as_str());
        let materials = materials.unwrap();
        info!(target: "aurora", "Models: {}, Materials: {}", models.len(), materials.len());

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut model_start_indices = Vec::new();
        let mut material_indices = Vec::new();

        for model in models {
            let start_index = indices.len() as u32;
            let start_vertex = vertices.len() as u32 / 3;
            model_start_indices.push(start_index);
            material_indices.push(model.mesh.material_id.unwrap_or(0) as u32);
            vertices.extend(model.mesh.positions);
            indices.extend(model.mesh.indices.iter().map(|i| i + start_vertex));
        }
        info!(target: "aurora", "Total Vertices: {}, Total triangles: {}", vertices.len(), indices.len() / 3);

        let mut ambient = Vec::new();
        let mut diffuse = Vec::new();
        let mut specular = Vec::new();

        for material in materials {
            ambient.extend(material.ambient.unwrap_or([0f32, 0f32, 0f32]));
            diffuse.extend(material.diffuse.unwrap_or([0f32, 0f32, 0f32]));
            specular.extend(material.specular.unwrap_or([0f32, 0f32, 0f32]));
        }

        Self {
            vertices,
            indices,
            model_start_indices,
            material_indices,
            ambient,
            diffuse,
            specular,
        }
    }
}

struct GpuSceneGeometry {
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub model_start_indices: wgpu::Buffer,
    pub material_indices: wgpu::Buffer,
    pub ambient: wgpu::Buffer,
    pub diffuse: wgpu::Buffer,
    pub specular: wgpu::Buffer,
}

impl GpuSceneGeometry {
    pub fn from(scene_geometry: &SceneGeometry, gpu: &GpuContext) -> Self {
        use wgpu::BufferUsages as BU;

        let vertices = gpu.create_buffer_init(
            "aurora_scene_vertices",
            &scene_geometry.vertices,
            BU::VERTEX | BU::STORAGE | BU::UNIFORM,
        );

        let indices = gpu.create_buffer_init(
            "aurora_scene_indices",
            &scene_geometry.indices,
            BU::INDEX | BU::STORAGE | BU::UNIFORM,
        );

        let model_start_indices = gpu.create_buffer_init(
            "aurora_scene_model_start_indices",
            &scene_geometry.model_start_indices,
            BU::STORAGE,
        );

        let material_indices = gpu.create_buffer_init(
            "aurora_scene_material_indices",
            &scene_geometry.material_indices,
            BU::STORAGE,
        );

        let ambient =
            gpu.create_buffer_init("aurora_scene_ambient", &scene_geometry.ambient, BU::STORAGE);

        let diffuse =
            gpu.create_buffer_init("aurora_scene_diffuse", &scene_geometry.diffuse, BU::STORAGE);

        let specular = gpu.create_buffer_init(
            "aurora_scene_specular",
            &scene_geometry.specular,
            BU::STORAGE,
        );

        Self {
            vertices,
            indices,
            model_start_indices,
            material_indices,
            ambient,
            diffuse,
            specular,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct SceneUniformBuffer {
    mvp: Mat4,
}

unsafe impl bytemuck::Pod for SceneUniformBuffer {}
unsafe impl bytemuck::Zeroable for SceneUniformBuffer {}

pub struct BasicScene3d {
    camera: Camera3d,
    pipeline: RenderPipeline,
    scene_geometry: SceneGeometry,
    gpu_scene_geometry: GpuSceneGeometry,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    staging_belt: wgpu::util::StagingBelt,
}

impl BasicScene3d {
    pub fn new(file: &str, gpu: Arc<GpuContext>, target: Arc<RenderTarget>) -> Self {
        let scene_geometry = SceneGeometry::new(file);

        let camera = Camera3d::Perspective {
            position: vec3(278f32, 273f32, -800f32),
            angle: vec3(FRAC_PI_2, -FRAC_PI_2, 0f32).into(),
            fov: 39.3f32,
            near: 500f32,
            far: 2000f32,
            aspect_ratio: 1.5f32,
        };

        let gpu_scene_geometry = GpuSceneGeometry::from(&scene_geometry, &gpu);

        let ub_contents = SceneUniformBuffer {
            mvp: camera.view_projection_matrix(),
        };

        let uniform_buffer = gpu.create_buffer_init(
            "aurora_scene_uniform",
            &vec![ub_contents],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        register_default!(gpu.shaders, "wireframe", "shader/wireframe.wgsl");

        let device = &gpu.device;

        use wgpu::BufferBindingType as BBT;
        use wgpu::ShaderStages as SS;
        let bind_group_layout = BindGroupLayoutBuilder::new(gpu.clone())
            .label("aurora_scene_bg_layout")
            .add_buffer(0, SS::VERTEX, BBT::Uniform)
            .add_buffer(1, SS::VERTEX, BBT::Storage { read_only: true })
            .add_buffer(2, SS::VERTEX, BBT::Storage { read_only: true })
            .build();

        let bind_group = bind_group_layout
            .bind_group_builder()
            .label("aurora_scene_bg")
            .buffer(0, uniform_buffer.clone())
            .buffer(1, gpu_scene_geometry.vertices.clone())
            .buffer(2, gpu_scene_geometry.indices.clone())
            .build()
            .unwrap();

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Basic 3D Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout.get()],
            push_constant_ranges: &[],
        });

        let pipeline = render_pipeline!(
            gpu, wireframe;
            &wgpu::RenderPipelineDescriptor {
                label: Some("Basic 3D Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &wireframe,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &wireframe,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Cw,
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
            }
        );

        let staging_belt = wgpu::util::StagingBelt::new(2048);

        Self {
            camera,
            pipeline,
            scene_geometry,
            gpu_scene_geometry,
            uniform_buffer,
            bind_group,
            staging_belt,
        }
    }
}

impl Scene for BasicScene3d {
    fn render<'a>(
        &'a mut self,
        gpu: Arc<GpuContext>,
        target: Arc<RenderTarget>,
    ) -> wgpu::CommandBuffer {
        let mut ce = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        self.staging_belt.recall();
        {
            let mut buffer_view = self.staging_belt.write_buffer(
                &mut ce,
                &self.uniform_buffer,
                0,
                NonZero::new(size_of::<SceneUniformBuffer>() as u64).unwrap(),
                &gpu.device,
            );

            let content: &mut [SceneUniformBuffer] = bytemuck::cast_slice_mut(&mut buffer_view);
            content[0] = SceneUniformBuffer {
                mvp: self.camera.view_projection_matrix(),
            };
        }
        self.staging_belt.finish();

        {
            // render pass
            let mut render_pass = ce.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rp_aurora_scene_3d"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.pipeline.get());
            render_pass.set_bind_group(0, &self.bind_group, &[]);

            //let geometry = &self.gpu_scene_geometry;
            render_pass.draw(0..(self.scene_geometry.indices.len() as u32), 0..1);
        }

        ce.finish()
    }

    fn draw_ui(&mut self, ui: &mut egui::Ui) {
        self.camera.draw_ui(ui);
    }
}
