use glam::{mat3, mat4, vec3, vec4};
use glam::{Mat3, Mat4, Vec3, Vec4};

use log::info;

use crate::buffers::{Buffer, BufferCopyContext, BufferCopyUtil};
use crate::shader::{BindGroupLayoutBuilder, RenderPipeline};
use crate::{register_default, render_pipeline, DebugUi, GpuContext, RenderTarget};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::f32::consts::*;
use std::num::NonZero;
use std::sync::Arc;

pub trait Scene {
    fn render<'a>(
        &'a mut self,
        gpu: Arc<GpuContext>,
        target: Arc<RenderTarget>,
    ) -> Vec<wgpu::CommandBuffer>;

    fn draw_ui(&mut self, _ui: &mut egui::Ui) {}

    fn update_target_parameters<'a>(
        &'a mut self,
        _gpu: Arc<GpuContext>,
        _target: Arc<RenderTarget>,
    ) {
    }
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

impl Default for Camera3d {
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

pub struct CameraWithBuffer {
    pub camera: Camera3d,
    buffer: Buffer<CameraBuffer>,
    is_buffer_current: bool,
}

impl CameraWithBuffer {
    pub fn new(camera: Camera3d, gpu: Arc<GpuContext>) -> Self {
        let ub_contents = CameraBuffer::from(&camera);

        let buffer = gpu.create_buffer_init(
            "aurora_scene_uniform",
            &[ub_contents],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        Self {
            camera,
            buffer,
            is_buffer_current: true,
        }
    }

    fn update_buffer(&mut self, ctx: &mut BufferCopyContext) {
        self.buffer.write(ctx, &[CameraBuffer::from(&self.camera)]);
    }

    pub fn update_resolution(&mut self, resolution: [u32; 2]) {
        let ratio = resolution[1] as f32 / resolution[0] as f32;
        let ar = match &mut self.camera {
            Camera3d::Centered { aspect_ratio, .. } => aspect_ratio,
            Camera3d::Perspective { aspect_ratio, .. } => aspect_ratio,
            Camera3d::Orthographic { aspect_ratio, .. } => aspect_ratio,
        };
        *ar = ratio;
        self.is_buffer_current = false;
    }
}

impl DebugUi for CameraWithBuffer {
    fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let camera = &mut self.camera;
        let mut changed = false;

        let current_type = match camera {
            &mut Camera3d::Centered { .. } => 0u32,
            &mut Camera3d::Perspective { .. } => 1u32,
            &mut Camera3d::Orthographic { .. } => 2u32,
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
                    camera.to_centered();
                }
                1 => {
                    camera.to_perspective();
                }
                2 => {
                    camera.to_orthographic();
                }
                _ => (),
            }
            changed = true;
        }

        match camera {
            Camera3d::Centered {
                position,
                angle,
                distance,
                fov,
                near,
                far,
                ..
            } => {
                changed |= position.draw_ui(ui);
                changed |= angle.draw_ui(ui);
                ui.horizontal(|ui| {
                    ui.label("distance");
                    changed |= ui.add(egui::DragValue::new(distance)).changed();
                });
                ui.horizontal(|ui| {
                    ui.label("near");
                    changed |= ui.add(egui::DragValue::new(near)).changed();
                    ui.label("far");
                    changed |= ui.add(egui::DragValue::new(far)).changed();
                    ui.label("fov");
                    changed |= ui.add(egui::DragValue::new(fov)).changed();
                });
            }
            Camera3d::Perspective {
                position,
                angle,
                fov,
                near,
                far,
                ..
            } => {
                changed |= position.draw_ui(ui);
                changed |= angle.draw_ui(ui);
                ui.horizontal(|ui| {
                    ui.label("near");
                    changed |= ui.add(egui::DragValue::new(near)).changed();
                    ui.label("far");
                    changed |= ui.add(egui::DragValue::new(far)).changed();
                    ui.label("fov");
                    changed |= ui.add(egui::DragValue::new(fov)).changed();
                });
            }
            Camera3d::Orthographic {
                position,
                angle,
                zoom,
                near,
                far,
                ..
            } => {
                changed |= position.draw_ui(ui);
                changed |= angle.draw_ui(ui);
                ui.horizontal(|ui| {
                    ui.label("near");
                    changed |= ui.add(egui::DragValue::new(near)).changed();
                    ui.label("far");
                    changed |= ui.add(egui::DragValue::new(far)).changed();
                    ui.label("zoom");
                    changed |= ui.add(egui::DragValue::new(zoom)).changed();
                });
            }
        }

        self.is_buffer_current &= !changed;

        changed
    }
}

impl DebugUi for Vec3 {
    fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("x");
            changed |= ui
                .add(egui::DragValue::new(&mut self.x).speed(1.0))
                .changed();
            ui.label("y");
            changed |= ui
                .add(egui::DragValue::new(&mut self.y).speed(1.0))
                .changed();
            ui.label("z");
            changed |= ui
                .add(egui::DragValue::new(&mut self.z).speed(1.0))
                .changed();
        });
        changed
    }
}

impl DebugUi for Angle {
    fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        let mut pitch = self.pitch.to_degrees();
        let mut yaw = self.yaw.to_degrees();
        let mut roll = self.roll.to_degrees();

        ui.horizontal(|ui| {
            ui.label("pitch");
            changed |= ui
                .add(egui::DragValue::new(&mut pitch).range(-90..=90).speed(1.0))
                .changed();
            ui.label("yaw");
            changed |= ui
                .add(egui::DragValue::new(&mut yaw).range(-180..=180).speed(1.0))
                .changed();
            ui.label("roll");
            changed |= ui
                .add(egui::DragValue::new(&mut roll).range(-180..=180).speed(1.0))
                .changed();
        });

        self.pitch = pitch.to_radians();
        self.yaw = yaw.to_radians();
        self.roll = roll.to_radians();
        changed
    }
}

pub struct SceneGeometry {
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
    pub model_start_indices: Vec<u32>,
    pub material_indices: Vec<u32>,
    pub ambient: Vec<f32>,
    pub diffuse: Vec<f32>,
    pub specular: Vec<f32>,
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

pub struct GpuSceneGeometry {
    pub vertices: Buffer<f32>,
    pub indices: Buffer<u32>,
    pub model_start_indices: Buffer<u32>,
    pub material_indices: Buffer<u32>,
    pub ambient: Buffer<f32>,
    pub diffuse: Buffer<f32>,
    pub specular: Buffer<f32>,
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
struct CameraBuffer {
    mvp: Mat4,
    origin: Vec4,
    direction: Vec4,
}

impl From<&Camera3d> for CameraBuffer {
    fn from(value: &Camera3d) -> Self {
        let (origin, direction) = match value {
            Camera3d::Centered {
                position,
                angle,
                distance,
                ..
            } => (position - distance * angle.direction(), angle.direction()),
            Camera3d::Perspective {
                position, angle, ..
            } => (*position, angle.direction()),
            Camera3d::Orthographic {
                position, angle, ..
            } => (*position, angle.direction()),
        };

        Self {
            mvp: value.view_projection_matrix(),
            origin: origin.extend(0.0f32),
            direction: direction.extend(0.0f32),
        }
    }
}

unsafe impl bytemuck::Pod for CameraBuffer {}
unsafe impl bytemuck::Zeroable for CameraBuffer {}

pub struct BasicScene3d {
    pub camera: CameraWithBuffer,
    pub scene_geometry: SceneGeometry,
    pub gpu_scene_geometry: GpuSceneGeometry,
    pub buffer_copy_util: BufferCopyUtil,
    views: BTreeMap<String, RefCell<Box<dyn Scene3dView>>>,
    current_view: String,
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
            aspect_ratio: target.size[1] as f32 / target.size[0] as f32,
        };
        let camera = CameraWithBuffer::new(camera, gpu.clone());

        let gpu_scene_geometry = GpuSceneGeometry::from(&scene_geometry, &gpu);

        register_default!(gpu.shaders, "wireframe", "shader/wireframe.wgsl");

        let buffer_copy_util = BufferCopyUtil::new(2048);

        let wireframe: Box<dyn Scene3dView> = Box::new(WireframeView::new(
            gpu,
            target,
            &camera,
            &gpu_scene_geometry,
        ));
        let views = BTreeMap::from([("wireframe".to_string(), RefCell::new(wireframe))]);
        let current_view = views.first_key_value().unwrap().0.clone();

        Self {
            camera,
            scene_geometry,
            gpu_scene_geometry,
            buffer_copy_util,
            views,
            current_view,
        }
    }

    pub fn add_view(&mut self, name: &str, view: Box<dyn Scene3dView>) {
        self.views.insert(name.to_string(), RefCell::new(view));
    }
}

impl Scene for BasicScene3d {
    fn render<'a>(
        &'a mut self,
        gpu: Arc<GpuContext>,
        target: Arc<RenderTarget>,
    ) -> Vec<wgpu::CommandBuffer> {
        let copy_cb = self.buffer_copy_util.create_copy_command(&gpu, |ctx| {
            self.camera.update_buffer(ctx);
        });

        let mut view = self
            .views
            .get(&self.current_view)
            .expect("Selected view does not exist.")
            .borrow_mut();
        let view_cb = view.render(gpu, target, self);

        vec![copy_cb, view_cb]
    }

    fn draw_ui(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Camera", |ui| {
            self.camera.draw_ui(ui);
        });

        egui::ComboBox::from_label("View")
            .selected_text(self.current_view.as_str())
            .show_ui(ui, |ui| {
                for (name, _) in self.views.iter() {
                    ui.selectable_value(&mut self.current_view, name.clone(), name.as_str());
                }
            });
    }
}

pub trait Scene3dView {
    fn render<'a>(
        &'a mut self,
        gpu: Arc<GpuContext>,
        target: Arc<RenderTarget>,
        scene: &BasicScene3d,
    ) -> wgpu::CommandBuffer;
}

struct WireframeView {
    pipeline: RenderPipeline,
    bind_group: wgpu::BindGroup,
}

impl WireframeView {
    fn new(
        gpu: Arc<GpuContext>,
        target: Arc<RenderTarget>,
        camera: &CameraWithBuffer,
        gpu_scene_geometry: &GpuSceneGeometry,
    ) -> Self {
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
            .buffer(0, &camera.buffer)
            .buffer(1, &gpu_scene_geometry.vertices)
            .buffer(2, &gpu_scene_geometry.indices)
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

        Self {
            pipeline,
            bind_group,
        }
    }
}

impl Scene3dView for WireframeView {
    fn render<'a>(
        &'a mut self,
        gpu: Arc<GpuContext>,
        target: Arc<RenderTarget>,
        scene: &BasicScene3d,
    ) -> wgpu::CommandBuffer {
        let mut ce = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

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
            render_pass.draw(0..(scene.scene_geometry.indices.len() as u32), 0..1);
        }

        ce.finish()
    }
}
