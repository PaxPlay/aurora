use glam::{Vec3, Vec4, Mat3, Mat4};
use glam::{vec3, vec4, mat3, mat4};

use log::{info};
use wgpu::util::DeviceExt;
use std::num::NonZero;

use std::f32::consts::*;
use crate::{register_default, AuroraWindow, GpuContext};
use std::sync::Arc;

pub trait Scene {
    fn build_pipeline(&mut self, gpu: Arc<GpuContext>, surface_format: wgpu::TextureFormat);
    fn render<'a>(&'a mut self, gpu: Arc<GpuContext>, view: &wgpu::TextureView) -> wgpu::CommandBuffer;
}

#[derive(Copy, Clone)]
struct Angle {
    pitch: f32,
    yaw: f32,
    roll: f32
}

impl From<Vec3> for Angle {
    fn from(value: Vec3) -> Self {
        Self { pitch: value.x, yaw: value.y, roll: value.z }
    }
}

impl Into<Vec3> for Angle {
    fn into(self) -> Vec3 {
        Vec3 { x: self.pitch, y: self.yaw, z: self.roll }
    }
}


impl Angle {
    fn direction(self) -> Vec3 {
        let sin_phi = self.yaw.sin();
        let cos_phi = self.yaw.cos();
        let sin_theta = self.pitch.sin();
        let cos_theta = self.pitch.cos();

        vec3(
            cos_phi * cos_theta,
            sin_phi * cos_theta,
            sin_theta
        )
    }

    fn normalize(self) -> Self {
        let mut res = self.clone();
        while res.pitch > PI {
            res.pitch -= 2.0f32 * PI;
        }
        while res.pitch < -PI {
            res.pitch += 2.0f32 * PI; }

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
        aspect_ratio: f32
    },
    Perspective {
        position: Vec3,
        angle: Angle,
        fov: f32,
        near: f32,
        far: f32,
        aspect_ratio: f32
    },
    Orthographic {
        position: Vec3,
        angle: Angle,
        zoom: f32,
        near: f32,
        far: f32,
        aspect_ratio: f32
    }
}

impl Camera3d {
    fn default() -> Self {
        Self::Centered {
            position: vec3(0.0f32, 0.0f32,0.0f32 ),
            angle: Angle::from(vec3(0.0f32, 0.0f32, 0.0f32)),
            distance: 10.0f32,
            fov: 60.0f32,
            near: 0.1f32,
            far: 50.0f32,
            aspect_ratio: 16.0f32 / 9.0f32
        }
    }
}

impl Camera3d {
    fn view_projection_matrix(&self) -> Mat4 {
        match self {
            Camera3d::Centered {
                position, angle, distance, fov, near, far, aspect_ratio
            } => {
                Mat4::perspective_lh(fov.to_radians(), *aspect_ratio, *near, *far)
                    * Mat4::look_at_lh(position - angle.direction() * distance, position.clone(), angle.up())
            },
            Camera3d::Perspective {
                position, angle, fov, near, far, aspect_ratio
            } => {
                Mat4::perspective_lh(fov.to_radians(), *aspect_ratio, *near, *far)
                    * Mat4::look_at_lh(position.clone(), position + angle.direction(), angle.up())
            }
            Camera3d::Orthographic {
                position, angle, zoom, near, far, aspect_ratio
            } => {
                let right = zoom * aspect_ratio;

                Mat4::orthographic_lh(-right, right, -zoom,  *zoom, *near, *far)
                    * Mat4::look_at_lh(position.clone(), position + angle.direction(), angle.up())
            }
        }
    }

    fn handle_keyboard_input() {

    }

    fn handle_mouse_movement() {

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
            BU::VERTEX | BU::STORAGE);

        let indices = gpu.create_buffer_init(
            "aurora_scene_indices", 
            &scene_geometry.indices, 
            BU::INDEX | BU::STORAGE);

        let model_start_indices  = gpu.create_buffer_init(
            "aurora_scene_model_start_indices", 
            &scene_geometry.model_start_indices, 
            BU::STORAGE);

        let material_indices = gpu.create_buffer_init(
            "aurora_scene_material_indices", 
            &scene_geometry.material_indices, 
            BU::STORAGE);

        let ambient = gpu.create_buffer_init(
            "aurora_scene_ambient", 
            &scene_geometry.ambient, 
            BU::STORAGE);

        let diffuse = gpu.create_buffer_init(
            "aurora_scene_diffuse", 
            &scene_geometry.diffuse, 
            BU::STORAGE);

        let specular = gpu.create_buffer_init(
            "aurora_scene_specular", 
            &scene_geometry.specular, 
            BU::STORAGE);

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
    mvp: Mat4
}

unsafe impl bytemuck::Pod for SceneUniformBuffer {}
unsafe impl bytemuck::Zeroable for SceneUniformBuffer {}

pub struct BasicScene3d {
    camera: Camera3d,
    pipeline: Option<wgpu::RenderPipeline>,
    scene_geometry: SceneGeometry,
    gpu_scene_geometry: Option<GpuSceneGeometry>,
    uniform_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
}

impl BasicScene3d {
    pub fn new(file: &str) -> Self {
        let scene_geometry = SceneGeometry::new(file);

        let camera = Camera3d::Perspective {
            position: vec3(278f32, 273f32, -800f32),
            angle: vec3(FRAC_PI_2, -FRAC_PI_2, 0f32).into(),
            fov: 39.3f32,
            near: 500f32,
            far: 2000f32,
            aspect_ratio: 1.5f32,
        };

        Self { camera, pipeline: None, scene_geometry, gpu_scene_geometry: None, uniform_buffer: None, bind_group: None }
    }
}

impl Scene for BasicScene3d {
    fn build_pipeline(&mut self, gpu: Arc<GpuContext>, surface_format: wgpu::TextureFormat) {
        let gpu_scene_geometry = GpuSceneGeometry::from(&self.scene_geometry, &gpu);
        self.gpu_scene_geometry = Some(gpu_scene_geometry);

        let ub_contents = SceneUniformBuffer {
            mvp: self.camera.view_projection_matrix(),
        };

        self.uniform_buffer = Some(gpu.create_buffer_init(
            "aurora_scene_uniform", 
            &vec![ub_contents], 
            wgpu::BufferUsages::UNIFORM));

        let device = &gpu.device;
        use crate::shader::ShaderManager;
        let mut sm = ShaderManager::new(gpu.clone());
        register_default!(sm, "basic3d", "shader/basic3d.wgsl");

        let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aurora_scene_bg_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZero::new(size_of::<SceneUniformBuffer>() as u64)
                },
                count: None,
            }]
        });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora_scene_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: self.uniform_buffer.as_ref().unwrap(),
                    offset: 0,
                    size: NonZero::new(size_of::<SceneUniformBuffer>() as u64),
                }),
            }]
        });

        self.bind_group = Some(bind_group);

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Basic 3D Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[]
        });

        let shader = sm.get_shader("basic3d").unwrap();
        let pipeline =  device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Basic 3D Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader.get_module().as_ref(),
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: size_of::<f32>() as wgpu::BufferAddress * 3,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                    },
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: shader.get_module().as_ref(),
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default()
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false
            },
            multiview: None,
            cache: None,
        });

        self.pipeline = Some(pipeline);
    }

    fn render<'a>(&'a mut self, gpu: Arc<GpuContext>, view: &wgpu::TextureView) -> wgpu::CommandBuffer {
        let mut ce = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut render_pass = ce.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rp_aurora_scene_3d"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(self.pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, &self.bind_group, &[]);

            let geometry = self.gpu_scene_geometry.as_ref().unwrap();
            render_pass.set_vertex_buffer(0, geometry.vertices.slice(..));
            render_pass.set_index_buffer(geometry.indices.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..(self.scene_geometry.indices.len() as u32), 0, 0..1);
        }
        
        ce.finish()
    }
}

