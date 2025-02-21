use glam::{Vec3, Vec4, Mat3, Mat4};
use glam::{vec3, vec4, mat3, mat4};

use std::f32::consts::*;
use wgpu::{PipelineCompilationOptions, PipelineLayout, RenderPass};
use crate::{register_default, AuroraWindow, GpuContext};
use std::sync::Arc;

pub trait Scene {
    fn render<'a>(&'a mut self, render_pass: &mut RenderPass<'a>);
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
    fn view_projection_matrix(self) -> Mat4 {
        match self {
            Camera3d::Centered {
                position, angle, distance, fov, near, far, aspect_ratio
            } => {
                Mat4::perspective_lh(fov.to_radians(), aspect_ratio, near, far)
                    * Mat4::look_at_lh(position - angle.direction() * distance, position, angle.up())
            },
            Camera3d::Perspective {
                position, angle, fov, near, far, aspect_ratio
            } => {
                Mat4::perspective_lh(fov.to_radians(), aspect_ratio, near, far)
                    * Mat4::look_at_lh(position, position + angle.direction(), angle.up())
            }
            Camera3d::Orthographic {
                position, angle, zoom, near, far, aspect_ratio
            } => {
                let right = zoom * aspect_ratio;

                Mat4::orthographic_lh(-right, right, -zoom,  zoom, near, far)
                    * Mat4::look_at_lh(position, position + angle.direction(), angle.up())
            }
        }
    }

    fn handle_keyboard_input() {

    }

    fn handle_mouse_movement() {

    }
}

pub struct BasicScene3d {
    camera: Camera3d,
    pipeline: wgpu::RenderPipeline
}

impl BasicScene3d {
    pub fn new(gpu: Arc<GpuContext>, window: &AuroraWindow) -> Self {
        let device = &gpu.device;
        use crate::shader::ShaderManager;
        let mut sm = ShaderManager::new(gpu.clone());
        register_default!(sm, "basic3d", "shader/basic3d.wgsl");

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Basic 3D Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[]
        });


        let shader = sm.get_shader("basic3d").unwrap();
        let pipeline =  device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Basic 3D Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader.get_module().as_ref(),
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: shader.get_module().as_ref(),
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: window.surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default()
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
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

        BasicScene3d {
            camera: Camera3d::default(),
            pipeline
        }
    }
}

impl Scene for BasicScene3d {
    fn render<'a>(&'a mut self, render_pass: &mut RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.draw(0..3, 0..1);
    }
}
