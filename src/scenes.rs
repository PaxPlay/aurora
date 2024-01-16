use glam::{Vec3, Vec4, Mat3, Mat4};
use glam::{vec3, vec4, mat3, mat4};

pub trait Scene {
    fn render();
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
        while res.pitch > f32::PI() {
            res.pitch -= 2.0f32 * f32::PI();
        }
        while res.pitch < -f32::PI() {
            res.pitch += 2.0f32 * f32::PI();
        }

        if res.pitch > f32::FRAC_PI_2() {
            res.pitch = f32::PI() - res.pitch;
            res.yaw += f32::PI();
        }
        if res.pitch < -f32::FRAC_PI_2() {
            res.pitch = -f32::PI() - res.pitch;
            res.yaw += f32::PI();
        }

        while res.yaw > f32::PI() {
            res.yaw -= 2.0f32 * f32::PI();
        }
        while res.yaw < -f32::PI() {
            res.yaw += 2.0f32 * f32::PI();
        }

        res
    }

    fn up(self) -> Vec3 {
        let mut res = self;
        res.pitch += f32::FRAC_PI_2();
        res.normalize().direction()
    }
}

pub enum Camera {
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

impl Camera {
    fn view_projection_matrix(self) -> Mat4 {
        match self {
            Camera::Centered {
                position, angle, distance, fov, near, far, aspect_ratio
            } => {
                Mat4::perspective_lh(fov.to_radians(), aspect_ratio, near, far)
                    * Mat4::look_at_lh(position - angle.direction() * distance, position, angle.up())
            },
            Camera::Perspective {
                position, angle, fov, near, far, aspect_ratio
            } => {
                Mat4::perspective_lh(fov.to_radians(), aspect_ratio, near, far)
                    * Mat4::look_at_lh(position, position + angle.direction(), angle.up())
            }
            Camera::Orthographic {
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
