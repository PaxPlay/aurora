struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
}

struct CameraBuffer {
    vp: mat4x4<f32>,
    vp_inv: mat4x4<f32>,
    origin: vec3<f32>,
    direction: vec3<f32>,
    up: vec3<f32>,
    resolution: vec2<u32>,
    fov: f32,
}

struct ScalarFieldParameters {
    model_matrix: mat4x4<f32>,
    world_to_field: mat4x4<f32>,
    field_size: vec3<u32>,
    origin: vec3<f32>,
    bb_size: vec3<f32>,
}

struct RenderParameters {
    isosurface_value: f32,
    isosurface_stddev: f32,
    step_size: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraBuffer;
@group(0) @binding(1) var<uniform> field_parameters: ScalarFieldParameters;
@group(0) @binding(2) var<uniform> render_parameters: RenderParameters;
@group(0) @binding(3) var field_texture: texture_3d<f32>;
@group(0) @binding(4) var field_sampler: sampler;

@group(1) @binding(0) var cmap_texture: texture_1d<f32>;
@group(1) @binding(1) var cmap_sampler: sampler;

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = (field_parameters.model_matrix * vec4(in.position, 1.0));
    out.position = camera.vp * world_pos;
    out.world_pos = world_pos.xyz;
    return out;
}

const EPSILON: f32 = 0.0001;

fn field_pos(world_pos: vec3<f32>) -> vec3<f32> {
    return (field_parameters.world_to_field * vec4(world_pos, 1.0)).xyz;
}

fn is_in_volume(field_pos: vec3<f32>) -> bool {
    return all(field_pos >= vec3(0.0))
        && all(field_pos <= vec3(1.0));
}

fn tf(value: f32) -> f32 {
    return exp(-pow(value - render_parameters.isosurface_value, 2.0) * pow(render_parameters.isosurface_stddev, 2.0));
}

@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    let ray_direction = normalize(in.world_pos - camera.origin);
    var position = in.world_pos + EPSILON * ray_direction; // World position
    var field_position = field_pos(position); // Coordinates within [0, 1]^3 cube, used for texture accesses

    var color = vec3<f32>(0.0);
    var transmittance = 1.0;

    let step_size = render_parameters.step_size;

    while is_in_volume(field_position) && transmittance > EPSILON {
        let value = textureSample(field_texture, field_sampler, field_position).r;
        position += step_size * ray_direction;
        field_position = field_pos(position);

        let local_transmittance = exp(-tf(value) * step_size);
        let emission = textureSample(cmap_texture, cmap_sampler, value).rgb;

        let alpha = 1.0 - local_transmittance;
        color += transmittance * alpha * emission;
        transmittance *= local_transmittance;
    }

    return vec4(color, 1.0 - transmittance);
}
