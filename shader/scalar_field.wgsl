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
    origin: vec3<f32>,
    bb_size: vec3<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraBuffer;
@group(0) @binding(1) var<uniform> field_parameters: ScalarFieldParameters;
@group(0) @binding(2) var field_texture: texture_3d<f32>;
@group(0) @binding(3) var field_sampler: sampler;

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = (field_parameters.model_matrix * vec4(in.position, 1.0));
    out.position = camera.vp * world_pos;
    out.world_pos = in.position;
    return out;
}

@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    return vec4(in.world_pos, 1.0);
}
