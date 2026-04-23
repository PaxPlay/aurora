struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) color: vec3<f32>,
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

@group(0) @binding(0) var<uniform> camera: CameraBuffer;

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = camera.vp * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    let instance = f32(in.vertex_index / 3) / 12;
    out.color = vec3(instance, 1.0 - instance, 1.0);
    return out;
}

@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    return vec4(in.world_pos, 1.0);
//    return vec4(in.color, 1.0);
}
