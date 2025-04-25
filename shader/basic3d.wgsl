struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) vert_pos: vec3<f32>,
}

struct UniformBuffer {
    mvp: mat4x4<f32>,
};

struct Vertex {
    @location(0) pos: vec3<f32>,
};

@group(0) @binding(0) var<uniform> ub: UniformBuffer;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    v: Vertex,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = ub.mvp * vec4<f32>(v.pos, 1.0);
    out.vert_pos = v.pos;
    return out;
}

@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    var d = in.vert_pos / 600;
    return vec4<f32>(d, 1.0);
}

