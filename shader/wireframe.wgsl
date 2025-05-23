struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) vert_pos: vec3<f32>,
    @location(1) bary_coord: vec3<f32>,
}

struct UniformBuffer {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> ub: UniformBuffer;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) index: u32,
) -> VertexOutput {
    var i = 3 * indices[index];
    var pos = vec3(vertices[i], vertices[i + 1], vertices[i + 2]);

    var bary_coord: vec3<f32> = vec3(0);
    bary_coord[index % 3] = 1;

    var out: VertexOutput;
    out.clip_position = ub.mvp * vec4<f32>(pos, 1.0);
    out.vert_pos = pos;
    out.bary_coord = bary_coord;
    return out;
}

fn edge_factor(bary: vec3<f32>) -> f32 {
  let d = fwidth(bary);
  let a3 = smoothstep(vec3f(0.0), d * 1, bary);
  return 1 - min(min(a3.x, a3.y), a3.z);
}

@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    let l = edge_factor(in.bary_coord);

    return vec4<f32>(l, l, l, l);
}

