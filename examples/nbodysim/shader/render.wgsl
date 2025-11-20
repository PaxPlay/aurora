@group(0) @binding(0) var<storage, read> positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> velocities: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> forces: array<vec3<f32>>;

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) relative_pos: vec2<f32>,
    @location(1) velocity: f32,
    @location(2) force: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let particle_index = vertex_index / 6u;
    let pos = positions[particle_index];

    const OFFSETS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );

    var output: VertexOutput;
    output.position = vec4<f32>((pos.xy * 2.0) - vec2(1.0) + OFFSETS[vertex_index % 6] * 0.003, 0.0, 1.0);
    output.relative_pos = OFFSETS[vertex_index % 6];
    output.velocity = length(velocities[particle_index]);
    output.force = length(forces[particle_index]);

    return output;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let alpha = 1.0 - smoothstep(0.8, 0.9, length(in.relative_pos));

    if (alpha < 0.01) {
        discard;
    }
    return vec4<f32>(vec3(1.0, 1.0, in.force / 10.0) * alpha, alpha);
}
