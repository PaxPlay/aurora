#include "common.wgsl"

#define VERTEX_COUNT 4

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Create a quad using vertex index
    let x = f32((vertex_index & 1u) * 2u) - 1.0;
    let y = f32(((vertex_index >> 1u) & 1u) * 2u) - 1.0;
    
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.tex_coords = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    
    return out;
}