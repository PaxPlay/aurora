#include "common.wgsl"

#define COLOR_RED vec3<f32>(1.0, 0.0, 0.0)
#define COLOR_GREEN vec3<f32>(0.0, 1.0, 0.0)
#define COLOR_BLUE vec3<f32>(0.0, 0.0, 1.0)

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let angle = atan2(in.tex_coords.y - 0.5, in.tex_coords.x - 0.5);
    let normalized_angle = (angle + PI) / TWO_PI;
    
    var color: vec3<f32>;
    if (normalized_angle < 0.333) {
        color = COLOR_RED;
    } else if (normalized_angle < 0.666) {
        color = COLOR_GREEN;
    } else {
        color = COLOR_BLUE;
    }
    
    return vec4<f32>(color, 1.0);
}