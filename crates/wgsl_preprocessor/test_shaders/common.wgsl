// Common shader definitions
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

// Common constants
#define PI 3.14159265359
#define TWO_PI 6.28318530718