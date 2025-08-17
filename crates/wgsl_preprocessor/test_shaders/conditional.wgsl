#include "common.wgsl"

// Example of conditional compilation in WGSL
#define PLATFORM_WEB
#define DEBUG

#ifdef PLATFORM_WEB
#define MAX_TEXTURE_SIZE 2048u
#else
#define MAX_TEXTURE_SIZE 8192u
#endif

#ifdef DEBUG
const VALIDATION_ENABLED: bool = true;
const LOG_LEVEL: u32 = 0u;  // Verbose logging
#else
const VALIDATION_ENABLED: bool = false;
const LOG_LEVEL: u32 = 2u;  // Error logging only
#endif

#ifndef RELEASE
const PROFILING_ENABLED: bool = true;
#endif

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    #ifdef DEBUG
    // Debug-specific code
    if (VALIDATION_ENABLED) {
        if (in.tex_coords.x < 0.0 || in.tex_coords.x > 1.0) {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0); // Red for invalid coords
        }
    }
    #endif
    
    let color = mix(COLOR_RED, COLOR_BLUE, in.tex_coords.x);
    return vec4<f32>(color, 1.0);
}