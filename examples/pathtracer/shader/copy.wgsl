struct CameraBuffer {
    vp: mat4x4<f32>,
    vp_inv: mat4x4<f32>,
    origin: vec3<f32>,
    direction: vec3<f32>,
    up: vec3<f32>,
    resolution: vec2<u32>,
    fov: f32,
}

struct PrimaryRayData {
    origin: vec3<f32>,
    direction: vec3<f32>,
    result_color: vec4<f32>,
    accumulated_color: vec4<f32>,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    weight: vec3<f32>,
    primary_ray: u32,
}

struct RayIntersectionData {
    pos: vec3<f32>,
    n: vec3<f32>,
    w_i: vec3<f32>,
    weight: vec3<f32>,
    t: f32,
    surface_id: u32,
    primary_ray: u32,
}

@group(0) @binding(0) var<uniform> camera : CameraBuffer;

@group(1) @binding(0) var<storage, read_write> primary_rays : array<PrimaryRayData>;
@group(1) @binding(1) var<storage, read_write> rays : array<Ray>;
@group(1) @binding(2) var<storage, read_write> ray_intersections: array<RayIntersectionData>;
@group(1) @binding(3) var<storage, read> rng_seeds: array<u32>;

@group(2) @binding(0) var<storage, read_write> output_buffer_f32: array<f32>; 
@group(2) @binding(1) var<storage, read_write> output_buffer_f16: array<u32>; // pack with pack2x16float

@compute
@workgroup_size(16, 16, 1)
fn copy_target(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if gid.x >= camera.resolution.x || gid.y >= camera.resolution.y {
        return;
    }

    let index = camera.resolution.x * gid.y + gid.x;
    var color = primary_rays[index].result_color;
    color += primary_rays[index].accumulated_color;
    primary_rays[index].accumulated_color = color;

    color /= color.a;

    output_buffer_f32[4 * index    ] = color.r;
    output_buffer_f32[4 * index + 1] = color.g;
    output_buffer_f32[4 * index + 2] = color.b;
    output_buffer_f32[4 * index + 3] = color.a;

    output_buffer_f16[2 * index    ] = pack2x16float(color.rg);
    output_buffer_f16[2 * index + 1] = pack2x16float(color.ba);
}

