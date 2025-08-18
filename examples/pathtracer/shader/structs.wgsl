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

struct InvocationSchedule {
    ray_intersection_groups: vec4<u32>,
    handle_intersections_groups: vec4<u32>,
}

struct ScheduleShade {
    num_rays: atomic<u32>,
    num_nee_rays: atomic<u32>,
    shade_invocations: u32,
    rng_seed_index: u32,
}

struct ScheduleIntersect {
    num_intersections: atomic<u32>,
    num_misses: atomic<u32>,
    intersect_invocations: u32,
    rng_seed_index: u32,
}

struct SceneGeometrySizes {
    vertices: u32,
    indices: u32,
    model_start_indices: u32,
    material_indices: u32,
    ambient: u32,
    diffuse: u32,
    specular: u32,
};

const F32_MAX: f32 = 3.4028e38;
const EPSILON: f32 = 1e-10;
const PI: f32 = 3.14159265359;
