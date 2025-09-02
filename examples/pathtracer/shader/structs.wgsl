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
    t_min: f32,
    t_max: f32,
    ray_type: u32, // 0: primary, 1: regular, 2: NEE
}

struct RayIntersectionData {
    pos: vec3<f32>,
    n: vec3<f32>,
    w_i: vec3<f32>,
    weight: vec3<f32>,
    t: f32,
    surface_id: u32,
    primary_ray: u32,
    event_type: u32, // 0: miss, 1: nee_hit, 2: nee_miss, 3: primary_hit,
                     // 4..7: reserved, 8..: intersection with bsdf id
}

struct InvocationSchedule {
    ray_intersection_groups: vec4<u32>,
    handle_intersections_groups: vec4<u32>,
    reorder_intersections_groups: vec4<u32>,
    nee_miss_groups: vec4<u32>,
}

struct ScheduleShade {
    num_rays: atomic<u32>,
    num_nee_rays: atomic<u32>,
    shade_invocations: u32,
    rng_seed_index: u32,
    isec_start: u32,
}

struct ScheduleIntersect {
    num_events: array<atomic<u32>, 16>,
    intersect_invocations: u32,
    rng_seed_index: u32,
}

struct ScheduleReorder {
    num_events: array<u32, 16>,
    index_in_event: array<atomic<u32>, 16>, // only used in reorder to determine indices, should equal num_events after
    event_type_start: array<u32, 16>, // determined by schedule, used by reorder and shade
    intersect_invocations: u32,
}

struct ScheduleNEE {
    nee_invocations: u32,
    nee_start: u32,
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

struct Settings {
    max_iterations: u32,
    selected_buffer: u32,
    accumulate: u32,
    nee: u32,
    rr_alpha: f32,
}

const F32_MAX: f32 = 3.4028e38;
const EPSILON: f32 = 1e-4;
const PI: f32 = 3.14159265359;
