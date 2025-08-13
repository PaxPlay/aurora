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

@group(3) @binding(0) var<storage, read_write> schedule: InvocationSchedule;
@group(3) @binding(1) var<storage, read_write> schedule_intersect: ScheduleIntersect;
@group(3) @binding(2) var<storage, read_write> schedule_shade: ScheduleShade;

@compute
@workgroup_size(16, 16, 1)
fn copy_target(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    if gid.x < camera.resolution.x && gid.y < camera.resolution.y {

        let index = camera.resolution.x * gid.y + gid.x;
        var result_color = max(primary_rays[index].result_color, vec4(0.0f));
        var accumulated_color = max(primary_rays[index].accumulated_color, vec4(0.0f));
        accumulated_color = accumulated_color + result_color;
        primary_rays[index].accumulated_color = accumulated_color;

        accumulated_color /= accumulated_color.a;

    // let num_rays = schedule_shade.shade_invocations;
    // let gidx = gid.x + gid.y * (16 * num_workgroups.x);
    // if gidx < num_rays {
        // let ray = rays[gidx];
        // let index = ray.primary_ray;
        // let accumulated_color = vec4(ray.weight, 1.0);
        // let isec = ray_intersections[gidx];
        // let index = isec.primary_ray;
        // let accumulated_color = vec4(abs(isec.n), 1.0);

        output_buffer_f32[4u * index     ] = accumulated_color.r;
        output_buffer_f32[4u * index + 1u] = accumulated_color.g;
        output_buffer_f32[4u * index + 2u] = accumulated_color.b;
        output_buffer_f32[4u * index + 3u] = accumulated_color.a;

        output_buffer_f16[2u * index     ] = pack2x16float(accumulated_color.rg);
        output_buffer_f16[2u * index + 1u] = pack2x16float(accumulated_color.ba);
    }
}
