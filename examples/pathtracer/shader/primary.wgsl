#import "structs.wgsl"
#import "random.wgsl"

@group(0) @binding(0) var<uniform> camera : CameraBuffer;

@group(1) @binding(0) var<storage, read_write> primary_rays : array<PrimaryRayData>;
@group(1) @binding(1) var<storage, read_write> rays : array<Ray>;
@group(1) @binding(2) var<storage, read_write> ray_intersections: array<RayIntersectionData>;
@group(1) @binding(3) var<storage, read> rng_seeds: array<u32>;

@group(2) @binding(0) var<storage, read_write> schedule: InvocationSchedule;
@group(2) @binding(1) var<storage, read_write> schedule_intersect: ScheduleIntersect;
@group(2) @binding(2) var<storage, read_write> schedule_shade: ScheduleShade;

fn world_pos_from_camera_space(camera_space: vec3<f32>) -> vec3<f32> {
    let world_pos = camera.vp_inv * vec4(camera_space, 1.0);
    return world_pos.xyz / world_pos.w;
}

@compute
@workgroup_size(16, 16, 1)
fn generate_rays(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    var screen_pos = (vec2<f32>(gid.xy) + vec2<f32>(0.5, 0.5)) / vec2<f32>(camera.resolution);
    screen_pos = vec2(1.0) - screen_pos;
    let world_pos = world_pos_from_camera_space(vec3(screen_pos * 2.0 - vec2(1.0), 0.0));
    let direction = normalize(world_pos - camera.origin);
    var ray_data: PrimaryRayData;
    ray_data.origin = camera.origin;
    ray_data.direction = direction;
    ray_data.result_color = vec4(0.0);

    if gid.x < camera.resolution.x && gid.y < camera.resolution.y {
        let idx = camera.resolution.x * gid.y + gid.x;

        primary_rays[idx].origin = ray_data.origin;
        primary_rays[idx].direction = ray_data.direction;
        primary_rays[idx].result_color = ray_data.result_color;

        var ray: Ray;
        ray.origin = ray_data.origin;
        ray.direction = ray_data.direction;
        ray.primary_ray = idx;
        ray.weight = vec3<f32>(1.0);
        ray.t_max = F32_MAX;
        ray.ray_type = 0u; // primary
        rays[idx] = ray;
    }

    if gid.x == 0u && gid.y == 0u {
        let num_rays = camera.resolution.x * camera.resolution.y;
        atomicStore(&schedule_shade.num_rays, 0u);
        atomicStore(&schedule_shade.num_nee_rays, 0u);

        schedule.ray_intersection_groups = vec4<u32>(
            (num_rays + 127u) / 128u,
            1u,
            1u,
            num_rays,
        );
        schedule_intersect.intersect_invocations = num_rays;

        schedule_shade.rng_seed_index = 0u;
        schedule_intersect.rng_seed_index = 0u;
    }

    if gid.x < 16u && gid.y == 0u {
        atomicStore(&schedule_intersect.num_events[gid.x], 0u);
    }
}
