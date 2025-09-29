#import "structs.wgsl"

@group(0) @binding(0) var<storage, read_write> primary_rays : array<PrimaryRayData>;
@group(0) @binding(1) var<storage, read_write> rays : array<Ray>;
@group(0) @binding(2) var<storage, read_write> ray_intersections: array<RayIntersectionData>;
@group(0) @binding(3) var<storage, read> rng_seeds: array<u32>;

@group(1) @binding(0) var<storage, read_write> schedule: ScheduleNEE;

@compute
@workgroup_size(256, 1, 1)
fn nee_miss(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    if gid.x < schedule.nee_invocations {
        let primary_ray = ray_intersections[gid.x + schedule.nee_start].primary_ray;
        let weight = ray_intersections[gid.x + schedule.nee_start].weight;
        var color = primary_rays[primary_ray].result_color;
        color.r += weight.r;
        color.g += weight.g;
        color.b += weight.b;
        primary_rays[primary_ray].result_color = color;
    }
}
