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
        let isec = ray_intersections[gid.x + schedule.nee_start];
        let primary_ray = primary_rays[isec.primary_ray];
        primary_rays[isec.primary_ray].result_color.r += isec.weight.r;
        primary_rays[isec.primary_ray].result_color.g += isec.weight.g;
        primary_rays[isec.primary_ray].result_color.b += isec.weight.b;
    }
}
