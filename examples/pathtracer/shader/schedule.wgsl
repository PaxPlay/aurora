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

@group(0) @binding(0) var<storage, read_write> schedule: InvocationSchedule;
@group(0) @binding(1) var<storage, read_write> schedule_intersect: ScheduleIntersect;
@group(0) @binding(2) var<storage, read_write> schedule_shade: ScheduleShade;

@compute
@workgroup_size(1, 1, 1)
fn schedule_invocations(
) {
    let num_rays = atomicLoad(&schedule_shade.num_rays);
    schedule.ray_intersection_groups = vec4<u32>(
        (num_rays + 127u) / 128u,
        1u,
        1u,
        num_rays,
    );
    schedule_intersect.intersect_invocations = num_rays;

    let num_intersections = atomicLoad(&schedule_intersect.num_intersections);
    schedule.handle_intersections_groups = vec4<u32>(
        (num_intersections + 255u) / 256u,
        1u,
        1u,
        num_intersections,
    );
    schedule_shade.shade_invocations = num_intersections;

    atomicStore(&schedule_shade.num_rays, 0u);
    atomicStore(&schedule_shade.num_nee_rays, 0u);
    atomicStore(&schedule_intersect.num_intersections, 0u);
    atomicStore(&schedule_intersect.num_misses, 0u);

    schedule_intersect.rng_seed_index += 1u;
    schedule_shade.rng_seed_index += 1u;
}

