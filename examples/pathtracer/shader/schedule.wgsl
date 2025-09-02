#import "structs.wgsl"

@group(0) @binding(0) var<storage, read_write> schedule: InvocationSchedule;
@group(0) @binding(1) var<storage, read_write> schedule_intersect: ScheduleIntersect;
@group(0) @binding(2) var<storage, read_write> schedule_shade: ScheduleShade;
@group(0) @binding(3) var<storage, read_write> schedule_reorder: ScheduleReorder;
@group(0) @binding(4) var<storage, read_write> schedule_nee: ScheduleNEE;

var<workgroup> num_events: array<u32, 16>;
var<workgroup> event_offsets: array<u32, 16>;

@compute
@workgroup_size(16, 1, 1)
fn schedule_invocations(
    @builtin(local_invocation_index) lidx: u32
) {
    if lidx < 16u {
        num_events[lidx] = atomicLoad(&schedule_intersect.num_events[lidx]);
    }

    if lidx == 0u {
        schedule.reorder_intersections_groups = schedule.ray_intersection_groups;
        schedule_reorder.intersect_invocations = schedule_intersect.intersect_invocations;

        let num_rays = atomicLoad(&schedule_shade.num_rays);
        schedule.ray_intersection_groups = vec4<u32>(
            (num_rays + 127u) / 128u,
            1u,
            1u,
            num_rays,
        );
        schedule_intersect.intersect_invocations = num_rays;

        let num_intersections = num_events[8] + num_events[3];
        schedule.handle_intersections_groups = vec4<u32>(
            (num_intersections + 127u) / 128u,
            1u,
            1u,
            num_intersections,
        );
        schedule_shade.shade_invocations = num_intersections;

        let num_nee_miss = num_events[2];
        schedule.nee_miss_groups = vec4<u32>(
            (num_nee_miss + 255u) / 256u,
            1u,
            1u,
            num_nee_miss,
        );
        schedule_nee.nee_invocations = num_nee_miss;

        atomicStore(&schedule_shade.num_rays, 0u);
        atomicStore(&schedule_shade.num_nee_rays, 0u);

        schedule_intersect.rng_seed_index += 1u;
        schedule_shade.rng_seed_index += 1u;
    }

    // We need to do a prefix sum over the event counts to determine the starting
    // index of each event type in the reordered intersection array.
    if lidx < 16u {
        atomicStore(&schedule_intersect.num_events[lidx], 0u);
        schedule_reorder.num_events[lidx] = num_events[lidx];
    }

    if lidx == 0u {
        // primary hits should already be handled, exclude from reordering 
        num_events[8] += num_events[3]; // TODO: bsdf index
        num_events[3] = 0u;
    }

    if lidx < 15u {
        let value = num_events[lidx];
        
        // very bad cumsum
        for (var i = lidx; i < 15u; i = i + 1u) {
            num_events[i + 1u] += value;
        }
    }

    if lidx == 0u {
        event_offsets[0] = 0u;
    } else {
        event_offsets[lidx] = num_events[lidx - 1u];
    }


    schedule_reorder.event_type_start[lidx] = event_offsets[lidx];
    atomicStore(&schedule_reorder.index_in_event[lidx], 0u);

    schedule_shade.isec_start = event_offsets[8]; // shade invocations need to start loading intersections from here
    schedule_nee.nee_start = event_offsets[2];
}

