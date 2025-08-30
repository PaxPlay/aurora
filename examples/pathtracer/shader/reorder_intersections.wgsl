#import "structs.wgsl"

@group(0) @binding(0) var<storage, read_write> schedule: ScheduleReorder;
@group(0) @binding(1) var<storage, read_write> isec_from: array<RayIntersectionData>;
@group(0) @binding(2) var<storage, read_write> isec_to: array<RayIntersectionData>;

var<workgroup> wg_num_events: array<atomic<u32>, 16>;
var<workgroup> event_offsets: array<u32, 16>;

@compute
@workgroup_size(256, 1, 1)
fn reorder_intersections(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32
) {
    if lidx < 16u {
        atomicStore(&wg_num_events[lidx], 0u);
        event_offsets[lidx] += schedule.event_type_start[lidx];
    }

    var index: u32 = 0u;
    var isec: RayIntersectionData;

    if gid.x < schedule.intersect_invocations {
        isec = isec_from[gid.x];
        index = atomicAdd(&wg_num_events[isec.event_type], 1u);
    }

    workgroupBarrier();

    if lidx < 16u {
        event_offsets[lidx] += atomicAdd(&schedule.index_in_event[lidx], atomicLoad(&wg_num_events[lidx]));
    }

    workgroupBarrier();

    if gid.x < schedule.intersect_invocations {
        index += event_offsets[isec.event_type];
        isec_to[index] = isec;
    }
}
