#import "structs.wgsl"

@group(0) @binding(0) var<storage, read_write> schedule: InvocationSchedule;
@group(0) @binding(1) var<storage, read_write> schedule_intersect: ScheduleIntersect;
@group(0) @binding(2) var<storage, read_write> schedule_shade: ScheduleShade;
@group(0) @binding(3) var<storage, read_write> schedule_reorder: ScheduleReorder;

@compute
@workgroup_size(16, 1, 1)
fn schedule_intersect(
    @builtin(local_invocation_index) lidx: u32
) {
}
