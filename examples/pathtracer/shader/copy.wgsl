#import "structs.wgsl"

@group(0) @binding(0) var<uniform> camera : CameraBuffer;
@group(0) @binding(1) var<uniform> settings: Settings;

@group(1) @binding(0) var<storage, read_write> primary_rays : array<PrimaryRayData>;
@group(1) @binding(1) var<storage, read_write> rays : array<Ray>;
@group(1) @binding(2) var<storage, read_write> ray_intersections: array<RayIntersectionData>;
@group(1) @binding(3) var<storage, read> rng_seeds: array<u32>;

@group(2) @binding(0) var<storage, read_write> output_buffer_f32: array<f32>; 
@group(2) @binding(1) var<storage, read_write> output_buffer_f16: array<u32>; // pack with pack2x16float

@group(3) @binding(0) var<storage, read_write> schedule: InvocationSchedule;
@group(3) @binding(1) var<storage, read_write> schedule_reorder: ScheduleReorder;

@compute
@workgroup_size(16, 16, 1)
fn copy_target(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    if gid.x < camera.resolution.x && gid.y < camera.resolution.y {
        var result_color = vec4(0.0f);
        let gidx = camera.resolution.x * gid.y + gid.x;
        var index = 0u;

        switch (settings.selected_buffer) {
            case 0u: { // primary rays
                index = camera.resolution.x * gid.y + gid.x;
                result_color = primary_rays[index].result_color;
                break;
            }
            case 1u: { // incidence
                let num_rays = schedule_reorder.num_events[8u];
                if gidx < num_rays {
                    let isec = ray_intersections[gidx + schedule_reorder.event_type_start[8u]];
                    index = isec.primary_ray;
                    result_color = vec4(isec.w_i * 0.5 + vec3(0.5), 1.0);
                }
                break;
            }
            case 2u: { // normal
                let num_rays = schedule_reorder.num_events[8u];
                if gidx < num_rays {
                    let isec = ray_intersections[gidx + schedule_reorder.event_type_start[8u]];
                    index = isec.primary_ray;
                    result_color = vec4(isec.n * 0.5 + vec3(0.5), 1.0);
                }
                break;
            }
            case 3u: { // w_o
                let num_ignored = schedule_reorder.index_in_event[7];
                let num_rays = schedule_reorder.intersect_invocations - num_ignored;
                if gidx < num_rays {
                    let ray = rays[num_ignored + gidx];
                    index = ray.primary_ray;
                    result_color = vec4(ray.direction * 0.5 + vec3(0.5), 1.0);
                }
                break;
            }
            case 4u { // weight
                let num_ignored = schedule_reorder.index_in_event[7];
                let num_rays = schedule_reorder.intersect_invocations - num_ignored;
                if gidx < num_rays {
                    let ray = ray_intersections[num_ignored + gidx];
                    index = ray.primary_ray;
                    result_color = vec4(ray.weight * 0.5 + vec3(0.5), 1.0);
                }
                break;
            }
            case 5u { // t
                let num_rays = schedule_reorder.num_events[8u];
                if gidx < num_rays {
                    let isec = ray_intersections[gidx + schedule_reorder.event_type_start[8u]];
                    index = isec.primary_ray;
                    result_color = vec4(vec3(isec.t / 1500), 1.0);
                }
                break;
            }
            default: {
                // do nothing
            }
        }

        var accumulated_color = result_color;
        if settings.accumulate != 0 {
            accumulated_color += primary_rays[index].accumulated_color;
            primary_rays[index].accumulated_color = accumulated_color;
            accumulated_color /= accumulated_color.a;
        }
        output_buffer_f32[4u * index     ] = accumulated_color.r;
        output_buffer_f32[4u * index + 1u] = accumulated_color.g;
        output_buffer_f32[4u * index + 2u] = accumulated_color.b;
        output_buffer_f32[4u * index + 3u] = accumulated_color.a;

        output_buffer_f16[2u * index     ] = pack2x16float(accumulated_color.rg);
        output_buffer_f16[2u * index + 1u] = pack2x16float(accumulated_color.ba);
    }
}
