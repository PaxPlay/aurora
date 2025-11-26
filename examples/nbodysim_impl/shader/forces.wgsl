#import "common.wgsl"

@group(0) @binding(0) var<storage, read> positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> forces: array<vec3<f32>>;

@group(1) @binding(0) var<uniform> settings: Settings;

const WORKGROUP_SIZE: u32 = 128u;
var<workgroup> wg_positions: array<vec3<f32>, WORKGROUP_SIZE>;

fn calculate_force(pos_a: vec3<f32>, pos_b: vec3<f32>) -> vec3<f32> {
    var direction = pos_b - pos_a;
    let distance_sq = max(dot(direction, direction), 0.0001);

    let force_magnitude = settings.gravitational_constant / distance_sq;
    return normalize(direction) * force_magnitude;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn cs_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    var accumulated_force = vec3<f32>(0.0, 0.0, 0.0);
    var position = vec3<f32>(0.0, 0.0, 0.0);
//    var accumulated_index = 0u;

    if (global_id.x < settings.num_particles) {
        position = positions[global_id.x];
    }
    wg_positions[local_id.x] = position;

    workgroupBarrier();

    let num_valid = min(settings.num_particles - workgroup_id.x * WORKGROUP_SIZE, WORKGROUP_SIZE);

    // First iteration needs to ignore own particle, thus it is done seperately
    for (var i: u32 = (local_id.x + 1u) % WORKGROUP_SIZE; i != local_id.x; i = (i + 1u) % WORKGROUP_SIZE) {
        let other_position = wg_positions[i];
        if i < num_valid {
            accumulated_force += calculate_force(position, other_position);
//            accumulated_index += workgroup_id.x * WORKGROUP_SIZE + i;
        }
    }

    // Remaining iterations
    for (var wg: u32 = (workgroup_id.x + 1u) % num_workgroups.x; wg != workgroup_id.x; wg = (wg + 1u) % num_workgroups.x) {
        workgroupBarrier();

        let base_index = wg * WORKGROUP_SIZE;
        wg_positions[local_id.x] = positions[base_index + local_id.x];

        workgroupBarrier();

        let num_valid = min(settings.num_particles - wg * WORKGROUP_SIZE, WORKGROUP_SIZE);
        for (var i: u32 = 0; i < WORKGROUP_SIZE; i = i + 1u) {
            let other_index = (local_id.x + i) % WORKGROUP_SIZE;
            let other_position = wg_positions[other_index];
            if other_index < num_valid {
                accumulated_force += calculate_force(position, other_position);
//                accumulated_index += wg * WORKGROUP_SIZE + other_index;
            }
        }
    }

    if (global_id.x < settings.num_particles) {
        forces[global_id.x] = accumulated_force;
    }
}
