#import "common.wgsl"

@group(0) @binding(0) var<storage, read_write> positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> forces: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> settings: Settings;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x < settings.num_particles) {
        let index = global_id.x;
        let time_step = settings.time_step;

        var velocity = velocities[index];
        var position = positions[index];
        var force = forces[index].xyz;

        if max(length(abs(force)), 0.0) == 0.0 {
            force = vec3<f32>(0.0, 0.0, 0.0);
        }

        // Update velocity
        velocity += force * time_step;

        // Update position
        position += velocity * time_step;

        if position.x < 0.0 {
            position.x = -position.x;
            velocity.x = -velocity.x;
        }
        if position.x > 1.0 {
            position.x = 2.0 - position.x;
            velocity.x = -velocity.x;
        }
        if position.y < 0.0 {
            position.y = -position.y;
            velocity.y = -velocity.y;
        }
        if position.y > 1.0 {
            position.y = 2.0 - position.y;
            velocity.y = -velocity.y;
        }

        velocities[index] = velocity;
        positions[index] = position;
    }
}
