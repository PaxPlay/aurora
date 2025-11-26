#import "common.wgsl"
#import "random.wgsl"

@group(0) @binding(0) var<storage, read_write> positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> forces: array<vec3<f32>>;

@group(1) @binding(0) var<uniform> settings: Settings;

const PI: f32 = 3.14159265359;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var pcg = pcg_seed(settings.seed, global_id.x);

    if (global_id.x < settings.num_particles) {
        let index = global_id.x;

        let theta = pcg_next_f32(&pcg) * 2.0 * PI;
        let r = sqrt(pcg_next_f32(&pcg)) * 0.5;
        let pos2d = vec2<f32>(
            r * cos(theta),
            r * sin(theta),
        );

        let pos = vec3<f32>(
            pos2d + vec2(0.5),
            0.0,
        );
        positions[index] = pos;

        let vel2d = vec2<f32>(
            -sin(theta),
            cos(theta),
        ) * r * 0.8;
        let vel = vec3<f32>(
            vel2d,
            0.0,
        );
        velocities[index] = vel;
    }
}