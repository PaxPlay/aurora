// Adapted from PCG, but with smaller state, since WGSL doesn't support 64-bit integers
// Results seem okay
struct PCG {
    state: u32,
    inc: u32,
}

fn pcg_seed(initstate: u32, initseq: u32) -> PCG {
    var pcg: PCG;
    pcg.state = 0u;
    pcg.inc = (initseq << 1u) | 1u;
    pcg_next_u32(&pcg);
    pcg.state += initstate;
    pcg_next_u32(&pcg);
    return pcg;
}

fn pcg_next_u32(pcg: ptr<function, PCG>) -> u32 {
    let state = (*pcg).state * 747796405u + (*pcg).inc;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    (*pcg).state = (word >> 22u) ^ word;
    return (*pcg).state;
}

fn pcg_next_f32(pcg: ptr<function, PCG>) -> f32 {
    return bitcast<f32>((pcg_next_u32(pcg) >> 9u) | 0x3f800000u) - 1.0f;
}

fn pcg_next_square(pcg: ptr<function, PCG>) -> vec2<f32> {
    return vec2(pcg_next_f32(pcg), pcg_next_f32(pcg));
}

fn warp_square_to_hemisphere(sample: vec2<f32>, n: vec3<f32>) -> vec3<f32> {
    let cosTheta = sample.x;
    let sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    let phi = 2.0 * PI * sample.y;
    let sinPhi = sin(phi);
    let cosPhi = cos(phi);

    let up_hemisphere = vec3<f32>(
        sinPhi * sinTheta,
        cosPhi * sinTheta,
        cosTheta,
    );

    if dot(up_hemisphere, n) < 0.0 {
        return -1.0 * up_hemisphere;
    } else {
        return up_hemisphere;
    }
}
