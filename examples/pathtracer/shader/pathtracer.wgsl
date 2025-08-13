struct CameraBuffer {
    vp: mat4x4<f32>,
    vp_inv: mat4x4<f32>,
    origin: vec3<f32>,
    direction: vec3<f32>,
    up: vec3<f32>,
    resolution: vec2<u32>,
    fov: f32,
}

struct PrimaryRayData {
    origin: vec3<f32>,
    direction: vec3<f32>,
    result_color: vec4<f32>,
    accumulated_color: vec4<f32>,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    weight: vec3<f32>,
    primary_ray: u32,
}

struct RayIntersectionData {
    pos: vec3<f32>,
    n: vec3<f32>,
    w_i: vec3<f32>,
    weight: vec3<f32>,
    t: f32,
    surface_id: u32,
    primary_ray: u32,
}

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

const F32_MAX: f32 = 3.4028e38;
const EPSILON: f32 = 1e-4;
const PI: f32 = 3.14159265359;

@group(0) @binding(0) var<uniform> camera : CameraBuffer;

@group(1) @binding(0) var<storage, read_write> primary_rays : array<PrimaryRayData>;
@group(1) @binding(1) var<storage, read_write> rays : array<Ray>;
@group(1) @binding(2) var<storage, read_write> ray_intersections: array<RayIntersectionData>;
@group(1) @binding(3) var<storage, read> rng_seeds: array<u32>;

@group(2) @binding(0) var<storage, read_write> schedule: ScheduleShade;

struct SceneGeometrySizes {
    vertices: u32,
    indices: u32,
    model_start_indices: u32,
    material_indices: u32,
    ambient: u32,
    diffuse: u32,
    specular: u32,
};

@group(3) @binding(0) var<uniform> vertices: array<vec3<f32>, 256>;
@group(3) @binding(1) var<storage> indices: array<u32>;
@group(3) @binding(2) var<storage> model_start_indices: array<u32>;
@group(3) @binding(3) var<storage> material_indices: array<u32>;
@group(3) @binding(4) var<uniform> ambient: array<vec3<f32>, 256>;
@group(3) @binding(5) var<uniform> diffuse: array<vec3<f32>, 256>;
@group(3) @binding(6) var<uniform> specular: array<vec3<f32>, 256>;
@group(3) @binding(7) var<uniform> sizes: SceneGeometrySizes;

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

fn bsdf_sample_phong(sample: vec2<f32>, w_i: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return warp_square_to_hemisphere(sample, n);
}

fn bsdf_pdf_phong(w_o: vec3<f32>) -> f32 {
    return 1.0 / (2.0 * PI);
}

fn bsdf_eval_phong(m_d: vec3<f32>, m_s: vec3<f32>, m_n: f32,
    w_i: vec3<f32>, n: vec3<f32>, w_o: vec3<f32>) -> vec3<f32> {
    let ideal = normalize(reflect(w_i, n));

    return m_d / PI + m_s * ((n + 2.0) / (2.0 * PI)) * pow(max(dot(ideal, w_o), 0.0), m_n);
}

var<workgroup> wg_rays: array<Ray, 256>;
var<workgroup> wg_num_rays: atomic<u32>;
var<workgroup> wg_ray_buffer_start: u32;

@compute
@workgroup_size(256, 1, 1)
fn handle_intersections(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32
) {
    if lidx == 0u {
        atomicStore(&wg_num_rays, 0u);
    }

    let total_num_intersections = schedule.shade_invocations;
    if gid.x < total_num_intersections {
        let isec = ray_intersections[gid.x];
        let mat_idx = material_indices[isec.surface_id];
        var pcg: PCG = pcg_seed(rng_seeds[schedule.rng_seed_index], isec.primary_ray);


        // Russian Roulette
        const alpha = 0.8;
        let xi = pcg_next_f32(&pcg);
        if xi <= alpha {
            let m_d = diffuse[mat_idx];
            let m_s = specular[mat_idx];

            let sample = pcg_next_square(&pcg);
            let w_o = bsdf_sample_phong(sample, isec.w_i, isec.n);
            let pdf = bsdf_pdf_phong(w_o);
            let f = bsdf_eval_phong(m_d, m_s, 1.0, isec.w_i, isec.n, w_o);

            let weight = isec.weight * f / pdf / alpha * dot(isec.n, w_o);
            var secondary_ray: Ray;
            secondary_ray.origin = isec.pos + EPSILON * w_o;
            secondary_ray.direction = w_o;
            secondary_ray.weight = weight;
            secondary_ray.primary_ray = isec.primary_ray;

            let ray_index = atomicAdd(&wg_num_rays, 1u);
            wg_rays[ray_index] = secondary_ray;
        }

        let primary = primary_rays[isec.primary_ray];
        let m_a = ambient[mat_idx];
        var result_color = primary_rays[isec.primary_ray].result_color;
        let delta = m_a * isec.weight * dot(isec.w_i, isec.n);
        result_color = max(result_color, vec4(0.0f));
        primary_rays[isec.primary_ray].result_color = vec4(
            delta + result_color.rgb,
            1.0f
        );
    }

    // write back rays into ray buffer
    workgroupBarrier();

    if lidx == 0u {
        wg_ray_buffer_start = atomicAdd(&schedule.num_rays, atomicLoad(&wg_num_rays));
    }

    workgroupBarrier();

    if lidx < atomicLoad(&wg_num_rays) {
        rays[wg_ray_buffer_start + lidx] = wg_rays[lidx];
    }
}

