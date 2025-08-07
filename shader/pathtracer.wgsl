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

struct Schedule {
    ray_intersection_groups: vec4<u32>,
    handle_intersections_groups: vec4<u32>,
    num_rays: atomic<u32>,
    num_nee_rays: atomic<u32>,
    num_intersections: atomic<u32>,
    num_misses: atomic<u32>,
    rng_seed_index: u32,
}

const F32_MAX: f32 = 3.4028e38;
const EPSILON: f32 = 1e-10;

@group(0) @binding(0) var<uniform> camera : CameraBuffer;

@group(1) @binding(0) var<storage, read_write> primary_rays : array<PrimaryRayData>;
@group(1) @binding(1) var<storage, read_write> rays : array<Ray>;
@group(1) @binding(2) var<storage, read_write> ray_intersections: array<RayIntersectionData>;
@group(1) @binding(3) var<storage, read_write> schedule: Schedule;
@group(1) @binding(4) var<storage, read> rng_seeds: array<u32>;

@group(2) @binding(0) var<storage, read_write> output_buffer_f32: array<f32>; 
@group(2) @binding(1) var<storage, read_write> output_buffer_f16: array<u32>; // pack with pack2x16float

struct SceneGeometrySizes {
    vertices: u32,
    indices: u32,
    model_start_indices: u32,
    material_indices: u32,
    ambient: u32,
    diffuse: u32,
    specular: u32,
};
@group(3) @binding(0) var<storage> vertices: array<f32>;
@group(3) @binding(1) var<storage> indices: array<u32>;
@group(3) @binding(2) var<storage> model_start_indices: array<u32>;
@group(3) @binding(3) var<storage> material_indices: array<u32>;
@group(3) @binding(4) var<storage> ambient: array<f32>;
@group(3) @binding(5) var<storage> diffuse: array<f32>;
@group(3) @binding(6) var<storage> specular: array<f32>;
@group(3) @binding(7) var<uniform> sizes: SceneGeometrySizes;

fn world_pos_from_camera_space(camera_space: vec3<f32>) -> vec3<f32> {
    let world_pos = camera.vp_inv * vec4(camera_space, 1.0);
    return world_pos.xyz / world_pos.w;
}

@compute
@workgroup_size(1, 1, 1)
fn schedule_invocations(
) {
    let num_rays = atomicLoad(&schedule.num_rays);
    schedule.ray_intersection_groups = vec4<u32>(
        (num_rays + 127) / 128,
        1,
        1,
        num_rays,
    );

    let num_intersections = atomicLoad(&schedule.num_intersections);
    schedule.handle_intersections_groups = vec4<u32>(
        (num_intersections + 255) / 256,
        1,
        1,
        num_intersections,
    );

    atomicStore(&schedule.num_rays, 0u);
    atomicStore(&schedule.num_nee_rays, 0u);
    atomicStore(&schedule.num_intersections, 0u);
    atomicStore(&schedule.num_misses, 0u);

    schedule.rng_seed_index += 1;
}

@compute
@workgroup_size(16, 16, 1)
fn generate_rays(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    var screen_pos = (vec2<f32>(gid.xy) + vec2<f32>(0.5, 0.5)) / vec2<f32>(camera.resolution);
    screen_pos.y = 1.0 - screen_pos.y;
    let world_pos = world_pos_from_camera_space(vec3(screen_pos * 2.0 - vec2(1.0), 0.0));
    let direction = normalize(world_pos - camera.origin);
    var ray_data: PrimaryRayData;
    ray_data.origin = camera.origin;
    ray_data.direction = direction;
    ray_data.result_color = vec4(0.0);

    if gid.x < camera.resolution.x && gid.y < camera.resolution.y {
        let idx = camera.resolution.x * gid.y + gid.x;

        primary_rays[idx].origin = ray_data.origin;
        primary_rays[idx].direction = ray_data.direction;
        primary_rays[idx].result_color = ray_data.result_color;

        var ray: Ray;
        ray.origin = ray_data.origin;
        ray.direction = ray_data.direction;
        ray.primary_ray = idx;
        ray.weight = vec3<f32>(1.0);
        rays[idx] = ray;
    }

    if gid.x == 0 && gid.y == 0 {
        let num_rays = camera.resolution.x * camera.resolution.y;
        atomicStore(&schedule.num_rays, 0u);
        atomicStore(&schedule.num_nee_rays, 0u);
        atomicStore(&schedule.num_intersections, 0u);
        atomicStore(&schedule.num_misses, 0u);

        schedule.ray_intersection_groups = vec4<u32>(
            (num_rays + 127) / 128,
            1,
            1,
            num_rays,
        );

        schedule.rng_seed_index = 0;
    }
}

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
    return bitcast<f32>((pcg_next_u32(pcg) >> 9) | 0x3f800000u) - 1.0f;
}

fn pcg_next_square(pcg: ptr<function, PCG>) -> vec2<f32> {
    return vec2(pcg_next_f32(pcg), pcg_next_f32(pcg));
}

const PI: f32 = 3.14159265359;

fn warp_square_to_hemisphere(sample: vec2<f32>, n: vec3<f32>) -> vec3<f32> {
    let cosTheta = sample.x;
    let sinTheta = sqrt(1 - cosTheta * cosTheta);

    let phi = 2.0 * PI * sample.y;
    let sinPhi = sin(phi);
    let cosPhi = cos(phi);

    let up_hemisphere = vec3<f32>(
        sinPhi * sinTheta,
        cosPhi * sinTheta,
        cosTheta,
    );

    if dot(up_hemisphere, n) < 0 {
        return -1.0 * up_hemisphere;
    } else {
        return up_hemisphere;
    }
}

fn bsdf_sample_phong(sample: vec2<f32>, w_i: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return warp_square_to_hemisphere(sample, n);
}

fn bsdf_pdf_phong(w_o: vec3<f32>) -> f32 {
    return 1 / (2 * PI);
}

fn bsdf_eval_phong(m_d: vec3<f32>, m_s: vec3<f32>, m_n: f32,
                   w_i: vec3<f32>, n: vec3<f32>, w_o: vec3<f32>) -> vec3<f32> {
    let ideal = normalize(reflect(w_i, n));

    return m_d / PI + m_s * (n + 2) / (2 * PI) * pow(dot(ideal, w_o), m_n);
}

fn get_ambient(index: u32) -> vec3<f32> {
    return vec3(
        ambient[3 * index],
        ambient[3 * index + 1],
        ambient[3 * index + 2],
    );
}

fn get_diffuse(index: u32) -> vec3<f32> {
    return vec3(
        diffuse[3 * index],
        diffuse[3 * index + 1],
        diffuse[3 * index + 2],
    );
}

fn get_specular(index: u32) -> vec3<f32> {
    return vec3(
        specular[3 * index],
        specular[3 * index + 1],
        specular[3 * index + 2],
    );
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
    if lidx == 0 {
        atomicStore(&wg_num_rays, 0u);
    }

    var pcg: PCG = pcg_seed(rng_seeds[schedule.rng_seed_index], gid.x);

    let total_num_intersections = schedule.handle_intersections_groups.w;
    if gid.x < total_num_intersections {
        let isec = ray_intersections[gid.x];
        let mat_idx = material_indices[isec.surface_id];

        // Russian Roulette
        const alpha = 0.6;
        let xi = pcg_next_f32(&pcg);
        if xi <= alpha {
            let m_d = get_diffuse(mat_idx);
            let m_s = get_specular(mat_idx);

            let sample = pcg_next_square(&pcg);
            let w_o = bsdf_sample_phong(sample, isec.w_i, isec.n);
            let pdf = bsdf_pdf_phong(w_o);
            let f = bsdf_eval_phong(m_d, m_s, 0.0, isec.w_i, isec.n, w_o);
            
            let weight = isec.weight * f / pdf / alpha * dot(isec.n, w_o);
            var secondary_ray: Ray;
            secondary_ray.origin = isec.pos + 0.0001 * w_o;
            secondary_ray.direction = w_o;
            secondary_ray.weight = weight;
            secondary_ray.primary_ray = isec.primary_ray;
            
            let ray_index = atomicAdd(&wg_num_rays, 1u);
            wg_rays[ray_index] = secondary_ray;
        }

        let primary = primary_rays[isec.primary_ray];
        let m_a = get_ambient(mat_idx);
        let result_color = &primary_rays[isec.primary_ray].result_color;
        let delta = m_a * isec.weight * dot(isec.w_i, isec.n);
        (*result_color).r += delta.r;
        (*result_color).g += delta.g;
        (*result_color).b += delta.b;
        (*result_color).a = 1.0f;
    }

    // write back rays into ray buffer
    workgroupBarrier();

    if lidx == 0 {
        wg_ray_buffer_start = atomicAdd(&schedule.num_rays, atomicLoad(&wg_num_rays));
    }

    workgroupBarrier();

    if lidx < wg_num_rays {
        rays[wg_ray_buffer_start + lidx] = wg_rays[lidx];
    }
}

@compute
@workgroup_size(16, 16, 1)
fn copy_target(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if gid.x >= camera.resolution.x || gid.y >= camera.resolution.y {
        return;
    }

    let index = camera.resolution.x * gid.y + gid.x;
    var color = primary_rays[index].result_color;
    color += primary_rays[index].accumulated_color;
    primary_rays[index].accumulated_color = color;

    color /= color.a;

    output_buffer_f32[4 * index    ] = color.r;
    output_buffer_f32[4 * index + 1] = color.g;
    output_buffer_f32[4 * index + 2] = color.b;
    output_buffer_f32[4 * index + 3] = color.a;

    output_buffer_f16[2 * index    ] = pack2x16float(color.rg);
    output_buffer_f16[2 * index + 1] = pack2x16float(color.ba);
}

