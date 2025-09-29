#import "structs.wgsl"
#import "random.wgsl"
#import "nee_sample.wgsl"

@group(0) @binding(0) var<uniform> camera : CameraBuffer;
@group(0) @binding(1) var<uniform> settings: Settings;

@group(1) @binding(0) var<storage, read_write> primary_rays : array<PrimaryRayData>;
@group(1) @binding(1) var<storage, read_write> rays : array<Ray>;
@group(1) @binding(2) var<storage, read_write> ray_intersections: array<RayIntersectionData>;
@group(1) @binding(3) var<storage, read> rng_seeds: array<u32>;

@group(2) @binding(0) var<storage, read_write> schedule: ScheduleShade;

@group(3) @binding(0) var<uniform> vertices: array<vec3<f32>, 256>;
@group(3) @binding(1) var<storage> indices: array<u32>;
@group(3) @binding(2) var<storage> model_start_indices: array<u32>;
@group(3) @binding(3) var<storage> material_indices: array<u32>;
@group(3) @binding(4) var<uniform> ambient: array<vec3<f32>, 256>;
@group(3) @binding(5) var<uniform> diffuse: array<vec3<f32>, 256>;
@group(3) @binding(6) var<uniform> specular: array<vec3<f32>, 256>;
@group(3) @binding(7) var<uniform> sizes: SceneGeometrySizes;

fn bsdf_sample_phong(sample: vec2<f32>, w_i: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return warp_square_to_hemisphere(sample, n);
}

fn bsdf_pdf_phong(w_o: vec3<f32>) -> f32 {
    return 1.0 / (2.0 * PI);
}

fn bsdf_eval_phong(mat_idx: u32,
    w_i: vec3<f32>, n: vec3<f32>, w_o: vec3<f32>) -> vec3<f32> {
    let ideal = normalize(reflect(w_i, n));

    let m_d = diffuse[mat_idx];
    let m_s = specular[mat_idx];
    let m_n = 1.0;

    return m_d / PI + m_s * ((n + 2.0) / (2.0 * PI)) * pow(max(dot(ideal, w_o), 0.0), m_n);
}

var<workgroup> wg_pcg: array<PCG, 128>;
var<workgroup> wg_num_rays_atomic: atomic<u32>;
var<workgroup> wg_num_nee_atomic: atomic<u32>;
var<workgroup> wg_num_rays: u32;
var<workgroup> wg_ray_indices: array<u32, 128>;
var<workgroup> wg_ray_buffer_start: u32;

@compute
@workgroup_size(128, 1, 1)
fn handle_intersections(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32
) {
    if lidx == 0u {
        atomicStore(&wg_num_rays_atomic, 0u);
        atomicStore(&wg_num_nee_atomic, 0u);
    }

    workgroupBarrier();

    let total_num_intersections = schedule.shade_invocations;
    var isec: RayIntersectionData;
    var mat_idx: u32;
    var rr_success = false;
    var pcg: PCG;

    if gid.x < total_num_intersections {
        pcg = pcg_seed(rng_seeds[schedule.rng_seed_index], ray_intersections[schedule.isec_start + gid.x].primary_ray);

        let xi = pcg_next_f32(&pcg);
        if xi <= settings.rr_alpha {
            let local_index = atomicAdd(&wg_num_rays_atomic, 1u);
            wg_ray_indices[local_index] = schedule.isec_start + gid.x;
            wg_pcg[local_index] = pcg;
            rr_success = true;
        }
    }

    workgroupBarrier();

    if lidx == 0u {
        wg_num_rays = atomicLoad(&wg_num_rays_atomic);
        wg_ray_buffer_start = atomicAdd(&schedule.num_rays, wg_num_rays);
    }

    workgroupBarrier();

    if gid.x < total_num_intersections && !rr_success {
        // we need the rest of the intersections for the nee rays later
        var local_index = atomicAdd(&wg_num_rays_atomic, 1u);
        wg_ray_indices[local_index] = schedule.isec_start + gid.x;
        wg_pcg[local_index] = pcg;
    }

    workgroupBarrier();

    if gid.x < total_num_intersections {
        pcg = wg_pcg[lidx];
        isec = ray_intersections[wg_ray_indices[lidx]];
        mat_idx = material_indices[isec.surface_id];
    }

    if lidx < wg_num_rays {
        let sample = pcg_next_square(&pcg);
        let w_o = bsdf_sample_phong(sample, isec.w_i, isec.n);
        let pdf = bsdf_pdf_phong(w_o);
        let f = bsdf_eval_phong(mat_idx, isec.w_i, isec.n, w_o);

        let weight = isec.weight * f / pdf / settings.rr_alpha * dot(isec.n, w_o);
        var secondary_ray: Ray;
        secondary_ray.origin = isec.pos + EPSILON * w_o;
        secondary_ray.direction = w_o;
        secondary_ray.weight = weight;
        secondary_ray.primary_ray = isec.primary_ray;
        secondary_ray.t_min = 0.01;
        secondary_ray.t_max = F32_MAX;
        secondary_ray.ray_type = 1u;
        rays[wg_ray_buffer_start + lidx] = secondary_ray;
    }

    workgroupBarrier();

    var nee_ray: Ray;
    var local_index: u32 = ~0u;

    if gid.x < total_num_intersections {
        if settings.nee == 0u {
            let primary = primary_rays[isec.primary_ray];
            let m_a = ambient[mat_idx];
            var result_color = primary_rays[isec.primary_ray].result_color;
            let delta = m_a * isec.weight * dot(isec.w_i, isec.n);
            primary_rays[isec.primary_ray].result_color = vec4(
                delta + result_color.rgb,
                result_color.a
            );
        } else {
            let sample = pcg_next_square(&pcg);
            let light_sample = nee_sample(sample);
            var direction = light_sample.position - isec.pos;
            let distance = length(direction);
            direction = direction / distance;

            let scalar = max(dot(isec.n, direction) * dot(-direction, light_sample.surface_normal), 0.0) / (distance * distance) / light_sample.pdf;
            let weight = light_sample.radiance * isec.weight * bsdf_eval_phong(mat_idx, isec.w_i, isec.n, direction) * scalar;
            if length(weight) > EPSILON {
                nee_ray.origin = isec.pos + EPSILON * direction;
                nee_ray.direction = direction;
                nee_ray.weight = weight;
                nee_ray.primary_ray = isec.primary_ray;
                nee_ray.t_min = 0.01;
                nee_ray.t_max = distance - 0.01;
                nee_ray.ray_type = 2u; // NEE ray

                local_index = atomicAdd(&wg_num_nee_atomic, 1u);
            }
        }
    }

    workgroupBarrier();

    if lidx == 0u {
        wg_ray_buffer_start = atomicAdd(&schedule.num_rays, atomicLoad(&wg_num_nee_atomic));
    }

    workgroupBarrier();

    if local_index != ~0u {
        rays[wg_ray_buffer_start + local_index] = nee_ray;
    }
}
