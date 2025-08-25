#import "structs.wgsl"

@group(0) @binding(0) var<storage, read_write> primary_rays : array<PrimaryRayData>;
@group(0) @binding(1) var<storage, read_write> rays : array<Ray>;
@group(0) @binding(2) var<storage, read_write> ray_intersections: array<RayIntersectionData>;
@group(0) @binding(3) var<storage, read> rng_seeds: array<u32>;

@group(1) @binding(0) var<storage, read_write> schedule: ScheduleIntersect;

@group(2) @binding(0) var<uniform> vertices: array<vec3<f32>, 256>;
@group(2) @binding(1) var<storage> indices: array<u32>;
@group(2) @binding(2) var<storage> model_start_indices: array<u32>;
@group(2) @binding(3) var<storage> material_indices: array<u32>;
@group(2) @binding(4) var<uniform> ambient: array<vec3<f32>, 256>;
@group(2) @binding(5) var<uniform> diffuse: array<vec3<f32>, 256>;
@group(2) @binding(6) var<uniform> specular: array<vec3<f32>, 256>;
@group(2) @binding(7) var<uniform> sizes: SceneGeometrySizes;

var<workgroup> local_vertices: array<f32, 128u * 3u * 3u>;
var<workgroup> wg_num_intersections: atomic<u32>;
var<workgroup> wg_num_misses: atomic<u32>;
var<workgroup> wg_intersections: array<RayIntersectionData, 128>;
var<workgroup> wg_misses: array<u32, 128>;
var<workgroup> wg_isec_group_start: u32;
var<workgroup> wg_miss_group_start: u32;

fn local_vertex(index: u32) -> vec3<f32> {
    return vec3<f32>(
        local_vertices[3u * index + 0u],
        local_vertices[3u * index + 1u],
        local_vertices[3u * index + 2u]
    );
}

@compute
@workgroup_size(128, 1, 1)
fn intersect_rays(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32
) {
    let num_rays = schedule.intersect_invocations;

    if gid.x == 0u {
        atomicStore(&wg_num_intersections, 0u);
        atomicStore(&wg_num_misses, 0u);
    }

    let ray = rays[gid.x];

    let num_triangles = sizes.indices / 3u;

    if lidx < (num_triangles * 3u) {
        let idx = indices[lidx];
        local_vertices[3u * lidx + 0u] = vertices[idx].x;
        local_vertices[3u * lidx + 1u] = vertices[idx].y;
        local_vertices[3u * lidx + 2u] = vertices[idx].z;
    }

    workgroupBarrier();

    if gid.x < num_rays {
        var isec: RayIntersectionData;
        isec.pos = vec3<f32>(-1.0, -1.0, -1.0);
        isec.t = ray.t_max;
        isec.primary_ray = ray.primary_ray;

        for (var i: u32; i < num_triangles; i++) {
            // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
            let a = local_vertex(3u * i + 0u);
            let b = local_vertex(3u * i + 1u);
            let c = local_vertex(3u * i + 2u);

            let e1 = b - a;
            let e2 = c - a;

            let ray_cross_e2 = cross(ray.direction, e2);
            let det = dot(e1, ray_cross_e2);

            if abs(det) < EPSILON {
                continue;
            }

            let inv_det = 1.0 / det;
            let s = ray.origin - a;
            let u = inv_det * dot(s, ray_cross_e2);

            if u < 0.0 || u > 1.0 {
                continue;
            }

            let s_cross_e1 = cross(s, e1);
            let v = inv_det * dot(ray.direction, s_cross_e1);
            if v < 0.0 || (u + v) > 1.0 {
                continue;
            }

            let t = inv_det * dot(e2, s_cross_e1);
            if t > 0.01 && t < isec.t {
                isec.t = t;
                isec.pos = ray.origin + t * ray.direction;
                isec.surface_id = i;
                isec.n = normalize(cross(e1, e2));
                if dot(isec.n, ray.direction) > 0.0 {
                    isec.n = -1.0 * isec.n;
                }
                isec.w_i = normalize(-ray.direction);
                isec.weight = ray.weight;
                isec.primary_ray = ray.primary_ray;
            }
        }

        // Sum up intersection in workgroup
        if isec.t < ray.t_max {
            let local_idx = atomicAdd(&wg_num_intersections, 1u);
            wg_intersections[local_idx] = isec;
        } else {
            let local_idx = atomicAdd(&wg_num_misses, 1u);
            wg_misses[local_idx] = isec.primary_ray;
        }
    }

    workgroupBarrier();

    let num_intersections = atomicLoad(&wg_num_intersections);
    let num_misses = atomicLoad(&wg_num_misses);

    // Sum up total intersections using global atomics
    // results of the atomics are the start indices in the respective buffers
    if lidx == 0u {
        wg_isec_group_start = atomicAdd(&schedule.num_intersections,
            num_intersections);
        wg_miss_group_start = atomicAdd(&schedule.num_misses,
            num_misses);
    }

    workgroupBarrier();

    // write intersection information into global buffers
    if lidx < num_intersections {
        ray_intersections[wg_isec_group_start + lidx] = wg_intersections[lidx];
    }
}
