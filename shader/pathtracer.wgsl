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
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    primary_ray: u32,
}

struct RayIntersectionData {
    pos: vec3<f32>,
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
}

const F32_MAX: f32 = 3.4028e38;
const EPSILON: f32 = 1e-10;

@group(0) @binding(0) var<uniform> camera : CameraBuffer;

@group(1) @binding(0) var<storage, read_write> primary_rays : array<PrimaryRayData>;
@group(1) @binding(1) var<storage, read_write> rays : array<Ray>;
@group(1) @binding(2) var<storage, read_write> ray_intersections: array<RayIntersectionData>;
@group(1) @binding(3) var<storage, read_write> schedule: Schedule;

@group(2) @binding(0) var output_texture: texture_storage_2d<rgba16float, write>; 

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
        (num_rays + 255) / 256,
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

        primary_rays[idx] = ray_data;

        var ray: Ray;
        ray.origin = ray_data.origin;
        ray.direction = ray_data.direction;
        ray.primary_ray = idx;
        rays[idx] = ray;
    }

    if gid.x == 0 && gid.y == 0 {
        let num_rays = camera.resolution.x * camera.resolution.y;
        atomicStore(&schedule.num_rays, 0u);
        atomicStore(&schedule.num_nee_rays, 0u);
        atomicStore(&schedule.num_intersections, 0u);
        atomicStore(&schedule.num_misses, 0u);

        schedule.ray_intersection_groups = vec4<u32>(
            (num_rays + 255) / 256,
            1,
            1,
            num_rays,
        );
    }
}

var<workgroup> local_vertices: array<vec3<f32>, 768>;
var<workgroup> wg_num_intersections: atomic<u32>;
var<workgroup> wg_num_misses: atomic<u32>;
var<workgroup> wg_intersections: array<RayIntersectionData, 256>;
var<workgroup> wg_misses: array<u32, 256>;
var<workgroup> wg_isec_group_start: u32;
var<workgroup> wg_miss_group_start: u32;

@compute
@workgroup_size(256, 1, 1)
fn intersect_rays(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32
) {
    let num_rays = schedule.ray_intersection_groups.w;

    if gid.x == 0 {
        atomicStore(&wg_num_intersections, 0u);
        atomicStore(&wg_num_misses, 0u);
    }

    if gid.x >= num_rays {
        return;
    }

    let ray = rays[gid.x];

    let num_triangles = sizes.indices / 3;
    if lidx < (num_triangles * 3) {
        let idx = indices[lidx];
        local_vertices[lidx] = vec3<f32>(
            vertices[idx * 3],
            vertices[idx * 3 + 1],
            vertices[idx * 3 + 2],
        );
    }

    workgroupBarrier();

    var isec: RayIntersectionData;
    isec.pos = vec3<f32>(-1, -1, -1);
    isec.t = F32_MAX;
    isec.primary_ray = ray.primary_ray;

    for (var i: u32; i < num_triangles; i++) {
        // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
        let a = local_vertices[3 * i];
        let b = local_vertices[3 * i + 1];
        let c = local_vertices[3 * i + 2];

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
        if t > EPSILON && t < isec.t {
            isec.t = t;
            isec.pos = ray.origin + t * ray.direction;
            isec.surface_id = i;
        }
    }

    // Sum up intersection in workgroup
    if isec.t < 1e38 {
        let local_idx = atomicAdd(&wg_num_intersections, 1u);
        wg_intersections[local_idx] = isec;
    } else {
        let local_idx = atomicAdd(&wg_num_misses, 1u);
        wg_misses[local_idx] = isec.primary_ray;
    }

    workgroupBarrier();

    let num_intersections = atomicLoad(&wg_num_intersections);
    let num_misses = atomicLoad(&wg_num_misses);

    // Sum up total intersections using global atomics
    // results of the atomics are the start indices in the respective buffers
    if lidx == 0 {
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

@compute
@workgroup_size(256, 1, 1)
fn handle_intersections(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let total_num_intersections = schedule.handle_intersections_groups.w;
    if gid.x < total_num_intersections {
        let isec = ray_intersections[gid.x];
        let mat_idx = material_indices[isec.surface_id];
        let diff = vec3<f32>(
            diffuse[3 * mat_idx],
            diffuse[3 * mat_idx + 1],
            diffuse[3 * mat_idx + 2],
        );
        let primary = primary_rays[isec.primary_ray];

        primary_rays[isec.primary_ray].result_color = vec4(diff, 1.0f);
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

    let color = primary_rays[camera.resolution.x * gid.y + gid.x].result_color;

    textureStore(output_texture, gid.xy, color);
}
