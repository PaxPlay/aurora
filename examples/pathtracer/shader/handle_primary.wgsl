#import "structs.wgsl"

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


// This shader assumes that the first x intersections are primary ray intersections
// This is ensured by the intersect shader if the primary rays are at the start of the ray buffer
// reorder_intersection.wgsl cannot be used between intersect and this shader

// This shader handles increasing the primary ray hit count and adding ambient light if NEE is enabled

@compute
@workgroup_size(16, 16, 1)
fn handle_primary(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32
) {
    if gid.x < camera.resolution.x && gid.y < camera.resolution.y {
        // direct illuminaton
        let idx = camera.resolution.x * gid.y + gid.x;
        let isec = ray_intersections[idx];

        if isec.event_type == 0u {
            primary_rays[idx].result_color.a += 1.0; // count number of primary hits

            // NEE requires ambient light to be added here
            if settings.nee != 0u {
                let mat_idx = material_indices[isec.surface_id];
                let ambient_color = ambient[mat_idx];
                if length(ambient_color) > 0.0 {
                    primary_rays[idx].result_color += vec4(ambient_color, 0.0);
                }
            }

            ray_intersections[idx].event_type = 8u; // set to normal intersection event; TODO bsdf id
        }
    }
}
