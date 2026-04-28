struct ScalarFieldParameters {
    model_matrix: mat4x4<f32>,
    world_to_field: mat4x4<f32>,
    field_size: vec3<u32>,
    origin: vec3<f32>,
    bb_size: vec3<f32>,
}

@group(0) @binding(0) var field_texture: texture_3d<f32>;
@group(0) @binding(1) var<uniform> field_parameters: ScalarFieldParameters;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>, 129>;

var<workgroup> wg_histogram: array<atomic<u32>, 129>;

@compute
@workgroup_size(8, 4, 4)
fn cs_histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    atomicStore(&wg_histogram[lidx], 0u);

    if lidx == 0u {
        atomicStore(&wg_histogram[128], 0u);
    }

    workgroupBarrier();

    let size = field_parameters.field_size;
    if gid.x < size.x && gid.y < size.y && gid.z < size.z {
        let value = textureLoad(field_texture, vec3<i32>(gid), 0).x;
        let index = u32(clamp(value * 128.0, 0.0, 127.0));
        atomicAdd(&wg_histogram[index], 1u);
    }

    workgroupBarrier();

    let value = atomicLoad(&wg_histogram[lidx]);
    atomicAdd(&histogram[lidx], value);
    atomicAdd(&wg_histogram[128], value);

    workgroupBarrier();

    if lidx == 0u {
        let total = atomicLoad(&wg_histogram[128]);
        atomicAdd(&histogram[128], total);
    }
}