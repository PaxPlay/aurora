@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f16>;
@group(0) @binding(2) var<uniform> size: array<u32>;


@compute
@workgroup_size(256, 1, 1)
fn copy_buffer(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if gid.x < size[0] {
        dst[gid.x] = f16(src[gid.x]);
    }
}

