struct TargetSizes {
    render: vec2<u32>,
    window: vec2<u32>,
    srgb: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) texture_coord: vec2<f32>,
}

@group(0) @binding(0) var renderTexture: texture_2d<f32>;
@group(0) @binding(1) var renderSampler: sampler;
@group(0) @binding(2) var<uniform> targetSizes: TargetSizes;

@vertex
fn vs_main(
    @builtin(vertex_index) i: u32,
) -> VertexOutput {
    var out: VertexOutput;
    var index = i;
    if index >= 3 {
        switch index {
            case 3u: {
                index = 1;
            }
            case 4u: {
                index = 3;
            }
            case 5u: {
                index = 2;
            }
            default: {}
        }
    }

    let render_aspect = f32(targetSizes.render.y) / f32(targetSizes.render.x);
    let window_aspect = f32(targetSizes.window.y) / f32(targetSizes.window.x);

    out.clip_position.x = f32(index & 1) * 2.0 - 1.0;
    out.clip_position.y = f32(index / 2) * 2.0 - 1.0;
    out.clip_position.z = 0.0;
    out.clip_position.w = 1.0;

    if render_aspect > window_aspect {
        out.clip_position.x *= window_aspect / render_aspect;
    } else {
        out.clip_position.y /= window_aspect / render_aspect;
    }

    out.texture_coord.x = f32(index & 1);
    out.texture_coord.y = 1 - f32(index / 2);
    return out;
}


fn linearToSrgb(c: f32) -> f32 {
    if c <= 0.0031308f {
        return c * 12.92f;
    } else {
        return 1.055f * pow(c, 1.0f / 2.4f) - 0.055f;
    }
}

fn linearToSrgbVec3(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linearToSrgb(c.x),
        linearToSrgb(c.y),
        linearToSrgb(c.z)
    );
}

@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    var color = textureSample(renderTexture, renderSampler, in.texture_coord);
    if targetSizes.srgb == 1u {
        color = vec4(linearToSrgbVec3(color.rgb), color.a);
    }

    return color;
}

