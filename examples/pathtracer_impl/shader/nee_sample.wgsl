struct NEESample {
    position: vec3<f32>,
    radiance: vec3<f32>,
    pdf: f32,
    surface_normal: vec3<f32>
}

// Sample any light source uniformly
fn nee_sample(sample: vec2<f32>) -> NEESample {
    // Single, hardcoded quad light source for now
    const A: vec3<f32> = vec3(0.343, 0.548, 0.227);
    const B: vec3<f32> = vec3(0.343, 0.548, 0.332);
    const C: vec3<f32> = vec3(0.213, 0.548, 0.227);
    const E1 = B - A;
    const E2 = C - A;

    var nee_sample: NEESample;
    nee_sample.position = A + sample.x * E1 + sample.y * E2;
    nee_sample.radiance = vec3(18.387, 13.9873, 6.75357);
    nee_sample.pdf = 1.0 / length(cross(E1, E2));
    nee_sample.surface_normal = normalize(cross(E1, E2));
    return nee_sample;
}
