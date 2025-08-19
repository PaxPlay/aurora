use wgsl_preprocessor::WgslPreprocessor;

fn main() {
    // Runtime preprocessing example
    runtime_example();

    // Conditional compilation example
    conditional_example();

    // Compile-time preprocessing example
    compile_time_example();
}

fn runtime_example() {
    println!("=== Runtime Preprocessing Example ===");

    let mut preprocessor = WgslPreprocessor::new();
    preprocessor
        .define("WORKGROUP_SIZE", "64")
        .define("PI", "3.14159265359");

    let shader_source = r#"
#define MAX_ITERATIONS 100

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let angle = f32(global_id.x) * PI / 180.0;
    
    for (var i: u32 = 0u; i < MAX_ITERATIONS; i++) {
        // Compute intensive work here
    }
}
"#;

    match preprocessor.process_string(shader_source, None) {
        Ok(processed) => {
            println!("Processed shader:");
            println!("{}", processed.content);
        }
        Err(e) => {
            eprintln!("Error processing shader: {}", e);
        }
    }
}

fn conditional_example() {
    println!("=== Conditional Compilation Example ===");

    let mut debug_preprocessor = WgslPreprocessor::new();
    debug_preprocessor.define("DEBUG", "1");

    let mut release_preprocessor = WgslPreprocessor::new();
    release_preprocessor.define("RELEASE", "1");

    let shader_source = r#"
#ifdef DEBUG
const LOG_LEVEL: u32 = 0u;  // Verbose
#else
const LOG_LEVEL: u32 = 2u;  // Errors only  
#endif

#ifndef RELEASE
const ENABLE_PROFILING: bool = true;
#endif

fn main() {}
"#;

    println!("Debug build:");
    match debug_preprocessor.process_string(shader_source, None) {
        Ok(processed) => println!("{}", processed.content),
        Err(e) => eprintln!("Error: {}", e),
    }

    println!("Release build:");
    match release_preprocessor.process_string(shader_source, None) {
        Ok(processed) => println!("{}", processed.content),
        Err(e) => eprintln!("Error: {}", e),
    }
}

fn compile_time_example() {
    println!("=== Compile-time Preprocessing Example ===");

    // This would work if we had the file:
    // const VERTEX_SHADER: &str = wgsl!("shaders/vertex.wgsl");

    println!("Compile-time preprocessing happens at build time using the wgsl! macro");
    println!(
        "Example: const SHADER: &str = wgsl_preprocessor_macro::wgsl!(\"path/to/shader.wgsl\");"
    );
}
