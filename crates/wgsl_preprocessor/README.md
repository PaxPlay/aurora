# WGSL Preprocessor

A preprocessing library for WGSL (WebGPU Shading Language) that supports `#include` and `#define` directives similar to C preprocessor. Provides both runtime and compile-time preprocessing capabilities.

## Features

- **#include directive**: Include other WGSL files with proper path resolution
- **#define directive**: Define constants and simple macros
- **Runtime preprocessing**: Process shaders at runtime with configurable defines and include paths
- **Compile-time preprocessing**: Process shaders at compile time using procedural macros
- **Circular include detection**: Prevents infinite recursion
- **Configurable include paths**: Support for multiple shader directories

## Usage

### Runtime Preprocessing

```rust
use wgsl_preprocessor::WgslPreprocessor;

let mut preprocessor = WgslPreprocessor::new();
preprocessor
    .define("WORKGROUP_SIZE", "64")
    .define("PI", "3.14159265359")
    .include_path("shaders/common");

let shader_source = r#"
#include "common.wgsl"
#define MAX_ITERATIONS 100

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main() {
    // shader code using PI and MAX_ITERATIONS
}
"#;

let processed = preprocessor.process_string(shader_source, None)?;
```

### Compile-time Preprocessing

```rust
use wgsl_preprocessor_macro::wgsl;

// Preprocess at compile time
const VERTEX_SHADER: &str = wgsl!("shaders/vertex.wgsl");
const FRAGMENT_SHADER: &str = wgsl!("shaders/fragment.wgsl");
```

## Include Directive

The `#include` directive supports both quoted and angle bracket syntax:

```wgsl
#include "common.wgsl"      // Relative to current file or include paths
#include <utils/math.wgsl>  // From include paths only
```

Include resolution order:
1. Relative to the current file's directory
2. Search through configured include paths in order

## Define Directive

The `#define` directive supports simple constant definitions:

```wgsl
#define PI 3.14159265359
#define WORKGROUP_SIZE 64
#define MAX_ITERATIONS 100

// Usage in shader code
let angle = PI * 2.0;
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn compute_main() {
    for (var i: u32 = 0u; i < MAX_ITERATIONS; i++) {
        // loop body
    }
}
```

## Conditional Compilation

The preprocessor supports conditional compilation with `#ifdef`, `#ifndef`, `#else`, and `#endif` directives:

```wgsl
#define DEBUG

#ifdef DEBUG
const LOG_LEVEL: u32 = 0u;  // Verbose logging
#else
const LOG_LEVEL: u32 = 2u;  // Error logging only
#endif

#ifndef RELEASE
const ENABLE_PROFILING: bool = true;
#endif

// Nested conditionals are supported
#ifdef PLATFORM_WEB
  #ifdef DEBUG
  const WEB_DEBUG_MODE: bool = true;
  #else
  const WEB_RELEASE_MODE: bool = true;
  #endif
#else
  #ifdef DEBUG
  const NATIVE_DEBUG_MODE: bool = true;
  #endif
#endif
```

### Conditional Features

- **#ifdef SYMBOL**: Include code if SYMBOL is defined
- **#ifndef SYMBOL**: Include code if SYMBOL is NOT defined  
- **#else**: Alternative branch for conditional blocks
- **#endif**: End conditional block
- **Nested conditionals**: Full support for nested conditional blocks
- **Error checking**: Validates matching #endif and prevents multiple #else

## Configuration

### Runtime Configuration

```rust
let mut preprocessor = WgslPreprocessor::new();

// Add global defines
preprocessor.define("DEBUG", "1");

// Add include search paths
preprocessor.include_path("shaders/common");
preprocessor.include_path("third_party/shaders");

// Set maximum include depth (default: 32)
preprocessor.max_include_depth(16);
```

## Examples

See the `examples/` directory for complete usage examples and the `test_shaders/` directory for shader examples demonstrating includes and defines.

Run the basic example:
```bash
cargo run --example basic_usage
```

Run tests:
```bash
cargo test
```

## Error Handling

The preprocessor provides detailed error information:

- `FileNotFound`: Include file could not be found
- `CircularInclude`: Detected circular include dependency
- `MaxIncludeDepthExceeded`: Include nesting too deep
- `FileReadError`: Failed to read include file
- `InvalidDefine`: Malformed define directive