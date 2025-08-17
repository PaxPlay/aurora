# WGSL Preprocessor Macros

Procedural macros for compile-time WGSL preprocessing. This crate provides the `wgsl!` macro for including and preprocessing WGSL shaders at compile time.

## Usage

Add both crates to your `Cargo.toml`:

```toml
[dependencies]
wgsl_preprocessor = { path = "../wgsl_preprocessor" }  # For runtime preprocessing
wgsl_preprocessor_macro = { path = "../wgsl_preprocessor_macro" }  # For compile-time macros
```

### Compile-time Preprocessing

```rust
use wgsl_preprocessor_macro::wgsl;

// Preprocess shader at compile time
const VERTEX_SHADER: &str = wgsl!("shaders/vertex.wgsl");

// The macro will:
// 1. Read the file at compile time
// 2. Process #include and #define directives
// 3. Embed the processed result as a string literal
```

### Example Shader with Includes

**shaders/common.wgsl:**
```wgsl
#define PI 3.14159265359
#define TWO_PI 6.28318530718

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}
```

**shaders/vertex.wgsl:**
```wgsl
#include "common.wgsl"

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let angle = f32(vertex_index) * PI / 4.0;
    // ... rest of shader
}
```

**Rust code:**
```rust
use wgsl_preprocessor_macro::wgsl;

const VERTEX_SHADER: &str = wgsl!("shaders/vertex.wgsl");
// At compile time, this becomes the fully processed shader with includes resolved
```

## Features

- **Zero runtime cost**: All preprocessing happens at compile time
- **#include support**: Include other WGSL files with proper path resolution
- **#define support**: Simple constant definitions
- **Error reporting**: Clear compile-time errors for preprocessing issues

## Notes

- The macro expects file paths relative to the crate root (`CARGO_MANIFEST_DIR`)
- Include files are resolved relative to the including file or from configured search paths
- Proc-macro crates cannot re-export regular items, so you need to import `wgsl_preprocessor` separately for runtime preprocessing