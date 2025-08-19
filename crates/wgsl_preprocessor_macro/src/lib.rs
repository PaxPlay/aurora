//! WGSL Preprocessor Macros
//!
//! Provides compile-time WGSL preprocessing using procedural macros.
//! The core functionality is provided by the `wgsl_preprocessor` crate.

use proc_macro::TokenStream;
use quote::quote;
use std::path::{self, Path, PathBuf};
use syn::{parse_macro_input, LitStr};
use wgsl_preprocessor::{PreprocessorError, ProcessedWgsl};

// Note: proc-macro crates cannot re-export items, so users must import
// wgsl_preprocessor directly for runtime preprocessing

/// Compile-time WGSL preprocessing macro
///
/// # Examples
///
/// ```
/// use wgsl_preprocessor_macro::wgsl;
///
/// const SHADER: &str = wgsl!("shader/vertex.wgsl");
/// ```
#[proc_macro]
pub fn wgsl(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as LitStr);
    let file_path = input.value();

    // Get the directory of the file that called this macro
    let calling_file = input
        .span()
        .local_file()
        .expect("Macro must be called from a file");
    let base_path = calling_file
        .parent()
        .expect("Calling file must have a parent directory");
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set");
    let crate_dir = PathBuf::from(crate_dir);

    // A list of sensible directories to search for the WGSL file
    // It might be reasonable for this to be configurable in the future
    // but I have genuinely no idea how to do that in a proc-macro
    let include_dir_candidates = [
        base_path,
        &base_path.join("shader"),
        &base_path.join("shaders"),
        &crate_dir,
        &crate_dir.join("shader"),
        &crate_dir.join("shaders"),
        &crate_dir.join("src"),
        &crate_dir.join("src").join("shader"),
        &crate_dir.join("src").join("shaders"),
    ];

    let full_path = search_candidate_dirs(&file_path, &include_dir_candidates).expect(&format!(
        "Failed to find WGSL file in any of the candidate directories, searched: {:?}",
        include_dir_candidates
    ));

    match process_file_compile_time(&full_path, &include_dir_candidates) {
        Ok(ProcessedWgsl {
            content,
            included_files,
        }) => {
            let included_files: Vec<String> = included_files
                .into_iter()
                .map(|p| {
                    // convert to absolute path, as `include_bytes!` uses path relative to the calling file
                    path::absolute(p)
                        .expect("Couldn't convert file path to absolute")
                        .to_string_lossy()
                        .to_string()
                })
                .collect();

            // Add a garbage include to ensure file changes are detected
            // This would be fixed by https://github.com/rust-lang/rust/issues/99515
            quote! {
                {
                    #( let _ = include_bytes!(#included_files); )*
                    #content
                }
            }
            .into()
        }
        Err(e) => {
            let error_msg = format!("WGSL preprocessing failed: {}", e);
            quote! {
                compile_error!(#error_msg)
            }
            .into()
        }
    }
}

fn search_candidate_dirs(file_path: &str, candidates: &[&Path]) -> Option<PathBuf> {
    for candidate in candidates {
        let candidate_path = candidate.join(file_path);
        if candidate_path.exists() {
            return Some(candidate_path);
        }
    }
    None
}

fn process_file_compile_time(
    path: &std::path::Path,
    candidates: &[&Path],
) -> Result<ProcessedWgsl, PreprocessorError> {
    let mut preprocessor = wgsl_preprocessor::WgslPreprocessor::new();
    for candidate in candidates {
        preprocessor.include_path(candidate);
    }
    preprocessor.process_file(path)
}
