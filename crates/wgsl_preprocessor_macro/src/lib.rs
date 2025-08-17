//! WGSL Preprocessor Macros
//! 
//! Provides compile-time WGSL preprocessing using procedural macros.
//! The core functionality is provided by the `wgsl_preprocessor` crate.

use proc_macro::TokenStream;
use quote::quote;
use std::path::PathBuf;
use syn::{parse_macro_input, LitStr};

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
    let calling_file = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());
    let base_path = PathBuf::from(calling_file);
    let full_path = base_path.join(&file_path);
    
    match process_file_compile_time(&full_path) {
        Ok(processed) => {
            quote! {
                #processed
            }.into()
        },
        Err(e) => {
            let error_msg = format!("WGSL preprocessing failed: {}", e);
            quote! {
                compile_error!(#error_msg)
            }.into()
        }
    }
}

fn process_file_compile_time(path: &std::path::Path) -> Result<String, wgsl_preprocessor::PreprocessorError> {
    let preprocessor = wgsl_preprocessor::WgslPreprocessor::new();
    preprocessor.process_file(path)
}