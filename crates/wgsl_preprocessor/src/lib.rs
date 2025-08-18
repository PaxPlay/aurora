//! WGSL Preprocessor
//!
//! A preprocessor for WGSL (WebGPU Shading Language) that supports #import and #define directives
//! similar to C preprocessor. This is the main library crate providing runtime preprocessing.

use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur during preprocessing
#[derive(Error, Debug)]
pub enum PreprocessorError {
    #[error("Failed to read file: {path}")]
    FileReadError { path: String },
    #[error("Include depth exceeded maximum of {max}")]
    MaxIncludeDepthExceeded { max: usize },
    #[error("Circular include detected: {file}")]
    CircularInclude { file: String },
    #[error("Invalid define syntax: {line}")]
    InvalidDefine { line: String },
    #[error("File not found: {path}")]
    FileNotFound { path: String },
    #[error("Unmatched #endif directive")]
    UnmatchedEndif,
    #[error("Unmatched #else directive")]
    UnmatchedElse,
    #[error("Missing #endif directive")]
    MissingEndif,
    #[error("Invalid conditional directive: {line}")]
    InvalidConditional { line: String },
    #[error("Multiple #else directives in same conditional block")]
    MultipleElse,
}

/// Represents the state of a conditional compilation block
#[derive(Debug, Clone)]
struct ConditionalState {
    /// Whether this block should include content
    should_include: bool,
    /// Whether we've seen an #else in this block
    has_else: bool,
    /// Whether any branch in this conditional has been taken
    any_branch_taken: bool,
}

/// Runtime WGSL preprocessor
pub struct WgslPreprocessor {
    defines: HashMap<String, String>,
    include_paths: Vec<PathBuf>,
    max_include_depth: usize,
}

pub struct ProcessedWgsl {
    pub content: String,
    pub included_files: Vec<PathBuf>,
}

impl Default for WgslPreprocessor {
    fn default() -> Self {
        Self {
            defines: HashMap::new(),
            include_paths: vec![PathBuf::from(".")],
            max_include_depth: 32,
        }
    }
}

impl WgslPreprocessor {
    /// Create a new preprocessor
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a define directive
    pub fn define(&mut self, name: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.defines.insert(name.into(), value.into());
        self
    }

    /// Add an include search path
    pub fn include_path(&mut self, path: impl Into<PathBuf>) -> &mut Self {
        self.include_paths.push(path.into());
        self
    }

    /// Set maximum include depth to prevent infinite recursion
    pub fn max_include_depth(&mut self, depth: usize) -> &mut Self {
        self.max_include_depth = depth;
        self
    }

    /// Process a WGSL string with preprocessing
    pub fn process_string(
        &self,
        input: &str,
        base_path: Option<&Path>,
    ) -> Result<ProcessedWgsl, PreprocessorError> {
        let mut include_stack = Vec::new();
        let mut included_files = Vec::new();
        self.process_recursive(input, base_path, &mut include_stack, &mut included_files, 0)
            .map(|result| ProcessedWgsl {
                content: result,
                included_files,
            })
    }

    /// Process a WGSL file with preprocessing
    pub fn process_file(&self, path: impl AsRef<Path>) -> Result<ProcessedWgsl, PreprocessorError> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|_| PreprocessorError::FileReadError {
            path: path.display().to_string(),
        })?;

        let mut include_stack = Vec::new();
        let mut included_files = vec![path.to_path_buf()];
        let base_path = path.parent();
        self.process_recursive(
            &content,
            base_path,
            &mut include_stack,
            &mut included_files,
            0,
        )
        .map(|result| ProcessedWgsl {
            content: result,
            included_files,
        })
    }

    fn process_recursive(
        &self,
        input: &str,
        base_path: Option<&Path>,
        include_stack: &mut Vec<PathBuf>,
        included_files: &mut Vec<PathBuf>,
        depth: usize,
    ) -> Result<String, PreprocessorError> {
        if depth >= self.max_include_depth {
            return Err(PreprocessorError::MaxIncludeDepthExceeded {
                max: self.max_include_depth,
            });
        }

        let mut result = String::new();
        let mut local_defines = self.defines.clone();
        let mut conditional_stack: Vec<ConditionalState> = Vec::new();

        for line in input.lines() {
            let trimmed = line.trim();

            if let Some(processed_line) = self.process_line(
                trimmed,
                &mut local_defines,
                base_path,
                include_stack,
                included_files,
                depth,
                &mut conditional_stack,
            )? {
                result.push_str(&processed_line);
                result.push('\n');
            }
        }

        // Check for unmatched conditionals
        if !conditional_stack.is_empty() {
            return Err(PreprocessorError::MissingEndif);
        }

        Ok(result)
    }

    fn process_line(
        &self,
        line: &str,
        local_defines: &mut HashMap<String, String>,
        base_path: Option<&Path>,
        include_stack: &mut Vec<PathBuf>,
        included_files: &mut Vec<PathBuf>,
        depth: usize,
        conditional_stack: &mut Vec<ConditionalState>,
    ) -> Result<Option<String>, PreprocessorError> {
        // Check if we should include this line based on current conditional state
        let should_include_line = self.should_include_line(conditional_stack);

        // Process conditional directives regardless of current state
        if line.starts_with("#ifdef") {
            return self.process_ifdef(line, local_defines, conditional_stack);
        }

        if line.starts_with("#ifndef") {
            return self.process_ifndef(line, local_defines, conditional_stack);
        }

        if line.starts_with("#else") {
            return self.process_else(conditional_stack);
        }

        if line.starts_with("#endif") {
            return self.process_endif(conditional_stack);
        }

        // If we're in a conditional block that shouldn't include content, skip non-conditional directives
        if !should_include_line {
            return Ok(None);
        }

        // Process other directives only if we should include this line
        if line.starts_with("#import") {
            return self.process_import(line, base_path, include_stack, included_files, depth);
        }

        if line.starts_with("#define") {
            self.process_define(line, local_defines)?;
            return Ok(None); // Don't include define directives in output
        }

        // Skip other preprocessor directives for now
        if line.starts_with('#') {
            return Ok(Some(line.to_string()));
        }

        // Apply defines to the line
        Ok(Some(self.apply_defines(line, local_defines)))
    }

    fn process_import(
        &self,
        line: &str,
        base_path: Option<&Path>,
        include_stack: &mut Vec<PathBuf>,
        included_files: &mut Vec<PathBuf>,
        depth: usize,
    ) -> Result<Option<String>, PreprocessorError> {
        let include_regex = Regex::new(r#"#import\s+[<"]([^">]+)[">]"#).unwrap();

        if let Some(captures) = include_regex.captures(line) {
            let filename = &captures[1];
            let file_path = self.resolve_include_path(filename, base_path)?;

            // Check for circular includes
            if included_files.contains(&file_path) {
                return Err(PreprocessorError::CircularInclude {
                    file: file_path.display().to_string(),
                });
            }

            let content =
                fs::read_to_string(&file_path).map_err(|_| PreprocessorError::FileReadError {
                    path: file_path.display().to_string(),
                })?;

            included_files.push(file_path.clone());
            include_stack.push(file_path.clone());
            let processed_content = self.process_recursive(
                &content,
                file_path.parent(),
                include_stack,
                included_files,
                depth + 1,
            )?;
            include_stack.pop();

            return Ok(Some(processed_content));
        }

        Err(PreprocessorError::InvalidDefine {
            line: line.to_string(),
        })
    }

    fn process_define(
        &self,
        line: &str,
        local_defines: &mut HashMap<String, String>,
    ) -> Result<(), PreprocessorError> {
        let define_regex = Regex::new(r"#define\s+(\w+)(?:\s+(.*))?").unwrap();

        if let Some(captures) = define_regex.captures(line) {
            let name = captures[1].to_string();
            let value = captures
                .get(2)
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_else(|| "1".to_string());

            local_defines.insert(name, value);
            return Ok(());
        }

        Err(PreprocessorError::InvalidDefine {
            line: line.to_string(),
        })
    }

    fn apply_defines(&self, line: &str, defines: &HashMap<String, String>) -> String {
        let mut result = line.to_string();

        // Sort by length descending to replace longer identifiers first
        let mut sorted_defines: Vec<_> = defines.iter().collect();
        sorted_defines.sort_by_key(|(k, _)| std::cmp::Reverse(k.len()));

        for (name, value) in sorted_defines {
            // Use word boundaries to avoid partial replacements
            let pattern = format!(r"\b{}\b", regex::escape(name));
            if let Ok(regex) = Regex::new(&pattern) {
                result = regex.replace_all(&result, value.as_str()).to_string();
            }
        }

        result
    }

    fn resolve_include_path(
        &self,
        filename: &str,
        base_path: Option<&Path>,
    ) -> Result<PathBuf, PreprocessorError> {
        // Try relative to current file first
        if let Some(base) = base_path {
            let path = base.join(filename);
            if path.exists() {
                return Ok(path);
            }
        }

        // Try include paths
        for include_path in &self.include_paths {
            let path = include_path.join(filename);
            if path.exists() {
                return Ok(path);
            }
        }

        Err(PreprocessorError::FileNotFound {
            path: filename.to_string(),
        })
    }

    /// Check if we should include content based on current conditional stack
    fn should_include_line(&self, conditional_stack: &[ConditionalState]) -> bool {
        // If stack is empty, always include
        if conditional_stack.is_empty() {
            return true;
        }

        // All conditions in the stack must be true to include the line
        conditional_stack.iter().all(|state| state.should_include)
    }

    /// Process #ifdef directive
    fn process_ifdef(
        &self,
        line: &str,
        local_defines: &HashMap<String, String>,
        conditional_stack: &mut Vec<ConditionalState>,
    ) -> Result<Option<String>, PreprocessorError> {
        let ifdef_regex = Regex::new(r"#ifdef\s+(\w+)").unwrap();

        if let Some(captures) = ifdef_regex.captures(line) {
            let define_name = &captures[1];
            let is_defined = local_defines.contains_key(define_name);

            conditional_stack.push(ConditionalState {
                should_include: is_defined,
                has_else: false,
                any_branch_taken: is_defined,
            });

            return Ok(None); // Don't include the directive in output
        }

        Err(PreprocessorError::InvalidConditional {
            line: line.to_string(),
        })
    }

    /// Process #ifndef directive
    fn process_ifndef(
        &self,
        line: &str,
        local_defines: &HashMap<String, String>,
        conditional_stack: &mut Vec<ConditionalState>,
    ) -> Result<Option<String>, PreprocessorError> {
        let ifndef_regex = Regex::new(r"#ifndef\s+(\w+)").unwrap();

        if let Some(captures) = ifndef_regex.captures(line) {
            let define_name = &captures[1];
            let is_not_defined = !local_defines.contains_key(define_name);

            conditional_stack.push(ConditionalState {
                should_include: is_not_defined,
                has_else: false,
                any_branch_taken: is_not_defined,
            });

            return Ok(None); // Don't include the directive in output
        }

        Err(PreprocessorError::InvalidConditional {
            line: line.to_string(),
        })
    }

    /// Process #else directive
    fn process_else(
        &self,
        conditional_stack: &mut Vec<ConditionalState>,
    ) -> Result<Option<String>, PreprocessorError> {
        if let Some(current_state) = conditional_stack.last_mut() {
            if current_state.has_else {
                return Err(PreprocessorError::MultipleElse);
            }

            current_state.has_else = true;
            // Include else branch only if no previous branch was taken
            current_state.should_include = !current_state.any_branch_taken;

            if current_state.should_include {
                current_state.any_branch_taken = true;
            }

            return Ok(None); // Don't include the directive in output
        }

        Err(PreprocessorError::UnmatchedElse)
    }

    /// Process #endif directive
    fn process_endif(
        &self,
        conditional_stack: &mut Vec<ConditionalState>,
    ) -> Result<Option<String>, PreprocessorError> {
        if conditional_stack.pop().is_some() {
            return Ok(None); // Don't include the directive in output
        }

        Err(PreprocessorError::UnmatchedEndif)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_basic_define() {
        let mut preprocessor = WgslPreprocessor::new();
        preprocessor.define("PI", "3.14159");

        let input = "let radius: f32 = PI * 2.0;";
        let result = preprocessor.process_string(input, None).unwrap();

        assert_eq!(result, "let radius: f32 = 3.14159 * 2.0;\n");
    }

    #[test]
    fn test_basic_include() {
        let temp_dir = TempDir::new().unwrap();
        let include_file = temp_dir.path().join("constants.wgsl");
        fs::write(&include_file, "const PI: f32 = 3.14159;").unwrap();

        let mut preprocessor = WgslPreprocessor::new();
        preprocessor.include_path(temp_dir.path());

        let input = "#include \"constants.wgsl\"\nlet radius: f32 = PI * 2.0;";
        let result = preprocessor.process_string(input, None).unwrap();

        assert!(result.contains("const PI: f32 = 3.14159;"));
        assert!(result.contains("let radius: f32 = PI * 2.0;"));
    }

    #[test]
    fn test_circular_include() {
        let temp_dir = TempDir::new().unwrap();
        let file_a = temp_dir.path().join("a.wgsl");
        let file_b = temp_dir.path().join("b.wgsl");

        fs::write(&file_a, "#include \"b.wgsl\"").unwrap();
        fs::write(&file_b, "#include \"a.wgsl\"").unwrap();

        let mut preprocessor = WgslPreprocessor::new();
        preprocessor.include_path(temp_dir.path());

        let result = preprocessor.process_file(&file_a);
        assert!(matches!(
            result,
            Err(PreprocessorError::CircularInclude { .. })
        ));
    }

    #[test]
    fn test_ifdef_defined() {
        let mut preprocessor = WgslPreprocessor::new();
        preprocessor.define("DEBUG", "1");

        let input = r#"
#ifdef DEBUG
const debug_enabled: bool = true;
#endif
fn main() {}
"#;
        let result = preprocessor.process_string(input, None).unwrap();

        assert!(result.contains("const debug_enabled: bool = true;"));
        assert!(result.contains("fn main() {}"));
    }

    #[test]
    fn test_ifdef_not_defined() {
        let preprocessor = WgslPreprocessor::new();

        let input = r#"
#ifdef DEBUG
const debug_enabled: bool = true;
#endif
fn main() {}
"#;
        let result = preprocessor.process_string(input, None).unwrap();

        assert!(!result.contains("const debug_enabled: bool = true;"));
        assert!(result.contains("fn main() {}"));
    }

    #[test]
    fn test_ifndef_not_defined() {
        let preprocessor = WgslPreprocessor::new();

        let input = r#"
#ifndef RELEASE
const debug_mode: bool = true;
#endif
fn main() {}
"#;
        let result = preprocessor.process_string(input, None).unwrap();

        assert!(result.contains("const debug_mode: bool = true;"));
        assert!(result.contains("fn main() {}"));
    }

    #[test]
    fn test_ifndef_defined() {
        let mut preprocessor = WgslPreprocessor::new();
        preprocessor.define("RELEASE", "1");

        let input = r#"
#ifndef RELEASE
const debug_mode: bool = true;
#endif
fn main() {}
"#;
        let result = preprocessor.process_string(input, None).unwrap();

        assert!(!result.contains("const debug_mode: bool = true;"));
        assert!(result.contains("fn main() {}"));
    }

    #[test]
    fn test_ifdef_else_defined() {
        let mut preprocessor = WgslPreprocessor::new();
        preprocessor.define("DEBUG", "1");

        let input = r#"
#ifdef DEBUG
const mode: u32 = 0u;
#else
const mode: u32 = 1u;
#endif
"#;
        let result = preprocessor.process_string(input, None).unwrap();

        assert!(result.contains("const mode: u32 = 0u;"));
        assert!(!result.contains("const mode: u32 = 1u;"));
    }

    #[test]
    fn test_ifdef_else_not_defined() {
        let preprocessor = WgslPreprocessor::new();

        let input = r#"
#ifdef DEBUG
const mode: u32 = 0u;
#else
const mode: u32 = 1u;
#endif
"#;
        let result = preprocessor.process_string(input, None).unwrap();

        assert!(!result.contains("const mode: u32 = 0u;"));
        assert!(result.contains("const mode: u32 = 1u;"));
    }

    #[test]
    fn test_nested_conditionals() {
        let mut preprocessor = WgslPreprocessor::new();
        preprocessor.define("PLATFORM_WEB", "1");
        preprocessor.define("DEBUG", "1");

        let input = r#"
#ifdef PLATFORM_WEB
  #ifdef DEBUG
  const web_debug: bool = true;
  #else
  const web_release: bool = true;
  #endif
#else
  #ifdef DEBUG
  const native_debug: bool = true;
  #else
  const native_release: bool = true;
  #endif
#endif
"#;
        let result = preprocessor.process_string(input, None).unwrap();

        assert!(result.contains("const web_debug: bool = true;"));
        assert!(!result.contains("const web_release: bool = true;"));
        assert!(!result.contains("const native_debug: bool = true;"));
        assert!(!result.contains("const native_release: bool = true;"));
    }

    #[test]
    fn test_unmatched_endif() {
        let preprocessor = WgslPreprocessor::new();

        let input = r#"
fn main() {}
#endif
"#;
        let result = preprocessor.process_string(input, None);
        assert!(matches!(result, Err(PreprocessorError::UnmatchedEndif)));
    }

    #[test]
    fn test_missing_endif() {
        let preprocessor = WgslPreprocessor::new();

        let input = r#"
#ifdef DEBUG
const debug: bool = true;
"#;
        let result = preprocessor.process_string(input, None);
        assert!(matches!(result, Err(PreprocessorError::MissingEndif)));
    }

    #[test]
    fn test_unmatched_else() {
        let preprocessor = WgslPreprocessor::new();

        let input = r#"
fn main() {}
#else
const debug: bool = true;
"#;
        let result = preprocessor.process_string(input, None);
        assert!(matches!(result, Err(PreprocessorError::UnmatchedElse)));
    }

    #[test]
    fn test_multiple_else() {
        let preprocessor = WgslPreprocessor::new();

        let input = r#"
#ifdef DEBUG
const debug: bool = true;
#else
const release1: bool = true;
#else
const release2: bool = true;
#endif
"#;
        let result = preprocessor.process_string(input, None);
        assert!(matches!(result, Err(PreprocessorError::MultipleElse)));
    }

    #[test]
    fn test_conditional_with_includes() {
        let temp_dir = TempDir::new().unwrap();
        let debug_file = temp_dir.path().join("debug.wgsl");
        let release_file = temp_dir.path().join("release.wgsl");

        fs::write(&debug_file, "const LOG_LEVEL: u32 = 0u;").unwrap();
        fs::write(&release_file, "const LOG_LEVEL: u32 = 3u;").unwrap();

        let mut preprocessor = WgslPreprocessor::new();
        preprocessor.include_path(temp_dir.path());
        preprocessor.define("DEBUG", "1");

        let input = r#"
#ifdef DEBUG
#import "debug.wgsl"
#else
#import "release.wgsl"
#endif
fn main() {}
"#;
        let result = preprocessor.process_string(input, None).unwrap();

        assert!(result.content.contains("const LOG_LEVEL: u32 = 0u;"));
        assert!(!result.content.contains("const LOG_LEVEL: u32 = 3u;"));
        assert!(result.content.contains("fn main() {}"));
    }
}
