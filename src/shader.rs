use std::collections::HashMap;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::fs;

use crate::GpuContext;

enum ShaderSource {
    Static(wgpu::ShaderSource<'static>),
    Dynamic {
        file: String
    }
}

impl ShaderSource {
    fn build_shader(&mut self, name: &str, device: &wgpu::Device) -> wgpu::ShaderModule {
        match self {
            ShaderSource::Static(source) => {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: source.clone()
                })
            }
            ShaderSource::Dynamic { file} => {
                Self::load_shader_source(name, file.as_str(), device)
                    .expect(format!("Failed to load the shader file \"{}\".", name).as_str())
            }
        }
    }

    fn load_shader_source(name: &str, file: &str, device: &wgpu::Device) -> std::io::Result<wgpu::ShaderModule> {
        let path = file; // maybe concatenate with some subdirectory
        let contents = fs::read_to_string(path)?;

        Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(contents.into())
        }))
    }
}

struct DynamicShaderModule {
    name: String,
    source: ShaderSource,
    module: Option<Arc<wgpu::ShaderModule>>,
    gpu: Arc<GpuContext>
}

impl DynamicShaderModule {
    fn new(name: String, source: ShaderSource, gpu: Arc<GpuContext>) -> Self {
        Self {
            name,
            source,
            module: None,
            gpu
        }
    }

    fn get_module(&mut self) -> Arc<wgpu::ShaderModule> {
        if self.module.is_none() {
            self.compile_shader();
        }

        match &self.module {
            None => panic!("Shader not available after compilation in get_module"),
            Some(module) => module.clone()
        }
    }

    fn compile_shader(&mut self) {
        self.module = Some(Arc::new(self.source.build_shader(self.name.as_str(),
                                                             &self.gpu.device)))
    }

    fn invalidate(&mut self) {
        self.module = None
    }
}

#[derive(Clone)]
pub struct DynamicShaderModuleHandle {
    dynamic_module: Rc<RefCell<DynamicShaderModule>>,
}

impl DynamicShaderModuleHandle {
    pub fn get_module(&self) -> Arc<wgpu::ShaderModule> {
        return self.dynamic_module.borrow_mut().get_module()
    }

    fn is_dynamic(&self) -> bool {
        match self.dynamic_module.borrow().source {
            ShaderSource::Static(_) => false,
            ShaderSource::Dynamic { .. } => true
        }
    }

    fn invalidate(&self) {
        self.dynamic_module.borrow_mut().invalidate()
    }
}

pub struct ShaderManager {
    shaders: HashMap<String, DynamicShaderModuleHandle>,
    gpu: Arc<GpuContext>
}

impl ShaderManager {
    pub fn new(gpu: Arc<GpuContext>) -> Self {
        Self {
            shaders: HashMap::new(),
            gpu
        }
    }

    pub fn get_shader(self, name: &str) -> Result<DynamicShaderModuleHandle, String> {
        match self.shaders.get(name) {
            Some(handle) => Ok(handle.clone()),
            None => {
                Err(format!("Specified shader \"{}\" does not exist", name).to_string())
            }
        }
    }
}

impl ShaderManager{
    fn add_module(&mut self, name: String, source: ShaderSource) {
        let handle = DynamicShaderModuleHandle {
            dynamic_module: Rc::new(RefCell::new(DynamicShaderModule::new(name.to_string(), source,
                                                                          self.gpu.clone())))
        };

        self.shaders.insert(name.to_string(), handle);
    }
    pub fn register_wgsl_static(&mut self, name: &str, source: wgpu::ShaderSource<'static>) {
        let source = ShaderSource::Static(source);

        self.add_module(name.to_string(), source);
    }
}

#[cfg(feature = "dynamic_shaders")]
impl ShaderManager  {
    pub fn register_wgsl(&mut self, name: String, file: String) {
        self.add_module(name, ShaderSource::Dynamic { file });
    }

    pub fn invalidate_dynamic_shaders(&mut self) {
        for (_, shader) in &self.shaders {
            if shader.is_dynamic() {
                shader.invalidate();
            }
        }
    }
}


type ShaderGetter = dyn Fn(String) -> Result<wgpu::ShaderModule, String>;
type PipelineConstructor = dyn Fn(&ShaderGetter) -> Result<wgpu::RenderPipelineDescriptor, String>;
struct PipelineHandle {
    constructor: &'static PipelineConstructor
}

impl PipelineHandle {
    pub fn new(constructor: &'static PipelineConstructor) -> Self {
        Self {
            constructor
        }
    }
}

/// Register a shader program from a relative path.
///
/// Depending on the run configuration, the shader will be retrieved dynamically or included
/// with the binary.
#[macro_export]
macro_rules! register_default {
    ($sm:expr, $name:expr, $file:expr) => {
        cfg_if::cfg_if! {
            if #[cfg(feature = "dynamic_shaders")] {
                $sm.register_wgsl(String::from($name), String::from($file))
            } else {
                $sm.register_wgsl_static($name,
                    wgpu::ShaderSource::Wgsl(include_str!(concat!("../", $file)).into())
                );
            }
        }
    };
}

