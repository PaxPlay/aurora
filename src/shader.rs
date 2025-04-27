use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::format;
use std::fs;
use std::sync::Arc;

use crate::GpuContext;

enum ShaderSource {
    Static(wgpu::ShaderSource<'static>),
    Dynamic { file: String },
}

impl ShaderSource {
    fn build_shader(&mut self, name: &str, device: &wgpu::Device) -> wgpu::ShaderModule {
        match self {
            ShaderSource::Static(source) => {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: source.clone(),
                })
            }
            ShaderSource::Dynamic { file } => Self::load_shader_source(name, file.as_str(), device)
                .expect(format!("Failed to load the shader file \"{}\".", name).as_str()),
        }
    }

    fn load_shader_source(
        name: &str,
        file: &str,
        device: &wgpu::Device,
    ) -> std::io::Result<wgpu::ShaderModule> {
        let path = file; // maybe concatenate with some subdirectory
        let contents = fs::read_to_string(path)?;

        Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(contents.into()),
        }))
    }
}

struct DynamicShaderModule {
    name: String,
    source: ShaderSource,
    module: Option<wgpu::ShaderModule>,
    device: wgpu::Device,
}

impl DynamicShaderModule {
    fn new(name: String, source: ShaderSource, device: wgpu::Device) -> Self {
        Self {
            name,
            source,
            module: None,
            device,
        }
    }

    fn get_module(&mut self) -> wgpu::ShaderModule {
        if self.module.is_none() {
            self.compile_shader();
        }

        match &self.module {
            None => panic!("Shader not available after compilation in get_module"),
            Some(module) => module.clone(),
        }
    }

    fn compile_shader(&mut self) {
        self.module = Some(self.source.build_shader(self.name.as_str(), &self.device))
    }

    fn invalidate(&mut self) {
        self.module = None
    }
}

#[derive(Clone)]
pub struct DynamicShaderModuleHandle {
    dynamic_module: Arc<RefCell<DynamicShaderModule>>,
}

impl DynamicShaderModuleHandle {
    pub fn get_module(&self) -> wgpu::ShaderModule {
        return self.dynamic_module.borrow_mut().get_module();
    }

    fn is_dynamic(&self) -> bool {
        match self.dynamic_module.borrow().source {
            ShaderSource::Static(_) => false,
            ShaderSource::Dynamic { .. } => true,
        }
    }

    fn invalidate(&self) {
        self.dynamic_module.borrow_mut().invalidate()
    }
}

pub struct ShaderManager {
    shaders: RefCell<HashMap<String, DynamicShaderModuleHandle>>,
    device: wgpu::Device,
}

impl ShaderManager {
    pub fn new(device: wgpu::Device) -> Self {
        Self {
            shaders: RefCell::new(HashMap::new()),
            device,
        }
    }

    pub fn get_shader(&self, name: &str) -> Result<DynamicShaderModuleHandle, String> {
        match self.shaders.borrow().get(name) {
            Some(handle) => Ok(handle.clone()),
            None => Err(format!("Specified shader \"{}\" does not exist", name).to_string()),
        }
    }
}

impl ShaderManager {
    fn add_module(&self, name: String, source: ShaderSource) {
        let handle = DynamicShaderModuleHandle {
            dynamic_module: Arc::new(RefCell::new(DynamicShaderModule::new(
                name.to_string(),
                source,
                self.device.clone(),
            ))),
        };

        self.shaders.borrow_mut().insert(name.to_string(), handle);
    }
    pub fn register_wgsl_static(&self, name: &str, source: wgpu::ShaderSource<'static>) {
        let source = ShaderSource::Static(source);

        self.add_module(name.to_string(), source);
    }
}

#[cfg(feature = "dynamic_shaders")]
impl ShaderManager {
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

type ShaderGetter = dyn Fn(&str) -> Result<wgpu::ShaderModule, String>;
type RenderPipelineConstructor =
    dyn Fn(&GpuContext, &ShaderGetter) -> Result<wgpu::RenderPipeline, String>;

pub struct RenderPipeline {
    gpu: Arc<GpuContext>,
    constructor: Box<RenderPipelineConstructor>,
    pipeline: Option<wgpu::RenderPipeline>,
}

impl RenderPipeline {
    pub fn new<T>(gpu: Arc<GpuContext>, constructor: T) -> Self
    where
        T: Fn(&GpuContext, &ShaderGetter) -> Result<wgpu::RenderPipeline, String> + 'static,
    {
        Self {
            gpu,
            constructor: Box::new(constructor),
            pipeline: None,
        }
    }
    fn build(&self) -> wgpu::RenderPipeline {
        let gpu = self.gpu.clone();
        let sg: Box<ShaderGetter> = Box::new(move |shader_name| {
            gpu.shaders
                .get_shader(shader_name)
                .clone()
                .map(|s| s.get_module())
        });
        (self.constructor)(&self.gpu, &sg).unwrap()
    }

    pub fn get(&mut self) -> wgpu::RenderPipeline {
        if self.pipeline.is_none() {
            self.pipeline = Some(self.build());
        }

        self.pipeline
            .as_ref()
            .expect("Pipeline not available???")
            .clone()
    }
}

#[derive(Clone)]
struct BindGroupEntry {
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BindingType,
}

pub struct BindGroupLayoutBuilder {
    gpu: Arc<GpuContext>,
    label: Option<String>,
    bindings: Vec<BindGroupEntry>,
    layout: Option<wgpu::BindGroupLayout>,
}

impl BindGroupLayoutBuilder {
    pub fn new(gpu: Arc<GpuContext>) -> Self {
        Self {
            gpu,
            label: None,
            bindings: Vec::new(),
            layout: None,
        }
    }

    pub fn add_buffer(
        mut self,
        binding: u32,
        visibility: wgpu::ShaderStages,
        ty: wgpu::BufferBindingType,
    ) -> Self {
        self.bindings.push(BindGroupEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        });

        self
    }

    pub fn label(mut self, l: &str) -> Self {
        self.label = Some(String::from(l));
        self
    }

    pub fn build(self) -> BindGroupLayout {
        let entries: Vec<_> = self
            .bindings
            .iter()
            .map(
                |BindGroupEntry {
                     binding,
                     visibility,
                     ty,
                 }| wgpu::BindGroupLayoutEntry {
                    binding: *binding,
                    visibility: *visibility,
                    ty: *ty,
                    count: None,
                },
            )
            .collect();

        let layout = self
            .gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: self.label.as_ref().map(|s| s.as_str()),
                entries: &entries,
            });
        BindGroupLayout {
            gpu: self.gpu,
            bindings: self.bindings,
            layout,
        }
    }
}

pub struct BindGroupLayout {
    gpu: Arc<GpuContext>,
    bindings: Vec<BindGroupEntry>,
    layout: wgpu::BindGroupLayout,
}

impl BindGroupLayout {
    pub fn bind_group_builder(&self) -> BindGroupBuilder {
        BindGroupBuilder {
            gpu: self.gpu.clone(),
            bindings: self.bindings.clone(),
            layout: self.layout.clone(),
            label: None,
            buffers: Vec::new(),
        }
    }

    pub fn get(&self) -> wgpu::BindGroupLayout {
        self.layout.clone()
    }
}

pub struct BindGroupBuilder {
    gpu: Arc<GpuContext>,
    bindings: Vec<BindGroupEntry>,
    layout: wgpu::BindGroupLayout,
    label: Option<String>,
    buffers: Vec<(u32, wgpu::Buffer)>,
}

impl BindGroupBuilder {
    pub fn label(mut self, l: &str) -> Self {
        self.label = Some(String::from(l));
        self
    }

    pub fn buffer(mut self, binding: u32, b: wgpu::Buffer) -> Self {
        self.buffers.push((binding, b));
        self
    }

    pub fn build(self) -> Result<wgpu::BindGroup, String> {
        let bindings: Vec<Result<_, String>> = self
            .bindings
            .iter()
            .map(|entry| {
                Ok(wgpu::BindGroupEntry {
                    binding: entry.binding,
                    resource: match entry.ty {
                        wgpu::BindingType::Buffer { .. } => {
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &self
                                    .buffers
                                    .iter()
                                    .filter(|(binding, _)| *binding == entry.binding)
                                    .next()
                                    .ok_or(format!(
                                        "Buffer type binding {} was not supplied",
                                        entry.binding
                                    ))?
                                    .1,
                                offset: 0,
                                size: None,
                            })
                        }
                        _ => Err(format!("Binding type {:?} not supported", entry.ty))?,
                    },
                })
            })
            .collect();

        let bindings: Result<Vec<wgpu::BindGroupEntry>, _> = bindings.into_iter().collect();

        Ok(self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: self.label.as_ref().map(|s| s.as_str()),
                layout: &self.layout,
                entries: &bindings?,
            }))
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

/// Create a PipelineHandle that can build a pipeline using dynamically provided shaders.
#[macro_export]
macro_rules! render_pipeline {
    ($gpu:expr, $($s:ident),+; $desc:expr) => {
        RenderPipeline::new($gpu.clone(), move |gpu, get_shader| {
            $(
                let $s = get_shader(stringify!($s))?;
            )+

            Ok(gpu.device.create_render_pipeline($desc))
        })
    };
}
