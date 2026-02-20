//! GPU dispatch utilities using wgpu.

use std::sync::Arc;

use bytemuck::cast_slice;
use futures_intrusive::channel::shared::oneshot_channel;
use wgpu::util::DeviceExt;

use crate::gpu::codegen::GpuProgram;

/// Executes WGSL programs on the GPU.
pub struct GpuExecutor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

#[derive(Debug, thiserror::Error)]
pub enum GpuExecutorError {
    #[error("no compatible GPU adapter detected")]
    NoAdapter,
    #[error("wgpu initialization failed: {0}")]
    Init(String),
    #[error("shader execution failed: {0}")]
    Execution(String),
}

impl GpuExecutor {
    pub async fn new() -> Result<Self, GpuExecutorError> {
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.ok_or(GpuExecutorError::NoAdapter)?;
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.map_err(|err| GpuExecutorError::Init(err.to_string()))?;
        Ok(Self { device: Arc::new(device), queue: Arc::new(queue) })
    }

    pub async fn execute(&self, program: &GpuProgram, inputs: &[f32], outputs: &mut [f32]) -> Result<(), GpuExecutorError> {
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("canon_shader"), source: wgpu::ShaderSource::Wgsl(program.shader.clone().into()) });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("canon_pipeline"), layout: None, module: &shader_module, entry_point: "main" });

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("gpu_inputs"), contents: bytemuck::cast_slice(inputs), usage: wgpu::BufferUsages::STORAGE });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_outputs"),
            size: (outputs.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_staging"),
            size: (outputs.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() }],
            label: Some("canon_bind_group"),
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (program.lanes + program.workgroup_size - 1) / program.workgroup_size;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, staging_buffer.size());
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (sender, receiver) = oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| sender.send(result).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        receiver.receive().await.ok_or_else(|| GpuExecutorError::Execution("map_async failed".into()))?.map_err(|_| GpuExecutorError::Execution("GPU mapping error".into()))?;

        {
            let data = slice.get_mapped_range();
            let floats: &[f32] = cast_slice(&data);
            outputs.copy_from_slice(floats);
        }
        staging_buffer.unmap();
        Ok(())
    }
}
