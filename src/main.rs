mod training;
mod data;
mod model;
 
use burn::optim::AdamConfig;
use burn::backend::{Autodiff, Wgpu, wgpu::AutoGraphicsApi};
use crate::model::ModelConfig;

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    crate::training::train::<MyAutodiffBackend>(
        "/tmp/guide",
        crate::training::TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device,
    );
}