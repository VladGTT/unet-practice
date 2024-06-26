mod data;
mod dataset;
mod model;
mod training;

use std::path::Path;

// use burn::backend::wgpu::*;
use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::optim::{AdamConfig, RmsPropConfig, SgdConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Data, Device, Shape, Tensor};
use dataset::CustomImage;
use model::ModelConfig;

fn main() {
    // type MyBackend = Wgpu<, f32, i32>;
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    
    let image = CustomImage::open("data/train/images/austin1.tif")
        .unwrap()
        .resize_with_mirroring();
    image.save("data/result.tif").expect("Cant save image");

    // test_model::<MyAutodiffBackend>(&device);

    // crate::training::train::<MyAutodiffBackend>(
    //     "/tmp/guide",
    //     crate::training::TrainingConfig::new(ModelConfig::new(),SgdConfig::new()),
    //     device,
    // );
}

#[allow(dead_code)]
fn test_model<B: Backend<FloatElem = f32>>(device: &Device<B>) {
    B::seed(42);
    let model: model::Model<B> = ModelConfig::new().init(device);
    let model = model
        .load("models/models.mpk", &device)
        .expect("Model not loaded");

    let image = CustomImage::open("data/train/images/austin1.tif")
        .unwrap()
        .resize_with_mirroring();
    let img_data = Data::new(image.into_bytes(), Shape::new([572, 572, 3]));
    let img_tensor = Tensor::<B, 3>::from_data(img_data.convert(), &device)
        .swap_dims(2, 1) // [H, C, W]
        .swap_dims(1, 0); // [C, H, W]

    let input = img_tensor.reshape(Shape::new([1, 3, 572, 572]));

    println!("{:?}", input.to_data().value);

    let output = model.forward(input);
    let val: Vec<u8> = output.into_data().value.iter().map(|f| *f as u8).collect();

    image::save_buffer(
        "data/result.tif",
        &val,
        388,
        388,
        image::ExtendedColorType::L8,
    )
    .unwrap()
}
