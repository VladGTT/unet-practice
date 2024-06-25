mod data;
mod dataset;
mod model;
mod training;

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

    // test_model::<MyAutodiffBackend>(&device);

    crate::training::train::<MyAutodiffBackend>(
        "/tmp/guide",
        crate::training::TrainingConfig::new(ModelConfig::new(1),SgdConfig::new()),
        device,
    );
}
fn test_model<B:Backend<FloatElem = f32>>(device: &Device<B>){
    B::seed(42);
    let model: model::Model<B> = ModelConfig::new(1).init(device);
    
    let model = model.load("models/models.mpk", &device).expect("Model not loaded");

    let image = CustomImage::open("data/images/0.png").unwrap();
    let img_data = Data::new(image.into_bytes(), Shape::new([572,572,3]));
    let img_tensor = Tensor::<B, 3>::from_data(img_data.convert(), &device)
        .swap_dims(2, 1) // [H, C, W]
        .swap_dims(1, 0); // [C, H, W]

    
    // println!("{:?}",img_tensor.to_data().value);
    let input = img_tensor.reshape(Shape::new([1,3,572,572]));

    let output = model.forward(input);    
    println!("{:?}",output.to_data().value);
    let val: Vec<u8> = output.into_data().value.iter().map(|f|*f as u8).collect();
    
    image::save_buffer("data/result.png", &val,388,388, image::ExtendedColorType::L8).unwrap()
}