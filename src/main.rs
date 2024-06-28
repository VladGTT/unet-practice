mod data;
mod dataset;
mod model;
mod training;


use burn::backend::wgpu::Vulkan;
use burn::backend::{Autodiff, LibTorch, NdArray, Wgpu};
use burn::optim::SgdConfig;
use burn::tensor::backend::Backend;
use burn::tensor::{Data, Device, Shape, Tensor};
use dataset::CustomImage;
use model::ModelConfig;

fn main() {
    
    // type MyBackend = Wgpu<Vulkan, f32, i32>;
    type Backend = Autodiff<LibTorch>;
    let device = burn::backend::libtorch::LibTorchDevice::default();

    let optimizer = SgdConfig::new(); 

    let training_config = crate::training::TrainingConfig::new(ModelConfig::new(),optimizer)
        .with_learning_rate(0.1)
        .with_num_epochs(5)
        .with_margin(20); 
    
    crate::training::train::<Backend>(
        "temp/1",
        training_config,
        device,
    );

    println!("Testing saved model");
    test_model::<Backend>(&device.clone());
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

    let img_tensor = img_tensor.div(Tensor::<B,3,_>::full(Shape::new([3,572,572]), 255, &device));
    let input = img_tensor.reshape(Shape::new([1, 3, 572, 572]));

    let output = model.forward(input).reshape([1,388,388]);

    println!("Output: {}",output);
    let out_img = output.mul(Tensor::<B,3,_>::full(Shape::new([1,388,388]), 255, &device));
    let img = CustomImage::from_bytes(out_img.to_data().value.iter().map(|f|*f as u8).collect()).expect("Cant create image");
    img.save("data/result.tif").expect("Cant save image");
}
