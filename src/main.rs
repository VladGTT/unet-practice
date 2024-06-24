mod data;
mod model;
mod training;
mod dataset;

use std::io::Cursor;
use std::path::Path;

// use crate::model::ModelConfig;
use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::data::dataset::vision::{ImageDatasetItem, ImageFolderDataset, PixelDepth};
use burn::optim::{AdamConfig, RmsPropConfig};
use burn::tensor::{Data, Shape, Tensor};

use model::ModelConfig;


fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    // let list = list_files_in_directory(Path::new("data/images")).unwrap();


    // let dataset = crate::dataset::CustomDataset::load("data");

    // println!("{:?}",dataset)

    crate::training::train::<MyAutodiffBackend>(
        "/tmp/guide",
        crate::training::TrainingConfig::new(ModelConfig::new(1),RmsPropConfig::new()),
        device,
    );
}
