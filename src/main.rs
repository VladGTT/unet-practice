mod data;
mod model;
mod training;
mod dataset;

use std::io::Cursor;
use std::path::Path;

// use crate::model::ModelConfig;
use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::data::dataset::transform::PartialDataset;
use burn::data::dataset::vision::{ImageDatasetItem, ImageFolderDataset, PixelDepth};
use burn::data::dataset::Dataset;
use burn::optim::AdamConfig;
use burn::tensor::{Data, Shape, Tensor};

use dataset::CustomDatasetItem;
use image::io::Reader as ImageReader;

fn list_files_in_directory(dir: &Path) -> std::io::Result<Vec<String>> {
    let mut file_list = Vec::new();

    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(file_name) = path.to_str() {
                    file_list.push(file_name.to_string());
                }
            }
        }
    }

    Ok(file_list)
}
fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    // let list = list_files_in_directory(Path::new("data/images")).unwrap();


    let dataset = crate::dataset::CustomDataset::load("data");

    println!("{:?}",dataset)


    // crate::training::train::<MyAutodiffBackend>(
    //     "/tmp/guide",
    //     crate::training::TrainingConfig::new(ModelConfig::new(1), AdamConfig::new()),
    //     device,
    // );
}
