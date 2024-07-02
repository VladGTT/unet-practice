mod data;
mod dataset;
mod model;
mod training;

use std::path::{Path, PathBuf};

use burn::backend::{Autodiff, LibTorch, NdArray, Wgpu};
use burn::config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::lr_scheduler::exponential::ExponentialLrSchedulerConfig;
use burn::lr_scheduler::linear::LinearLrSchedulerConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::momentum::MomentumConfig;
use burn::optim::{AdamConfig, RmsPropConfig, SgdConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Data, Device, Shape, Tensor};
use burn::train::ValidStep;
use data::DataBatcher;
use dataset::{CustomDataset, CustomImage};
use model::ModelConfig;
use training::TrainingConfig;

fn main() {
    type Backend = Autodiff<LibTorch>;
    let device = burn::backend::libtorch::LibTorchDevice::default();

    let training_config = crate::training::TrainingConfig::new(
        ModelConfig::new(),
        SgdConfig::new(),
        ExponentialLrSchedulerConfig::new(0.3, 1.0),
        (
            PathBuf::from("./data/train/images"),
            PathBuf::from("./data/train/masks"),
        ),
        (
            PathBuf::from("./data/test/images"),
            PathBuf::from("./data/test/masks"),
        ),
        PathBuf::from("./temp/1"),
    )
    .with_num_epochs(5)
    .with_margin(150);

    crate::training::train::<Backend>(training_config.clone(), device);
}

#[cfg(test)]
mod tests {
    use config::Config;

    use super::*;
    
    #[derive(Config)]
    struct TestConfig{
        pub model: ModelConfig,
    
    // In tuple first element corresponds to path to images, second one to masks
        pub test_data_path: (PathBuf,PathBuf), 
        pub artifact_dir: PathBuf,
        pub output_dir: PathBuf,

        #[config(default = 1)]
        pub num_epochs: usize,
        #[config(default = 1)]
        pub batch_size: usize,
        #[config(default = 4)]
        pub num_workers: usize,
        #[config(default = 42)]
        pub seed: u64,
    }

    #[test]
    fn test_model() {
        type Backend = LibTorch;
        let device = burn::backend::libtorch::LibTorchDevice::default();

        let config = TestConfig::new(
            ModelConfig::new(),
            (
                PathBuf::from("./data/test/images"),
                PathBuf::from("./data/test/masks"),
            ),
            PathBuf::from("./temp/1"),
            PathBuf::from("./results"),
        );

        Backend::seed(config.seed);

        let model: model::Model<Backend> = ModelConfig::new().init(&device);
        let model = model
            .load(config.artifact_dir.join("model.mpk").as_path(), &device)
            .expect("Model not loaded");

        let batcher_valid = DataBatcher::<Backend>::new(device.clone());

        println!("Loading dataset...");
        let dataset_test = CustomDataset::load(
            config.test_data_path.0.as_path(),
            config.test_data_path.1.as_path(),
        )       
        .expect("Cant load test data");

        println!("Dataset loaded");

        let dataloader_test = DataLoaderBuilder::new(batcher_valid)
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .num_workers(config.num_workers)
            .build(dataset_test);

        create_artifact_dir(&config.output_dir);
        for (iteration, batch) in dataloader_test.iter().enumerate() {
            println!("Iter-{iteration}");
            let output = ValidStep::step(&model, batch);

            let out_img = output
                .output
                .reshape([1, 388, 388])
                .mul(Tensor::<Backend, 3, _>::full(
                    Shape::new([1, 388, 388]),
                    255,
                    &device,
                ));
            let img =
                CustomImage::from_bytes(out_img.to_data().value.iter().map(|f| *f as u8).collect())
                    .expect("Cant create image");

            img.save(PathBuf::from(config.output_dir.join(format!("result-{iteration}.tif"))).as_path())
                .expect("Cant save image");
        }
    }
    fn create_artifact_dir(artifact_dir: &Path) {
        // Remove existing artifacts before to get an accurate learner summary
        std::fs::remove_dir_all(artifact_dir).ok();
        std::fs::create_dir_all(artifact_dir).ok();
    }
}
