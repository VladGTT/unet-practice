use std::{io::Write, path::{Path, PathBuf}, thread::sleep, time::Duration};

use burn::{config::Config, data::{dataloader::DataLoaderBuilder, dataset::transform::PartialDataset}, lr_scheduler::{exponential::ExponentialLrSchedulerConfig, linear::LinearLrSchedulerConfig, LrScheduler}, nn::loss::BinaryCrossEntropyLossConfig, optim::{AdamConfig, GradientsParams, Optimizer, RmsPropConfig, SgdConfig}, record::CompactRecorder, tensor::backend::AutodiffBackend, train::{logger::FileMetricLogger, metric::{store::Aggregate, CpuMemory, CpuTemperature, LearningRateMetric, LossMetric}, EarlyStoppingStrategy, LearnerBuilder, MetricEarlyStoppingStrategy, TrainStep, ValidStep}};

use crate::{data::DataBatcher, dataset::CustomDataset, model::{Model, ModelConfig}};

// impl<B: Backend> Model<B> {
//     pub fn forward_classification(
//         &self,
//         images: Tensor<B, 3>,
//         targets: Tensor<B, 1, Int>,
//     ) -> ClassificationOutput<B> {
//         let output = self.forward(images);
//         let loss = CrossEntropyLoss::new(None, &output.device()).forward(output.clone(), targets.clone());

//         ClassificationOutput::new(loss, output, targets)
//     }
// }

// impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
//     fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
//         let item = self.forward_classification(batch.images, batch.targets);

//         TrainOutput::new(self, item.loss.backward(), item)
//     }
// }

// impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
//     fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
//         self.forward_classification(batch.images, batch.targets)
//     }
// }

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: SgdConfig,
    pub learning_rate: ExponentialLrSchedulerConfig, 
    
    // In tuple first element corresponds to path to images, second one to masks
    pub train_data_path: (PathBuf,PathBuf),
    pub test_data_path: (PathBuf,PathBuf), 
    pub artifact_dir: PathBuf,

    #[config(default = 1)]
    pub num_epochs: usize,
    #[config(default = 1)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 10)]
    pub stop_num_epoch: usize,

    #[config(default = 0)]
    pub start_index: usize,
    #[config(default = 1)]
    pub margin: usize
}

fn create_artifact_dir(artifact_dir: &Path) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

// pub fn train<B: AutodiffBackend>(artifact_dir: &str,config: TrainingConfig, device: B::Device) {
//     create_artifact_dir(artifact_dir);
//     config   
//         .save(format!("{artifact_dir}/config.json"))
//         .expect("Config should be saved successfully");

//     B::seed(config.seed);

//     let batcher_train = DataBatcher::<B>::new(device.clone());
//     let batcher_valid = DataBatcher::<B>::new(device.clone());

//     println!("Loading datasets...");
//     let dataset_train = CustomDataset::load("data/train").expect("Cant load train data");
//     let dataset_test = CustomDataset::load("data/test").expect("Cant load test data");
//     println!("Datasets Loaded");
//     let dataloader_train = DataLoaderBuilder::new(batcher_train)
//         .batch_size(config.batch_size)
//         .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(PartialDataset::new(dataset_train,config.start_index,config.start_index+config.margin));

//     let dataloader_test = DataLoaderBuilder::new(batcher_valid)
//         .batch_size(config.batch_size)
//         .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(PartialDataset::new(dataset_test,config.start_index+config.margin+1,config.start_index+2*config.margin));


//     // Create the model and optimizer.
//     let mut model: Model<B> = config.model.init(&device);
//     let mut optim = config.optimizer.init::<B,_>();
//     // let lossfn = BinaryCrossEntropyLossConfig::new().init(&device);
//     // Iterate over our training and validation loop for X epochs.
//     let mut iter_train_res:Vec<String> = Vec::default();
//     let mut iter_test_res:Vec<String> = Vec::default();
//     for epoch in 1..config.num_epochs + 1 {
//         // Implement our training loop.
//         for (iteration, batch) in dataloader_train.iter().enumerate() {
//             let output = TrainStep::step(&model,batch);
//             // println!("Output: {}",output.item.output);
//             // iter_train_res.push(format!("Output: {}",output.item.output));
//             let outstr=format!("[Train - Epoch {epoch} - Iteration {iteration}] Loss {:.5}",output.item.loss.clone().into_scalar());
//             iter_train_res.push(outstr.clone());
//             println!("{outstr}");            

//             // Update the model using the optimizer.
//             model = optim.step(config.learning_rate, model, output.grads);            
//         }
//         for (iteration, batch) in dataloader_test.iter().enumerate() {
//             let output = ValidStep::step(&model,batch);
//             // println!("Output: {}",output.clone());

//             let outstr=format!("[Test - Epoch {epoch} - Iteration {iteration}] Loss {:.5}",output.loss.clone().into_scalar());
//             iter_test_res.push(outstr.clone());
//             // println!("{outstr}");
//         }
//         model.clone().save("models/models").expect("model not saved");
//     }

//     let mut file = std::fs::File::create("output.txt").expect("Cant create file");
//     println!("============================= TRAIN SUMMARY =====================================");
//     for item in iter_train_res{
//         file.write_all(format!("{item}").as_bytes()).expect("Cant write to file");
//         println!("{}",item)
//     }


//     println!("============================= TEST SUMMARY =====================================");
//     for item in iter_test_res{
//         file.write_all(format!("{item}").as_bytes()).expect("Cant write to file");
//         println!("{}",item)
//     }
// }






pub fn train<B: AutodiffBackend>(config: TrainingConfig, device: B::Device) {

    create_artifact_dir(config.artifact_dir.as_path());

    config
        .save(config.artifact_dir.join("config.json"))
        .expect("Config was not saved successfully");

    B::seed(config.seed);

    let batcher_train = DataBatcher::<B>::new(device.clone());
    let batcher_valid = DataBatcher::<B::InnerBackend>::new(device.clone());

    println!("Loading datasets...");
    let dataset_train = CustomDataset::load(config.train_data_path.0.as_path(),config.train_data_path.1.as_path()).expect("Cant load train data");
    let dataset_test = CustomDataset::load(config.test_data_path.0.as_path(),config.test_data_path.1.as_path()).expect("Cant load test data");
    println!("Datasets Loaded");
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(PartialDataset::new(dataset_train,config.start_index,config.start_index+config.margin));
        // .build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_test);
        // .build(PartialDataset::new(dataset_test,config.start_index+config.margin+1,config.start_index+2*config.margin));



    let model = config.model.init::<B>(&device);

    let learner = LearnerBuilder::new(&config.artifact_dir.display().to_string())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .metric_train_numeric(CpuMemory::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>
            (
                Aggregate::Mean, 
                burn::train::metric::store::Direction::Lowest, 
                burn::train::metric::store::Split::Train, 
                burn::train::StoppingCondition::NoImprovementSince { n_epochs: config.stop_num_epoch }
            )
        )
        .with_file_checkpointer(CompactRecorder::new())
        .log_to_file(true)
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            model,
            config.optimizer.init(),
            config.learning_rate.init(),
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained.save(config.artifact_dir.join("model").as_path()).expect("model not saved");
}