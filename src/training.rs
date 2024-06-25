use std::{thread::sleep, time::Duration};

use burn::{config::Config, data::dataloader::DataLoaderBuilder, module::Module, nn::loss::{BinaryCrossEntropyLossConfig, MseLoss}, optim::{AdamConfig, GradientsParams, Optimizer}, record::CompactRecorder, tensor::backend::AutodiffBackend};

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
    pub optimizer: AdamConfig,
    #[config(default = 1)]
    pub num_epochs: usize,
    #[config(default = 1)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-1)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend<IntElem = i32>>(artifact_dir: &str,config: TrainingConfig, device: B::Device) {
    create_artifact_dir("/tmp/guide");
    config   
        .save(format!("/tmp/guide/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = DataBatcher::<B>::new(device.clone());
    // let batcher_valid = DataBatcher::<B::InnerBackend>::new(device.clone());
    let dataset = CustomDataset::load("data",[388,388],10).unwrap();
    
    
    
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset);


    // Create the model and optimizer.
    let mut model: Model<B> = config.model.init(&device);
    let mut optim = config.optimizer.init::<B,_>();

    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {
        // Implement our training loop.
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward(batch.images);
            
            // println!("{:?}",batch.targets.clone().int().to_data().value);
            let loss = BinaryCrossEntropyLossConfig::new().init(&device)
                .forward(output, batch.targets.int());
            // let loss = MseLoss::new()
                //  .forward_no_reduction(output, batch.targets);


            println!(
                "[Train - Epoch {} - Iteration {}]",
                epoch,
                iteration,
                // loss.clone().into_scalar()
            );
            // println!("Sleeping");
            // sleep(Duration::from_secs(20));
 
            // Gradients for the current backward pass
           // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            // println!("Sleeping");
            // sleep(Duration::from_secs(30));
 
            // Update the model using the optimizer.
            model = optim.step(config.learning_rate, model, grads);

        }
    }
    let _ = model.save("models/models").expect("model not saved");


    
}



// pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
//     create_artifact_dir(artifact_dir);
//     config
//         .save(format!("{artifact_dir}/config.json"))
//         .expect("Config should be saved successfully");

//     B::seed(config.seed);

//     let batcher_train = DataBatcher::<B>::new(device.clone());
//     let batcher_valid = DataBatcher::<B::InnerBackend>::new(device.clone());

//     let dataset = CustomDataset::load("data",[388,388],30).unwrap();

//     let dataloader_train = DataLoaderBuilder::new(batcher_train)
//         .batch_size(config.batch_size)
//         .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(PartialDataset::new(dataset,0,20));

//     let dataloader_test = DataLoaderBuilder::new(batcher_valid)
//         .batch_size(config.batch_size)
//         .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(PartialDataset::new(dataset,20,30));

//     let learner = LearnerBuilder::new(artifact_dir)
//         .metric_train_numeric(AccuracyMetric::new())
//         .metric_valid_numeric(AccuracyMetric::new())
//         .metric_train_numeric(LossMetric::new())
//         .metric_valid_numeric(LossMetric::new())
//         .with_file_checkpointer(CompactRecorder::new())
//         .devices(vec![device.clone()])
//         .num_epochs(config.num_epochs)
//         .summary()
//         .build(
//             config.model.init::<B>(&device),
//             config.optimizer.init(),
//             config.learning_rate,
//         );

//     let model_trained = learner.fit(dataloader_train, dataloader_test);

//     model_trained
//         .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
//         .expect("Trained model should be saved successfully");
// }
