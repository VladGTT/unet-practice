use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
};
use crate::dataset::CustomDatasetItem;

#[derive(Clone)]
pub struct DataBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> DataBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct DataBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> Batcher<CustomDatasetItem, DataBatch<B>> for DataBatcher<B> {
    fn batch(&self, items: Vec<CustomDatasetItem>) -> DataBatch<B> {
        let dims = [480,480,3]; 
        let (mut images,mut targets): (Vec<Tensor<B, 3>>,Vec<Tensor<B, 3>>) = (Vec::new(),Vec::new());
        for item in items.iter(){   
            let img_data = Data::new(item.image.clone(),Shape::new(dims));
            let img_tensor = Tensor::<B, 3>::from_data(img_data.convert(), &self.device);

            let msk_data = Data::new(item.mask.clone(),Shape::new(dims));
            let msk_tensor = Tensor::<B, 3>::from_data(msk_data.convert(), &self.device);
        
            images.push(img_tensor);
            targets.push(msk_tensor);
        }


        // let images: Vec<Tensor<B,2>> = items
        //     .into_iter()
        //     .map(|item| Data::new(item.image,Shape::new([1280,720])))
        //     .map(|data| Tensor::<B, 2>::from_data(data.convert(), &self.device))
        //     .collect();

        // let targets: Vec<Tensor<B,2>> = items
        //     .into_iter()
        //     .map(|item| Data::new(item.mask,Shape::new([1280,720])))
        //     .map(|data| Tensor::<B, 2>::from_data(data.convert(), &self.device))
        //     .collect();


        // let images = Tensor::cat(images, 0).to_device(&self.device);
        // let targets = Tensor::cat(targets, 0).to_device(&self.device);

        DataBatch { 
            images: Tensor::stack(images, 0).to_device(&self.device),
            targets: Tensor::stack(targets, 0).to_device(&self.device)
        }
    }
}
