use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::*,
};
use nn::{conv::{ConvTranspose2d, ConvTranspose2dConfig}, pool::{MaxPool2d, MaxPool2dConfig}};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    //Encoder layers
    conve1: Conv2d<B>,
    conve2: Conv2d<B>,

    conve3: Conv2d<B>,
    conve4: Conv2d<B>,

    conve5: Conv2d<B>,
    conve6: Conv2d<B>,

    conve7: Conv2d<B>,
    conve8: Conv2d<B>,

    //Bottleneck
    convb1: Conv2d<B>,
    convb2: Conv2d<B>,

    //Decoder layers
    convdt1: ConvTranspose2d<B>,
    convd1: Conv2d<B>,
    convd2: Conv2d<B>,
    
    convdt2: ConvTranspose2d<B>,
    convd3: Conv2d<B>,
    convd4: Conv2d<B>,
    
    convdt3: ConvTranspose2d<B>,
    convd5: Conv2d<B>,
    convd6: Conv2d<B>,
    
    convdt4: ConvTranspose2d<B>,
    convd7: Conv2d<B>,
    convd8: Conv2d<B>,


    activation: Relu,
    pool: MaxPool2d,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
}

impl ModelConfig {
    // Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            //Encoder
            conve1: Conv2dConfig::new([1,64], [3, 3]).init(device),
            conve2: Conv2dConfig::new([64,64], [3, 3]).init(device),
            
            conve3: Conv2dConfig::new([32,128], [3, 3]).init(device),
            conve4: Conv2dConfig::new([128,128], [3, 3]).init(device),
            
            conve5: Conv2dConfig::new([64,256], [3, 3]).init(device),
            conve6: Conv2dConfig::new([256,256], [3, 3]).init(device),
            
            conve7: Conv2dConfig::new([128,512], [3, 3]).init(device),
            conve8: Conv2dConfig::new([512,512], [3, 3]).init(device),

            //Bottleneck
            convb1: Conv2dConfig::new([256,1024], [3, 3]).init(device),
            convb2: Conv2dConfig::new([1024,1024], [3, 3]).init(device),

            //Decoder  
            convdt1: ConvTranspose2dConfig::new([1024,512], [2, 2]).with_stride([2,2]).init(device),
            convd1: Conv2dConfig::new([1024,512], [3, 3]).init(device),
            convd2: Conv2dConfig::new([512,512], [3, 3]).init(device),
            
            convdt2: ConvTranspose2dConfig::new([512,256], [2, 2]).with_stride([2,2]).init(device),
            convd3: Conv2dConfig::new([512,256], [3, 3]).init(device),
            convd4: Conv2dConfig::new([256,256], [3, 3]).init(device),
            
            convdt3: ConvTranspose2dConfig::new([256,128], [2, 2]).with_stride([2,2]).init(device),
            convd5: Conv2dConfig::new([256,128], [3, 3]).init(device),
            convd6: Conv2dConfig::new([128,128], [3, 3]).init(device),
           
            convdt4: ConvTranspose2dConfig::new([128,64], [2, 2]).with_stride([2,2]).init(device),
            convd7: Conv2dConfig::new([128,64], [3, 3]).init(device),
            convd8: Conv2dConfig::new([64,64], [3, 3]).init(device),
  
            activation: Relu::new(),
            pool: MaxPool2dConfig::new([2,2]).init() 
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, 1, height, width]);
        
        //Encoder
        let x = self.conve1.forward(x);
        let x = self.activation.forward(x);
        
        let x = self.conve2.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);


        let x = self.conve3.forward(x);
        let x = self.activation.forward(x);
        
        let x = self.conve4.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        
        let x = self.conve5.forward(x);
        let x = self.activation.forward(x);
        
        let x = self.conve6.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        
        let x = self.conve7.forward(x);
        let x = self.activation.forward(x);
        
        let x = self.conve8.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        //Bottleneck
        let x = self.convb1.forward(x);        
        let x = self.activation.forward(x);
        let x = self.convb2.forward(x);        
        let x = self.activation.forward(x);

        //Decoder
        let x = self.convdt1.forward(x);

        let x = self.convdt2.forward(x);
        
        let x = self.convdt3.forward(x);
    
        let x = self.convdt4.forward(x);
   }
}
