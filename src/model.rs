use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::*,
};
use nn::{
    conv::{ConvTranspose2d, ConvTranspose2dConfig},
    pool::{MaxPool2d, MaxPool2dConfig},
    PaddingConfig2d,
};

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

    convex: Conv2d<B>,

    activation: Relu,
    pool: MaxPool2d,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
}

impl ModelConfig {
    // Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            //Encoder
            conve1: Conv2dConfig::new([3, 64], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            conve2: Conv2dConfig::new([64, 64], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),

            conve3: Conv2dConfig::new([64, 128], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            conve4: Conv2dConfig::new([128, 128], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),

            conve5: Conv2dConfig::new([128, 256], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            conve6: Conv2dConfig::new([256, 256], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),

            conve7: Conv2dConfig::new([256, 512], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            conve8: Conv2dConfig::new([512, 512], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),

            //Bottleneck
            convb1: Conv2dConfig::new([512, 1024], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            convb2: Conv2dConfig::new([1024, 1024], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),

            //Decoder
            convdt1: ConvTranspose2dConfig::new([1024, 512], [2, 2])
                .with_stride([2, 2])
                // .with_padding_out([1,1])
                .init(device),
            convd1: Conv2dConfig::new([1024, 512], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            convd2: Conv2dConfig::new([512, 512], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),

            convdt2: ConvTranspose2dConfig::new([512, 256], [2, 2])
                .with_stride([2, 2])
                .init(device),
            convd3: Conv2dConfig::new([512, 256], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            convd4: Conv2dConfig::new([256, 256], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),

            convdt3: ConvTranspose2dConfig::new([256, 128], [2, 2])   
                .with_stride([2, 2])
                .init(device),
            convd5: Conv2dConfig::new([256, 128], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            convd6: Conv2dConfig::new([128, 128], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),

            convdt4: ConvTranspose2dConfig::new([128, 64], [2, 2])
                .with_stride([2, 2])
                .init(device),
            convd7: Conv2dConfig::new([128, 64], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            convd8: Conv2dConfig::new([64, 64], [3, 3])
                // .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),

            convex: Conv2dConfig::new([64, self.num_classes], [1, 1])
                .init(device),
            activation: Relu::new(),
            pool: MaxPool2dConfig::new([2, 2])
                .with_strides([2,2])
                .init(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 4> {
        // let [batch_size, height, width] = images.dims();

        // Create a channel at the second dimension.
        // let x = images.reshape([batch_size, 1, height, width]);

        //Encoder
        let e1 = self.conve1.forward(images);
        let e1 = self.activation.forward(e1);
        let e1 = self.conve2.forward(e1);
        let e1 = self.activation.forward(e1);
        let e1 = self.pool.forward(e1);

        let e2 = self.conve3.forward(e1.clone()); 
        let e2 = self.activation.forward(e2);
        let e2 = self.conve4.forward(e2);
        let e2 = self.activation.forward(e2);
        let pe2 = self.pool.forward(e2.clone());

        let e3 = self.conve5.forward(pe2);
        let e3 = self.activation.forward(e3);
        let e3 = self.conve6.forward(e3);
        let e3 = self.activation.forward(e3);
        let pe3 = self.pool.forward(e3.clone());

        let e4 = self.conve7.forward(pe3);
        let e4 = self.activation.forward(e4);
        let e4 = self.conve8.forward(e4);
        let e4 = self.activation.forward(e4);
        let pe4 = self.pool.forward(e4.clone());

        //Bottleneck
        let b1 = self.convb1.forward(pe4);
        let b1 = self.activation.forward(b1);
        let b1 = self.convb2.forward(b1);
        let b1 = self.activation.forward(b1);

        //Decoder
        let d1 = self.convdt1.forward(b1);
        let d1 = Tensor::cat(vec![d1.clone(), e4.slice([0..1 , 0..512 ,4..60, 4..60])], 1);
        let d1 = self.convd1.forward(d1);
        let d1 = self.activation.forward(d1);
        let d1 = self.convd2.forward(d1);
        let d1 = self.activation.forward(d1);

        let d2 = self.convdt2.forward(d1);
        let d2 = Tensor::cat(vec![d2.clone(), e3.slice([0..1, 0..256, 16..120,16..120])],1);
        let d2 = self.convd3.forward(d2);
        let d2 = self.activation.forward(d2);
        let d2 = self.convd4.forward(d2);
        let d2 = self.activation.forward(d2);

        let d3 = self.convdt3.forward(d2);
        let d3 = Tensor::cat(vec![d3.clone(), e2.slice([0..1,0..128,40..240,40..240])],1);
        let d3 = self.convd5.forward(d3);
        let d3 = self.activation.forward(d3);
        let d3 = self.convd6.forward(d3);
        let d3 = self.activation.forward(d3);
        
        let d4 = self.convdt4.forward(d3);
        println!("{:?}",d4.dims());
        println!("{:?}",e1.dims());
        let d4 = Tensor::cat(vec![d4.clone(), e1.slice([0..1,0..64,54..338,54..338])], 1);
        let d4 = self.convd7.forward(d4);
        let d4 = self.activation.forward(d4);
        let d4 = self.convd8.forward(d4);
        let d4 = self.activation.forward(d4);

        let result = burn::tensor::activation::sigmoid(self.convex.forward(d4));
        result
        // result.reshape([batch_size, height, width])
    }
}
