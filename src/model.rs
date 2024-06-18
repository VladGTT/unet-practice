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
pub struct Encoder<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    activation: Relu,
    pool: MaxPool2d,
}

impl<B: Backend> Encoder<B> {
    pub fn init(&self, device: &B::Device,channels: usize) -> Encoder<B> {
        Encoder { 
            conv1: Conv2dConfig::new([1, channels], [3, 3]).init(device),
            conv2: Conv2dConfig::new([channels, channels], [3, 3]).init(device),
            activation: Relu::new(),
            pool: MaxPool2dConfig::new([2,2]).init() 
        }
    }
}


#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    conv1: ConvTranspose2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    activation: Relu,
}

impl<B: Backend> Decoder<B> {
    pub fn init(&self, device: &B::Device,channels: usize) -> Decoder<B> {
        Decoder { 
            conv1: ConvTranspose2dConfig::new([, channels], [3, 3]).with_stride([2,2]).init(device),
            conv2: Conv2dConfig::new([channels, channels], [3, 3]).init(device),
            conv3: Conv2dConfig::new([channels, channels], [3, 3]).init(device),
            activation: Relu::new(),
        }
    }
}


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    enc1: Encoder<B>,
    enc2: Encoder<B>,
    enc3: Encoder<B>,
    enc4: Encoder<B>,

    conv1: Conv2d<B>,
    conv2: Conv2d<B>,

    dec1: Decoder<B>,
    dec2: Decoder<B>,
    dec3: Decoder<B>,
    dec4: Decoder<B>,

    activation: Relu
}

#[derive(Config, Debug)]
pub struct ModelConfig {
}

impl ModelConfig {
    // Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
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

        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
    }
}
