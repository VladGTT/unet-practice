use std::{
    io::{Cursor, Error}, iter::zip, ops::Range, path::Path
};

use burn::data::dataset::Dataset;
use image::{codecs::png::PngDecoder, io::Reader as ImageReader, DynamicImage, ImageBuffer, Pixel, Rgb};

#[derive(Debug,Clone)]
pub struct CustomDataset<I> {
    data: Vec<I>,
}
#[derive(Debug,Clone)]
pub struct CustomDatasetItem {
    pub image: Vec<u8>,
    pub mask: Vec<u8>,
}
impl<I: Send + Sync + Clone> Dataset<I> for CustomDataset<I> {
    fn get(&self, index: usize) -> Option<I> {
        if index <= self.len() {
            Some(self.data[index].clone())
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn iter(&self) -> burn::data::dataset::DatasetIterator<'_, I>
    where
        Self: Sized,
    {
        burn::data::dataset::DatasetIterator::new(self)
    }
}

impl CustomDataset<CustomDatasetItem> {
    //implementation assumes that images are contained in {path}/images and masks in {path}/masks
    pub fn load(path: &str,shape: [u32;2],len: usize) -> Result<Self, Error> {
        let mut list_images = Self::list_files_in_directory(Path::new(path).join("images").as_path())?;
        let mut list_masks = Self::list_files_in_directory(Path::new(path).join("masks").as_path())?;

        list_images.truncate(len);
        list_masks.truncate(len);
        
        let mut data: Vec<CustomDatasetItem> = Vec::new();
        for (image, mask) in zip(list_images, list_masks) {

            let image = CustomImage::open(&image)?.resize().resize_with_mirroring();
            let mask = CustomImage::open(&mask)?.resize().into_grayscale();

            let new_item = CustomDatasetItem {
                image: image.into_bytes(),
                mask: mask.into_bytes(),
            };

            data.push(new_item);
        }

        Ok(Self { data: data })
    }
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

}
pub struct CustomImage{
    image: DynamicImage
}

impl CustomImage {
    pub fn open(path: &str)->Result<Self, Error>{
        let img = ImageReader::open(path)?.decode().map_err(|err| Error::other(err))?; 
        Ok(Self{image: img})
    }
    pub fn resize(self)->Self{
        Self{image: self.image.resize_exact(388,388, image::imageops::FilterType::Lanczos3)}
    }

    pub fn resize_with_mirroring(self)->Self{
        let mut new_img = ImageBuffer::new(572,572);
        let image_buffer = self.image.to_rgb8(); 
        for (x,y,pixel) in image_buffer.enumerate_pixels(){
            new_img.put_pixel(x+92, y+92, pixel.clone());
        };
    
        let mirror = |img: &mut ImageBuffer<Rgb<u8>,_>,xrange: Range<u32>,yrange: Range<u32>|{
            for i in yrange{
                for j in xrange.clone(){
                    let pixel = img.get_pixel(j, i);
                    img.put_pixel(j, i-(i-92)*2+1, pixel.clone());
                }
            };    
        };
        mirror(&mut new_img,92..480,92..184);
        let mut new_img = DynamicImage::from(new_img).rotate180().to_rgb8();    
        mirror(&mut new_img,92..480,92..184);
        let mut new_img = DynamicImage::from(new_img).rotate90().to_rgb8();
        mirror(&mut new_img,0..572,92..184);
        let mut new_img = DynamicImage::from(new_img).rotate180().to_rgb8(); 
        mirror(&mut new_img,0..572,92..184);
        let new_img = DynamicImage::from(new_img).rotate270().to_rgb8();
        Self{image: DynamicImage::from(new_img)}
    }
    pub fn save(self,path:&str)->Result<(),Error>{
        self.image.save(path).map_err(|err|Error::other(err))
    }
    pub fn into_bytes(self)->Vec<u8>{
        self.image.into_bytes()
    }

    pub fn from_bytes(buf: Vec<u8>)->Option<Self>{
        let img_buf: ImageBuffer<Rgb<_>, Vec<_>> = ImageBuffer::from_vec(388, 388,buf)?;
        let image = DynamicImage::from(img_buf); 

        // let decoder = PngDecoder::new(Cursor::new(&buf)).map_err(|e|Error::other(e))?;
        // let image = DynamicImage::from_decoder(decoder).map_err(|e|Error::other(e))?;
        
        Some(
            Self{
                image: image
            }
        )
    }
    pub fn into_grayscale(self)->Self{
        let img = self.image.into_luma8();
        Self { image: DynamicImage::from(img) }
    }
}