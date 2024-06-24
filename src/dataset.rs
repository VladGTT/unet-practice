use std::{
    io::{Cursor, Error}, iter::zip, ops::Range, path::Path
};

use burn::data::dataset::Dataset;
use image::{io::Reader as ImageReader, DynamicImage, ImageBuffer, Rgb};

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
            let image = Self::open_image(&image,shape)?;
            let prepared_image = Self::resize_with_mirroring(image);

            let mask = Self::open_image(&mask, shape)?;
            let prepared_mask = DynamicImage::from(mask.into_luma8());

            let new_item = CustomDatasetItem {
                image: prepared_image.into_bytes(),
                mask: prepared_mask.into_bytes(),
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
    fn open_image(path: &str,shape: [u32;2]) -> Result<DynamicImage, Error> {
        let img = ImageReader::open(path)?.decode().map_err(|err| Error::other(err))?; 
        let img = img.resize_exact(shape[0],shape[1], image::imageops::FilterType::Lanczos3);
        Ok(img)
    }
    
    fn resize_with_mirroring(image: DynamicImage)->DynamicImage{
        let mut new_img = ImageBuffer::new(572,572);
        let image_buffer = image.to_rgb8(); 
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
        DynamicImage::from(new_img)   
    }


}
