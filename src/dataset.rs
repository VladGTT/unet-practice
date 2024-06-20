use std::{io::{Cursor, Error}, iter::zip, path::Path, sync::Mutex};

use burn::data::dataset::Dataset;
use image::io::Reader as ImageReader;

pub struct CustomDataset<I> {
    data: Vec<I>
}
pub struct CustomDatasetItem {
    image: Vec<u8>,
    mask: Vec<u8>
}
impl<I: Send + Sync + Clone> Dataset<I> for CustomDataset<I>{
    fn get(&self, index: usize) -> Option<I> {
        if index<=self.len(){
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

impl CustomDataset<CustomDatasetItem>{
    //implementation assumes that images are contained in {path}/images and masks in {path}/masks
    pub fn load(path: &str)->Result<Self,Error>{
        let list_images = Self::list_files_in_directory(Path::new(path).join("images").as_path())?;   
        let list_masks = Self::list_files_in_directory(Path::new(path).join("masks").as_path())?;   

        let mut data: Vec<CustomDatasetItem> = Vec::new();
        for (image,mask) in zip(list_images, list_masks){
            let new_item = CustomDatasetItem{
                image: Self::open_image(&image)?,
                mask: Self::open_image(&mask)?
            };     
            data.push(new_item);
        }     
     
        Ok(Self { data: data})
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
    fn open_image(path: &str)->Result<Vec<u8>,Error>{
        let mut buf: Vec<u8> = Vec::new();
        let img = ImageReader::open(path)?;
        let decoded_img = img.decode().map_err(|err|err.into())?;
        decoded_img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png).map_err(|err|err.into())?;
        Ok(buf)        
    }

}