use std::io::Cursor;

use image::{self, DynamicImage, GenericImageView, ImageBuffer, ImageFormat, Luma, Pixel};

use jpeg_encoder::JpegColorType;

use crate::LuminanceImage;

pub(crate) trait ToLumaZero {
    fn to_luma32f_zero(&self) -> LuminanceImage;
}

#[inline]
fn rgb_to_luma_zero(rgb: &[u8]) -> f64 {
    let r = f64::from(rgb[0]);
    let g = f64::from(rgb[1]);
    let b = f64::from(rgb[2]);

    (b.mul_add(0.114, r.mul_add(0.299, g * 0.587))).round()
}

impl ToLumaZero for DynamicImage {
    fn to_luma32f_zero(&self) -> LuminanceImage {
        let mut buffer: ImageBuffer<Luma<f64>, Vec<f64>> =
            ImageBuffer::new(self.width(), self.height());
        for (to, from) in buffer.pixels_mut().zip(self.pixels()) {
            let gray = to.channels_mut();
            let rgb = from.2.channels();
            gray[0] = rgb_to_luma_zero(rgb);
        }

        LuminanceImage {
            image: buffer.into_raw().into_boxed_slice(),
            width: self.width(),
            height: self.height(),
        }
    }
}

impl jpeg_encoder::ImageBuffer for &LuminanceImage {
    fn get_jpeg_color_type(&self) -> JpegColorType {
        JpegColorType::Luma
    }

    fn width(&self) -> u16 {
        self.width as u16
    }

    fn height(&self) -> u16 {
        self.height as u16
    }

    fn fill_buffers(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
        for x in 0..self.width {
            let &pixel = unsafe { self.unsafe_get_pixel(x, u32::from(y)) };

            buffers[0].push(pixel as u8);
        }
    }
}

impl LuminanceImage {
    pub(crate) fn to_jpeg_99_luminance(&self) -> Result<Self, crate::Error> {
        let mut buffer = Vec::new();
        let encoder = jpeg_encoder::Encoder::new(&mut buffer, 99);
        encoder.encode_image(self)?;

        let jpeg_99 =
            image::io::Reader::with_format(Cursor::new(buffer), ImageFormat::Jpeg).decode()?;

        Ok(jpeg_99.to_luma32f_zero())
    }
}
