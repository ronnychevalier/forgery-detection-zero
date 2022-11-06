use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Pixel};

pub type LuminanceImage = ImageBuffer<Luma<f64>, Vec<f64>>;

pub trait ToLumaZero {
    fn to_luma32f_zero(&self) -> ImageBuffer<Luma<f64>, Vec<f64>>;
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
        let mut buffer: LuminanceImage = ImageBuffer::new(self.width(), self.height());
        for (to, from) in buffer.pixels_mut().zip(self.pixels()) {
            let gray = to.channels_mut();
            let rgb = from.2.channels();
            gray[0] = rgb_to_luma_zero(rgb);
        }

        buffer
    }
}
