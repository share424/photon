use crate::conv::laplace_1channel;
use crate::PhotonImage;
use crate::monochrome::grayscale_human_corrected_1channel;

#[cfg(feature = "enable_wasm")]
use wasm_bindgen::prelude::*;

#[cfg_attr(feature = "enable_wasm", wasm_bindgen)]
pub fn blur_score(photon_image: PhotonImage) -> f32 {
    let mut grayscale_img = grayscale_human_corrected_1channel(photon_image);
    laplace_1channel(&mut grayscale_img);

    let variance = calculate_variance(&grayscale_img.raw_pixels);

    variance
}

// calulcate variance using Welford's method for faster calculation
fn calculate_variance(array: &[u8]) -> f32 {
    let n = array.len() as f32;
    if n <= 1.0 {
        return 0.0;
    }

    let mut mean = array[0] as f32;
    let mut m2 = 0.0;

    for &x in array.iter().skip(1) {
        let delta = x as f32 - mean;
        mean += delta / n;
        let delta2 = x as f32 - mean;
        m2 += delta * delta2;
    }

    m2 / n
}

#[cfg_attr(feature = "enable_wasm", wasm_bindgen)]
pub fn light_score(photon_image: PhotonImage) -> f32 {
    // convert to hsv
    let end = photon_image.get_raw_pixels().len();

    let mut sum = 0.0;
    let mut count = 0.0;
    for i in (0..end).step_by(4) {
        let r = photon_image.raw_pixels[i] as u32;
        let g = photon_image.raw_pixels[i + 1] as u32;
        let b = photon_image.raw_pixels[i + 2] as u32;

        let value = u32::max(u32::max(r, g), b);
        sum += value as f32;
        count += 1.0;
    }

    let avg = (sum / count) * 100.0 / 255.0;
    avg
}