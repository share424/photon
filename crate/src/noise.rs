//! Add noise to images.

use image;
use image::Pixel;
use image::{GenericImage, GenericImageView};
use rand;
use rand::Rng;
// use wasm_bindgen::prelude::*;
use crate::helpers;
use crate::iter::ImageIterator;
use crate::PhotonImage;

/// Add randomized noise to an image.
/// This function adds a Gaussian Noise Sample to each pixel through incrementing each channel by a randomized offset.
/// This randomized offset is generated by creating a randomized thread pool.
/// **[WASM SUPPORT NOT AVAILABLE]**: Randomized thread pools cannot be created with WASM using the code used currently, but
/// a workaround is oncoming.
/// # Arguments
/// * `img` - A PhotonImage.
///
/// # Example
///
/// ```no_run
/// // For example:
/// use photon_rs::native::open_image;
/// use photon_rs::noise::add_noise_rand;
/// use photon_rs::PhotonImage;
///
/// let img = open_image("img.jpg").expect("File should open");
/// let result: PhotonImage = add_noise_rand(img);
/// ```
pub fn add_noise_rand(mut photon_image: PhotonImage) -> PhotonImage {
    let mut img = helpers::dyn_image_from_raw(&photon_image);
    let mut rng = rand::thread_rng();

    for (x, y) in ImageIterator::with_dimension(&img.dimensions()) {
        let offset = rng.gen_range(0, 150);
        let px =
            img.get_pixel(x, y).map(
                |ch| {
                    if ch <= 255 - offset {
                        ch + offset
                    } else {
                        255
                    }
                },
            );
        img.put_pixel(x, y, px);
    }
    photon_image.raw_pixels = img.to_bytes();
    photon_image
}

/// Add pink-tinted noise to an image.
///
/// **[WASM SUPPORT NOT AVAILABLE]**: Randomized thread pools cannot be created using the code used currently, but
/// support is oncoming.
/// # Arguments
/// * `name` - A PhotonImage that contains a view into the image.
///
/// # Example
///
/// ```no_run
/// // For example, to add pink-tinted noise to an image:
/// use photon_rs::native::open_image;
/// use photon_rs::noise::pink_noise;
///
/// let mut img = open_image("img.jpg").expect("File should open");
/// pink_noise(&mut img);
/// ```
pub fn pink_noise(mut photon_image: &mut PhotonImage) {
    let mut img = helpers::dyn_image_from_raw(&photon_image);
    let mut rng = rand::thread_rng();

    for (x, y) in ImageIterator::with_dimension(&img.dimensions()) {
        let ran1: f64 = rng.gen(); // generates a float between 0 and 1
        let ran2: f64 = rng.gen();
        let ran3: f64 = rng.gen();

        let ran_color1: f64 = 0.6 + ran1 * 0.6;
        let ran_color2: f64 = 0.6 + ran2 * 0.1;
        let ran_color3: f64 = 0.6 + ran3 * 0.4;

        let mut px = img.get_pixel(x, y);
        let channels = px.channels();

        let new_r_val = (channels[0] as f64 * 0.99 * ran_color1) as u8;
        let new_g_val = (channels[1] as f64 * 0.99 * ran_color2) as u8;
        let new_b_val = (channels[2] as f64 * 0.99 * ran_color3) as u8;
        px = image::Rgba([new_r_val, new_g_val, new_b_val, 255]);
        img.put_pixel(x, y, px);
    }
    photon_image.raw_pixels = img.to_bytes();
}
