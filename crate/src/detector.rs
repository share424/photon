use crate::PhotonImage;
use crate::transform::{resize, SamplingFilter};
// use js_sys::Array;
use serde::{Deserialize, Serialize};
use image::DynamicImage::ImageLuma8;
use image::{ImageBuffer, GrayImage};
use imageproc::contours::{find_contours, Contour};

#[cfg(feature = "enable_wasm")]
use wasm_bindgen::prelude::*;

#[cfg_attr(feature = "enable_wasm", wasm_bindgen)]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PreprocessedImage {
    data: Vec<f32>,
    width: u32,
    height: u32,
}

#[cfg_attr(feature = "enable_wasm", wasm_bindgen)]
impl PreprocessedImage {
    pub fn get_data(&self) -> Vec<f32> {
        self.data.clone()
    }

    pub fn get_width(&self) -> u32 {
        self.width
    }

    pub fn get_height(&self) -> u32 {
        self.height
    }
}

#[cfg_attr(feature = "enable_wasm", wasm_bindgen)]
pub fn preprocess_detector(photon_image: PhotonImage, det_shape: f32) -> PreprocessedImage {
    // resize
    let mut ratio = 1.0;
    let height = photon_image.get_height() as f32;
    let width = photon_image.get_width() as f32;

    if f32::max(height, width) > det_shape {
        ratio = if height > width { det_shape / height } else { det_shape / width };
    }

    let resize_h: u32 = (height * ratio / 32.0).round() as u32 * 32;
    let resize_w: u32 = (width * ratio / 32.0).round() as u32 * 32;

    let resized_img = resize(&photon_image, resize_w, resize_h, SamplingFilter::Nearest);

    let h = resized_img.get_height();
    let w = resized_img.get_width();

    // Normalize the pixel data and convert to float32
    let mut float32_data: Vec<f32> = vec![0.0; (h * w * 3) as usize];

    // Normalize mean and std
    let mean: [f32; 3] = [0.485, 0.456, 0.406];
    let std: [f32; 3] = [0.229, 0.224, 0.225];

    let size = (h * w) as usize;
    let mut x = 0;
    let mut y = size;
    let mut z = size * 2;

    for i in (0..resized_img.raw_pixels.len()).step_by(4) {
        let r = resized_img.raw_pixels[i] as f32 / 255.0;
        let g = resized_img.raw_pixels[i + 1] as f32 / 255.0;
        let b = resized_img.raw_pixels[i + 2] as f32 / 255.0;

        // put in R G B order
        float32_data[x] = (r - mean[0]) / std[0]; // R
        float32_data[y] = (g - mean[1]) / std[1]; // G
        float32_data[z] = (b - mean[2]) / std[2]; // B

        x += 1;
        y += 1;
        z += 1;
        // skip data[i + 3] to filter out the alpha channel
    }

    PreprocessedImage {
        data: float32_data,
        width: w,
        height: h,
    }
}

fn create_integral_image(input: &Vec<f32>, height: u32, width: u32) -> Option<Vec<f32>> {
    // Check if the height and width are valid
    if (height * width) as usize != input.len() {
        return None; // The input dimensions do not match the input length
    }

    // Create an empty integral image with the same dimensions as the input array
    let mut integral_image = vec![0.0; (width * height) as usize];

    // Compute the first row of the integral image
    integral_image[0] = input[0];
    for j in 1..width as usize {
        integral_image[j] = integral_image[j - 1] + input[j];
    }

    // Compute the first column of the integral image
    for i in 1..height as usize {
        integral_image[i * width as usize] = integral_image[(i - 1) * width as usize] + input[i * width as usize];
    }

    // Compute the rest of the integral image
    for i in 1..height as usize {
        for j in 1..width as usize {
            let index = i * width as usize + j;
            integral_image[index] = integral_image[index - 1] + integral_image[index - width as usize] - integral_image[index - width as usize - 1] + input[index];
        }
    }

    Some(integral_image)
}

fn get_sum_of_area(integral_image: &Vec<f32>, width: u32, height: u32, top: u32, left: u32, bottom: u32, right: u32) -> f32 {
    if top >= height || left >= width || bottom >= height || right >= width || top > bottom || left > right {
        return 0.0; // Invalid area coordinates
    }

    let top = top + 1;
    let left = left + 1;
    let bottom = bottom + 1;
    let right = right + 1;

    // The sum of the area can be computed using four precomputed values from the integral image
    let index_bottom_right = (bottom * width + right) as usize;
    let index_top_left = ((top - 1) * width + (left - 1)) as usize;
    let index_bottom_left = (bottom * width + (left - 1)) as usize;
    let index_top_right = ((top - 1) * width + right) as usize;

    let sum = integral_image[index_bottom_right] + integral_image[index_top_left]
        - integral_image[index_bottom_left]
        - integral_image[index_top_right];

    sum
}

fn get_mean_of_area(integral_image: &Vec<f32>, width: u32, height: u32, top: u32, left: u32, bottom: u32, right: u32) -> f32 {
    let sum = get_sum_of_area(integral_image, width, height, top, left, bottom, right);
    let area_width = (right - left + 1) as f32;
    let area_height = (bottom - top + 1) as f32;
    let area_size = area_width * area_height;
    let mean = sum / (area_size + 0.000001) as f32;
    mean
}

#[cfg_attr(feature = "enable_wasm", wasm_bindgen)]
pub fn postprocess_detector(
    data: Vec<f32>,
    height: u32,
    width: u32,
    original_height: u32,
    original_width: u32,
    _thresh: f32,
    score_threshold: f32,
    max_candidates: u32,
    height_threshold: f32,
    width_threshold: f32,
    num_threshold: u32,
) -> bool {
    let mask = data
        .iter()
        .map(|&f| if f > _thresh { 255 as u8 } else { 0 as u8 })
        .collect();

    let image_buffer = ImageBuffer::from_vec(width, height, mask).unwrap();
    let mask: GrayImage = ImageLuma8(image_buffer).to_luma8();

    let contours: Vec<Contour<u32>> = find_contours(&mask);
    
    let num_contour = u32::min(max_candidates, contours.len() as u32);

    if num_contour < num_threshold {
        return true;
    }

    // let bbox: Array = Array::new();

    // create integral image of data
    // so we can calculate the mean of any rectangle quickly
    let integral_image = create_integral_image(&data, height, width).unwrap();

    let mut total_bbox = 0;
    for i in 0..num_contour {
        let contour = &contours[i as usize];
        let mut x_min = width;
        let mut x_max = 0;
        let mut y_min = height;
        let mut y_max = 0;

        for point in contour.points.iter() {
            if point.x < x_min {
                x_min = point.x;
            }
            if point.x > x_max {
                x_max = point.x;
            }
            if point.y < y_min {
                y_min = point.y;
            }
            if point.y > y_max {
                y_max = point.y;
            }
        }

        if x_min >= x_max || y_min >= y_max {
            continue;
        }

        // get value from data within the bounding box
        // let mut sum = 0.0;
        // let mut count = 0;

        // for i in 0..data.len() {
        //     let x = i % width as usize;
        //     let y = i / width as usize;

        //     if x >= x_min as usize && x < x_max as usize && y >= y_min as usize && y < y_max as usize {
        //         sum += data[i];
        //         count += 1;
        //     }
        // }

        // // calculate the mean
        // let score = sum / (count as f32 + 0.000001) as f32;
        let score = get_mean_of_area(&integral_image, width, height, y_min, x_min, y_max, x_max);

        if score < score_threshold {
            continue;
        }

        let h = (y_max - y_min) as f32 / height as f32 * original_height as f32;
        let w = (x_max - x_min) as f32 / width as f32 * original_width as f32;

        if h < height_threshold || w < width_threshold {
            return true;
        }

        total_bbox += 1;

        if total_bbox > num_threshold {
            return false;
        }

        // let bbox_obj: Array = Array::new();
        // bbox_obj.push(&JsValue::from(x_min as f32 / width as f32 * original_width as f32));
        // bbox_obj.push(&JsValue::from(y_min as f32 / height as f32 * original_height as f32));
        // bbox_obj.push(&JsValue::from(x_max as f32 / width as f32 * original_width as f32));
        // bbox_obj.push(&JsValue::from(y_max as f32 / height as f32 * original_height as f32));

        // bbox.push(&bbox_obj);
    }
    

    return false;
}