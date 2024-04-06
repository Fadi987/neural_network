//! A module for neural network layers.
use crate::matrix;

pub mod activation;
pub mod fully_connected;
/// A trait for neural network layers.
pub trait Layer {
    fn forward(&mut self, input: &matrix::Matrix) -> matrix::Matrix;

    fn backward(&mut self, gradient: &matrix::Matrix) -> matrix::Matrix;

    fn update(&mut self, learning_rate: f32);
}
