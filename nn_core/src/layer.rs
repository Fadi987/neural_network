//! A module for neural network layers.
use crate::matrix;

pub mod activation;
pub mod fully_connected;
/// A trait for neural network layers.
pub trait Layer {
    fn forward(&self, input: &matrix::Matrix) -> matrix::Matrix;

    fn backward(&self, gradient: &matrix::Matrix) -> matrix::Matrix;
}
