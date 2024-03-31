//! A module for neural network layers.
use crate::matrix;

pub mod activation;
pub mod fully_connected;
/// A trait for neural network layers.
trait Layer {
    fn forward(&self, input: &matrix::Matrix) -> matrix::Matrix;
}
