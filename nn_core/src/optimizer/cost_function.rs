//! Interface for a cost function to be applied to a single sample.
pub mod squared_error;
use crate::matrix::Matrix;

pub trait CostFunction {
    /// Computes the cost of the given input and target matrices.
    ///
    /// # Arguments
    ///
    /// * `input` - The input matrix.
    /// * `target` - The target matrix.
    ///
    /// # Returns
    ///
    /// The computed cost as a `f32` value.
    fn compute(&mut self, input: &Matrix, target: &Matrix) -> f32;

    /// Computes the gradient of the cost function with respect to the input matrix.
    ///
    /// # Arguments
    ///
    /// * `input` - The input matrix.
    /// * `target` - The target matrix.
    ///
    /// # Returns
    ///
    /// The computed gradient as a `Matrix`.
    fn gradient(&mut self, input: &Matrix, target: &Matrix) -> Matrix;
}
