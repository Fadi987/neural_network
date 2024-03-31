pub mod activation;
pub mod fully_connected;

use crate::matrix;

trait Layer {
    fn forward(&self, input: &matrix::Matrix) -> matrix::Matrix;
}
