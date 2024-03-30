use crate::matrix;

trait Layer {
    fn forward(&self) -> matrix::Matrix;
}

struct FullyConnectedLayer {
    weights: matrix::Matrix,
    biases: matrix::Matrix,
}

impl FullyConnectedLayer {
    pub fn new() {}
}
