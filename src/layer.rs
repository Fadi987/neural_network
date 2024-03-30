use crate::matrix;
use rand::Rng;

trait Layer {
    fn forward(&self) -> matrix::Matrix;
}

struct FullyConnectedLayer {
    weights: matrix::Matrix,
    biases: matrix::Matrix,
}

impl FullyConnectedLayer {
    // TODO: think about how to test
    // TODO: add ability to use other methods of initialization
    pub fn new((input_size, output_size): (usize, usize)) {
        // Xavier Glorot Initialization
        let boundary = (6.0 / ((input_size + output_size) as f32)).sqrt();
        let mut rng = rand::thread_rng();

        let generator = || rng.gen_range(-boundary..boundary);
        matrix::Matrix::random((input_size, output_size), generator);
    }
}
