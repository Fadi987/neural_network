use crate::matrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

trait Layer {
    fn forward(&self) -> matrix::Matrix;
}

struct FullyConnectedLayer {
    weights: matrix::Matrix,
    biases: matrix::Matrix,
}

impl FullyConnectedLayer {
    // TODO: add ability to use other methods of initialization
    // Passing random rng as input allows for seeding and thus testing
    pub fn new(shape: (usize, usize), mut rng: impl Rng) -> Self {
        // Xavier Glorot Initialization
        let boundary = (6.0 / ((shape.0 + shape.1) as f32)).sqrt();

        let generator = || rng.gen_range(-boundary..boundary);
        let weights = matrix::Matrix::random(shape, generator);

        FullyConnectedLayer {
            weights,
            biases: matrix::Matrix::new(vec![0.0; shape.0], (shape.0, 1)),
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_fully_connected_layer_initialization() {
//         let seed = [42; 32];
//         let rng = StdRng::from_seed(seed);

//         let layer = FullyConnectedLayer::new((2, 2), rng);
//         // assert_eq!(layer.weights, matrix::Matrix::zeros((2, 2)));
//     }
// }
