use crate::layer;
use crate::matrix;
use rand::Rng;

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

impl layer::Layer for FullyConnectedLayer {
    fn forward(&self, input: &matrix::Matrix) -> matrix::Matrix {
        matrix::Matrix::add(&matrix::Matrix::dot(&self.weights, input), &self.biases)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_fully_connected_layer_random_initialization() {
        let mut previous_weight_variance: Option<f32> = None;

        for i in 0..100 {
            let seed = [i; 32];
            let rng = StdRng::from_seed(seed);
            let layer = FullyConnectedLayer::new((3, 3), rng);
            let layer_shape = layer.weights.get_shape();

            let layer_values: Vec<f32> = (0..layer_shape.0)
                .flat_map(|i| (0..layer_shape.1).map(move |j| (i, j)))
                .map(|indices| layer.weights.get_value(indices))
                .collect();

            println!("{:?}", layer_values);

            // Asserts values are in [-1.0, 1.0] which should be true for Xavier/Clorot initialization in case of a layer of size (3, 2)
            for value in &layer_values {
                assert!(value.abs() <= 1.0);
            }

            let variance: f32 =
                layer_values.iter().map(|v| v * v).sum::<f32>() / (layer_values.len() as f32);

            // Assert values are not identically 0.0
            assert!(variance > 0.0);

            // Assert we're sampling different values every iteration
            match previous_weight_variance {
                None => previous_weight_variance = Some(variance),
                Some(prev_variance) => {
                    assert!((prev_variance - variance).abs() > 1e-7);
                    previous_weight_variance = Some(variance);
                }
            }
        }
    }
}
