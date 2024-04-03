use crate::layer;
use crate::matrix;
use rand::Rng;

/// Represents a fully connected layer in a neural network.
pub struct FullyConnectedLayer {
    weights: matrix::Matrix,
    biases: matrix::Matrix,

    // Temporary storage for backpropagation
    weights_gradient: Option<matrix::Matrix>,
    biases_gradient: Option<matrix::Matrix>,
    forward_pass_input: Option<matrix::Matrix>,
}

impl FullyConnectedLayer {
    /// Creates a new fully connected layer with the specified shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - A tuple representing the shape of the layer (number of rows, number of columns).
    /// * `rng` - A random number generator used for weight initialization.
    ///
    /// # Returns
    ///
    /// A new `FullyConnectedLayer` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    /// use nn_core::layer::fully_connected::FullyConnectedLayer;
    ///
    /// let seed = [42; 32];
    /// let rng = StdRng::from_seed(seed);
    /// let layer = FullyConnectedLayer::new((3, 3), rng);
    /// ```
    pub fn new(shape: (usize, usize), mut rng: impl Rng) -> Self {
        // Xavier Glorot Initialization
        let boundary = (6.0 / ((shape.0 + shape.1) as f32)).sqrt();

        let generator = || rng.gen_range(-boundary..boundary);
        let weights = matrix::Matrix::random(shape, generator);

        FullyConnectedLayer {
            weights,
            biases: matrix::Matrix::new(vec![0.0; shape.0], (shape.0, 1)),
            weights_gradient: None,
            biases_gradient: None,
            forward_pass_input: None,
        }
    }
}

impl layer::Layer for FullyConnectedLayer {
    /// Performs the forward pass of the fully connected layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The input matrix to the layer.
    ///
    /// # Returns
    ///
    /// The output matrix of the layer.
    fn forward(&mut self, input: &matrix::Matrix) -> matrix::Matrix {
        self.forward_pass_input = Some(input.clone());
        matrix::Matrix::add(&matrix::Matrix::dot(&self.weights, input), &self.biases)
    }

    /// Performs the backward pass of the fully connected layer.
    ///
    /// # Arguments
    ///
    /// * `gradient` - The gradient matrix/vector from the next layer.
    ///
    /// # Returns
    ///
    /// The gradient matrix/vector for the current layer.
    fn backward(&mut self, gradient: &matrix::Matrix) -> matrix::Matrix {
        match &self.forward_pass_input {
            None => panic!("No forward pass input found."),
            Some(forward_pass_input) => {
                self.weights_gradient = Some(matrix::Matrix::dot(
                    gradient,
                    &forward_pass_input.transpose(),
                ));
                self.biases_gradient = Some(gradient.clone());
            }
        }

        matrix::Matrix::dot(&self.weights.transpose(), gradient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::Layer;
    use approx::{assert_relative_eq, assert_relative_ne};
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
                    assert_relative_ne!(prev_variance, variance, epsilon = 1e-4);
                    previous_weight_variance = Some(variance);
                }
            }
        }
    }

    #[test]
    fn test_fully_connected_layer_forward() {
        let mut layer = FullyConnectedLayer {
            weights: matrix::Matrix::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2)),
            biases: matrix::Matrix::new(vec![1.0, 2.0], (2, 1)),
            weights_gradient: None,
            biases_gradient: None,
            forward_pass_input: None,
        };

        let input = matrix::Matrix::new(vec![1.0, 2.0], (2, 1));
        let output = layer.forward(&input);

        assert_eq!(output.get_shape(), (2, 1));
        assert_relative_eq!(output.get_value((0, 0)), 6.0, epsilon = 1e-4);
        assert_relative_eq!(output.get_value((1, 0)), 13.0, epsilon = 1e-4);
    }

    #[test]
    fn test_fully_connected_layer_backward() {
        let mut layer = FullyConnectedLayer {
            weights: matrix::Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3)),
            biases: matrix::Matrix::new(vec![1.0, 2.0], (2, 1)),
            weights_gradient: None,
            biases_gradient: None,
            forward_pass_input: None,
        };

        let input = matrix::Matrix::new(vec![1.0, 2.0, 3.0], (3, 1));
        let output = layer.forward(&input);

        assert_eq!(output.get_shape(), (2, 1));
        assert_relative_eq!(output.get_value((0, 0)), 15.0, epsilon = 1e-4);
        assert_relative_eq!(output.get_value((1, 0)), 34.0, epsilon = 1e-4);

        let gradient = matrix::Matrix::new(vec![1.0, 2.0], (2, 1));
        let backward_output = layer.backward(&gradient);

        assert_eq!(backward_output.get_shape(), (3, 1));
        assert_relative_eq!(backward_output.get_value((0, 0)), 9.0, epsilon = 1e-4);
        assert_relative_eq!(backward_output.get_value((1, 0)), 12.0, epsilon = 1e-4);
        assert_relative_eq!(backward_output.get_value((2, 0)), 15.0, epsilon = 1e-4);

        let w_gradient = layer.weights_gradient.unwrap();

        assert_eq!(w_gradient.get_shape(), (2, 3));
        assert_relative_eq!(w_gradient.get_value((0, 0)), 1.0, epsilon = 1e-4);
        assert_relative_eq!(w_gradient.get_value((0, 1)), 2.0, epsilon = 1e-4);
        assert_relative_eq!(w_gradient.get_value((0, 2)), 3.0, epsilon = 1e-4);
        assert_relative_eq!(w_gradient.get_value((1, 0)), 2.0, epsilon = 1e-4);
        assert_relative_eq!(w_gradient.get_value((1, 1)), 4.0, epsilon = 1e-4);
        assert_relative_eq!(w_gradient.get_value((1, 2)), 6.0, epsilon = 1e-4);

        let b_gradient = layer.biases_gradient.unwrap();
        assert_eq!(b_gradient.get_shape(), (2, 1));
        assert_relative_eq!(b_gradient.get_value((0, 0)), 1.0, epsilon = 1e-4);
        assert_relative_eq!(b_gradient.get_value((1, 0)), 2.0, epsilon = 1e-4);
    }
}
