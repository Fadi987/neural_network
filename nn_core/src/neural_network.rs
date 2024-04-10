use crate::matrix;
pub mod layer;

pub struct NeuralNetwork {
    layers: Vec<Box<dyn layer::Layer>>,
}

/// Represents a neural network.
impl NeuralNetwork {
    /// Creates a new instance of the neural network.
    pub fn new() -> Self {
        NeuralNetwork { layers: Vec::new() }
    }

    pub fn add_layer<L: layer::Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    /// Performs the forward pass of the neural network.
    pub fn forward(&mut self, input: &matrix::Matrix) -> matrix::Matrix {
        // TODO: understand usage of &mut
        let mut intermediate_result = input.clone();

        for layer in self.layers.iter_mut() {
            intermediate_result = layer.forward(&intermediate_result);
        }

        intermediate_result
    }

    /// Performs the backward pass of the neural network.
    pub fn backward(&mut self, gradient: &matrix::Matrix) -> matrix::Matrix {
        let mut intermediate_result = gradient.clone();

        for layer in self.layers.iter_mut().rev() {
            intermediate_result = layer.backward(&intermediate_result);
        }

        intermediate_result
    }

    /// Updates the weights and biases of the neural network.
    pub fn update(&mut self, learning_rate: f32) {
        for layer in self.layers.iter_mut() {
            layer.update(learning_rate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use layer::{activation, fully_connected};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_neural_network_forward() {
        let mut network = NeuralNetwork::new();
        let seed = [42; 32];
        let mut rng = StdRng::from_seed(seed);

        let fully_connected_layer = fully_connected::FullyConnectedLayer::new((2, 3), &mut rng);
        let activation_layer =
            activation::ActivationLayer::new(activation::ActivationFunction::ReLU);

        network.add_layer(fully_connected_layer);
        network.add_layer(activation_layer);

        let input = matrix::Matrix::from_row_major(vec![1.0, 2.0, 3.0], (3, 1));
        let output = network.forward(&input);

        assert_eq!(output.get_shape(), (2, 1));
    }

    #[test]
    fn test_neural_network_forward_multiple_layers() {
        let mut network = NeuralNetwork::new();
        let seed = [42; 32];
        let mut rng = StdRng::from_seed(seed);

        let fully_connected_layer_1 = fully_connected::FullyConnectedLayer::new((2, 3), &mut rng);
        let activation_layer_1 =
            activation::ActivationLayer::new(activation::ActivationFunction::ReLU);
        let fully_connected_layer_2 = fully_connected::FullyConnectedLayer::new((2, 2), &mut rng);
        let activation_layer_2 =
            activation::ActivationLayer::new(activation::ActivationFunction::ReLU);

        network.add_layer(fully_connected_layer_1);
        network.add_layer(activation_layer_1);
        network.add_layer(fully_connected_layer_2);
        network.add_layer(activation_layer_2);

        let input = matrix::Matrix::from_row_major(vec![1.0, 2.0, 3.0], (3, 1));
        let output = network.forward(&input);

        assert_eq!(output.get_shape(), (2, 1));
    }

    #[test]
    fn test_neural_network_backward() {
        let mut network = NeuralNetwork::new();
        let seed = [42; 32];
        let mut rng = StdRng::from_seed(seed);

        let fully_connected_layer = fully_connected::FullyConnectedLayer::new((2, 3), &mut rng);
        let activation_layer =
            activation::ActivationLayer::new(activation::ActivationFunction::ReLU);

        let fully_connected_layer_pre_output =
            fully_connected::FullyConnectedLayer::new((1, 2), &mut rng);

        network.add_layer(fully_connected_layer);
        network.add_layer(activation_layer);
        network.add_layer(fully_connected_layer_pre_output);

        let input = matrix::Matrix::from_row_major(vec![1.0, 2.0, 3.0], (3, 1));
        let _ = network.forward(&input);

        let output_gradient = matrix::Matrix::from_row_major(vec![2.0], (1, 1));
        let input_gradient = network.backward(&output_gradient);

        assert_eq!(input_gradient.get_shape(), (3, 1));
    }
}
