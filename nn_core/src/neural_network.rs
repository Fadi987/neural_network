use crate::layer::Layer;
use crate::matrix;

struct NeuralNetwork {
    layers: Vec<Box<dyn Layer>>,
}

/// Represents a neural network.
impl NeuralNetwork {
    /// Creates a new instance of the neural network.
    pub fn new() -> Self {
        NeuralNetwork { layers: Vec::new() }
    }

    pub fn add_layer<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    pub fn forward(&self, input: &matrix::Matrix) -> matrix::Matrix {
        self.layers
            .iter()
            .fold(input.clone(), |acc, layer| layer.forward(&acc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{activation, fully_connected, Layer};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_neural_network_forward() {
        let mut network = NeuralNetwork::new();
        let seed = [42; 32];
        let rng = StdRng::from_seed(seed);

        let fully_connected_layer = fully_connected::FullyConnectedLayer::new((2, 3), rng);
        let activation_layer =
            activation::ActivationLayer::new(activation::ActivationFunction::ReLU);

        network.add_layer(fully_connected_layer);
        network.add_layer(activation_layer);

        let input = matrix::Matrix::new(vec![1.0, 2.0, 3.0], (3, 1));
        let output = network.forward(&input);

        assert_eq!(output.get_shape(), (2, 1));
    }
}
