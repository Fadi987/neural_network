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
