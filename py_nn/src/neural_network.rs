use crate::matrix;
use nn_core::neural_network;
use pyo3::prelude::*;

// pub mod layer;

#[pyclass]
pub struct NeuralNetwork {
    pub neural_network: neural_network::NeuralNetwork,
}

#[pymethods]
impl NeuralNetwork {
    #[new]
    pub fn new() -> Self {
        NeuralNetwork {
            neural_network: neural_network::NeuralNetwork::new(),
        }
    }

    pub fn add_fully_connected_layer(&mut self, shape: (usize, usize)) {
        let fully_connected_layer =
            neural_network::layer::fully_connected::FullyConnectedLayer::new(
                shape,
                &mut rand::thread_rng(),
            );

        self.neural_network.add_layer(fully_connected_layer);
    }

    pub fn add_activation_layer(&mut self, activation_function: String) {
        let activation_layer = match activation_function.as_str() {
            "relu" => neural_network::layer::activation::ActivationLayer::new(
                neural_network::layer::activation::ActivationFunction::ReLU,
            ),
            "sigmoid" => neural_network::layer::activation::ActivationLayer::new(
                neural_network::layer::activation::ActivationFunction::Sigmoid,
            ),
            "tanh" => neural_network::layer::activation::ActivationLayer::new(
                neural_network::layer::activation::ActivationFunction::Tanh,
            ),
            _ => panic!("Invalid activation function {}.", activation_function),
        };

        self.neural_network.add_layer(activation_layer);
    }

    pub fn forward(&mut self, input: &matrix::Matrix) -> matrix::Matrix {
        matrix::Matrix {
            matrix: self.neural_network.forward(&input.matrix),
        }
    }

    pub fn backward(&mut self, output: &matrix::Matrix) -> matrix::Matrix {
        matrix::Matrix {
            matrix: self.neural_network.backward(&output.matrix),
        }
    }

    pub fn update(&mut self, learning_rate: f32) {
        self.neural_network.update(learning_rate);
    }
}
