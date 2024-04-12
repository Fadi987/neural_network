use crate::matrix;
use nn_core::neural_network;
use pyo3::prelude::*;

#[pyclass]
pub struct NeuralNetwork {
    neural_network: neural_network::NeuralNetwork,
}

#[pymethods]
impl NeuralNetwork {
    #[new]
    pub fn new() -> Self {
        NeuralNetwork {
            neural_network: neural_network::NeuralNetwork::new(),
        }
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
