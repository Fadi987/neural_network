use nn_core::neural_network::layer;
use pyo3::prelude::*;

#[pyclass]
pub struct Layer {
    layer: Box<dyn layer::Layer>,
}

#[pymethods]
impl Layer {
    #[staticmethod]
    pub fn new_dense(shape: (usize, usize)) -> Self {
        Layer {
            layer: Box::new(layer::fully_connected::FullyConnectedLayer::new(
                shape,
                &mut rand::thread_rng(),
            )),
        }
    }

    #[staticmethod]
    pub fn new_activation(activation_function: String) -> Self {
        match activation_function.as_str() {
            "relu" => Layer {
                layer: Box::new(layer::activation::ActivationLayer::new(
                    layer::activation::ActivationFunction::ReLU,
                )),
            },
            "sigmoid" => Layer {
                layer: Box::new(layer::activation::ActivationLayer::new(
                    layer::activation::ActivationFunction::Sigmoid,
                )),
            },
            "tanh" => Layer {
                layer: Box::new(layer::activation::ActivationLayer::new(
                    layer::activation::ActivationFunction::Tanh,
                )),
            },
            _ => panic!("Invalid activation function {}.", activation_function),
        }
    }

    pub fn forward(&mut self, input: &crate::matrix::Matrix) -> crate::matrix::Matrix {
        crate::matrix::Matrix {
            matrix: self.layer.forward(&input.matrix),
        }
    }

    pub fn backward(&mut self, output: &crate::matrix::Matrix) -> crate::matrix::Matrix {
        crate::matrix::Matrix {
            matrix: self.layer.backward(&output.matrix),
        }
    }

    pub fn update(&mut self, learning_rate: f32) {
        self.layer.update(learning_rate);
    }
}
