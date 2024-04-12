use nn_core::neural_network::layer;
use pyo3::prelude::*;

#[pyclass]
pub struct Layer {
    layer: Box<dyn layer::Layer>,
}

#[pymethods]
impl Layer {
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
