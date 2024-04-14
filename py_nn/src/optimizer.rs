use crate::matrix;
use nn_core::optimizer;
use pyo3::prelude::*;

#[pyclass]
pub struct MseOptimizer {
    optimizer: optimizer::Optimizer<optimizer::cost_function::squared_error::SquaredError>,
}

#[pymethods]
impl MseOptimizer {
    #[new]
    pub fn new(learning_rate: f32) -> Self {
        MseOptimizer {
            optimizer: optimizer::Optimizer::new(
                learning_rate,
                optimizer::cost_function::squared_error::SquaredError,
            ),
        }
    }

    pub fn train_on_sample(
        &mut self,
        neural_network: &mut crate::neural_network::NeuralNetwork,
        input: &matrix::Matrix,
        target: &matrix::Matrix,
    ) {
        self.optimizer.train_on_sample(
            &mut neural_network.neural_network,
            &input.matrix,
            &target.matrix,
        );
    }
}
