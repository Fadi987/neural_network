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
}
