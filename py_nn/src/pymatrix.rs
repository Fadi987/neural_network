use nn_core::matrix;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyclass]
pub struct PyMatrix {
    matrix: matrix::Matrix,
}

#[pymethods]
impl PyMatrix {
    #[staticmethod]
    pub fn zeros(shape: &Bound<'_, PyTuple>) -> Self {
        let shape: (usize, usize) = shape.extract().unwrap();
        PyMatrix {
            matrix: matrix::Matrix::zeros(shape),
        }
    }

    #[new]
    pub fn new(shape: &Bound<'_, PyTuple>, value: f32) -> Self {
        let shape: (usize, usize) = shape.extract().unwrap();
        PyMatrix {
            matrix: matrix::Matrix::new(vec![value; shape.0 * shape.1], shape),
        }
    }

    fn __str__(&self) -> String {
        format!("{}", self.matrix)
    }
}
