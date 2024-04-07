use crate::matrix;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyclass]
struct PyClosure {
    f: Box<dyn FnMut() -> f32 + Send>,
}

#[pyclass]
pub struct PyMatrix {
    matrix: matrix::Matrix,
}

#[pymethods]
impl PyMatrix {
    #[staticmethod]
    pub fn zeros(shape: &PyTuple) -> Self {
        let shape: (usize, usize) = shape.extract().unwrap();
        PyMatrix {
            matrix: matrix::Matrix::zeros(shape),
        }
    }

    fn __str__(&self) -> String {
        format!("{}", self.matrix)
    }
}
