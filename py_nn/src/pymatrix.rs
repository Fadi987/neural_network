use nn_core::matrix;
use pyo3::prelude::*;
use pyo3::types::PyFunction;
use pyo3::Python;

#[pyclass]
pub struct PyMatrix {
    matrix: matrix::Matrix,
}

#[pymethods]
impl PyMatrix {
    #[staticmethod]
    pub fn zeros(shape: (usize, usize)) -> Self {
        PyMatrix {
            matrix: matrix::Matrix::zeros(shape),
        }
    }

    pub fn get_row(&self, index: usize) -> PyMatrix {
        PyMatrix {
            matrix: self.matrix.get_row(index),
        }
    }

    pub fn get_column(&self, index: usize) -> PyMatrix {
        PyMatrix {
            matrix: self.matrix.get_column(index),
        }
    }

    pub fn transpose(&self) -> PyMatrix {
        PyMatrix {
            matrix: self.matrix.transpose(),
        }
    }

    pub fn add(&self, other: &PyMatrix) -> PyMatrix {
        PyMatrix {
            matrix: matrix::Matrix::add(&self.matrix, &other.matrix),
        }
    }

    pub fn sub(&self, other: &PyMatrix) -> PyMatrix {
        PyMatrix {
            matrix: matrix::Matrix::sub(&self.matrix, &other.matrix),
        }
    }

    pub fn mul(&self, other: &PyMatrix) -> PyMatrix {
        PyMatrix {
            matrix: matrix::Matrix::mul(&self.matrix, &other.matrix),
        }
    }

    pub fn mul_scalar(&self, scalar: f32) -> PyMatrix {
        PyMatrix {
            matrix: matrix::Matrix::mul_scalar(&self.matrix, scalar),
        }
    }

    pub fn dot(&self, other: &PyMatrix) -> PyMatrix {
        PyMatrix {
            matrix: matrix::Matrix::dot(&self.matrix, &other.matrix),
        }
    }

    pub fn map(&self, lambda: &Bound<'_, PyFunction>) -> PyMatrix {
        let f = |x: f32| -> f32 { lambda.call1((x,)).unwrap().extract().unwrap() };

        PyMatrix {
            matrix: matrix::Matrix::map(&self.matrix, f),
        }
    }

    pub fn sum(&self) -> f32 {
        self.matrix.sum()
    }

    pub fn __str__(&self) -> String {
        format!("{}", self.matrix)
    }
}
