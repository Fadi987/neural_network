use nn_core::matrix;
use pyo3::prelude::*;
use pyo3::types::{PyFunction, PyList};

#[pyclass]
pub struct Matrix {
    pub matrix: matrix::Matrix,
}

impl Matrix {
    pub fn get_matrix(&self) -> &matrix::Matrix {
        &self.matrix
    }
}

#[pymethods]
impl Matrix {
    #[staticmethod]
    pub fn zeros(shape: (usize, usize)) -> Self {
        Matrix {
            matrix: matrix::Matrix::zeros(shape),
        }
    }

    #[staticmethod]
    pub fn from_row_major(data: &Bound<'_, PyList>, shape: (usize, usize)) -> Self {
        Matrix {
            matrix: matrix::Matrix::from_row_major(data.extract().unwrap(), shape),
        }
    }

    #[staticmethod]
    pub fn from_2d_vector(data: &Bound<'_, PyList>) -> Self {
        Matrix {
            matrix: matrix::Matrix::from_2d_vector(data.extract().unwrap()),
        }
    }

    pub fn get_shape(&self) -> (usize, usize) {
        self.matrix.get_shape()
    }

    pub fn get_value(&self, indices: (usize, usize)) -> f32 {
        self.matrix.get_value(indices)
    }

    pub fn get_row(&self, index: usize) -> Matrix {
        Matrix {
            matrix: self.matrix.get_row(index),
        }
    }

    pub fn get_column(&self, index: usize) -> Matrix {
        Matrix {
            matrix: self.matrix.get_column(index),
        }
    }

    pub fn transpose(&self) -> Matrix {
        Matrix {
            matrix: self.matrix.transpose(),
        }
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        Matrix {
            matrix: matrix::Matrix::add(&self.matrix, &other.matrix),
        }
    }

    pub fn sub(&self, other: &Matrix) -> Matrix {
        Matrix {
            matrix: matrix::Matrix::sub(&self.matrix, &other.matrix),
        }
    }

    pub fn mul(&self, other: &Matrix) -> Matrix {
        Matrix {
            matrix: matrix::Matrix::mul(&self.matrix, &other.matrix),
        }
    }

    pub fn mul_scalar(&self, scalar: f32) -> Matrix {
        Matrix {
            matrix: matrix::Matrix::mul_scalar(&self.matrix, scalar),
        }
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        Matrix {
            matrix: matrix::Matrix::dot(&self.matrix, &other.matrix),
        }
    }

    pub fn map(&self, lambda: &Bound<'_, PyFunction>) -> Matrix {
        let f = |x: f32| -> f32 { lambda.call1((x,)).unwrap().extract().unwrap() };

        Matrix {
            matrix: matrix::Matrix::map(&self.matrix, f),
        }
    }

    pub fn to_2d_list(&self) -> Vec<Vec<f32>> {
        self.matrix.to_2d_vec()
    }

    pub fn sum(&self) -> f32 {
        self.matrix.sum()
    }

    pub fn __str__(&self) -> String {
        format!("{}", self.matrix)
    }
}
