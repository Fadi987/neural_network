struct Matrix {
    data: Vec<f32>,
    shape: (usize, usize),
}

impl Matrix {
    pub fn zeros(shape: (usize, usize)) -> Matrix {
        Matrix {
            data: vec![0.0; shape.0 * shape.1],
            shape,
        }
    }

    pub fn new(data: Vec<f32>, shape: (usize, usize)) -> Matrix {
        assert_eq!(
            data.len(),
            shape.0 * shape.1,
            "Size of input data is not compatible with input shape."
        );

        Matrix { data, shape }
    }

    pub fn add(matrix_a: &Matrix, matrix_b: &Matrix) -> Matrix {
        assert_eq!(
            matrix_a.shape, matrix_b.shape,
            "Cannot add two matrixs of different shapes"
        );

        let data = matrix_a
            .data
            .iter()
            .zip(matrix_b.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Matrix {
            data,
            shape: matrix_a.shape,
        }
    }

    pub fn mul(matrix_a: &Matrix, matrix_b: &Matrix) -> Matrix {
        assert_eq!(
            matrix_a.shape, matrix_b.shape,
            "Cannot add two matrixs of different shapes"
        );

        let data = matrix_a
            .data
            .iter()
            .zip(matrix_b.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Matrix {
            data,
            shape: matrix_a.shape,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let result = Matrix::zeros((2, 2));
        assert_eq!(result.data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_addition() {
        let matrix_a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let matrix_b = Matrix::new(vec![4.0, 3.0, 2.0, 1.0], (2, 2));
        let result = Matrix::add(&matrix_a, &matrix_b);
        assert_eq!(result.data, vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_multiplication() {
        let matrix_a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let matrix_b = Matrix::new(vec![4.0, 3.0, 2.0, 1.0], (2, 2));
        let result = Matrix::mul(&matrix_a, &matrix_b);
        assert_eq!(result.data, vec![4.0, 6.0, 6.0, 4.0]);
    }

    // Add more tests for multiplication, dot product, etc.
}
