pub struct Matrix {
    data: Vec<f32>,
    shape: (usize, usize),
}

impl Matrix {
    pub fn zeros(shape: (usize, usize)) -> Self {
        Matrix {
            data: vec![0.0; shape.0 * shape.1],
            shape,
        }
    }

    pub fn random<F>(shape: (usize, usize), generator: F) -> Self
    where
        F: Fn() -> f32,
    {
        let data: Vec<f32> = (0..shape.0 * shape.1).map(|_| generator()).collect();
        Matrix { data: data, shape }
    }

    pub fn new(data: Vec<f32>, shape: (usize, usize)) -> Self {
        assert_eq!(
            data.len(),
            shape.0 * shape.1,
            "Size of input data is not compatible with input shape."
        );

        Matrix { data, shape }
    }

    pub fn add(matrix_a: &Matrix, matrix_b: &Matrix) -> Self {
        assert_eq!(
            matrix_a.shape, matrix_b.shape,
            "Cannot perform element-wise addition on two matrixs of different shapes"
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

    pub fn mul(matrix_a: &Matrix, matrix_b: &Matrix) -> Self {
        assert_eq!(
            matrix_a.shape, matrix_b.shape,
            "Cannot perform element-wise multiplication on two matrixs of different shapes"
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

    pub fn dot(matrix_a: &Matrix, matrix_b: &Matrix) -> Self {
        assert_eq!(
            matrix_a.shape.1, matrix_b.shape.0,
            "Cannot multiply left-hand matrix with number of columns different than number of rows of right-hand matrix."
        );

        let mut data = vec![0.0; matrix_a.shape.0 * matrix_b.shape.1];

        for row_a in 0..matrix_a.shape.0 {
            for col_b in 0..matrix_b.shape.1 {
                for k in 0..matrix_a.shape.1 {
                    data[row_a * matrix_b.shape.1 + col_b] += matrix_a.data
                        [row_a * matrix_a.shape.1 + k]
                        * matrix_b.data[k * matrix_b.shape.1 + col_b]
                }
            }
        }

        Matrix {
            data,
            shape: (matrix_a.shape.0, matrix_b.shape.1),
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
        assert_eq!(result.shape, (2, 2));
    }

    #[test]
    fn test_random() {
        // We pass in a deterministic generator for testing
        let result = Matrix::random((2, 2), || 1.0);
        assert_eq!(result.data, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(result.shape, (2, 2));
    }

    #[test]
    fn test_addition() {
        let matrix_a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let matrix_b = Matrix::new(vec![4.0, 3.0, 2.0, 1.0], (2, 2));
        let result = Matrix::add(&matrix_a, &matrix_b);
        assert_eq!(result.data, vec![5.0, 5.0, 5.0, 5.0]);
        assert_eq!(result.shape, (2, 2));
    }

    #[test]
    fn test_multiplication() {
        let matrix_a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let matrix_b = Matrix::new(vec![4.0, 3.0, 2.0, 1.0], (2, 2));
        let result = Matrix::mul(&matrix_a, &matrix_b);
        assert_eq!(result.data, vec![4.0, 6.0, 6.0, 4.0]);
        assert_eq!(result.shape, (2, 2));
    }

    #[test]
    fn test_dot_product() {
        // Create two matrixs for the test
        let matrix_a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2)); // 2x2 matrix
        let matrix_b = Matrix::new(vec![2.0, 0.0, 1.0, 3.0], (2, 2)); // 2x2 matrix

        // Expected result of the dot product
        let expected = Matrix::new(vec![4.0, 6.0, 10.0, 12.0], (2, 2)); // 2x2 matrix

        // Perform the dot product
        let result = Matrix::dot(&matrix_a, &matrix_b);

        // Check if the result matches the expected outcome
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }
}
