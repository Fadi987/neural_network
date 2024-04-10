//! This module contains the implementation of the `Matrix` struct and its associated methods.

use std::fmt;

/// Represents a matrix of floating-point numbers.
#[derive(Clone)]
pub struct Matrix {
    data: Vec<f32>,
    shape: (usize, usize),
}

impl Matrix {
    /// Creates a new matrix filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the matrix (number of rows, number of columns).
    ///
    /// # Example
    ///
    /// ```
    /// use nn_core::matrix::Matrix;
    /// let matrix = Matrix::zeros((2, 3));
    /// ```
    pub fn zeros(shape: (usize, usize)) -> Self {
        Matrix {
            data: vec![0.0; shape.0 * shape.1],
            shape,
        }
    }

    /// Creates a new matrix filled with random values.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the matrix (number of rows, number of columns).
    /// * `generator` - A closure that generates random values.
    ///
    /// # Example
    ///
    /// ```
    /// use nn_core::matrix::Matrix;
    /// let matrix = Matrix::random((2, 3), || rand::random());
    /// ```
    pub fn random<F>(shape: (usize, usize), mut generator: F) -> Self
    where
        F: FnMut() -> f32,
    {
        let data: Vec<f32> = (0..shape.0 * shape.1).map(|_| generator()).collect();
        Matrix { data: data, shape }
    }

    /// Creates a new matrix from existing data.
    ///
    /// # Arguments
    ///
    /// * `data` - The data of the matrix, stored in row-major order.
    /// * `shape` - The shape of the matrix (number of rows, number of columns).
    ///
    /// # Panics
    ///
    /// This function will panic if the size of the input data is not compatible with the input shape.
    ///
    /// # Example
    ///
    /// ```
    /// use nn_core::matrix::Matrix;
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let matrix = Matrix::from_row_major(data, (2, 2));
    /// ```
    pub fn from_row_major(data: Vec<f32>, shape: (usize, usize)) -> Self {
        assert_eq!(
            data.len(),
            shape.0 * shape.1,
            "Size of input data is not compatible with input shape."
        );

        Matrix { data, shape }
    }

    /// Returns the shape of the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use nn_core::matrix::Matrix;
    /// let matrix = Matrix::zeros((2, 3));
    /// let shape = matrix.get_shape();
    /// assert_eq!(shape, (2, 3));
    /// ```
    pub fn get_shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Returns the value at the specified indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - The indices of the value to retrieve (row index, column index).
    ///
    /// # Panics
    ///
    /// This function will panic if the indices are out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use nn_core::matrix::Matrix;
    /// let matrix = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    /// let value = matrix.get_value((1, 0));
    /// assert_eq!(value, 3.0);
    /// ```
    pub fn get_value(&self, indices: (usize, usize)) -> f32 {
        assert!(indices.0 < self.shape.0 && indices.1 < self.shape.1);
        self.data[indices.0 * self.shape.1 + indices.1]
    }

    /// Returns the row at the specified index.
    /// # Arguments
    ///
    /// * `row` - The index of the row to retrieve.
    ///
    /// # Panics
    /// This function will panic if the row index is out of bounds.
    ///
    /// # Example
    /// ```
    /// use nn_core::matrix::Matrix;
    /// let matrix = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    /// let row = matrix.get_row(1);
    /// ```

    pub fn get_row(&self, row: usize) -> Matrix {
        assert!(row < self.shape.0);
        let data = self.data[row * self.shape.1..(row + 1) * self.shape.1].to_vec();
        Matrix {
            data,
            shape: (1, self.shape.1),
        }
    }

    /// Returns the column at the specified index.
    /// # Arguments
    ///
    /// * `column` - The index of the column to retrieve.
    ///
    /// # Panics
    ///
    /// This function will panic if the column index is out of bounds.
    ///
    /// # Example
    /// ```
    /// use nn_core::matrix::Matrix;
    /// let matrix = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    /// let column = matrix.get_column(1);
    /// ````
    pub fn get_column(&self, column: usize) -> Matrix {
        assert!(column < self.shape.1);
        let data = (0..self.shape.0)
            .map(|i| self.data[i * self.shape.1 + column])
            .collect();
        Matrix {
            data,
            shape: (self.shape.0, 1),
        }
    }

    /// Transposes the matrix.
    ///
    /// # Returns
    ///
    /// The transposed matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use nn_core::matrix::Matrix;
    /// let matrix = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    /// let transposed = matrix.transpose();
    /// ```
    pub fn transpose(&self) -> Self {
        let data = (0..self.shape.1)
            .flat_map(|j| (0..self.shape.0).map(move |i| (i, j)))
            .map(|indices| self.get_value(indices))
            .collect();

        Matrix {
            data,
            shape: (self.shape.1, self.shape.0),
        }
    }

    /// Performs element-wise addition of two matrices.
    ///
    /// # Arguments
    ///
    /// * `matrix_a` - The first matrix.
    /// * `matrix_b` - The second matrix.
    ///
    /// # Panics
    ///
    /// This function will panic if the shapes of the matrices are different.
    ///
    /// # Example
    ///
    /// ```
    /// use nn_core::matrix::Matrix;
    /// let matrix_a = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    /// let matrix_b = Matrix::from_row_major(vec![4.0, 3.0, 2.0, 1.0], (2, 2));
    /// let result = Matrix::add(&matrix_a, &matrix_b);
    /// ```
    pub fn add(matrix_a: &Matrix, matrix_b: &Matrix) -> Self {
        assert_eq!(
            matrix_a.shape, matrix_b.shape,
            "Cannot perform element-wise addition on two matrices of different shapes"
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

    /// Performs element-wise subtraction of two matrices.
    ///
    /// # Arguments
    ///
    /// * `matrix_a` - The first matrix.
    /// * `matrix_b` - The second matrix.
    ///
    /// # Panics
    ///
    /// This function will panic if the shapes of the matrices are different.
    ///
    /// # Example
    ///
    /// ```
    /// use nn_core::matrix::Matrix;
    /// let matrix_a = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    /// let matrix_b = Matrix::from_row_major(vec![4.0, 3.0, 2.0, 1.0], (2, 2));
    /// let result = Matrix::sub(&matrix_a, &matrix_b);
    /// ```
    pub fn sub(matrix_a: &Matrix, matrix_b: &Matrix) -> Self {
        assert_eq!(
            matrix_a.shape, matrix_b.shape,
            "Cannot perform element-wise subtraction on two matrices of different shapes"
        );

        let data = matrix_a
            .data
            .iter()
            .zip(matrix_b.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Matrix {
            data,
            shape: matrix_a.shape,
        }
    }

    /// Performs element-wise multiplication of two matrices.
    ///
    /// # Arguments
    ///
    /// * `matrix_a` - The first matrix.
    /// * `matrix_b` - The second matrix.
    ///
    /// # Panics
    ///
    /// This function will panic if the shapes of the matrices are different.
    ///
    /// # Example
    ///
    /// ```
    /// use nn_core::matrix::Matrix;
    /// let matrix_a = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    /// let matrix_b = Matrix::from_row_major(vec![4.0, 3.0, 2.0, 1.0], (2, 2));
    /// let result = Matrix::mul(&matrix_a, &matrix_b);
    /// ```
    pub fn mul(matrix_a: &Matrix, matrix_b: &Matrix) -> Self {
        assert_eq!(
            matrix_a.shape, matrix_b.shape,
            "Cannot perform element-wise multiplication on two matrices of different shapes"
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

    /// Performs matrix multiplication of two matrices.
    ///
    /// # Arguments
    ///
    /// * `matrix_a` - The left-hand matrix.
    /// * `matrix_b` - The right-hand matrix.
    ///
    /// # Panics
    ///
    /// This function will panic if the number of columns of `matrix_a` is different than the number of rows of `matrix_b`.
    ///
    /// # Example
    ///
    /// ```
    /// use nn_core::matrix::Matrix;
    /// let matrix_a = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    /// let matrix_b = Matrix::from_row_major(vec![2.0, 0.0, 1.0, 3.0], (2, 2));
    /// let result = Matrix::dot(&matrix_a, &matrix_b);
    /// ```
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

    /// Applies a function element-wise to the matrix.
    ///
    /// # Arguments
    ///
    /// * `f` - The function to apply to each element of the matrix.
    ///
    /// # Returns
    ///
    /// The resulting matrix after applying the function element-wise.
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        let data = self.data.iter().map(|&x| f(x)).collect();
        Matrix {
            data,
            shape: self.shape,
        }
    }

    /// Multiplies all the elements of the matrix by a scalar.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar to multiply the elements of the matrix by.
    ///
    /// # Returns
    ///
    /// The resulting matrix after multiplying all the elements by the scalar.
    pub fn mul_scalar(&self, scalar: f32) -> Self {
        self.map(|x| x * scalar)
    }

    /// Sums all the elements of the matrix.
    ///
    /// # Returns
    ///
    /// The sum of all the elements in the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use nn_core::matrix::Matrix;
    /// let matrix = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    /// let sum = matrix.sum();
    /// assert_eq!(sum, 10.0);
    /// ```
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for row in 0..self.shape.0 {
            for col in 0..self.shape.1 {
                write!(f, "{: >8.2}", self.data[row * self.shape.1 + col])?;
            }
            // After printing each row, add a newline
            writeln!(f)?;
        }
        Ok(())
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
    fn test_transpose() {
        let matrix = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let result = matrix.transpose();
        assert_eq!(result.data, vec![1.0, 3.0, 2.0, 4.0]);
        assert_eq!(result.shape, (2, 2));
    }

    #[test]
    fn test_get_row() {
        let matrix = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let result = matrix.get_row(1);
        assert_eq!(result.data, vec![3.0, 4.0]);
        assert_eq!(result.shape, (1, 2));
    }

    #[test]
    fn test_get_column() {
        let matrix = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let result = matrix.get_column(1);
        assert_eq!(result.data, vec![2.0, 4.0]);
        assert_eq!(result.shape, (2, 1));
    }

    #[test]
    fn test_addition() {
        let matrix_a = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let matrix_b = Matrix::from_row_major(vec![4.0, 3.0, 2.0, 1.0], (2, 2));
        let result = Matrix::add(&matrix_a, &matrix_b);
        assert_eq!(result.data, vec![5.0, 5.0, 5.0, 5.0]);
        assert_eq!(result.shape, (2, 2));
    }

    #[test]
    fn test_subtraction() {
        let matrix_a = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let matrix_b = Matrix::from_row_major(vec![4.0, 3.0, 2.0, 1.0], (2, 2));
        let result = Matrix::sub(&matrix_a, &matrix_b);
        assert_eq!(result.data, vec![-3.0, -1.0, 1.0, 3.0]);
        assert_eq!(result.shape, (2, 2));
    }

    #[test]
    fn test_multiplication() {
        let matrix_a = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let matrix_b = Matrix::from_row_major(vec![4.0, 3.0, 2.0, 1.0], (2, 2));
        let result = Matrix::mul(&matrix_a, &matrix_b);
        assert_eq!(result.data, vec![4.0, 6.0, 6.0, 4.0]);
        assert_eq!(result.shape, (2, 2));
    }

    // Test applying a function element-wise to a matrix
    #[test]
    fn test_map() {
        let matrix = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let result = matrix.map(|x| x * 2.0);
        assert_eq!(result.data, vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(result.shape, (2, 2));
    }

    #[test]
    fn test_mul_scalar() {
        let matrix = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let result = matrix.mul_scalar(2.0);
        assert_eq!(result.data, vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(result.shape, (2, 2));
    }

    #[test]
    fn test_sum() {
        let matrix = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let result = matrix.sum();
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_dot_product() {
        // Create two matrices for the test
        let matrix_a = Matrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], (2, 2)); // 2x2 matrix
        let matrix_b = Matrix::from_row_major(vec![2.0, 0.0, 1.0, 3.0], (2, 2)); // 2x2 matrix

        // Expected result of the dot product
        let expected = Matrix::from_row_major(vec![4.0, 6.0, 10.0, 12.0], (2, 2)); // 2x2 matrix

        // Perform the dot product
        let result = Matrix::dot(&matrix_a, &matrix_b);

        // Check if the result matches the expected outcome
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }
}
