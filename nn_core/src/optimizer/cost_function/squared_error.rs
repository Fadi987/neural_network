use crate::matrix;
use crate::optimizer::cost_function::CostFunction;

/// Represents the squared error cost function.
pub struct SquaredError;

impl CostFunction for SquaredError {
    fn compute(&mut self, input: &matrix::Matrix, target: &matrix::Matrix) -> f32 {
        // squared error for a single sample shoudl be a scalar
        assert_eq!(input.get_shape(), (1, 1));
        assert_eq!(input.get_shape(), target.get_shape());

        let diff_squared = matrix::Matrix::sub(input, target).map(|x| x.powi(2));
        diff_squared.sum()
    }

    fn gradient(&mut self, input: &matrix::Matrix, target: &matrix::Matrix) -> matrix::Matrix {
        assert_eq!(input.get_shape(), (1, 1));
        assert_eq!(input.get_shape(), target.get_shape());

        matrix::Matrix::sub(input, target).map(|x| 2.0 * x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;
    use approx::assert_relative_eq;

    #[test]
    fn test_squared_error_compute() {
        let input = Matrix::from_row_major(vec![1.0], (1, 1));
        let target = Matrix::from_row_major(vec![3.0], (1, 1));

        let cost = SquaredError.compute(&input, &target);
        assert_relative_eq!(cost, 4.0, epsilon = 1e-4);
    }

    #[test]
    fn test_squared_error_gradient() {
        let input = Matrix::from_row_major(vec![1.0], (1, 1));
        let target = Matrix::from_row_major(vec![0.0], (1, 1));

        let gradient = SquaredError.gradient(&input, &target);

        assert_relative_eq!(gradient.get_value((0, 0)), 2.0, epsilon = 1e-4);
    }
}
