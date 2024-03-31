/// This module contains the implementation of activation functions and an activation layer.
/// Activation functions are mathematical functions that introduce non-linearity to neural networks.
/// The activation layer applies an activation function element-wise to the input matrix.
/// Supported activation functions are Sigmoid, ReLU, and Tanh.
use crate::layer::Layer;
use crate::matrix::Matrix;

/// Enum representing different activation functions.
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Tanh,
}

/// Computes the sigmoid function for a given input.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Computes the hyperbolic tangent function for a given input.
fn tanh(x: f32) -> f32 {
    x.tanh()
}

/// Computes the rectified linear unit function for a given input.
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Struct representing an activation layer.
pub struct ActivationLayer {
    activation_function: ActivationFunction,
}

impl ActivationLayer {
    /// Creates a new activation layer with the specified activation function.
    pub fn new(activation_function: ActivationFunction) -> Self {
        ActivationLayer {
            activation_function,
        }
    }
}

impl Layer for ActivationLayer {
    /// Applies the activation function element-wise to the input matrix and returns the result.
    fn forward(&self, input: &Matrix) -> Matrix {
        let data = (0..input.get_shape().0)
            .flat_map(|i| (0..input.get_shape().1).map(move |j| (i, j)))
            .map(|indices| {
                let value = input.get_value(indices);
                match self.activation_function {
                    ActivationFunction::Sigmoid => sigmoid(value),
                    ActivationFunction::ReLU => relu(value),
                    ActivationFunction::Tanh => tanh(value),
                }
            })
            .collect::<Vec<f32>>();

        Matrix::new(data, input.get_shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

    #[test]
    fn test_activation_layer_forward() {
        let layer = ActivationLayer::new(ActivationFunction::Sigmoid);
        let input = Matrix::new(vec![-1.0, 0.0, 1.0, 2.0], (2, 2));
        let output = layer.forward(&input);

        assert!((output.get_value((0, 0)) - 0.2689414).abs() < 1e-4);
        assert!((output.get_value((0, 1)) - 0.5).abs() < 1e-4);
        assert!((output.get_value((1, 0)) - 0.7310586).abs() < 1e-4);
        assert!((output.get_value((1, 1)) - 0.8807971).abs() < 1e-4);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(-1.0) - 0.2689414).abs() < 1e-4);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-4);
        assert!((sigmoid(1.0) - 0.7310586).abs() < 1e-4);
        assert!((sigmoid(2.0) - 0.8807971).abs() < 1e-4);
    }

    #[test]
    fn test_tanh() {
        assert!((tanh(-1.0) - -0.76159416).abs() < 1e-4);
        assert!((tanh(0.0) - 0.0).abs() < 1e-4);
        assert!((tanh(1.0) - 0.76159416).abs() < 1e-4);
        assert!((tanh(2.0) - 0.9640276).abs() < 1e-4);
    }

    #[test]
    fn test_relu() {
        assert!((relu(-1.0) - 0.0).abs() < 1e-4);
        assert!((relu(0.0) - 0.0).abs() < 1e-4);
        assert!((relu(1.0) - 1.0).abs() < 1e-4);
        assert!((relu(2.0) - 2.0).abs() < 1e-4);
    }
}
