/// This module contains the implementation of activation functions and an activation layer.
/// Activation functions are mathematical functions that introduce non-linearity to neural networks.
/// The activation layer applies an activation function element-wise to the input matrix.
/// Supported activation functions are Sigmoid, ReLU, and Tanh.
use crate::layer::Layer;
use crate::matrix;

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

/// Computes the derivative of the sigmoid function for a given input.
fn sigmoid_derivative(x: f32) -> f32 {
    let sigmoid_x = sigmoid(x);
    sigmoid_x * (1.0 - sigmoid_x)
}

/// Computes the hyperbolic tangent function for a given input.
fn tanh(x: f32) -> f32 {
    x.tanh()
}

/// Computes the derivative of the hyperbolic tangent function for a given input.
fn tanh_derivative(x: f32) -> f32 {
    1.0 - x.tanh().powi(2)
}

/// Computes the rectified linear unit function for a given input.
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Computes the derivative of the rectified linear unit function for a given input.
fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Struct representing an activation layer.
pub struct ActivationLayer {
    activation_function: ActivationFunction,
    forward_pass_input: Option<matrix::Matrix>,
}

impl ActivationLayer {
    /// Creates a new activation layer with the specified activation function.
    pub fn new(activation_function: ActivationFunction) -> Self {
        ActivationLayer {
            activation_function,
            forward_pass_input: None,
        }
    }
}

impl Layer for ActivationLayer {
    /// Applies the activation function element-wise to the input matrix and returns the result.
    fn forward(&mut self, input: &matrix::Matrix) -> matrix::Matrix {
        // Store the input for the backward pass.
        self.forward_pass_input = Some(input.clone());

        match self.activation_function {
            ActivationFunction::Sigmoid => input.map(sigmoid),
            ActivationFunction::ReLU => input.map(relu),
            ActivationFunction::Tanh => input.map(tanh),
        }
    }

    /// Computes the element-wise derivative of the activation function at forward pass input and multiplies it with the gradient.
    fn backward(&mut self, gradient: &matrix::Matrix) -> matrix::Matrix {
        match &self.forward_pass_input {
            None => panic!("No forward pass input found."),
            Some(forward_pass_input) => matrix::Matrix::mul(
                &gradient,
                &match self.activation_function {
                    ActivationFunction::Sigmoid => forward_pass_input.map(sigmoid_derivative),
                    ActivationFunction::ReLU => forward_pass_input.map(relu_derivative),
                    ActivationFunction::Tanh => forward_pass_input.map(tanh_derivative),
                },
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

    #[test]
    fn test_activation_layer_forward() {
        let mut layer = ActivationLayer::new(ActivationFunction::Sigmoid);
        let input = Matrix::new(vec![-1.0, 0.0, 1.0, 2.0], (2, 2));
        let output = layer.forward(&input);

        assert!((output.get_value((0, 0)) - 0.2689).abs() < 1e-4);
        assert!((output.get_value((0, 1)) - 0.5000).abs() < 1e-4);
        assert!((output.get_value((1, 0)) - 0.7310).abs() < 1e-4);
        assert!((output.get_value((1, 1)) - 0.8807).abs() < 1e-4);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(-1.0) - 0.2689).abs() < 1e-4);
        assert!((sigmoid(0.0) - 0.5000).abs() < 1e-4);
        assert!((sigmoid(1.0) - 0.7310).abs() < 1e-4);
        assert!((sigmoid(2.0) - 0.8807).abs() < 1e-4);
    }

    #[test]
    fn test_tanh() {
        assert!((tanh(-1.0) - -0.7615).abs() < 1e-4);
        assert!((tanh(0.0) - 0.0000).abs() < 1e-4);
        assert!((tanh(1.0) - 0.7615).abs() < 1e-4);
        assert!((tanh(2.0) - 0.9640).abs() < 1e-4);
    }

    #[test]
    fn test_relu() {
        assert!((relu(-1.0) - 0.0000).abs() < 1e-4);
        assert!((relu(0.0) - 0.0000).abs() < 1e-4);
        assert!((relu(1.0) - 1.0000).abs() < 1e-4);
        assert!((relu(2.0) - 2.0000).abs() < 1e-4);
    }
}
