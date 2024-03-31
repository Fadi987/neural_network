use crate::layer::Layer;
use crate::matrix::Matrix;

pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Tanh,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh(x: f32) -> f32 {
    x.tanh()
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub struct ActivationLayer {
    activation_function: ActivationFunction,
}

impl ActivationLayer {
    pub fn new(activation_function: ActivationFunction) -> Self {
        ActivationLayer {
            activation_function,
        }
    }
}

impl Layer for ActivationLayer {
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
