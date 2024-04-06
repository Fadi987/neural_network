pub mod cost_function;
use crate::matrix;
use crate::neural_network;

pub struct Optimizer<T: cost_function::CostFunction> {
    learning_rate: f32,
    cost_function: Box<T>,
    neural_network: neural_network::NeuralNetwork,
}

impl<T: cost_function::CostFunction> Optimizer<T> {
    pub fn new(
        learning_rate: f32,
        cost_function: T,
        neural_network: neural_network::NeuralNetwork,
    ) -> Self {
        Optimizer {
            learning_rate,
            cost_function: Box::new(cost_function),
            neural_network,
        }
    }

    pub fn train_on_example(&mut self, input: &matrix::Matrix, target: &matrix::Matrix) {
        let output = self.neural_network.forward(input);
        let gradient = self.cost_function.gradient(&output, target);
        let _ = self.neural_network.backward(&gradient);
        self.neural_network.update(self.learning_rate);
    }

    pub fn train_on_sample(&mut self, rhs: &matrix::Matrix, lhs: &matrix::Matrix) {
        assert_eq!(rhs.get_shape().0, lhs.get_shape().0);
        assert_eq!(lhs.get_shape().1, 1);

        for i in 0..rhs.get_shape().0 {
            self.train_on_example(&rhs.get_row(i).transpose(), &lhs.get_row(i));
        }
    }
}
