pub mod cost_function;
use crate::matrix;
use crate::neural_network;

/// The `Optimizer` struct represents an optimizer for a neural network.
pub struct Optimizer<T: cost_function::CostFunction> {
    learning_rate: f32,
    cost_function: Box<T>,
    neural_network: neural_network::NeuralNetwork,
}

impl<T: cost_function::CostFunction> Optimizer<T> {
    /// Creates a new `Optimizer` instance.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for the optimizer.
    /// * `cost_function` - The cost function to be used by the optimizer.
    /// * `neural_network` - The neural network to be optimized.
    ///
    /// # Returns
    ///
    /// A new `Optimizer` instance.
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

    /// Trains the neural network on a single example.
    ///
    /// # Arguments
    ///
    /// * `input` - The input matrix representing the example.
    /// * `target` - The target matrix representing the expected output.
    pub fn train_on_example(&mut self, input: &matrix::Matrix, target: &matrix::Matrix) {
        let output = self.neural_network.forward(input);
        let gradient = self.cost_function.gradient(&output, target);
        let _ = self.neural_network.backward(&gradient);
        self.neural_network.update(self.learning_rate);
    }

    /// Trains the neural network on a sample of examples.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The input matrix representing the sample.
    /// * `lhs` - The target matrix representing the expected outputs.
    ///
    /// # Panics
    ///
    /// This method will panic if the number of rows in `rhs` is not equal to the number of rows in `lhs`,
    /// or if the number of columns in `lhs` is not equal to 1.
    pub fn train_on_sample(&mut self, rhs: &matrix::Matrix, lhs: &matrix::Matrix) {
        assert_eq!(rhs.get_shape().0, lhs.get_shape().0);
        assert_eq!(lhs.get_shape().1, 1);

        for i in 0..rhs.get_shape().0 {
            self.train_on_example(&rhs.get_row(i).transpose(), &lhs.get_row(i));
        }
    }
}
