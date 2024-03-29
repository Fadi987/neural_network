struct Tensor {
    data: Vec<f32>,
    shape: (usize, usize),
}

impl Tensor {
    pub fn zeros(shape: (usize, usize)) -> Tensor {
        Tensor {
            data: vec![0.0; shape.0 * shape.1],
            shape,
        }
    }

    pub fn new(data: Vec<f32>, shape: (usize, usize)) -> Tensor {
        assert_eq!(
            data.len(),
            shape.0 * shape.1,
            "Size of input data is not compatible with input shape."
        );

        Tensor { data, shape }
    }

    pub fn add(tensor_a: &Tensor, tensor_b: &Tensor) -> Tensor {
        assert_eq!(
            tensor_a.shape, tensor_b.shape,
            "Cannot add two tensors of different shapes"
        );

        let data = tensor_a
            .data
            .iter()
            .zip(tensor_b.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Tensor {
            data,
            shape: tensor_a.shape,
        }
    }

    pub fn mul(tensor_a: &Tensor, tensor_b: &Tensor) -> Tensor {
        assert_eq!(
            tensor_a.shape, tensor_b.shape,
            "Cannot add two tensors of different shapes"
        );

        let data = tensor_a
            .data
            .iter()
            .zip(tensor_b.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Tensor {
            data,
            shape: tensor_a.shape,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let result = Tensor::zeros((2, 2));
        assert_eq!(result.data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_addition() {
        let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let tensor_b = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], (2, 2));
        let result = Tensor::add(&tensor_a, &tensor_b);
        assert_eq!(result.data, vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_multiplication() {
        let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let tensor_b = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], (2, 2));
        let result = Tensor::mul(&tensor_a, &tensor_b);
        assert_eq!(result.data, vec![4.0, 6.0, 6.0, 4.0]);
    }

    // Add more tests for multiplication, dot product, etc.
}
