use nn_core::matrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    let seed = [42; 32]; // A fixed seed for reproducibility
    let mut rng = StdRng::from_seed(seed);
    let generator = || rng.gen();

    let matrix = matrix::Matrix::random((2, 2), generator);
    println!("{}", matrix);
}
