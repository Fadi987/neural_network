use pyo3::prelude::*;
pub mod matrix;
pub mod neural_network;

#[pymodule]
fn py_nn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<matrix::Matrix>()?;
    Ok(())
}
