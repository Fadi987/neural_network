pub mod matrix;
pub mod neural_network;
pub mod optimizer;
mod py_bindings;

use pyo3::prelude::*;

#[pymodule]
fn nn_core(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<py_bindings::PyMatrix>()?;
    Ok(())
}
