use pyo3::prelude::*;
pub mod pymatrix;

#[pymodule]
fn py_nn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<pymatrix::PyMatrix>()?;
    Ok(())
}
