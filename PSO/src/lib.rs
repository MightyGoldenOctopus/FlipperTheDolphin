pub mod pso;
pub mod particle;

use pyo3::prelude::*;

#[pymodule]
fn pso(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<pso::PSOConstraints>()?;
    m.add_class::<pso::PSOHyperparameters>()?;
    m.add_class::<pso::ParticleSwarmOptimizer>()?;

    Ok(())
}
