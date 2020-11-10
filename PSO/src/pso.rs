use crate::particle::*;
use rayon::prelude::*;

use ndarray::*;

use std::sync::RwLock;

use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};

use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyDict};
use pyo3::exceptions;

#[pyclass]
#[derive(Clone)]
pub struct PSOHyperparameters {
    pub inertia: f64,
    pub inertia_dampening: f64,
    pub cognitive_acceleration: f64,
    pub social_acceleration: f64,
}

#[pymethods]
impl PSOHyperparameters {
    #[new]
    pub fn new(inertia: f64, inertia_dampening: f64, cognitive_acceleration: f64, social_acceleration: f64) -> Self {
        PSOHyperparameters {
            inertia,
            inertia_dampening,
            cognitive_acceleration,
            social_acceleration,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PSOConstraints {
    pub n_var: usize,
    pub var_min: f64,
    pub var_max: f64,
}

#[pymethods]
impl PSOConstraints {
    #[new]
    pub fn new(n_var: usize, var_min: f64, var_max: f64) -> Self {
        PSOConstraints {
            n_var,
            var_min,
            var_max,
        }
    }
}

#[pyclass]
pub struct ParticleSwarmOptimizer {
    constraints: PSOConstraints,
    hyperparameters: PSOHyperparameters,
    particles: RwLock<Vec<Particle>>,

    values: Array1<f64>,
    covariance_matrix: Array2<f64>,

    best_particle: Particle,
}

impl ParticleSwarmOptimizer {
    /**
     * compute a particle fitness
     */
    pub fn compute_fitness(&self, particle: &Particle) -> f64 {
        let x = &particle.position;

        //FIXME replace 0
        return (x.t().dot(&self.values) - 0.) / (self.covariance_matrix.dot(x).dot(&x.t()))
    }

    /**
     * check if a position is valid (constraints)
     */
    pub fn is_valid_position(&self, position: &Array1<f64>) -> bool {
        //FIXME
        true || position.iter()
                .all(|&v| v >= self.constraints.var_min && v <= self.constraints.var_max)
    }

    /**
     * generate a random move for a particle, which is valid
     */
    pub fn generate_valid_move(&self, particle: &Particle) -> Option<(Array1<f64>, Array1<f64>)> {
        for _ in 0..10 {
            let (velocity, position) = particle.random_move(
                self.constraints.n_var, 
                &self.best_particle.best_position,
                &self.hyperparameters
            );

            if self.is_valid_position(&position) {
                return Some((velocity, position));
            }
        }

        None
    }

    /**
     * retrieve particle with the best fitness
     */
    fn retrieve_best_particle(&mut self) {
        self.best_particle = self.particles.read().unwrap()
                                 .par_iter()
                                 .max().unwrap().clone().max(self.best_particle.clone());
    }

    /**
     * internal function: initialize particles fitness
     */
    fn initialize(&mut self) {
        self.particles.write().unwrap().par_iter_mut()
                      .for_each(|p| {
                            p.apply_move(
                                p.position.clone(),
                                p.velocity.clone(),
                                self.compute_fitness(&p)
                            );
                      });
        self.retrieve_best_particle();
    }
}

#[pymethods]
impl ParticleSwarmOptimizer {
    #[new]
    /**
     * create a new ParticleSwarmOptimizer object
     */
    pub fn new(population_size: usize,
               hyperparameters: PSOHyperparameters,
               constraints: PSOConstraints,
               values: PyReadonlyArrayDyn<f64>,
               covariance_matrix: PyReadonlyArrayDyn<f64>) -> PyResult<Self> {

        if values.shape().len() != 1 {
            return Err(exceptions::PyValueError::new_err("values should be a 1d array"));
        }

        if covariance_matrix.shape().len() != 2 {
            return Err(exceptions::PyValueError::new_err("covariance_matrix should be a 2d array"));
        }

        let values = Array1::from_shape_vec(
            (values.shape()[0],),
            values.to_vec()?
        ).unwrap();

        let covariance_matrix = Array2::from_shape_vec(
            (covariance_matrix.shape()[0], covariance_matrix.shape()[1]),
            covariance_matrix.to_vec()?
        ).expect("value error");

        let particles = (0..population_size).into_par_iter().map(
                |_| Particle::create_random_particle(&constraints)
        ).collect();

        Ok(Self {
            best_particle: Particle::create_random_particle(&constraints),

            constraints,
            hyperparameters,
            particles: RwLock::new(particles),
            values,
            covariance_matrix,
        })
    }

    /**
     * run {nb_iterations} of the algorithm
     */
    pub fn run<'py>(&mut self, _py: Python<'py>, nb_iterations: usize) -> PyResult<(&'py PyArrayDyn<f64>, &'py PyArrayDyn<f64>)> {
        // create a tqdm progressbar
        let dict = PyDict::new(_py);
        dict.set_item("bar_format", "{desc}{percentage:3.0f}%")?;
        let tqdm = _py.import("tqdm.notebook")?.get("trange")?.call((nb_iterations,), Some(dict))?;
        let iterator: PyIterator = PyIterator::from_object(_py, tqdm)?;

        // algorithm
        self.initialize();
        let mut history = vec![self.best_particle.best_fitness];

        for _ in iterator {
            // update particle positions
            self.particles.write().unwrap().par_iter_mut()
                          .for_each(|p| {
                              if let Some((vel, pos)) = self.generate_valid_move(&p) {
                                  let fitness = self.compute_fitness(&p);
                                  p.apply_move(pos, vel, fitness);
                              }
                          });

            // get best particle
            self.retrieve_best_particle();
            history.push(self.best_particle.best_fitness);

            // update progressbar
            tqdm.call_method("set_description", (&format!("|best fitness: {:.2}|", self.best_particle.best_fitness),), None)?;
            tqdm.call_method("refresh", (), None)?;
        }

        // convert ndarray to numpy
        let history = ArrayD::from_shape_vec(
            IxDyn(&[history.len()]),
            history,
        ).unwrap();

        let result = ArrayD::from_shape_vec(
            IxDyn(&[self.best_particle.best_position.shape()[0]]),
            self.best_particle.best_position.clone().into_raw_vec(),
        ).unwrap();

        Ok((result.into_pyarray(_py), history.into_pyarray(_py)))
    }
}
