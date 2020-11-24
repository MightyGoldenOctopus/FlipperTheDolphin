use crate::particle::*;
use rayon::prelude::*;

use ndarray::*;

use std::sync::RwLock;

use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};

use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyDict};
use pyo3::exceptions;

const ASSET_EPSILON: f64 = 1e-5;

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
    pub min_diff_titles: usize,
}

#[pymethods]
impl PSOConstraints {
    #[new]
    pub fn new(n_var: usize, var_min: f64, var_max: f64, min_diff_titles: usize) -> Self {
        PSOConstraints {
            n_var,
            var_min,
            var_max,
            min_diff_titles,
        }
    }
}

#[pyclass]
pub struct ParticleSwarmOptimizer {
    constraints: PSOConstraints,
    hyperparameters: PSOHyperparameters,
    particles: RwLock<Vec<Particle>>,

    disabled_particules: Array1<f64>,
    values: Array1<f64>,
    std: Array1<f64>,

    best_particle: Particle,
    best_valid_particle: Option<Particle>,

    current_min: f64,
}

impl ParticleSwarmOptimizer {
    /**
     * compute a particle fitness
     */
    pub fn compute_fitness(&self, x: &Array1<f64>) -> f64 {
        //FIXME replace 0
        (x.t().dot(&self.values) - 0.0) / x.t().dot(&self.std)
    }

    pub fn asset_to_remove(&self) -> usize {
        self.best_particle.position.iter()
            .enumerate()
            .filter(|(i, _)| self.disabled_particules[*i] > 0.3)
            .fold((0, &std::f64::INFINITY), |a, b| {
                if a.1 < b.1 {
                    a
                } else {
                    b
                }
            }).0
    }

    pub fn remove_asset(&mut self) {
        let to_remove = self.asset_to_remove();

        if self.disabled_particules[to_remove] < 0.5 {
            panic!("rah {}", to_remove);
        }

        self.current_min = self.particles
            .write().unwrap()
            .par_iter_mut()
            .map(|p| {
                p.position[to_remove] = 0.0;
                p.velocity[to_remove] = 0.0;
                p.position /= p.position.scalar_sum();

                p.position.iter()
                 .filter(|&&v| v > ASSET_EPSILON)
                 .fold(std::f64::INFINITY, |a, b| a.min(*b))
            })
            .reduce(|| std::f64::INFINITY, |a, b| a.min(b));

        self.disabled_particules[to_remove] = 0.0;
    }

    pub fn is_fully_valid_position(&self, position: &Array1<f64>) -> bool {
        position.iter().all(|&v|
            (v < ASSET_EPSILON || v >= self.constraints.var_min) && v <= self.constraints.var_max
        ) && position.iter().filter(|&&v| v >= self.constraints.var_min).count() > self.constraints.min_diff_titles
    }

    /**
     * check if a position is valid (constraints)
     */
    pub fn is_valid_position(&self, position: &Array1<f64>) -> bool {
        position.iter()
                .all(|&v| v >= self.current_min && v <= self.constraints.var_max)
    }

    /**
     * generate a random move for a particle, which is valid
     */
    pub fn generate_valid_move(&self, particle: &Particle) -> Option<(Array1<f64>, Array1<f64>)> {
        for _ in 0..10 {
            let (velocity, position) = particle.random_move(
                self.constraints.n_var, 
                &self.best_particle.best_position,
                &self.hyperparameters,
                &self.disabled_particules,
            );

            if self.is_valid_position(&position) {
                return Some((velocity, position));
            }
        }

        None
    }

    fn current_best_particle(&self) -> Particle {
        self.particles.read().unwrap()
                      .par_iter()
                      .max().unwrap().clone()
    }

    fn current_best_valid_particle(&self) -> Option<Particle> {
        if let Some(v) = self.particles.read().unwrap()
                              .par_iter()
                              .filter(|p| self.is_fully_valid_position(&p.position))
                              .max() {
            Some(v.clone())
        } else {
            None
        }
    }

    /**
     * retrieve particle with the best fitness
     */
    fn retrieve_best_particle(&mut self) {
        self.best_particle = self.current_best_particle().max(self.best_particle.clone());

        if let Some(best) = self.current_best_valid_particle() {
            let best = best.clone();

            if let Some(particle) = &mut self.best_valid_particle {
                *particle = particle.clone().max(best);
            } else {
                self.best_valid_particle = Some(best);
            }
        }
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
                                self.compute_fitness(&p.position)
                            );
                      });
        self.retrieve_best_particle();
    }

    fn run_iteration<'py>(&mut self, tqdm: &PyAny, iterator: &mut PyIterator, nb_iterations: usize, history: &mut Vec<f64>) -> PyResult<()> {
        // algorithm
        self.initialize();

        let mut i = 0;

        for _ in iterator {
            // update particle positions
            self.particles.write().unwrap().par_iter_mut()
                          .for_each(|p| {
                              if let Some((vel, pos)) = self.generate_valid_move(&p) {
                                  let fitness = self.compute_fitness(&p.position);
                                  p.apply_move(pos, vel, fitness);
                              }
                          });

            // get best particle
            self.retrieve_best_particle();

            let true_best = if let Some(best) = &self.best_valid_particle {
                history.push(best.best_fitness);
                best.best_fitness
            } else {
                history.push(self.best_particle.best_fitness);
                0.0
            };

            // update progressbar
            tqdm.call_method("set_description", (&format!("|(best, best valid): {:.2} {:.2} |", self.best_particle.best_fitness, true_best),), None)?;
            tqdm.call_method("refresh", (), None)?;

            i += 1;
            if i == nb_iterations {
                break;
            }
        }

        self.remove_asset();
        self.retrieve_best_particle();

        Ok(())
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
               std: PyReadonlyArrayDyn<f64>) -> PyResult<Self> {

        if values.shape().len() != 1 {
            return Err(exceptions::PyValueError::new_err("values should be a 1d array"));
        }

        if std.shape().len() != 1 {
            return Err(exceptions::PyValueError::new_err("std should be a 1d array"));
        }

        let shape = (values.shape()[0], );

        let values = Array1::from_shape_vec(
            shape,
            values.to_vec()?
        ).unwrap();

        let std = Array1::from_shape_vec(
            (std.shape()[0],),
            std.to_vec()?
        ).expect("value error");

        let particles = (0..population_size).into_par_iter().map(
                |_| Particle::create_random_particle(&constraints)
        ).collect();

        Ok(Self {
            best_particle: Particle::create_random_particle(&constraints),
            best_valid_particle: None,

            current_min: 0.0,

            constraints,
            hyperparameters,
            particles: RwLock::new(particles),
            values,
            std,
            disabled_particules: Array1::ones(shape),
        })
    }

    /**
     * run {nb_iterations} of the algorithm
     */
    pub fn run<'py>(&mut self, _py: Python<'py>, iteration_per_epoch: usize) -> PyResult<(&'py PyArrayDyn<f64>, &'py PyArrayDyn<f64>)> {
        let epoch_count = self.constraints.n_var - self.constraints.min_diff_titles;

        let dict = PyDict::new(_py);
        dict.set_item("bar_format", "{desc}{percentage:3.0f}%")?;
        let tqdm = _py.import("tqdm.notebook")?.get("trange")?.call((iteration_per_epoch*epoch_count,), Some(dict))?;
        let mut iterator: PyIterator = PyIterator::from_object(_py, tqdm)?;

        let mut history = vec![];

        let min_to_remove = self.constraints.n_var - (1. / self.constraints.var_min) as usize;

        for i in 0..epoch_count {
            self.current_min = i.min(min_to_remove) as f64 / 100.0 * self.constraints.var_min;
            self.run_iteration(&tqdm, &mut iterator, iteration_per_epoch, &mut history)?;
        }

        // complete iterator
        for _ in iterator {}

        // convert ndarray to numpy
        let history = ArrayD::from_shape_vec(
            IxDyn(&[history.len()]),
            history,
        ).unwrap();

        let best = self.best_valid_particle.clone().unwrap_or(self.best_particle.clone());

        let result = ArrayD::from_shape_vec(
            IxDyn(&[best.position.shape()[0]]),
            best.position.into_raw_vec(),
        ).unwrap();

        if self.best_valid_particle.is_none() {
            println!("no valid particle found");
        }

        Ok((result.into_pyarray(_py), history.into_pyarray(_py)))
    }

    pub fn fitness(&self, particle: PyReadonlyArrayDyn<f64>) -> PyResult<f64> {
        if particle.shape().len() != 1 {
            return Err(exceptions::PyValueError::new_err("particle should be a 1d array"));
        }

        let shape = (particle.shape()[0], );

        let particle = Array1::from_shape_vec(
            shape,
            particle.to_vec()?
        ).unwrap();

        Ok(self.compute_fitness(&particle))
    }
}
