use ndarray_rand::RandomExt;
use ndarray::Array1;

use ndarray_rand::rand_distr::Uniform;

use crate::pso::{PSOConstraints, PSOHyperparameters};
use std::cmp::Ordering;

#[derive(Clone)]
pub struct Particle {
    pub position: Array1<f64>,
    pub velocity: Array1<f64>,
    pub best_fitness: f64,
    pub best_position: Array1<f64>,
}

impl PartialEq for Particle {
    fn eq(&self, other: &Self) -> bool {
        self.best_fitness == other.best_fitness
    }
}

impl Eq for Particle { }

impl PartialOrd for Particle {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.best_fitness.partial_cmp(&other.best_fitness)
    }
}

impl Ord for Particle {
    fn cmp(&self, other: &Self) -> Ordering {
        self.best_fitness.partial_cmp(&other.best_fitness).unwrap_or(Ordering::Equal)
    }
}

impl Particle {
    pub fn create_random_particle(constraints: &PSOConstraints) -> Particle {
        let position = Array1::random((constraints.n_var,), Uniform::new(constraints.var_min, constraints.var_max));
        let velocity = Array1::random((constraints.n_var,), Uniform::new(-0.1, 0.1));

        Particle {
            best_position: position.clone(),
            position,
            velocity,
            best_fitness: 0.0,
        }
    }

    pub fn random_move(&self, n_var: usize, best_position: &Array1<f64>, hp: &PSOHyperparameters) -> (Array1<f64>, Array1<f64>) {
        let dv1 = hp.social_acceleration * Array1::random((n_var,), Uniform::new(0., 1.));
        let dv2 = hp.cognitive_acceleration * Array1::random((n_var,), Uniform::new(0., 1.));

        let velocity = &self.velocity * hp.inertia_dampening + dv1*best_position + dv2*&self.best_position;

        let mut position = &self.position + &velocity;
        position /= position.scalar_sum();

        (velocity, position)
    }

    pub fn apply_move(&mut self, position: Array1<f64>, velocity: Array1<f64>, fitness: f64) {
        self.position = position;
        self.velocity = velocity;

        if fitness > self.best_fitness {
            self.best_position = self.position.clone();
            self.best_fitness = fitness;
        }
    }
}
