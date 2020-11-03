import numpy as np


class Particle:
    def __init__(self, fitness, x, v, misc=None):
        self.fitness = fitness
        self.x = x
        self.v = v
        self.misc = misc
        self.best_p = {
            "position:": x,
            "fitness": fitness
        }

    def distance(self, target):
        return np.linalg.norm(np.subtract(target, self.x))


class ParticleSwarmOptimizer:
    def __init__(self, params, constraints, fitness_function):
        # PSO parameters
        self.max_it = params["max_it"]
        self.pop_size = params["pop_size"]
        self.population = {}
        self.w = params["hyperparameters"]["initertia"]
        self.c1 = params["hyperparameters"]["cognitive_acceleration"]
        self.c2 = params["hyperparameters"]["social_acceleration"]
        # Problem constraints
        self.n_var = constraints["n_var"]
        self.var_min = constraints["var_min"]
        self.var_max = constraints["var_max"]
        # Fitness function
        self.fitness_function = fitness_function
        # Global best position
        self.best_g = {
            "position": None,
            "fitness": None
        }
        self.best_g_history = []

    def initialize(self, particles):
        if len(particles.keys()) > self.pop_size:
            raise AttributeError("Population size and particles dict size don't match")
        for id in particles.keys():
            position, velocity = particles[id]["position"], particles[id]["velocity"]
            if not position:
                position = np.random.uniform(low=self.var_min,
                                             high=self.var_max,
                                             size=(self.n_var,))
            if not velocity:
                velocity = np.zeros(shape=(self.n_var,))
            misc = particles[id]["misc"]
            self.population[id] = Particle(None, position, velocity, misc)
            self.evaluate_fitness(self.population[id])

    def evaluate_fitness(self, particle):
        particle.fitness = self.fitness_function(particle.x)
        if not particle.best_p["fitness"] or particle.fitness > particle.best_p["fitness"]:
            particle.best_p = {
                "position": particle.x,
                "fitness": particle.fitness
            }
        if not self.best_g["fitness"] or particle.fitness > self.best_g["fitness"]:
            self.best_g = {
                "position": particle.x,
                "fitness": particle.fitness
            }

    def run(self, verbose=False):
        # Do stuff...