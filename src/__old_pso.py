import numpy as np
from tqdm import trange

class Particle:
    def __init__(self, id, fitness, x, v):
        self.id = id
        self.fitness = fitness
        self.x = x
        self.v = v
        self.best_p = {
            "position": x,
            "fitness": fitness
        }


    def distance(self, target):
        return np.subtract(target, self.x)


class ParticleSwarmOptimizer:
    def __init__(self, params, constraints, fitness_function):
        self.population = {}

        # PSO parameters
        self.max_it = params["max_it"]
        self.pop_size = params["pop_size"]
        self.w = params["hyperparameters"]["inertia"]
        self.w_damp = params["hyperparameters"]["inertia_dampening"]
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
            "id": None,
            "position": None,
            "fitness": None
        }
        self.best_g_history = []


    def initialize(self):
        for i in range(self.pop_size):
            position = np.random.uniform(low=self.var_min,
                                             high=self.var_max,
                                             size=(self.n_var,))
            velocity = np.random.uniform(low=-0.1, high=0.1, size=(self.n_var,))
            self.population[i] = Particle(i, None, position, velocity)
            self.evaluate_fitness(self.population[i])


    def evaluate_fitness(self, particle):
        particle.fitness = self.fitness_function(particle.x)

        if not particle.best_p["fitness"] or particle.fitness > particle.best_p["fitness"]:
            particle.best_p = {
                "position": particle.x,
                "fitness": particle.fitness
            }
        if not self.best_g["fitness"] or particle.fitness > self.best_g["fitness"]:
            self.best_g = {
                "id": particle.id,
                "position": particle.x,
                "fitness": particle.fitness
            }


    def run(self, verbose=False):
        t = trange(self.max_it, desc='PSO Run', leave=True)

        for i in t:
            for id, particle in self.population.items():
                valid_move, new_position, timeout = False, None, 10

                while not valid_move and timeout > 0:
                    timeout -= 1
                    new_velocity = (
                          self.w * particle.v
                        + self.c1 * np.random.uniform(size=(self.n_var,)) *
                            particle.distance(particle.best_p["position"])
                        + self.c2 * np.random.uniform(size=(self.n_var,)) *
                            particle.distance(self.best_g["position"])
                    )
                    new_position = particle.x + new_velocity
                    new_position /= np.sum(new_position)
                    valid_move = np.all(new_position >= self.var_min) and np.all(new_position <= self.var_max)

                if valid_move:
                    particle.x = new_position
                    self.evaluate_fitness(particle)

            self.best_g_history.append(self.best_g)
            self.w *= self.w_damp

            t.set_description(f"PSO run (best particle id: {self.best_g['id']}, best fitness: {self.best_g['fitness']:.3f})")
            t.refresh() # to show immediately the update

        return self.best_g, self.best_g_history
