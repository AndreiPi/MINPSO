import numpy as np

class ParticleSwarm(object):
    def __init__(self, cost_func, num_dimensions, num_particles, chi=0.72984, phi_p=2.05, phi_g=2.05,decay=0):
        self.cost_func = cost_func
        self.num_dimensions = num_dimensions

        self.num_particles = num_particles
        self.chi = chi
        self.phi_p = phi_p
        self.phi_g = phi_g

        self.X = np.random.uniform(size=(self.num_particles, self.num_dimensions))
        self.V = np.random.uniform(size=(self.num_particles, self.num_dimensions))

        self.P = self.X.copy()
        self.S = self.cost_func(self.X)
        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()
        self.decay = decay


    def _update(self):
        # Velocities update
        R_p = np.random.uniform(size=(self.num_particles, self.num_dimensions))
        R_g = np.random.uniform(size=(self.num_particles, self.num_dimensions))

        self.V = self.chi * (self.V \
                + self.phi_p * R_p * (self.P - self.X) \
                + self.phi_g * R_g * (self.g - self.X))


        self.X = self.X + self.V

        # Best scores
        scores = self.cost_func(self.X)

        better_scores_idx = scores < self.S
        self.P[better_scores_idx] = self.X[better_scores_idx]
        self.S[better_scores_idx] = scores[better_scores_idx]

        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()
