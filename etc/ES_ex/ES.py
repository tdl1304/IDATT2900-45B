import numpy as np


# Use ES to find a specific vector
# Steps:
# Fitness F(w)
# Solution w
# Initial vector
# Loop with gaussian distribution
class ES:
    """"
    Setup variables for training with evolutionary strategies
        sigma: radius for exploration
        alpha: learning rate
        model: model with forward pass, object
            TODO implement a method to change model weights after each iteration
        solution: solution for forward pass from model, nparray (N,1)
        maxIt: maxiumum amount of iterations - default 10000
        p: population of exploration, int - default 50
    """

    def __init__(self, sigma, alpha, model, solution, maxIt=10000, p=50):
        self.maxIt = maxIt
        self.size = len(solution)
        self.solution = solution
        self.sigma = sigma
        self.alpha = alpha
        self.p = p
        self.model = model
        self.W = np.random.randn(self.size)

    # fitness with standard loss function
    def f(self, w):
        return - np.sum(np.square(self.solution - w))

    # move w closer towards direction of rewards
    # return new weights
    def update_rule(self, w, N, A):
        return w + self.alpha / (self.p * self.sigma) * np.dot(N.T, A)

    def train(self):
        for i in range(self.maxIt):
            if i % 20 == 0:
                print('iter %d. w: %s, solution: %s, reward: %f' %
                      (i, str(self.W), str(self.solution), self.f(self.W)))

            N = np.random.randn(self.p, self.size)  # random points with shape (population, size)
            R = np.zeros(self.p)  # store rewards per point in N

            for j in range(self.p):
                R[j] = self.f(self.W + self.sigma * N[j])  # calculate fitness for every point in Gaussian
                # distribution away from centre w

            A = (R - np.mean(R)) / np.std(R)  # z-score normalization
            wNew = self.update_rule(w=self.W, N=N, A=A)
            if np.abs(self.f(w=wNew)) < np.abs(self.f(w=self.W)):  #
                self.W = wNew
            elif np.abs(self.f(self.W)) <= 1e-6:  # stop when within margin of error
                break


if __name__ == '__main__':
    sol = np.array([3, 1, 2, 3])
    es = ES(sigma=0.1, alpha=0.001, model=0, solution=sol)
    es.train()
