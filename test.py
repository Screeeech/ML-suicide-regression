import numpy as np

# This is the much faster class that uses numpy arrays for calculations.
# Use this for reference; don't use learning.py


def get_hyp(x, theta_0, theta_1):
    return theta_1 * x + theta_0


class LearnTest:
    def __init__(self, data_set_x, data_set_y, hypothesis, alpha):
        a = np.array([data_set_x]).T
        self.data_set_x = np.ones((len(data_set_x), 2))
        self.data_set_x[:, :-1] = np.array([a]).T

        a = np.array([data_set_y]).T
        self.data_set_y = np.ones((len(data_set_y), 2))
        self.data_set_y[:, :-1] = np.array([a]).T

        self.hypothesis = hypothesis

        self.alpha = alpha

    def get_hyp_theta(self):
        self.hyp_theta = np.dot(self.data_set_x, self.hypothesis)
        return self.hyp_theta

    def get_cost(self, theta_0, theta_1):
        j = 0
        i = 0
        for data in self.data_set_x:
            j += ((get_hyp(data[0], theta_0, theta_1) - self.data_set_y[i][0]) ** 2) * (1 / (len(self.data_set_x) * 2))
            i += 1
        return j

    def adjust_by_gradient(self):
        self.theta_0_gradient = (self.get_hyp_theta() - self.data_set_y[:, [0]]) * (1 / len(self.data_set_x))
        self.theta_1_gradient = np.multiply((self.get_hyp_theta() - self.data_set_y[:, [0]]),
                                            self.data_set_x[:, [0]]) * (1 / len(self.data_set_x))

        self.hypothesis -= ((np.array([self.theta_1_gradient.sum(axis=0),
                                       self.theta_0_gradient.sum(axis=0)])) * self.alpha)
