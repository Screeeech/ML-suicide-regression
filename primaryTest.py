# This class uses simple lists to do calculations.
# In fact, it's so simple that it's painfully slow.
# Go to regression.py for a faster experience.


def get_hyp(x, theta_0, theta_1):
    return theta_1 * x + theta_0


class Learn:
    def __init__(self, data_set_x, data_set_y, theta_0, theta_1, alpha):
        self.data_set_x = data_set_x
        self.data_set_y = data_set_y

        self.theta_0 = theta_0
        self.theta_1 = theta_1

        self.alpha = alpha

    def get_hyp_value(self, x):
        return self.theta_1 * x + self.theta_0

    def get_cost(self, theta_0, theta_1):
        j = 0
        i = 0
        for data in self.data_set_x:
            j += ((get_hyp(data, theta_0, theta_1) - self.data_set_y[i]) ** 2) * (1 / (len(self.data_set_x) * 2))
            i += 1

        return j

    def adjust_by_gradient(self):
        theta_0_gradient = 0
        theta_1_gradient = 0

        i = 0
        for data in self.data_set_x:
            theta_0_gradient += (self.get_hyp_value(data) - self.data_set_y[i]) * (1 / len(self.data_set_x))
            theta_1_gradient += (self.get_hyp_value(data) - self.data_set_y[i]) * (1 / len(self.data_set_x)) * data
            i += 1

        self.theta_0 = self.theta_0 - (theta_0_gradient * self.alpha)
        self.theta_1 = self.theta_1 - (theta_1_gradient * self.alpha)
