import unittest
from ES import ES
import numpy as np


class ES_test(unittest.TestCase):

    def setUp(self):
        self.sol = np.array([3, 1, 2, 3])
        self.es = ES(sigma=0.1, alpha=0.001, model=object, solution=self.sol, p=5)

    def test_f(self):
        w_test = np.array([0.1089199, 1.87523603, 0.90693119, 1.64611404])
        r_test = self.es.f(w_test)
        self.assertEqual(round(r_test, 6), -12.152189)

    def test_update_rule(self):
        N = np.array([[0.07052918, -0.67430821, 0.56289697, -0.49114206],
                      [-1.54743999, 0.04462818, -1.85270286, -0.45707769],
                      [-1.37443185, 0.11013093, -0.3011854, -1.08428862],
                      [-0.79658387, -0.5883068, 0.50728512, -0.11242235],
                      [0.22824909, -0.94495168, -0.99398151, -0.45583229]])
        A = np.array([1.1323958, -1.31810334, -1.05935213, 0.82620803, 0.41885163])
        w = np.array([-0.71171093, -0.22085498, -0.21947476, -2.09151483])
        w_test = self.es.update_rule(w, N, A)
        w_test = np.round(w_test, decimals=6)
        self.assertTrue(np.alltrue(w_test == np.array([-0.705685, -0.224497, -0.212672, -2.089693])))


if __name__ == '__main__':
    unittest.main()
