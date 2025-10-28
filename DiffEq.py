import tensorflow as tf

class DiffEq:
    def __init__(self, diffeq, x, y, dydx, d2ydx2, d3ydx3, d4ydx4, d5ydx5, d6ydx6):
        # Example 1: y^(6) = -6 e^x + y  => residual = y^(6) + 6 e^x - y
        if diffeq == "sixth":
            self.eqf = d6ydx6 + 6.0 * tf.exp(x) - y
        else:
            raise ValueError(f"Unknown diffeqf '{diffeq}'")

