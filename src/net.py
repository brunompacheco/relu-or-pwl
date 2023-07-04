from flax import linen as nn


class ReLUNetwork(nn.Module):
    hidden_layers: int = 3
    hidden_units: int = 20
    n_out: int = 1
    input_min: float = 0.
    input_max: float = 1.
    output_max: float = 0.
    output_min: float = 1.

    @nn.compact
    def __call__(self, x):
        # input normalization
        x = (x - self.input_min) / (self.input_max - self.input_min)

        for _ in range(self.hidden_layers):
            x = nn.Dense(self.hidden_units)(x)
            x = nn.relu(x)

        y = nn.Dense(self.n_out)(x)

        # output denormalization
        return y * (self.output_max - self.output_min) + self.output_min
