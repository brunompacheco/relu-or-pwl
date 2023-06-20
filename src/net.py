import jax.numpy as jnp
import optax

from optax import GradientTransformation
from jax import value_and_grad, jit, vmap, nn, random


class NeuralNetwork():
    def __init__(self, sizes, key, init_params_scale=1e-2) -> None:
        keys = random.split(key, len(sizes))

        self.params = list()
        for m, n, layer_key in zip(sizes[:-1], sizes[1:], keys):
            # m = layer's input size
            # n = layer's output size
            w_key, b_key = random.split(layer_key)
            self.params.append((init_params_scale * random.normal(w_key, (n, m)),  # W
                                init_params_scale * random.normal(b_key, (n,))))   # b

    @staticmethod
    def _predict(x, params):
        l = x  # layer activation

        for w, b in params[:-1]:  # last layer has no activation
            l = nn.relu(jnp.dot(w, l) + b)

        w, b = params[-1]
        return jnp.dot(w, l) + b

    def __call__(self, x):
        return self._predict(x, self.params)
    
    def compile(self):
        """Fix params and speed up inference.
        """
        return jit(vmap(self.__call__))

    def compile_training_step(self, loss_fn, optimizer):
        predict_fun = vmap(self._predict, in_axes=(0, None))

        def compute_mean_loss(params, batch, target):
            pred = predict_fun(batch, params)

            loss_value = loss_fn(pred, target).sum(axis=-1)

            return loss_value.mean()

        def step(params, opt_state, batch, targets):
            loss_value, grads = value_and_grad(compute_mean_loss)(params, batch, targets)

            updates, opt_state = optimizer.update(grads, opt_state, params)

            params = optax.apply_updates(params, updates)

            return params, opt_state, loss_value

        return jit(step)

    def fit(self, optimizer: GradientTransformation, loss_fn, X, y, epochs=100):
        opt_state = optimizer.init(self.params)

        step = self.compile_training_step(loss_fn, optimizer)

        loss_values = list()
        for _ in range(epochs):
            self.params, opt_state, loss_value = step(self.params, opt_state, X, y)

            loss_values.append(loss_value)

        return loss_values
