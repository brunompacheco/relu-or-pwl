import jax.numpy as jnp
import optax

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.estimator_checks import check_is_fitted
from jax import value_and_grad, jit, vmap, nn, random
from jax.random import PRNGKey


def _compile_NN_predict(net):
    # I tried making NeuralNetwork picklable, but failed :(

    # global compiled_predict
    def compiled_predict(x):
        return net._predict_from_params(
            x, net.Wbs_,
            input_range=net.input_range_,
            output_range=net.output_range_,
        )

    return jit(vmap(compiled_predict))

class NeuralNetwork(BaseEstimator,RegressorMixin):
    def __init__(self, h_layers=3, h_units=20, optimizer=optax.adam,
                 loss_fn=optax.l2_loss, epochs=100, learning_rate=1e-3,
                 random_key=0, init_params_scale=1e-2,
                 warm_start=False) -> None:
        self.h_layers = h_layers
        self.h_units = h_units

        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.epochs = epochs
        self.learning_rate = learning_rate

        self.init_params_scale = init_params_scale

        self.random_key = random_key

        self.warm_start = warm_start

    def _initialize_weights_and_biases(self, sizes):
        key = PRNGKey(self.random_key)
        keys = random.split(key, len(sizes))
            
        self.Wbs_ = list()
        for m, n, layer_key in zip(sizes[:-1], sizes[1:], keys):
            # m = layer's input size
            # n = layer's output size
            W_key, b_key = random.split(layer_key)
            W = self.init_params_scale * random.normal(W_key, (n, m))
            b = self.init_params_scale * random.normal(b_key, (n,))

            self.Wbs_.append((W, b))

    @staticmethod
    def _predict_from_params(x, params, input_range, output_range):
        in_min = input_range[0]
        in_max = input_range[1]
        l = (x - in_min) / (in_max - in_min)  # layer activation

        for w, b in params[:-1]:  # last layer has no activation
            l = nn.relu(jnp.dot(w, l) + b)

        w, b = params[-1]
        y = jnp.dot(w, l) + b

        out_min = output_range[0]
        out_max = output_range[1]
        return y * (out_max - out_min) + out_min

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)

        X_ = jnp.array(X)
        y_hat = self._predict(X_)

        if self.n_features_out_ == 0:
            return y_hat.flatten()
        else:
            return y_hat.reshape(-1, self.n_features_out_)

    def compile_training_step(self, optimizer):
        predict_fun = vmap(self._predict_from_params, in_axes=(0, None, None, None))

        def compute_mean_loss(params, input_range, output_range, batch, target):
            pred = predict_fun(batch, params, input_range, output_range)

            loss_value = self.loss_fn(pred, target).sum(axis=-1)

            return loss_value.mean()

        def step(params, input_range, output_range, opt_state, batch, targets):
            loss_value, grads = value_and_grad(compute_mean_loss)(params,
                                                                  input_range,
                                                                  output_range,
                                                                  batch,
                                                                  targets)

            updates, opt_state = optimizer.update(grads, opt_state, params)

            params = optax.apply_updates(params, updates)

            return params, opt_state, loss_value

        return jit(step)

    def compile_validation_step(self):
        predict_fun = vmap(self._predict_from_params, in_axes=(0, None, None, None))

        def compute_mean_loss(params, input_range, output_range, batch, target):
            pred = predict_fun(batch, params, input_range, output_range)

            loss_value = self.loss_fn(pred, target).sum(axis=-1)

            return loss_value.mean()

        return jit(compute_mean_loss)

    def fit(self, X, y, X_val=None, y_val=None):
        X, y = check_X_y(X, y)

        try:
            X_ = jnp.array(X)
            y_ = jnp.array(y)
        except TypeError:
            X_ = jnp.array(X.astype(float))
            y_ = jnp.array(y.astype(float))
        
        y_ = y_.reshape(y_.shape[0], -1)

        if not (self.warm_start and hasattr(self, 'Wbs_')):
            # initialize weights and biases
            self.n_features_in_ = X_.shape[1]
            self.n_features_out_ = y_.shape[1]

            self._initialize_weights_and_biases([
                self.n_features_in_,
                *([self.h_units,]*self.h_layers),
                self.n_features_out_ if self.n_features_out_ > 0 else 1
            ])

        # fit scaler to data
        self.input_range_ = jnp.stack([X_.min(0), X_.max(0)])
        self.output_range_ = jnp.stack([y_.min(0), y_.max(0)])

        # TRAIN
        optimizer = self.optimizer(learning_rate=self.learning_rate)
        opt_state = optimizer.init(self.Wbs_)
        step = self.compile_training_step(optimizer)

        if (X_val is not None) and (y_val is not None):
            assert len(X_val.shape) == 2, 'expects `X` to have shape (n_samples, n_feats)'
            val_step = self.compile_validation_step()
        else:
            val_step = None

        self.train_loss_values_ = list()
        self.val_loss_values_ = list()
        for _ in range(self.epochs):
            self.Wbs_, opt_state, loss_value = step(self.Wbs_,
                                                    self.input_range_,
                                                    self.output_range_,
                                                    opt_state, X_, y_)
            self.train_loss_values_.append(loss_value)

            if val_step is not None:
                val_loss_value = val_step(self.Wbs_, self.input_range_,
                                          self.output_range_, X_val, y_val)
                self.val_loss_values_.append(val_loss_value)

        self._predict = _compile_NN_predict(self)

        return self
