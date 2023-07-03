import os
import pickle
import jax.numpy as jnp
import optax
import wandb

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.estimator_checks import check_is_fitted
from jax import value_and_grad, jit, vmap, nn, random
from jax.random import PRNGKey


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

        y_hat = self._predict(X)

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

    def _maybe_init_wandb(self, wandb_kwargs: dict):
        if wandb_kwargs:
            run = wandb.init(**wandb_kwargs)

            for k, v in self.get_params().items():
                run.config[k] = v
            run.config['n_data_points'] = self.n_data_points_
            run.config['n_features'] = self.n_features_in_
            run.config['input_range'] = self.input_range_
            run.config['output_range'] = self.output_range_

            self._wandb_run = run

    def _maybe_log_to_wandb(self, epoch_log: dict, step=None, commit=None):
        try:
            self._wandb_run.log(epoch_log, step=step, commit=commit)
        except AttributeError:
            pass  # wandb was not initialized :shrug:

    def _maybe_finish_wandb(self):
        try:
            # upload model to wandb
            tmp_file_fpath = 'models/'+self._wandb_run.name+'.pkl'
            self.save(tmp_file_fpath)
            self._wandb_run.save(tmp_file_fpath)
            # os.remove(tmp_file_fpath)

            # finish wandb
            self._wandb_run.finish()
        except AttributeError:
            pass  # wandb was not initialized :shrug:

    def fit(self, X, y, X_val=None, y_val=None, bandit: int = jnp.inf, **kwargs):
        """Keyword arguments to be passed to W&B should be prefixed with
        `wandb_`, e.g., `wandb_project='project-name'` will become
        `wandb.init(project='project-name')`.
        """
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        if not (self.warm_start and hasattr(self, 'Wbs_')):
            self._fit_to_Xy_shape(X, y)

        self._maybe_init_wandb({k[len('wandb_'):]:v for k,v in kwargs.items()
                                if k.startswith('wandb_')})

        # initialize optimizer
        optimizer = self.optimizer(learning_rate=self.learning_rate)
        opt_state = optimizer.init(self.Wbs_)

        # compile training and validation steps, for faster training
        step = self.compile_training_step(optimizer)
        if (X_val is not None) and (y_val is not None):
            assert len(X_val.shape) == 2, 'expects `X` to have shape (n_samples, n_feats)'
            val_step = self.compile_validation_step()
        else:
            val_step = None

        self.train_loss_values_ = list()
        self.val_loss_values_ = list()
        best_val = jnp.inf
        best_val_epoch = -1
        for epoch in range(self.epochs):  # TRAINING LOOP
            self.Wbs_, opt_state, loss_value = step(self.Wbs_,
                                                    self.input_range_,
                                                    self.output_range_,
                                                    opt_state, X, y)
            self.train_loss_values_.append(loss_value)

            epoch_log = {
                'opt_state': opt_state,
                'train_loss': loss_value,
            }
            if val_step is not None:
                val_loss_value = val_step(self.Wbs_, self.input_range_,
                                          self.output_range_, X_val, y_val)
                self.val_loss_values_.append(val_loss_value)
                epoch_log['val_loss'] = val_loss_value

                if val_loss_value < 0.95 * best_val:
                    best_val = val_loss_value
                    best_val_epoch = epoch
                elif epoch - best_val_epoch > bandit:
                    # ensure that _maybe_log_to_wandb runs one last time
                    epoch = self.epochs + 1

            self._maybe_log_to_wandb(epoch_log)

        # compile predict function, for faster inference
        # TODO: lazy compilation
        self._predict = jit(vmap(lambda x: self._predict_from_params(
            x, self.Wbs_,
            input_range=self.input_range_,
            output_range=self.output_range_,
        )))

        self._maybe_finish_wandb()

        return  self

    def _fit_to_Xy_shape(self, X, y):
        # initialize weights and biases
        self.n_data_points_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = y.shape[1]

        # fit scaler to data
        self.input_range_ = jnp.stack([X.min(0), X.max(0)])
        self.output_range_ = jnp.stack([y.min(0), y.max(0)])

        self._initialize_weights_and_biases([
            self.n_features_in_,
            *([self.h_units,]*self.h_layers),
            self.n_features_out_ if self.n_features_out_ > 0 else 1
        ])

    def save(self, fpath):
        check_is_fitted(self)

        with open(fpath, 'wb') as f:
            pickle.dump(dict(
                Wbs_=self.Wbs_,
                input_range_=self.input_range_,
                output_range_=self.output_range_,
                train_loss_values_=self.train_loss_values_,
                val_loss_values_=self.train_loss_values_,
                n_features_in_=self.n_features_in_,
                n_features_out_=self.n_features_out_,
                **self.get_params()
            ), f)

    @classmethod
    def load(cls, fpath):
        self = cls()
        with open(fpath, 'rb') as f:
            all_params = pickle.load(f)
        # npzfile = jnp.load(fpath)

        self.set_params(**{k: v for k, v in all_params.items() if not k.endswith('_')})

        # "fits" the model
        self.Wbs_ = all_params['Wbs_']
        self.input_range_ = all_params['input_range_']
        self.output_range_ = all_params['output_range_']
        self.train_loss_values_ = all_params['train_loss_values_']
        self.val_loss_values_ = all_params['val_loss_values_']
        self.n_features_in_ = all_params['n_features_in_']
        self.n_features_out_ = all_params['n_features_out_']

        # compile prediction
        self._predict = jit(vmap(lambda x: self._predict_from_params(
            x, self.Wbs_,
            input_range=self.input_range_,
            output_range=self.output_range_,
        )))

        return self
