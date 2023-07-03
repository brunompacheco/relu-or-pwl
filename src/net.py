import os
import pickle
from typing import Any, List
import jax.numpy as jnp
import optax
import wandb

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.estimator_checks import check_is_fitted
from jax import value_and_grad, jit, vmap, nn, random
from jax.random import PRNGKey
from flax import linen as nn
from flax.training.train_state import TrainState


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

class NetworkTrainer(BaseEstimator,RegressorMixin):
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

    def _initialize_net_and_train_state(self, X):
        key = PRNGKey(self.random_key)

        self.net_ = ReLUNetwork(
            hidden_layers=self.h_layers,
            hidden_units=self.h_units,
            n_out=self.n_features_out_,
            input_min=self.input_range_[0],
            input_max=self.input_range_[1],
            output_min=self.output_range_[0],
            output_max=self.output_range_[1],
        )
        Wbs = self.net_.init(key, X)  # weights and biases

        # initialize optimizer
        optimizer = self.optimizer(learning_rate=self.learning_rate)

        self.train_state_ = TrainState.create(
            apply_fn=self.net_.apply,
            params=Wbs['params'],
            tx=optimizer,
        )

    @staticmethod
    def _predict_from_params(x, state: TrainState, input_range, output_range):
        in_min = input_range[0]
        in_max = input_range[1]
        l = (x - in_min) / (in_max - in_min)  # layer activation

        y = state.apply_fn({'params': state.params}, l)

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

    def compile_training_step(self):
        def step(state: TrainState, inputs, targets):
            def compute_mean_loss(params):
                pred = state.apply_fn({'params': params}, inputs)

                loss_value = self.loss_fn(pred, targets).sum(axis=-1)

                return loss_value.mean()

            loss_value, grads = value_and_grad(compute_mean_loss)(state.params)

            state = state.apply_gradients(grads=grads)

            return state, loss_value

        return jit(step)

    def compile_validation_step(self):
        # TODO: refactor loss function to avoid double definition (validation
        # and training steps). Maybe compile using jit+partial
        def compute_mean_loss(state: TrainState, inputs, targets):
            pred = state.apply_fn({'params': state.params}, inputs)

            loss_value = self.loss_fn(pred, targets).sum(axis=-1)

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
        
    def _shuffle_data(self, X, y):
        key = PRNGKey(self.random_key)
        idx = jnp.arange(X.shape[0])
        shuffled_idx = random.shuffle(key, idx)

        return X[shuffled_idx], y[shuffled_idx]

    def fit(self, X, y, X_val=None, y_val=None, bandit: int = jnp.inf, **kwargs):
        """Keyword arguments to be passed to W&B should be prefixed with
        `wandb_`, e.g., `wandb_project='project-name'` will become
        `wandb.init(project='project-name')`.
        """
        X_, y_ = check_X_y(X, y, multi_output=True, y_numeric=True)

        if not (self.warm_start and hasattr(self, 'train_state_')):
            self._fit_to_Xy_shape(X_, y_)
            self._initialize_net_and_train_state(X_)
        
        X_, y_ = self._shuffle_data(X_, y_)

        self._maybe_init_wandb({k[len('wandb_'):]:v for k,v in kwargs.items()
                                if k.startswith('wandb_')})

        # compile training and validation steps, for faster training
        step = self.compile_training_step()
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
            val_loss_value = val_step(self.train_state_, X_val, y_val)
            self.train_state_, loss_value = step(self.train_state_, X_, y_)
            self.train_loss_values_.append(loss_value)

            epoch_log = {
                'train_state': self.train_state_,
                'train_loss': loss_value,
            }

            if val_step is not None:
                val_loss_value = val_step(self.train_state_, X_val, y_val)
                self.val_loss_values_.append(val_loss_value)
                epoch_log['val_loss'] = val_loss_value

                if val_loss_value < 0.95 * best_val:
                    best_val = val_loss_value
                    best_val_epoch = epoch
                elif epoch - best_val_epoch > bandit:
                    # ugly way to ensure that _maybe_log_to_wandb runs one last
                    # time
                    epoch = self.epochs + 1

            self._maybe_log_to_wandb(epoch_log)

        self._predict = jit(lambda x: self.train_state_.apply_fn(
            {'params': self.train_state_.params}, x
        ))
        self._maybe_finish_wandb()

        return self

    def _fit_to_Xy_shape(self, X, y):
        # initialize weights and biases
        self.n_data_points_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = y.shape[1]

        # fit scaler to data
        self.input_range_ = jnp.stack([X.min(0), X.max(0)])
        self.output_range_ = jnp.stack([y.min(0), y.max(0)])

    def save(self, fpath):
        check_is_fitted(self)

        with open(fpath, 'wb') as f:
            pickle.dump(dict(
                Wbs_=self.train_state_,
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
        self.train_state_ = all_params['Wbs_']
        self.input_range_ = all_params['input_range_']
        self.output_range_ = all_params['output_range_']
        self.train_loss_values_ = all_params['train_loss_values_']
        self.val_loss_values_ = all_params['val_loss_values_']
        self.n_features_in_ = all_params['n_features_in_']
        self.n_features_out_ = all_params['n_features_out_']

        # compile prediction
        self._predict = jit(vmap(lambda x: self._predict_from_params(
            x, self.train_state_,
            input_range=self.input_range_,
            output_range=self.output_range_,
        )))

        return self
