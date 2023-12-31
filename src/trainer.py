import pickle
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.estimator_checks import check_is_fitted
from jax import grad, value_and_grad, jit, random
from jax.tree_util import tree_leaves
from jax.random import PRNGKey
from flax.training.train_state import TrainState
from flax.serialization import to_state_dict, from_state_dict

from .net import ReLUNetwork


def l2_reg(x, alpha):
    return alpha * jnp.mean(jnp.square(x))

class NetworkTrainer(BaseEstimator,RegressorMixin):
    def __init__(self, h_layers=3, h_units=20, optimizer=optax.adamw,
                 loss_fn=optax.l2_loss, epochs=100, learning_rate=1e-3,
                 l2_reg_alpha=0., weight_decay=0., random_key=0,
                 init_params_scale=1e-2, warm_start=False, l2_reg=0.01,
                 l2_second_deriv_reg=0, eps_reg=1e-3) -> None:
        self.h_layers = h_layers
        self.h_units = h_units

        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.epochs = epochs
        self.learning_rate = learning_rate

        self.l2_reg_alpha = l2_reg_alpha
        self.weight_decay = weight_decay

        self.init_params_scale = init_params_scale

        self.random_key = random_key

        self.warm_start = warm_start

        self.l2_reg = l2_reg
        self.l2_second_deriv_reg = l2_second_deriv_reg
        self.eps_reg = eps_reg

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
        optimizer = self.optimizer(learning_rate=self.learning_rate,
                                   weight_decay=self.weight_decay)

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
            def compute_train_loss(params):
                pred = state.apply_fn({'params': params}, inputs)

                loss_value = self.loss_fn(pred, targets).sum(axis=-1).mean()

                # l2 regularization
                if self.l2_reg_alpha > 0:
                    loss_value += jnp.mean(l2_reg(w, alpha=self.l2_reg_alpha)
                                        for w in tree_leaves(params))

                # numerical second derivative regularization
                if self.l2_second_deriv_reg > 0:
                    n_dim = inputs.shape[1]
                    batch_size = inputs.shape[0]

                    second_derivative_penalty = 0
                    for i in range(n_dim):  # input dimensions
                        # euclidean vector along i-th dimension
                        e_i = jnp.eye(n_dim)[i][None,:].repeat(batch_size,0)

                        inputs_plus = inputs + self.eps_reg * e_i
                        inputs_minus = inputs - self.eps_reg * e_i

                        pred_plus = state.apply_fn({'params': params}, inputs_plus)
                        pred_minus = state.apply_fn({'params': params}, inputs_minus)

                        second_derivative = (pred_plus - 2 * pred + pred_minus) / (2 * self.eps_reg)
                        second_derivative_penalty += jnp.sum(jnp.square(second_derivative)) / n_dim

                    loss_value += self.l2_second_deriv_reg * second_derivative_penalty

                return loss_value

            loss_value, grads = value_and_grad(compute_train_loss)(state.params)

            state = state.apply_gradients(grads=grads)

            return state, loss_value

        return jit(step)

    def compile_validation_step(self):
        # TODO: refactor loss function to avoid double definition (validation
        # and training steps). Maybe compile using jit+partial
        def compute_val_loss(state: TrainState, inputs, targets):
            pred = state.apply_fn({'params': state.params}, inputs)

            loss_value = self.loss_fn(pred, targets).sum(axis=-1)

            return loss_value.mean()

        return jit(compute_val_loss)

    def _shuffle_data(self, X, y):
        key = PRNGKey(self.random_key)
        idx = jnp.arange(X.shape[0])
        shuffled_idx = random.permutation(key, idx, independent=True)

        return X[shuffled_idx], y[shuffled_idx]

    def fit(self, X, y, X_val=None, y_val=None, bandit: int = jnp.inf, **kwargs):
        """Keyword arguments to be passed to W&B should be prefixed with
        `wandb_`, e.g., `wandb_project='project-name'` will become
        `wandb.init(project='project-name')`.
        """
        X_, y_ = check_X_y(X, y, multi_output=True, y_numeric=True)

        X_, y_, step, val_step = self._initialize(X_, y_, X_val, y_val, **kwargs)

        self.train_loss_values_ = list()
        self.val_loss_values_ = list()
        best_val = jnp.inf
        best_val_epoch = -1
        for self.curr_epoch_ in range(self.epochs):  # TRAINING LOOP
            self._run_epoch(X_, y_, X_val, y_val, bandit, step, val_step,
                            best_val, best_val_epoch)

        self._finish_fit(X_val, y_val)

        return self

    def _initialize(self, X, y, X_val, y_val, **kwargs):
        if not (self.warm_start and hasattr(self, 'train_state_')):
            self._fit_to_Xy_shape(X, y)
            self._initialize_net_and_train_state(X)
        else:
            raise NotImplementedError('warm-starting :(')

        X, y = self._shuffle_data(X, y)

        # compile training and validation steps, for faster training
        step = self.compile_training_step()
        if (X_val is not None) and (y_val is not None):
            assert len(X_val.shape) == 2, 'expects `X` to have shape (n_samples, n_feats)'
            val_step = self.compile_validation_step()
        else:
            val_step = None

        return X, y, step, val_step

    def _run_epoch(self, X_, y_, X_val, y_val, bandit, step, val_step, best_val,
                   best_val_epoch):
        self.train_state_, loss_value = step(self.train_state_, X_, y_)
        self.train_loss_values_.append(loss_value)

        if val_step is not None:
            val_loss_value = val_step(self.train_state_, X_val, y_val)
            self.val_loss_values_.append(val_loss_value)

            if val_loss_value < 0.95 * best_val:
                best_val = val_loss_value
                best_val_epoch = self.curr_epoch_
            elif self.curr_epoch_ - best_val_epoch > bandit:
                # ugly way to ensure that _maybe_log_to_wandb runs one last
                # time
                self.curr_epoch_ = self.epochs + 1
        else:
            val_loss_value = None

        return loss_value, val_loss_value

    def _finish_fit(self, X, y):
        self._compile_predict()

        if X is not None:
            y_hat = self.predict(X)
            percentage_error = (y_hat - y) / y

            self.mape_ = np.abs(percentage_error).mean()
            self.max_ape_ = np.abs(percentage_error).max()

    def _compile_predict(self):
        self._predict = jit(lambda x: self.train_state_.apply_fn(
            {'params': self.train_state_.params}, x
        ))

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
            # unpicklable
            attributes = {v: getattr(self, v) for v in vars(self) if v.endswith('_')}
            train_state = attributes.pop('train_state_')
            attributes.pop('net_')

            pickle.dump(dict(
                __state_dict=to_state_dict(train_state),
                **attributes,
                **self.get_params()
            ), f)

    @classmethod
    def load(cls, fpath):
        self = cls()
        with open(fpath, 'rb') as f:
            all_params = pickle.load(f)

        # only non-param in dict
        state_dict = all_params.pop('__state_dict')

        # load (hyper-)parameters
        self.set_params(**{k: v for k, v in all_params.items() if not k.endswith('_')})

        # "fits"
        for k, v in all_params.items():
            if k.endswith('_'):
                setattr(self, k, v)

        # load weights
        self._initialize_net_and_train_state(jnp.ones((self.n_data_points_,
                                                       self.n_features_in_)))

        self.train_state_ = from_state_dict(self.train_state_, state_dict)

        self._compile_predict()

        return self

class NetworkTrainerWandB(NetworkTrainer):
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

            if hasattr(self, 'mape_'):
                self._wandb_run.summary['MAPE'] = self.mape_
            if hasattr(self, 'max_ape_'):
                self._wandb_run.summary['MaxAPE'] = self.max_ape_

            # finish wandb
            self._wandb_run.finish()
        except AttributeError:
            pass  # wandb was not initialized :shrug:

    def _initialize(self, X, y, X_val, y_val, **kwargs):
        r = super()._initialize(X, y, X_val, y_val, **kwargs)

        self._maybe_init_wandb({k[len('wandb_'):]:v for k,v in kwargs.items()
                                if k.startswith('wandb_')})

        return r

    def _run_epoch(self, X_, y_, X_val, y_val, bandit, step, val_step, best_val,
                   best_val_epoch):
        loss_value, val_loss_value = super()._run_epoch(X_, y_, X_val, y_val,
                                                        bandit, step, val_step,
                                                        best_val, best_val_epoch)

        self._maybe_log_to_wandb({'train_loss': loss_value,
                                  'val_loss': val_loss_value,})

        return loss_value, val_loss_value

    def _finish_fit(self, *args, **kwargs):
        r = super()._finish_fit(*args, **kwargs)

        self._maybe_finish_wandb()

        return r