import pickle
from pathlib import Path
from time import time

import numpy as np
import optax
from tqdm import tqdm

from src.data import get_X_y, load_riser_data, split_curve
from src.net import NetworkTrainerWandB

N_ITER = 100
N_RUNS = 5
EPOCHS = 5000


if __name__ == '__main__':
    results_fpath = Path('medium_nn_hp_tuning_2_results.pkl')
    if results_fpath.exists():
        with open(results_fpath, 'rb') as f:
            results = pickle.load(f)
    else:
        results = list()

    hp_ranges = {
        'h_layers': [2, 5, 10, 20],
        'h_units': [20, 50, 200, 500],
        'learning_rate': [1e-2, 1e-3, 1e-4],
    }

    # load data
    df_riser = load_riser_data('data/riser.csv')
    df_train, df_test = split_curve(df_riser, 0.5)
    X_train, y_train = get_X_y(df_train)
    X_test, y_test = get_X_y(df_test)

    for i in tqdm(list(range(N_ITER))):
        hps = {k: np.random.choice(vs) for k, vs in hp_ranges.items()}

        # check whether this set of hps has already been tested
        new = True
        for r in results:
            if hps == r['hps']:
                new = False
                break
        if not new:
            i -= 1
            continue

        # train and check results
        val_results = list()
        train_results = list()
        start = time()
        for key in range(N_RUNS):
            net = NetworkTrainerWandB(
                **hps,
                optimizer=optax.adam,
                loss_fn=optax.l2_loss,
                epochs=EPOCHS,
                random_key=key
            )

            net.fit(X_train, y_train, X_val=X_test, y_val=y_test, bandit=1000,
                    wandb_project='relu-pwl', wandb_group='hp-tuning-2',)

            val_results.append(net.val_loss_values_[-1])
            train_results.append(net.train_loss_values_[-1])

        results.append({
            'epochs': EPOCHS,
            'hps': hps,
            'val_results': val_results,
            'train_results': train_results,
            'time': time() - start,
        })

        with open(results_fpath, 'wb') as f:
            pickle.dump(results, f)
