import pickle
from pathlib import Path
from random import shuffle
from time import time

import numpy as np
import optax
from tqdm import tqdm

from src.data import get_X_y, load_riser_data, split_curve
from src.trainer import NetworkTrainerWandB

N_RUNS = 5
EPOCHS = 5000


if __name__ == '__main__':
    results_fpath = Path('medium_nn_hp_tuning_3_results.pkl')
    if results_fpath.exists():
        with open(results_fpath, 'rb') as f:
            results = pickle.load(f)
    else:
        results = list()

    hp_ranges = {
        'h_layers': [2, 5, 10, 20],
        'h_units': [20, 200, 500],
        'learning_rate': [1e-3, 1e-4, 1e-5],
        'weight_decay': [0., 1e-3, 1e-4, 1e-5],
    }

    all_hps = [
        {'h_layers': hl,
         'h_units': hu,
         'learning_rate': lr,
         'weight_decay': wd,}
        for hl in hp_ranges['h_layers']
        for hu in hp_ranges['h_units']
        for lr in hp_ranges['learning_rate']
        for wd in hp_ranges['weight_decay']
    ]
    shuffle(all_hps)

    # load data
    df_riser = load_riser_data('data/riser.csv')
    df_train, df_test = split_curve(df_riser, 0.5)
    X, y = get_X_y(df_riser)
    X_train, y_train = get_X_y(df_train)
    X_test, y_test = get_X_y(df_test)

    for hps in tqdm(all_hps):
        # check whether this set of hps has already been tested
        new = True
        for r in results:
            if hps == r['hps']:
                new = False
                break
        if not new:
            continue

        # train and check results
        val_results = list()
        train_results = list()
        mapes = list()
        max_apes = list()
        start = time()
        for key in range(N_RUNS):
            net = NetworkTrainerWandB(
                **hps,
                optimizer=optax.adamw,
                loss_fn=optax.l2_loss,
                epochs=EPOCHS,
                random_key=key
            )

            net.fit(X_train, y_train, X_val=X_test, y_val=y_test, bandit=1000,
                    wandb_project='relu-pwl', wandb_group='hp-tuning-3',)

            val_results.append(net.val_loss_values_[-1])
            train_results.append(net.train_loss_values_[-1])
            mapes.append(net.mape_)
            max_apes.append(net.max_ape_)

        results.append({
            'epochs': EPOCHS,
            'hps': hps,
            'val_results': val_results,
            'train_results': train_results,
            'MAPEs': mapes,
            'MaxAPEs': max_apes,
            'time': time() - start,
        })

        with open(results_fpath, 'wb') as f:
            pickle.dump(results, f)
