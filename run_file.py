import torch.nn as nn
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm.auto import trange, tqdm

from data_generator import DataGenerator
from payoff_spec import payoff_dict
from spanningengine import SpanningEngine
from tool import sphere_regular, sphere_uniform, predict_net, fit_net, RMSE, MAE, W_dist

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-payoff', '--payoff', required=True, type=str)
    parser.add_argument('-device', '--device', default='cuda:0', type=str)
    parser.add_argument('-epochs', '--epochs', default=1000, type=int)
    parser.add_argument('-lr', '--lr', default=0.01, type=float)
    parser.add_argument('-BS', '--BS', default=None, type=int)
    parser.add_argument('-decay_weight',
                        '--decay_weight',
                        default=0.001,
                        type=float)
    parser.add_argument('-nb_run', '--nb_run', default=50, type=int)

    parsed = parser.parse_args()

    device = torch.device(parsed.device)
    payoff = parsed.payoff
    epochs = parsed.epochs
    lr = parsed.lr
    decay_weight = parsed.decay_weight
    BS = parsed.BS
    nb_run = parsed.nb_run
    seed = 0

    loop = tqdm(
        zip([2, 3, 4, 5, 20, 50], [6, 4, 4, 4, 3, 3], [5, 5, 4, 3, 10, 8]))

    N_train = 100000
    N_test = 200000

    full_report_table = list()
    
    for dim, nb_1D, nb_radius in loop:
        if dim == 2:
            N_grid = 15
            N_test = 50000
        else:
            N_grid = 10

        data_generator = DataGenerator(payoff_dict, payoff, dim)
        if dim < 20:
            X_train, y_train = data_generator.sample_grid(N_grid)
            X_test, y_test = data_generator.sample_uniform(N_test, seed=seed)
        else:
            X_test, y_test = data_generator.sample_lognormal(N_test, seed=seed)

        for i in range(nb_run):
            if dim >= 20:
                X_train, y_train = data_generator.sample_lognormal(N_train,
                                                                   seed=seed + i)
            norm_X_sup = float(X_train.norm(dim=1).max())
            span_engine = SpanningEngine(device, dim, 1., nb_1D, nb_radius,
                                         1 / norm_X_sup, 3, seed + i)
            span_engine.fit_all(X_train, y_train, epochs, lr, BS, decay_weight)
            span_engine.predict_all(X_train, y_train, X_test, y_test)
            span_engine.print_report(y_train, y_test)
            full_report_table.append(span_engine.report_table)
            if dim >= 20: 
                print(span_engine.y_uniform.norm(dim = 1).max())

    full_report_table = pd.concat(full_report_table, axis=0)

    with open('./result/report_table_' + payoff + '.pickle', 'wb') as f:
        pickle.dump(full_report_table, f, -1)


if __name__ == "__main__":
    main()