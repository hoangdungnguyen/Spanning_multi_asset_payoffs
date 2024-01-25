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
    epochs = parsed.epochs
    lr = parsed.lr
    decay_weight = parsed.decay_weight
    BS = parsed.BS
    nb_run = parsed.nb_run
    seed = 0

    
    dim = 2
    nb_1D = 6
    nb_radius = 5
    
    N_train = 100000
    N_test = 200000

    full_report = dict()
    
    for payoff in payoff_dict.keys():
        N_grid = 15
        N_test = 50000
        
        data_generator = DataGenerator(payoff_dict, payoff, dim)
        
        X_train, y_train = data_generator.sample_grid(N_grid)
        X_test, y_test = data_generator.sample_uniform(N_test, seed=seed)


        norm_X_sup = float(X_train.norm(dim=1).max())
        span_engine = SpanningEngine(device, dim, 1., nb_1D, nb_radius,
                                     1 / norm_X_sup, 3, seed )
        span_engine.fit_all(X_train, y_train, epochs, lr, BS, decay_weight)
        span_engine.predict_all(X_train, y_train, X_test, y_test)
        span_engine.print_report(y_train, y_test)
        
        sub_dict = dict()
        sub_dict['y_test'] = np.array(y_test)
        sub_dict['X_test'] = np.array(X_test)
        sub_dict['NN_pred'] = np.array(span_engine.pred_test_NN)
        sub_dict['abs_err'] =  np.abs(sub_dict['y_test'] -  sub_dict['NN_pred'])
        if payoff == 'BOC':
            sub_dict['X_train'] = np.array(X_train)
            sub_dict['y_train'] = np.array(y_train)
        full_report[payoff] = sub_dict

    with open('./result/report_dim2.pickle', 'wb') as f:
        pickle.dump(full_report, f, -1)


if __name__ == "__main__":
    main()