import torch.nn as nn
import numpy as np
import pandas as pd
import torch
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tool import sphere_regular, sphere_uniform, predict_net, fit_net, RMSE, MAE, W_dist
from model import NN_SingleAsset, NN_Longonly, LS_GD, NN

class SpanningEngine(nn.Module):
    def __init__(self,device,
                 input_dim,
                 strike,
                 nb_point_1D,
                 nb_point_rad,
                 b_inf,
                 b_sup=3,
                 seed=None):
        super(SpanningEngine, self).__init__()
        self.input_dim = input_dim
        self.device = device
        self.strike = strike

        self.N_basket = int(
            ((nb_point_1D - 2)**(input_dim - 1) + input_dim * 2) *
            nb_point_rad)
        print('Number of spanning basket =', self.N_basket)
        self.nb_point_1D = nb_point_1D  
        self.nb_point_rad = nb_point_rad 

        self.r_regular = np.linspace(b_inf, b_sup, self.nb_point_rad)
        
        self.r_uniform = np.random.RandomState(seed).uniform(
            b_inf, b_sup, self.N_basket)

        self.y_regular = sphere_regular(self.nb_point_1D, input_dim)
        self.y_uniform = sphere_uniform(self.N_basket, input_dim, seed)

        self.y_regular = np.concatenate(
            [r * self.y_regular for r in self.r_regular], axis=0)
        self.y_uniform *= self.r_uniform.reshape(-1, 1)

        self.y_uniform = torch.tensor(self.y_uniform, dtype=torch.float32)
        self.y_regular = torch.tensor(self.y_regular, dtype=torch.float32)

        self.NN = NN(input_dim, self.y_uniform.T).to(device)
        self.LS_GD = LS_GD(input_dim, self.y_regular.T).to(device)
        self.NN_longonly = NN_Longonly(input_dim, self.y_uniform.T).to(device)
        self.NN_SingleAsset = NN_SingleAsset(input_dim,
                                             self.y_uniform.T).to(device)

    def fit_LS_regular(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.scaler_y_regular = StandardScaler(with_mean=True)
        self.scaler_x_regular = StandardScaler(with_mean=True)

        X_train = np.concatenate([X_train, np.maximum(
            np.matmul(X_train,
                      self.y_regular.numpy().T) - self.strike, 0)],
                                 axis =1)

        self.LS_regular = LinearRegression(fit_intercept=True)
        self.LS_regular.fit(self.scaler_x_regular.fit_transform(X_train),
                            self.scaler_y_regular.fit_transform(y_train))

        print('Model with regular weights is calibrated by SVD')

    def predict_LS_regular(self, X):
        X = np.array(X)
        X = np.concatenate([X, np.maximum(
            np.matmul(X,
                      self.y_regular.numpy().T) - self.strike, 0)],
                                 axis =1)

        pred = self.scaler_y_regular.inverse_transform(
            self.LS_regular.predict(self.scaler_x_regular.transform(X)))

        return pred

    def fit_LS_uniform(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.scaler_y_uniform = StandardScaler(with_mean=True)
        self.scaler_x_uniform = StandardScaler(with_mean=True)

        X_train = np.concatenate([X_train, np.maximum(
            np.matmul(X_train,
                      self.y_uniform.numpy().T) - self.strike, 0)],
                                 axis =1)

        self.LS_uniform = LinearRegression(fit_intercept=True)
        self.LS_uniform.fit(self.scaler_x_uniform.fit_transform(X_train),
                            self.scaler_y_uniform.fit_transform(y_train))

        print('Model with random weights is calibrated by SVD')

    def predict_LS_uniform(self, X):
        X = np.array(X)
        X = np.concatenate([X, np.maximum(
            np.matmul(X,
                      self.y_uniform.numpy().T) - self.strike, 0)],
                                 axis =1)

        pred = self.scaler_y_uniform.inverse_transform(
            self.LS_uniform.predict(self.scaler_x_uniform.transform(X)))

        return pred


    def fit_all(self,
                X_train,
                y_train,
                epochs=1000,
                lr=0.01,
                BS=None,
                weight_decay=0.001):
        idx = np.arange(0, X_train.shape[0])
        np.random.shuffle(idx)
        X_train = X_train[idx]
        y_train = y_train[idx]
        
        self.fit_LS_regular(X_train, y_train)
        self.fit_LS_uniform(X_train, y_train)
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        self.X_mean = torch.maximum(X_train.mean(dim=0, keepdim=True),
                                    torch.tensor(0.))
        self.X_std = X_train.std(dim=0, keepdim=True) + 1e-16
        self.y_std = y_train.std(dim=0, keepdim=True) + 1e-16
        t = time.time()
        fit_net(self.NN, (X_train - self.X_mean) / self.X_std,
                     y_train / self.y_std, self.device, epochs, lr, BS, weight_decay)
        fit_net(self.LS_GD, (X_train - self.X_mean) / self.X_std,
                     y_train / self.y_std, self.device, epochs, lr, BS, weight_decay)
        fit_net(self.NN_longonly, (X_train - self.X_mean) / self.X_std,
                     y_train / self.y_std, self.device, epochs, lr, BS, weight_decay)
        fit_net(self.NN_SingleAsset, (X_train - self.X_mean) / self.X_std,
                     y_train / self.y_std, self.device, epochs, lr, BS, weight_decay)
        self.time = time.time()-t

    def predict_all(self, X_train, y_train, X_test, y_test):
        
        self.pred_train_LS_regular = self.predict_LS_regular(X_train)
        self.pred_test_LS_regular = self.predict_LS_regular(X_test)
        self.pred_train_LS_uniform = self.predict_LS_uniform(X_train)
        self.pred_test_LS_uniform = self.predict_LS_uniform(X_test)
        with torch.no_grad():
            self.pred_train_NN = predict_net(self.NN, X_train, self.X_mean, self.X_std, self.y_std, self.device)
            self.pred_test_NN = predict_net(self.NN, X_test, self.X_mean, self.X_std, self.y_std, self.device)
            
            self.pred_train_LS_GD = predict_net(self.LS_GD, X_train, self.X_mean, self.X_std, self.y_std, self.device)
            self.pred_test_LS_GD = predict_net(self.LS_GD, X_test, self.X_mean, self.X_std, self.y_std, self.device)
            
            self.pred_train_NN_longonly = predict_net(self.NN_longonly, X_train, self.X_mean, self.X_std, self.y_std, self.device)
            self.pred_test_NN_longonly = predict_net(self.NN_longonly, X_test, self.X_mean, self.X_std, self.y_std, self.device)
            
            self.pred_train_NN_SingleAsset = predict_net(self.NN_SingleAsset, X_train, self.X_mean, self.X_std, self.y_std, self.device)
            self.pred_test_NN_SingleAsset = predict_net(self.NN_SingleAsset, X_test, self.X_mean, self.X_std, self.y_std, self.device)

    def print_report(self, y_train, y_test):
        report_table = np.empty((6, 13))
        for i, pred in enumerate([
                self.pred_train_LS_regular, self.pred_train_LS_uniform,
                self.pred_train_NN, self.pred_train_LS_GD,
                self.pred_train_NN_longonly, self.pred_train_NN_SingleAsset
        ]):
            report_table[i, 0] = RMSE(pred, y_train)
            report_table[i, 2] = W_dist(pred, y_train)
            report_table[i, 4] = MAE(pred, y_train)
            resi_abs = abs(y_train - pred).numpy()  
            report_table[i, 6] = np.quantile(resi_abs, 0.95)
            report_table[i, 8] = (resi_abs[resi_abs >= report_table[i, 6]]).mean()

        for i, pred in enumerate([
                self.pred_test_LS_regular, self.pred_test_LS_uniform,
                self.pred_test_NN, self.pred_test_LS_GD,
                self.pred_test_NN_longonly, self.pred_test_NN_SingleAsset
        ]):
            report_table[i, 1] = RMSE(pred, y_test)
            report_table[i, 3] = W_dist(pred, y_test)
            report_table[i, 5] = MAE(pred, y_test)
            resi_abs = abs(y_test - pred).numpy()  
            report_table[i, 7] = np.quantile(resi_abs, 0.95)
            report_table[i, 9] = (resi_abs[resi_abs >= report_table[i, 7]]).mean()

        report_table[:, 10] = self.N_basket
        report_table[:, 11] = self.input_dim
        report_table[:, 12] = self.time  
        report_table = pd.DataFrame(
            report_table,
            index=['LS-regular', 'LS-uniform', 'NN', 'LS-GD', 'NN-Long only', 'NN-Single asset'],
            columns=([
                'RMSE-train', 'RMSE-test', 'Wass-train', 'Wass-test',
                'MAE-train', 'MAE-test', 'VaR_95_resi_abs-train',
                'VaR_95_resi_abs-test', 'ES_95_resi_abs-train',
                'ES_95_resi_abs-test', 'Nb baskets', 'Dim', 'Time'
            ]))
        self.report_table = report_table
        print(report_table)
