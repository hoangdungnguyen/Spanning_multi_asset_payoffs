import torch
import math
import itertools
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange
from scipy.stats import wasserstein_distance as wass_d
import torch.nn.functional as F

# batch_iterate, predict_net, fit_net, RMSE, NRMSE, MAE, W_dist, VaR_err, ES_err

def batch_iterate(features, labels, dest_features, dest_labels, batch_size):
    # Thanks to Bouazza Saadeddine for this function
    for batch_idx in range((features.shape[0] + batch_size - 1) // batch_size):
        start_idx = batch_idx * batch_size
        tmp_features_batch = features[start_idx:(batch_idx + 1) * batch_size]
        eff_batch_size = tmp_features_batch.shape[0]
        dest_features[:eff_batch_size] = tmp_features_batch
        if labels is None:
            yield start_idx, eff_batch_size, dest_features[:
                                                           eff_batch_size], None
        else:
            tmp_labels_batch = labels[start_idx:(batch_idx + 1) * batch_size]
            dest_labels[:eff_batch_size] = tmp_labels_batch
            yield start_idx, eff_batch_size, dest_features[:
                                                           eff_batch_size], dest_labels[:
                                                                                        eff_batch_size]


def predict_net(model, X, X_mean, X_std, Y_std, device, batchsize=1000):
    pred = torch.empty((len(X), 1), dtype=torch.float32)
    t_x_batch = torch.empty((batchsize, X.shape[1]),
                            dtype=torch.float32,
                            device=device)
    with torch.no_grad():
        model.eval()
        for idx, eff, x_batch, _ in batch_iterate((X - X_mean) / X_std,
                                                  None,
                                                  t_x_batch,
                                                  None,
                                                  batch_size=batchsize):
            pred[idx:idx + eff].copy_(model(x_batch))
        pred *= Y_std
    return pred


def fit_net(model,
            X_train,
            y_train,
            device,
            epochs=1000,
            lr=0.01,
            BS=None,
            weight_decay=0.01,
            regulator = 'l2'):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay if regulator == 'l2' else 0.)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=300,
                                                   gamma=0.8,
                                                   last_epoch=-1,
                                                   verbose=False)

    best_err = np.inf
    if BS is None:
        BS = int(X_train.shape[0] // 10)
    t_x_batch = torch.empty((BS, X_train.shape[1]),
                            dtype=torch.float32,
                            device=device)
    t_y_batch = torch.empty((BS, y_train.shape[1]),
                            dtype=torch.float32,
                            device=device)
    loss_list = []
    bar = trange(epochs)

    loss_func = F.mse_loss
    #loss_func = F.l1_loss
    #loss_func = lambda pred, true: (model.q + 1 / (1 - 0.95) * F.relu(
    #    torch.abs(true - pred) - model.q)).mean()
    for i in bar:
        for _, _, x_batch, y_batch in batch_iterate(X_train,
                                                    y_train,
                                                    t_x_batch,
                                                    t_y_batch,
                                                    batch_size=BS):
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_func(pred, y_batch)
            if regulator == 'l1':
                loss+= weight_decay*torch.norm(model.v, p =1)
                #for p in model.parameters():
                #    loss+= weight_decay*torch.norm(p, p =1)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            err = 0
            for _, eff, x_batch, y_batch in batch_iterate(X_train,
                                                          y_train,
                                                          t_x_batch,
                                                          t_y_batch,
                                                          batch_size=BS):
                pred = model(x_batch)
                err += loss_func(pred, y_batch) * eff
            err = err / X_train.shape[0]
            err = float(err.data.item())
            if err < best_err:
                best_err = err
                best_state = copy.deepcopy(model.state_dict())
        bar.set_postfix(loss=f'{err :.6f}')
        loss_list.append(err)
        if lr_scheduler.get_last_lr()[-1] >= 0.001:
            lr_scheduler.step()
    plt.plot(np.log(loss_list))
    plt.ylabel('Log loss')
    plt.xlabel('epochs')
    plt.show()

    model.load_state_dict(best_state)
    print('Best error = {}'.format(best_err))


def RMSE(pred, true):
    pred = np.array(pred).squeeze()
    true = np.array(true).squeeze()
    return np.sqrt(np.mean((pred - true)**2, axis=0))


def NRMSE(pred, true):
    pred = np.array(pred).squeeze()
    true = np.array(true).squeeze()
    return np.sqrt(np.mean((pred - true)**2, axis=0)) / true.std()


def MAE(pred, true):
    pred = np.array(pred).squeeze()
    true = np.array(true).squeeze()
    return np.mean(abs(pred - true))


def W_dist(pred, true):
    pred = np.asarray(pred).squeeze()
    true = np.asarray(true).squeeze()
    return wass_d(true, pred)


def VaR_err(pred, true, alpha=0.95):
    pred = np.asarray(pred).squeeze()
    true = np.asarray(true).squeeze()
    return np.quantile(np.abs(true - pred), alpha)


def ES_err(pred, true, alpha=0.95):
    pred = np.asarray(pred).squeeze()
    true = np.asarray(true).squeeze()
    err = np.abs(true - pred)
    q = np.quantile(err, alpha)
    return (err[err >= q]).mean()


def sphere_regular(nb_point, input_dim):
    phi_list = []
    for i in range(input_dim - 2):
        phi_list.append(np.linspace(0, math.pi, nb_point)[1:-1])
    phi_list.append(np.linspace(0, 2 * math.pi, nb_point)[1:-1])

    phi = []
    for i in itertools.product(*phi_list):
        phi.append(i)
    phi = np.array(phi)

    y = np.empty_like(phi)
    y[:, 0] = np.cos(phi[:, 0])
    for i in range(1, input_dim - 1):
        y[:, i] = np.prod(np.sin(phi[:, :i]), axis=1) * np.cos(phi[:, i])
    y = np.concatenate([y, np.prod(np.sin(phi), axis=1).reshape(-1, 1)],
                       axis=1)
    for i in range(input_dim):
        a = np.zeros(input_dim)
        a[i] = 1.
        y = np.concatenate([y, a.reshape(1, -1), (-1 * a).reshape(1, -1)],
                           axis=0)
    return y


def sphere_uniform(nb_point, input_dim, seed=None):
    phi = np.random.RandomState(seed).uniform(0, math.pi,
                                              (nb_point, input_dim - 1))
    phi[:, -1] *= 2

    y = np.empty_like(phi)
    y[:, 0] = np.cos(phi[:, 0])
    for i in range(1, input_dim - 1):
        y[:, i] = np.prod(np.sin(phi[:, :i]), axis=1) * np.cos(phi[:, i])
    y = np.concatenate([y, np.prod(np.sin(phi), axis=1).reshape(-1, 1)],
                       axis=1)
    return y