import deepchem as dc
import numpy as np
import pandas as pd
import itertools
from sklearn.ensemble import RandomForestRegressor

def print_progress(i, n):
    print('\r[%s%s]' % (''.join(['#']*int(20*((i+1)/n))), ''.join([' ']*int(20*(((n-i+1)/n))))), end='')

def grid_search_graph_conv(train_set, hyper_params, folds=5):

    params = list(map(lambda key: hyper_params[key], hyper_params.keys()))
    n_of_tries = len(list(itertools.product(*params)))
    search_results = []

    # split dataset into folds
    splitter = dc.splits.RandomSplitter()
    fold_sets = splitter.k_fold_split(train_set, folds)

    # save best hyperparams
    best_score = 1e10
    best_params = None
    
    # try all possible combinations of hyperparams
    for i, (batch_size, conv_layers, layer_sizes, dropout_rate) in enumerate(itertools.product(*params)):

        rmse_scores = []

        for train, valid in fold_sets:
            transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=train, move_mean=True)]

            # preprocess data
            for transformer in transformers:
                train = transformer.transform(train)
                valid = transformer.transform(valid)

            # intantiate and fit model
            model = dc.models.GraphConvModel(1, mode='regression', batch_size=batch_size, graph_conv_layers=conv_layers, dense_layer_size=layer_sizes, dropout=dropout_rate)
            model.fit(train, nb_epoch=50)
            
            # evaluate model
            metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
            rmse = model.evaluate(valid, [metric], transformers)['mean-rms_score']
            rmse_scores.append(rmse)
        
        average_rmse = np.mean(rmse_scores)
        
        # save best hyperparams
        if average_rmse < best_score:
            best_score = average_rmse
            best_params = (batch_size, conv_layers, layer_sizes, dropout_rate)

        search_results.append([average_rmse, batch_size, conv_layers, layer_sizes, dropout_rate])
        print_progress(i, n_of_tries)

    search_results = pd.DataFrame(search_results, columns=['rmse', 'batch_size', 'conv_layers', 'layer_sizes', 'dropout_rate']).sort_values(by='rmse')
    return search_results, best_params


def grid_search_mpnn(train_set, hyper_params, folds=5):

    params = list(map(lambda key: hyper_params[key], hyper_params.keys()))
    n_of_tries = len(list(itertools.product(*params)))
    search_results = []

    # split dataset into folds
    splitter = dc.splits.RandomSplitter()
    fold_sets = splitter.k_fold_split(train_set, folds)

    # save best hyperparams
    best_score = 1e10
    best_params = None
    
    # try all possible combinations of hyperparams
    for i, (batch_size, n_atom_feat, n_pair_feat, n_hidden) in enumerate(itertools.product(*params)):

        rmse_scores = []

        for train, valid in fold_sets:
            transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=train, move_mean=True)]

            # preprocess data
            for transformer in transformers:
                train = transformer.transform(train)
                valid = transformer.transform(valid)

            # intantiate and fit model
            model = dc.models.MPNNModel(1, mode='regression', batch_size=batch_size, use_queue=False, n_atom_feat=n_atom_feat, n_pair_feat=n_pair_feat, n_hidden=n_hidden, learning_rate=0.0001, T=3, M=5)
            model.fit(train, nb_epoch=50, checkpoint_interval=100)
            
            # evaluate model
            metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
            rmse = model.evaluate(valid, [metric], transformers)['mean-rms_score']
            rmse_scores.append(rmse)
        
        average_rmse = np.mean(rmse_scores)

        # save best hyperparams
        if average_rmse < best_score:
            best_score = average_rmse
            best_params = (batch_size, n_atom_feat, n_pair_feat, n_hidden)

        search_results.append([average_rmse, batch_size, n_atom_feat, n_pair_feat, n_hidden])
        print_progress(i, n_of_tries)

    search_results = pd.DataFrame(search_results, columns=['rmse', 'batch_size', 'n_atom_feat', 'n_pair_feat', 'n_hidden']).sort_values(by='rmse')
    return search_results, best_params


def grid_search_random_forest(train_set, hyper_params, folds=5):

    params = list(map(lambda key: hyper_params[key], hyper_params.keys()))
    n_of_tries = len(list(itertools.product(*params)))
    search_results = []

    # split dataset into folds
    splitter = dc.splits.RandomSplitter()
    fold_sets = splitter.k_fold_split(train_set, folds)

    # save best hyperparams
    best_score = 1e10
    best_params = None
    
    # try all possible combinations of hyperparams
    for i, (n_estimators, criterion, max_features) in enumerate(itertools.product(*params)):

        rmse_scores = []

        for train, valid in fold_sets:

            # intantiate and fit model
            sklearn_model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_features=max_features, random_state=0)
            model = dc.models.SklearnModel(sklearn_model)
            model.fit(train)
            
            # evaluate model
            metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
            rmse = model.evaluate(valid, [metric], [])['mean-rms_score']
            rmse_scores.append(rmse)
        
        average_rmse = np.mean(rmse_scores)

        # save best hyperparams
        if average_rmse < best_score:
            best_score = average_rmse
            best_params = (n_estimators, criterion, max_features)

        search_results.append([average_rmse, n_estimators, criterion, max_features])
        print_progress(i, n_of_tries)

    search_results = pd.DataFrame(search_results, columns=['rmse', 'n_estimators', 'criterion', 'max_features']).sort_values(by='rmse')
    return search_results, best_params
