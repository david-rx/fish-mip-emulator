from collections import defaultdict
import os
import pickle
from typing import Callable, List
from emulator.models.model import Model
from sklearn import tree
from sklearn import metrics
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from emulator.models.mean_guesser import MeanGuesser
from emulator.models.nn.lstm import make_lstm_features, make_lstm_features_3d
from emulator.models.pca import PcaSingleton
from emulator.models.random_guesser import RandomGuesser
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

from emulator.dataloading.macroecological import INPUTS_PATH_TOS, INPUTS_PATH_INTPP, OUTPUTS_PATH, MacroecologicalDataLoader, TEST_INPUTS_PATH_TOS, TEST_INPUTS_PATH_INTPP, TEST_OUTPUTS_PATH
from emulator.dataloading.boats import INPUTS_PATH_INTPP_BOATS, TEST_INPUTS_PATH_TOS_BOATS, INPUTS_PATH_TOS_BOATS, TEST_INPUTS_PATH_INTPP_BOATS, TEST_OUTPUTS_PATH_BOATS, OUTPUTS_PATH_BOATS, BoatsConfig, BoatsDataloader
from emulator.models.nn.nn_model import NNRegressor
from emulator.training.data_models import DatasetName
from emulator.training.evaluation_helpers import by_period_overall_results, evaluate_model_by_period, evaluate_models

def train(dataset_name: str, train_config: BoatsConfig, test_config: BoatsConfig) -> None:
    """
    Train and evaluate emulator models on the given dataset.
    """
    if dataset_name == DatasetName.MACROECOGICAL.value:
        dataloader = MacroecologicalDataLoader(inputs_path_tos = INPUTS_PATH_TOS, outputs_path = OUTPUTS_PATH, inputs_path_intpp = INPUTS_PATH_INTPP)
        test_dataloader = MacroecologicalDataLoader(inputs_path_tos = TEST_INPUTS_PATH_TOS, outputs_path = TEST_OUTPUTS_PATH, inputs_path_intpp = TEST_INPUTS_PATH_INTPP)
    elif dataset_name == DatasetName.BOATS.value:
        print("loading train")
        dataloader = BoatsDataloader(boats_config = train_config)
        print("loading test")
        test_dataloader = BoatsDataloader(boats_config = test_config)
    else:
        raise NotImplementedError("No dataloader implemented for dataset with this name!")
    train_features, eval_features, train_labels, eval_labels = dataloader.load_train_eval()
    test_features, test_labels = test_dataloader.features, test_dataloader.labels
    

    pca = None
    scaler = MinMaxScaler()
    if train_config.contextual:
        pca = PcaSingleton()
        train_features = scaler.fit_transform(pca.run(train_features))
        test_features = scaler.transform(pca.run(test_features))
        eval_features = scaler.transform(pca.run(eval_features))

    # test_features = pca.run(ss.transform(test_features))
    # eval_features = pca.run(ss.transform(eval_features))

    # WARNING: not doing this causes issues with default eps values in NN training
    # train_features[:, 1] *= 10 ** 8
    # eval_features[:, 1] *= 10 ** 8
    # test_features[:, 1] *= 10 ** 8

    # train_df = pd.DataFrame.from_dict({"surface temp": train_features[:3000, 0], "intpp": train_features[:3000, 1], "tcb": train_labels[:3000, 0, ], "prev tcb": train_features[:3000, 2],
    #     "lat or lon": train_features[:3000, 3], "lon or lat": train_features[:3000, 4]})
    # train_df.to_csv("train_data_boats.csv")
    # flattened_test_features = test_features.reshape(-1, test_features.shape[-1])
    # flattened_test_labels = test_labels.reshape(-1, test_labels.shape[-1])
    # test_df = pd.DataFrame.from_dict({"surface temp": flattened_test_features[:3000, 0], "intpp": flattened_test_features[:3000, 1], "tcb": flattened_test_labels[:3000, 0], "prev tcb": flattened_test_features[:3000, 2]})
    # test_df.to_csv("test_data_boats.csv")

    
    print(f"DATA METRICS: train mean is {train_labels.mean()} with stdev {train_labels.std()}.")
    print(f"train feat shape is {train_features.shape}")
    print(f"eval mean is {eval_labels.mean()} with stdev {eval_labels.std()}")
    print(f"test mean is {test_labels.mean()} with stdev {test_labels.std()}")
    print(f"train features mean is {train_features.mean()} with stdev {train_features.std()}")
    print(f"test features mean is {test_features.mean()} with stdev {test_features.std()}")

    #fit models
    eval_metrics = [metrics.mean_squared_error, metrics.mean_absolute_error, metrics.r2_score]
    # models = [tree.DecisionTreeRegressor(), XGBRegressor(objective='reg:squarederror'), NNRegressor(input_size = train_features.shape[1], output_size = train_labels.shape[1])]
        # RandomForestRegressor(), 
        # HistGradientBoostingRegressor(max_iter=100), NNRegressor(input_size = train_features.shape[1], output_size = train_labels.shape[1])]
    if train_config.contextual:
        nn_features_train, nn_labels_train = make_lstm_features(train_features, train_labels)
        nn_features_eval, nn_labels_eval = make_lstm_features(eval_features, eval_labels)
        num_labels = nn_labels_train.shape[-1]
    elif train_config.by_period:
        nn_features_train, nn_labels_train = make_lstm_features_3d(train_features, train_labels)
        nn_features_eval, nn_labels_eval = make_lstm_features_3d(eval_features, eval_labels)
        num_labels = 1


    models = [NNRegressor(input_size=nn_features_train.shape[-1], output_size=num_labels, model="lstm")]
    for model in models:
        print("model", model)
        if model.__class__ == NNRegressor:
            model.fit(nn_features_eval, nn_labels_eval, nn_features_eval, nn_labels_eval)
        # elif model.__class__ in [RandomForestRegressor, HistGradientBoostingRegressor]:
        #     model.fit(train_features, train_labels.reshape(-1))
        else:
            model.fit(train_features, train_labels)
        eval_by_period(model=model, features=test_features.copy(), labels=test_labels, metrics=eval_metrics, dataset_name=dataset_name,
            teacher_forcing=False, eval_delta=test_config.predict_delta, predict_delta=train_config.predict_delta, pca=pca, lat_features=train_config.flat,
            autoregressive=train_config.flat)
    
    evaluate_models(models, eval_features, eval_labels, eval_metrics, dataset_name=dataset_name)
    save_models(models, dataset_name)

def save_models(models, dataset_name: str):
    for model in models:
        model_name = model.__class__.__name__
        model_path = os.path.join("outputs/models", f"{model_name}_{dataset_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

def eval_by_period(model: Model, features: np.ndarray, labels: np.ndarray, metrics, dataset_name: str, predict_delta: bool = True,
    teacher_forcing = False, eval_delta=False, pca: PcaSingleton = None, lat_features: bool = False, autoregressive=False):
    """
    Evaluates a model on a dataset by period, and outputs the results to a csv file.
    """

    autoregessive_feature = None
    all_predictions: List[np.ndarray] = []
    all_inter_predictions: List[np.ndarray] = []
    first_third = int(features[0].shape[0] / 3)

    if model.__class__ == NNRegressor and model.model == "lstm":
        if features.ndim == 3:
            lstm_features, lstm_labels = make_lstm_features_3d(features, labels)
        else:
            lstm_features, lstm_labels = make_lstm_features(features, labels)
        all_predictions = model.predict(lstm_features).reshape(lstm_labels.shape)
        labels = lstm_labels
    else:
        for period_features, period_labels in zip(features, labels):

            if autoregessive_feature is not None:
                period_features[:, 2] = autoregessive_feature
            
            if pca:
                period_features = pca.run(period_features.reshape(1, period_features.shape[0]))
            elif period_features.ndim == 1:
                period_features = period_features.reshape(1, -1)

            predictions = model.predict(period_features)
            all_inter_predictions.append(predictions)
            if autoregressive:
                autoregessive_feature = get_autoregressive_feature(eval_delta = eval_delta, predict_delta=predict_delta, teacher_forcing=teacher_forcing, predictions=predictions, period_labels=period_labels, period_features=period_features)
            final_predictions = get_final_predictions(predict_delta=predict_delta, eval_delta=eval_delta, predictions=predictions, period_features=period_features)
            all_predictions.append(final_predictions)

    evaluate_model_by_period(all_predictions, labels, metrics=metrics, model_name=model.__class__.__name__, dataset_name = dataset_name, eval_delta=eval_delta, teacher_forcing=teacher_forcing, features=features, lat_features=lat_features)

def get_autoregressive_feature(predict_delta: bool, eval_delta: bool, teacher_forcing: bool, period_labels, predictions, period_features):
    """
    Returns the autoregressive feature for the next period, which is either the true label, or the predicted label.
    """
    if teacher_forcing:
        if eval_delta:
            return period_labels.reshape(-1) + period_features[:, -1]
        return period_labels.reshape(-1)
    if predict_delta:
        return predictions.reshape(-1) + period_features[:, -1]
    else:
        return predictions.reshape(-1)

def get_final_predictions(predict_delta: bool, eval_delta: bool, predictions, period_features):
    if predict_delta and not eval_delta:
        return predictions.reshape(-1) + period_features[:, -1]
    else:
        return predictions.reshape(-1)

if __name__ == "__main__":
    boats_config_train = BoatsConfig(
        inputs_path_tos = INPUTS_PATH_TOS_BOATS,
        outputs_path = OUTPUTS_PATH_BOATS,
        inputs_path_intpp = INPUTS_PATH_INTPP_BOATS,
        predict_delta=False,
        by_period=False,
        flat=False,
        contextual=True,
        debug=False
    )
    boats_config_test = BoatsConfig(
        inputs_path_tos = TEST_INPUTS_PATH_TOS_BOATS,
        outputs_path = TEST_OUTPUTS_PATH_BOATS,
        inputs_path_intpp = TEST_INPUTS_PATH_INTPP_BOATS,
        by_period=False,
        predict_delta=False,
        flat=False,
        contextual=True,
        debug=False
    )

    train(dataset_name=DatasetName.BOATS.value, train_config = boats_config_train, test_config = boats_config_test)