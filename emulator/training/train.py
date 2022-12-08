from collections import defaultdict
import os
from typing import Callable, List
from emulator.models.model import Model
from sklearn import tree
from sklearn import metrics
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from emulator.models.mean_guesser import MeanGuesser
from emulator.models.random_guesser import RandomGuesser
from emulator.dataloading.macroecological import INPUTS_PATH_TOS, INPUTS_PATH_INTPP, OUTPUTS_PATH, MacroecologicalDataLoader, TEST_INPUTS_PATH_TOS, TEST_INPUTS_PATH_INTPP, TEST_OUTPUTS_PATH
from emulator.dataloading.boats import INPUTS_PATH_INTPP_BOATS, TEST_INPUTS_PATH_TOS_BOATS, INPUTS_PATH_TOS_BOATS, TEST_INPUTS_PATH_INTPP_BOATS, TEST_OUTPUTS_PATH_BOATS, OUTPUTS_PATH_BOATS, BoatsDataloader
from emulator.models.simple_nn import NNRegressor
from emulator.training.data_models import DatasetName

def plot(predictions: List[float], labels: List[float], model_name: str, dataset_name: str, num_to_plot: int = 5000):
    indices_to_plot = np.random.randint(low = 0, high=len(labels), size=num_to_plot)
    predictions, labels = predictions[indices_to_plot], labels[indices_to_plot]
    fig, ax = plt.subplots()
    ax.scatter(predictions, labels)
    ax.set_xlabel("emulated biomass density (g / m^2)")
    ax.set_ylabel("expected biomass density (g / m ^2)")
    ax.set_title(f"MACROECOLOGICAL Emulator: {model_name}")
    fig.savefig(os.path.join("outputs/visualizations", f"{model_name}_{dataset_name}"))

def train(dataset_name: str, test: bool = False) -> None:
    if dataset_name == DatasetName.MACROECOGICAL.value:
        dataloader = MacroecologicalDataLoader(inputs_path_tos = INPUTS_PATH_TOS, outputs_path = OUTPUTS_PATH, inputs_path_intpp = INPUTS_PATH_INTPP)
        test_dataloader = MacroecologicalDataLoader(inputs_path_tos = TEST_INPUTS_PATH_TOS, outputs_path = TEST_OUTPUTS_PATH, inputs_path_intpp = TEST_INPUTS_PATH_INTPP)
    elif dataset_name == DatasetName.BOATS.value: #boats, for now
        dataloader = BoatsDataloader(inputs_path_tos = INPUTS_PATH_TOS_BOATS, outputs_path = OUTPUTS_PATH_BOATS, inputs_path_intpp = INPUTS_PATH_INTPP_BOATS, debug=False, predict_delta=True)
        test_dataloader = BoatsDataloader(inputs_path_tos = TEST_INPUTS_PATH_TOS_BOATS, outputs_path = TEST_OUTPUTS_PATH_BOATS, inputs_path_intpp = TEST_INPUTS_PATH_INTPP_BOATS, debug = False, by_period=True, predict_delta=False)
    else:
        raise NotImplementedError("No dataloader implemented for dataset with this name!")
    train_features, eval_features, train_labels, eval_labels = dataloader.load_train_eval()
    test_features, test_labels = test_dataloader.features, test_dataloader.labels
        # eval_features, eval_labels = second_test_dataloader.features, second_test_dataloader.labels

    # WARNING: not doing this causes issues with default eps values in NN training
    train_features[:, 1] *= 10 ** 8
    eval_features[:, 1] *= 10 ** 8
    test_features[:, :, 1] *= 10 ** 8
    # print(f"intpp mean is {train_features[0:5, 1]}")

    # train_df = pd.DataFrame.from_dict({"surface temp": train_features[:, 0], "intpp": train_features[:, 1], "tcb": train_labels[:, 0, ], "prev tcb": train_features[:, 2]})
    # train_df.to_csv("train_data_macro.csv")

    print(f"DATA METRICS: train mean is {train_labels.mean()} with stdev {train_labels.std()}.")
    print(f"eval mean is {eval_labels.mean()} with stdev {eval_labels.std()}")

    #fit models

    eval_metrics = [metrics.mean_squared_error, metrics.mean_absolute_error, metrics.r2_score]
    # NNRegressor(input_size = train_features.shape[1], output_size = train_labels.shape[1]
    # NNRegressor(input_size = train_features.shape[1], output_size = train_labels.shape[1]), HistGradientBoostingRegressor(max_iter=100)
    models = [RandomGuesser(), MeanGuesser(), LinearRegression(), tree.DecisionTreeRegressor(), HistGradientBoostingRegressor(max_iter=100), NNRegressor(input_size = train_features.shape[1], output_size = train_labels.shape[1])]
    for model in models:
        print("model", model)
        if model.__class__ == NNRegressor:
            model.fit(train_features, train_labels, eval_features, eval_labels)
        else:
            model.fit(train_features, train_labels)
        eval_by_year(model=model, features=test_features, labels=test_labels, metrics=eval_metrics, dataset_name=dataset_name,
            teacher_forcing=False, eval_delta=False, predict_delta=True)

    model_names = []
    mse_scores = []
    mae_scores = []
    r2_scores = []

    #make predictions, output results
    features = test_features if test else eval_features
    labels = test_labels if test else eval_labels
    print(features.shape)
    print(labels.shape)

    for model in models:

        model_name = model.__class__.__name__
        model_names.append(model_name)
        eval_predictions = model.predict(features)
        print(eval_predictions[0:5])
        print("preds shape", eval_predictions.shape)
        print("labels shape", labels.shape)

        plot(eval_predictions, labels, model_name=model_name, dataset_name=dataset_name)

        for metric in eval_metrics:
            score = metric(eval_predictions, labels)
            if metric.__name__ == "mean_squared_error":
                mse_scores.append(score)
            elif metric.__name__ == "mean_absolute_error":
                mae_scores.append(score)
            elif metric.__name__ == "r2_score":
                r2_scores.append(score)
            print(f"model {model_name} getting score {score} on metric {metric.__name__}")


    df = pd.DataFrame.from_dict({"Name": model_names, "MSE_Score": mse_scores, "MAE_Score": mae_scores, "R2_Score": r2_scores})
    df.to_csv(os.path.join("outputs/evaluation_results", f"results_{dataset_name}{'_test' if test else ''}.csv"))

def evaluate_model_by_period(all_predictions, all_labels, metrics: List[Callable], model_name: str, dataset_name: str, eval_delta: bool, teacher_forcing: bool):
    scores = defaultdict(list)
    for predictions, labels in zip(all_predictions, all_labels):
        for metric in metrics:
            score = metric(predictions, labels)
            scores[metric.__name__].append(score)
            sample_index = random.randint(0, len(predictions) - 1)
        scores["sample pred"].append(predictions[sample_index])
        scores["sample label"].append(labels[sample_index])
    df = pd.DataFrame.from_dict(scores)
    df.to_csv(os.path.join("outputs/evaluation_results", f"period_results_{dataset_name}_{model_name}{'_delta' if eval_delta else ''}{'_forced' if teacher_forcing else ''}.csv"))


def eval_by_year(model: Model, features: np.ndarray, labels: np.ndarray, metrics, dataset_name: str, predict_delta: bool = True, teacher_forcing = False, eval_delta=False):

    autoregessive_feature = None
    all_predictions: List[np.ndarray] = []



    for period_features, period_labels in zip(features, labels):

        if autoregessive_feature is not None:
            if teacher_forcing:
                assert np.all(period_features[:, -1] == autoregessive_feature)
            period_features[:, -1] = autoregessive_feature

        predictions = model.predict(period_features)

        autoregessive_feature = get_autoregressive_feature(eval_delta = eval_delta, predict_delta=predict_delta, teacher_forcing=teacher_forcing, predictions=predictions, period_labels=period_labels, period_features=period_features)
        final_predictions = get_final_predictions(predict_delta=predict_delta, eval_delta=eval_delta, predictions=predictions, period_features=period_features)
        all_predictions.append(final_predictions)

    evaluate_model_by_period(all_predictions, labels, metrics=metrics, model_name=model.__class__.__name__, dataset_name = dataset_name, eval_delta=eval_delta, teacher_forcing=teacher_forcing)

def get_autoregressive_feature(predict_delta: bool, eval_delta: bool, teacher_forcing: bool, period_labels, predictions, period_features):
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
    train(dataset_name=DatasetName.BOATS.value, test=False)