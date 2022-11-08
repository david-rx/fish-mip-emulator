import os
from typing import Callable, List
from sklearn import tree
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from emulator.models.mean_guesser import MeanGuesser
from emulator.models.random_guesser import RandomGuesser
from emulator.dataloading.macroecological import INPUTS_PATH_TOS, INPUTS_PATH_INTPP, OUTPUTS_PATH, MacroecologicalDataLoader, TEST_INPUTS_PATH_TOS, TEST_INPUTS_PATH_INTPP, TEST_OUTPUTS_PATH


def evaluate(metrics: List[Callable], models: List):
    #TODO: separate eval from training
    pass

def plot(predictions: List[float], labels: List[float], model_name: str, num_to_plot: int = 1000):
    indices_to_plot = np.random.randint(low = 0, high=len(labels), size=num_to_plot)
    predictions, labels = predictions[indices_to_plot], labels[indices_to_plot]
    fig, ax = plt.subplots()
    ax.scatter(predictions, labels)
    ax.set_xlabel("emulated biomass density (g / m^2)")
    ax.set_ylabel("expected biomass density (g / m ^2)")
    ax.set_title(model_name)
    fig.savefig(os.path.join("outputs", model_name))

def train(test: bool = False) -> None:
    dataloader = MacroecologicalDataLoader(inputs_path_tos = INPUTS_PATH_TOS, outputs_path = OUTPUTS_PATH, inputs_path_intpp = INPUTS_PATH_INTPP)
    train_features, eval_features, train_labels, eval_labels = dataloader.load_train_eval()
    test_dataloader = MacroecologicalDataLoader(inputs_path_tos = TEST_INPUTS_PATH_TOS, outputs_path = TEST_OUTPUTS_PATH, inputs_path_intpp = TEST_INPUTS_PATH_INTPP)
    test_features, test_labels = test_dataloader.features, test_dataloader.labels
    print(f"DATA METRICS: train mean is {train_labels.mean()} with stdev {train_labels.std()}.")
    print(f"eval mean is {eval_labels.mean()} with stdev {eval_labels.std()}")

    #fit models
    models = [RandomGuesser(), MeanGuesser(), LinearRegression(), tree.DecisionTreeRegressor(), RandomForestRegressor(n_estimators=3), HistGradientBoostingRegressor(max_iter=50)]
    for model in models:
        model.fit(train_features, train_labels)

    eval_metrics = [metrics.mean_squared_error, metrics.mean_absolute_error, metrics.r2_score]

    model_names = []
    mse_scores = []
    mae_scores = []
    r2_scores = []

    #make predictions, output results
    features = test_features if test else eval_features
    labels = test_labels if test else eval_labels

    for model in models:

        model_name = model.__class__.__name__
        model_names.append(model_name)
        eval_predictions = model.predict(features)

        plot(eval_predictions, labels, model_name)

        for metric in eval_metrics:
            score = metric(eval_predictions, labels)
            if metric.__name__ == "mean_squared_error":
                mse_scores.append(score)
            elif metric.__name__ == "mean_absolute_error":
                mae_scores.append(score)
            elif metric.__name__ == "r2_score":
                r2_scores.append(score)
            print(f"model {model.__class__.__name__} getting score {score} on metric {metric.__name__}")

    df = pd.DataFrame.from_dict({"Name": model_names, "MSE_Score": mse_scores, "MAE_Score": mae_scores, "R2_Score": r2_scores})
    df.to_csv(f"results_MACROEOLOGICAL{'_test' if test else ''}.csv")


if __name__ == "__main__":
    train(test=True)