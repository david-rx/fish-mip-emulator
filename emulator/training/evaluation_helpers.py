from collections import defaultdict
import pandas as pd
import numpy as np
import os
from typing import List, Callable

from emulator.training.plotting import plot_animated_map, plot_map

def by_period_overall_results(all_predictions, all_labels, metrics):
    # Calculate overall scores for the model's performance on the entire dataset
    overall_scores = defaultdict(float)

    # Loop through each period in the dataset
    for predictions, labels in zip(all_predictions, all_labels):
        # Loop through each evaluation metric
        for metric in metrics:
            # Calculate the score for the current period
            score = metric(predictions, labels)
            # Add the score to the overall scores, using the metric's name as the key
            overall_scores[metric.__name__] += score

    # Divide the overall scores by the number of periods to get the average score across all periods
    for metric_name, overall_score in overall_scores.items():
        overall_scores[metric_name] = overall_score / len(all_predictions)

    # Output the overall scores to the console
    for metric_name, overall_score in overall_scores.items():
        print(f"{metric_name}: {overall_score}")

def evaluate_model_by_period(all_predictions, all_labels, metrics: List[Callable], model_name: str, dataset_name: str, eval_delta: bool, teacher_forcing: bool, features: np.ndarray, lat_features: bool):
    """
    Evaluates the predictions by period, and saves the results to a csv file.
    """
    scores = defaultdict(list)
    print("true overall results: ")
    print(f"pred shapes: len: {len(all_predictions)}, 1st shape{all_predictions[0].shape}")
    # print(f"label shapes: len: {len(all_labels)}, 1st shape{all_labels[0].shape}")
    by_period_overall_results(all_predictions, all_labels, metrics)
    print(f"results on first 120: ")
    by_period_overall_results(all_predictions[0:120], all_labels[0:120], metrics)
    print(f"results on last 120: ")
    by_period_overall_results(all_predictions[:-120], all_labels[:-120], metrics)
    print(f"model {model_name} has prediction shape {all_predictions[0].shape} ---------------------------------------")

    yearly_predictions = all_predictions[0:len(all_predictions):12]
    yearly_labels = all_labels[0:len(all_labels):12]

    latitude_features_single_period  = np.repeat(np.expand_dims(np.expand_dims(np.arange(90, -89.5, -1), axis=1), axis=0), 360, axis=-1)
    num_periods = len(yearly_labels)
    if not lat_features:
        latitudes = np.repeat(np.expand_dims(latitude_features_single_period, axis=0), num_periods, axis=0).reshape(num_periods, -1, 1)
        longitude_features_single_period = np.repeat(np.expand_dims(np.expand_dims(np.arange(-179.5, 180, 1), axis=0), axis=0), 180, axis=0)
        longitudes = np.repeat(np.expand_dims(longitude_features_single_period, axis=0), num_periods, axis=0).reshape(num_periods, -1, 1)
        plottable_predictions = yearly_predictions
        plottable_labels = yearly_labels
        print(f"shapes are lat {latitudes.shape} long {longitudes.shape} preds {plottable_predictions.shape} labels {plottable_labels.shape}")
    else:
        latitudes = features[0, :, 3]
        longitudes = features[0, :, 4]

        latitudes = np.tile(latitudes, (len(yearly_predictions), 1))
        longitudes = np.tile(longitudes, (len(yearly_predictions), 1))
        plottable_predictions = np.array(yearly_predictions).reshape(len(yearly_predictions), latitudes.shape[1], -1)
        plottable_labels = yearly_labels.reshape(len(yearly_labels), latitudes.shape[1], -1)
        print(f"shapes are lat {latitudes.shape} long {longitudes.shape} preds {plottable_predictions.shape} labels {plottable_labels.shape}")
    
    plot_animated_map(predictions=plottable_predictions, labels=plottable_labels, latitudes=latitudes, longitudes=longitudes, model_name=model_name, dataset_name=dataset_name, contextual=not lat_features)

    plotting_period = 120

    for period_index, (predictions, labels) in enumerate(zip(yearly_predictions, yearly_labels)):

        if period_index % plotting_period == 0:
            plot_map(predictions=predictions, labels=labels, latitudes=latitudes[period_index], longitudes=longitudes[period_index], model_name=model_name, dataset_name=dataset_name, year=period_index / 12)
        # by_period_overall_results([predictions], [labels], metrics)
        for metric in metrics:
            score = metric(predictions, labels)
            scores[metric.__name__].append(score)
        sample_index = 100
        scores["sample pred"].append(predictions[sample_index])
        scores["sample label"].append(labels[sample_index])
        scores["sample metric"].append(metrics[1]([predictions[sample_index]], [labels[sample_index]]))
    df = pd.DataFrame.from_dict(scores)
    df.to_csv(os.path.join("outputs/evaluation_results", f"period_results_{dataset_name}_{model_name}{'_delta' if eval_delta else ''}{'_forced' if teacher_forcing else ''}.csv"))

    #output a sample of the the features and predictions in a csv file.
    #There are many examples, so we only output
    features_array = np.stack(features)
    labels_array = np.stack(all_labels)
    all_predictions_array = np.stack(all_predictions)
    features_subset = features_array[:, 10]
    labels_subset = labels_array[:, 10]
    all_predictions_subset = all_predictions_array[:, 10]
    # sample_df = pd.DataFrame.from_dict({"surface temp": features_subset[:, 0], "intpp": features_subset[:, 1], "prev tcb": features_subset[:, 2], "predicted_tcb": all_predictions_subset, "tcb": labels_subset[:, 0],})

    
    # # sample_df = pd.DataFrame.from_dict({"surface temp": features[:, 0], "intpp": features[:, 1], "tcb": labels[:, 0, ], "prev tcb": features[:, 2]})
    # sample_df.to_csv(os.path.join("outputs/evaluation_results", f"sample_{dataset_name}_{model_name}{'_delta' if eval_delta else ''}{'_forced' if teacher_forcing else ''}.csv"))

def evaluate_models(models, features, labels, metrics: List[Callable], dataset_name: str):
    """
    Evaluates the models on the given features and labels, and saves the results to a csv file.
    """
    scores = defaultdict(list)
    for model in models:
        model_name = model.__class__.__name__
        predictions = model.predict(features)
        for metric in metrics:
            score = metric(predictions, labels)
            scores[metric.__name__].append(score)
        scores["sample pred"].append(predictions[0])
        scores["sample label"].append(labels[0])

        df = pd.DataFrame.from_dict(scores)
        df.to_csv(os.path.join("outputs/evaluation_results", f"results_{dataset_name}_{model_name}.csv"))

def evaluate_model_spatial_by_period_spatial(predictions, labels, metrics: List[Callable], model_name: str, dataset_name: str, eval_delta: bool, teacher_forcing: bool, all_input_deltas, features: np.ndarray):
    """
    Evaluates the global (all lat and long) predictions by period, and saves the results to a csv file.
    Predictions and labels are 3D arrays, with dimensions (period, lat, long)
    """
    for prediction, label, input_delta in zip(predictions, labels, all_input_deltas):
        scores = defaultdict(list)
        for metric in metrics:
            score = metric(prediction, label)
            scores[metric.__name__].append(score)
        scores["sample pred"].append(prediction[0, 0])
        scores["sample label"].append(label[0, 0])
        scores["predicted delta"].append(input_delta[0, 0])
        scores["sample metric"].append(metrics[1]([prediction[0, 0]], [label[0, 0]]))
        df = pd.DataFrame.from_dict(scores)
        df.to_csv(os.path.join("outputs/evaluation_results", f"spatial_period_results_{dataset_name}_{model_name}{'_delta' if eval_delta else ''}{'_forced' if teacher_forcing else ''}.csv"))