from matplotlib import pyplot as plt
from matplotlib import animation
from typing import List
import numpy as np
import os
# import geopandas as gpd

def plot(predictions: List[float], labels: List[float], model_name: str, dataset_name: str, num_to_plot: int = 5000):
    indices_to_plot = np.random.randint(low = 0, high=len(labels), size=num_to_plot)
    predictions, labels = predictions[indices_to_plot], labels[indices_to_plot]
    fig, ax = plt.subplots()
    ax.scatter(predictions, labels)
    ax.set_xlabel("emulated biomass density (g / m^2)")
    ax.set_ylabel("expected biomass density (g / m ^2)")
    ax.set_title(f"MACROECOLOGICAL Emulator: {model_name}")
    fig.savefig(os.path.join("outputs/visualizations", f"{model_name}_{dataset_name}"))

def plot_map(predictions: List[float], labels: List[float], latitudes: List[float], longitudes: List[float],
    model_name: str, dataset_name: str, year: int, num_to_plot: int = 5000):
    """Plots two graphs: one for the predictions, and one for the labels. 
    The predictions and labels are plotted on the same map, with the same colorbar.
    """
    print("Plotting map")
    indices_to_plot = np.random.randint(low = 0, high=len(labels), size=num_to_plot)
    predictions, labels, latitudes, longitudes = predictions[indices_to_plot], labels[indices_to_plot], latitudes[indices_to_plot], longitudes[indices_to_plot]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(longitudes, latitudes, c=predictions, cmap="viridis")
    ax1.set_title(f"Predicted Biomass Density {year}")
    ax1.set_xlabel("longitude")
    ax1.set_ylabel("latitude")
    ax2.scatter(longitudes, latitudes, c=labels, cmap="viridis")
    ax2.set_title(f"Expected Biomass Density {year}")
    ax2.set_xlabel("longitude")
    ax2.set_ylabel("latitude")
    fig.suptitle(f"MACROECOLOGICAL Emulator: {model_name}")
    fig.savefig(os.path.join("outputs/visualizations", f"{model_name}_{dataset_name}_map_{year}.png"))
    plt.close(fig)

def plot_animated_map(predictions: List[float], labels: List[float], latitudes: List[float], longitudes: List[float], model_name: str, dataset_name: str, num_to_plot: int = 8000, contextual=False):
    """
    Plots an animated map of the predictions and labels.
    """
    print("num years", len(predictions))
    indices_to_plot = np.random.randint(low = 0, high=len(labels), size=num_to_plot)
    # latitudes, longitudes = latitudes[indices_to_plot], longitudes[indices_to_plot]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    yearly_difference = np.average(np.absolute(np.array(predictions) - labels.reshape(np.array(predictions).shape)))
    predictions = np.array(predictions)
    predictions = np.array(predictions).reshape(predictions.shape[0], -1)
    labels = np.array(labels).reshape(labels.shape[0], -1)
    print(f"yearly diff: {yearly_difference}")

    print(f"predictions shape: {predictions.shape} and labels shape: {labels.shape} lat shape: {latitudes.shape} long shape: {longitudes.shape}")

    def update(year):
        # Use the year to index into your prediction and label arrays

        prediction = predictions[year]
        label = labels[year]
        lats = latitudes[year]  
        lons = longitudes[year]
        # prediction, label = prediction[indices_to_plot], label[indices_to_plot]

        # Update the first plot with the current prediction
        ax1.clear()
        
        ax1.scatter(lons, lats, c=np.absolute(prediction - labels[0].reshape(label.shape)), cmap='plasma')
        ax1.set_title('Predicted Biomass Density for year {}'.format(year + 2015))

        # Update the second plot with the current label
        ax2.clear()
        ax2.scatter(lons, lats, c=np.abs(label - labels[0].reshape(label.shape)), cmap='plasma')
        ax2.set_title('Expected Biomass Density for year {}'.format(year + 2015))
        ax2.set_xlabel("longitude")
        ax2.set_ylabel("latitude")
        ax1.set_xlabel("longitude")
        ax1.set_ylabel("latitude")


    anim = animation.FuncAnimation(fig, update, frames=np.arange(0, len(predictions)), interval=500)

    anim.save(os.path.join(f"outputs/visualizations{'_contextual' if contextual else ''}", f"animated_{model_name}_{dataset_name}_map.gif"), dpi=80, writer='Pillow')
    plt.close(fig)

def plot_global_integral(yearly_predictions, yearly_labels, model_name) -> None:
    """
    Plots the predictions and labels by year.
    """
    fig, ax = plt.subplots()
    ax.plot(yearly_predictions, label="predictions")
    ax.plot(yearly_labels, label="labels")
    ax.set_xlabel("year")
    ax.set_ylabel("global biomass integral (g)")
    ax.set_title("Global Biomass Integral")
    ax.legend()
    fig.savefig(os.path.join("outputs/visualizations", f"global_integral_{model_name}.png"))
