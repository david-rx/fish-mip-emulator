"""
Module to load and process the input data and predictions of MACROECOLOGICAL model,
for use with ML models for prediction.
MACROECOLOGICAL Paper:
"""
from typing import Tuple
import netCDF4
import numpy as np

from emulator.dataloading.dataloader import Dataloader, filter_fill, filter_fill_by_period


INPUTS_PATH_TOS = "../Emulator/Inputs/MACROECOLOGICAL/gfdl-esm4_r1i1p1f1_historical_tos_onedeg_global_annual_1950_2014.nc"
INPUTS_PATH_INTPP = "../Emulator/Inputs/MACROECOLOGICAL/gfdl-esm4_r1i1p1f1_historical_intpp_onedeg_global_annual_1950_2014.nc"
OUTPUTS_PATH = "../Emulator/Outputs/MACROECOLOGICAL/macroecological_gfdl-esm4_nobasd_historical_nat_default_tcb_global_annual_1950_2014.nc"

TEST_INPUTS_PATH_TOS = "../Emulator/Inputs/MACROECOLOGICAL/gfdl-esm4_r1i1p1f1_ssp585_tos_onedeg_global_annual_2015_2100.nc"
TEST_INPUTS_PATH_INTPP = "../Emulator/Inputs/MACROECOLOGICAL/gfdl-esm4_r1i1p1f1_ssp585_intpp_onedeg_global_annual_2015_2100.nc"
TEST_OUTPUTS_PATH = "../Emulator/Outputs/MACROECOLOGICAL/macroecological_gfdl-esm4_nobasd_ssp585_nat_default_tcb_global_annual_2015_2100.nc"


class MacroecologicalDataLoader(Dataloader):
    def __init__(self, inputs_path_tos: str, outputs_path: str, inputs_path_intpp, mask_tos: bool = False, mask_intpp: bool = False) -> None:
        """
        Read NetCDF datasets.
        """
        inputs_dataset_tos = netCDF4.Dataset(inputs_path_tos)
        inputs_dataset_intpp = netCDF4.Dataset(inputs_path_intpp)
        outputs_dataset = netCDF4.Dataset(outputs_path)

        tos = np.asarray(inputs_dataset_tos["tos"]).flatten().reshape(-1, 1) # tos is temperature of surface (input feature)
        intpp = np.asarray(inputs_dataset_intpp["intpp"]).flatten().reshape(-1, 1) #primary production (input feature)
        tcb = np.asarray(outputs_dataset["tcb"]).flatten().reshape(-1, 1) # tcb is the main output to predict

        labels = tcb
        if mask_intpp:
            features_array = tos
            fill_values = (inputs_dataset_tos["tos"]._FillValue,)
        elif mask_tos:
            features_array = intpp
            fill_values = (inputs_dataset_intpp["intpp"]._FillValue,)
        else:
            features_array = np.concatenate([tos, intpp], axis=1)
            fill_values = (inputs_dataset_tos["tos"]._FillValue, inputs_dataset_intpp["intpp"]._FillValue)

        features_array, self.labels = filter_fill(features_array, labels, fill_values=fill_values)

        self.features = features_array.reshape(-1, 2 if not mask_tos and not mask_intpp else 1)
        self.labels = self.labels.reshape(-1, 1)

        print(f"features shape: {self.features.shape}")
        print(f"labels shape: {self.labels.shape}")

    def get_features_flat(self, inputs_dataset_tos, inputs_dataset_intpp, outputs_dataset, predict_delta: bool) -> Tuple[np.ndarray, np.ndarray]:
        shifted_unflattened_tcb = outputs_dataset["tcb"][:-1]
        tos = np.asarray(inputs_dataset_tos["tos"][1:]).flatten().reshape(-1, 1) # tos is temperature of surface (input feature)
        intpp = np.asarray(inputs_dataset_intpp["intpp"][1:]).flatten().reshape(-1, 1) #primary production (input feature)
        shifted_tcb = shifted_unflattened_tcb.flatten().reshape(-1, 1)
        tcb = np.asarray(outputs_dataset["tcb"][1:]).flatten().reshape(-1, 1) # tcb is the main output to predict
        delta_tcb = tcb - shifted_tcb
        features = np.concatenate([tos, intpp], axis= -1)
        latitude_features_single_period = np.repeat(np.expand_dims(np.expand_dims(np.arange(90, -89.5, -1), axis=1), axis=0), 360, axis=0)
        latitude_features = np.repeat(np.expand_dims(latitude_features_single_period, axis=0), inputs_dataset_tos["tos"][1:].shape[0], axis=0).flatten().reshape(-1, 1)
        longitude_features_single_period = np.repeat(np.expand_dims(np.expand_dims(np.arange(-179.5, 180, 1), axis=0), axis=0), 180, axis=0)
        longitude_features = np.repeat(np.expand_dims(longitude_features_single_period, axis=0), inputs_dataset_tos["tos"][1:].shape[0], axis=0).flatten().reshape(-1, 1)
        features = np.concatenate([features, latitude_features, longitude_features], axis= -1)

        labels = delta_tcb if predict_delta else tcb
        fill_values = [inputs_dataset_tos["tos"]._FillValue, inputs_dataset_intpp["intpp"]._FillValue, outputs_dataset["tcb"]._FillValue]
        features, labels = filter_fill(features, labels, fill_values = fill_values,
                label_fill_value = outputs_dataset["tcb"]._FillValue)
        return features, labels

    def get_features_by_period(self, inputs_dataset_tos, inputs_dataset_intpp, outputs_dataset, predict_delta: bool) -> Tuple[np.ndarray, np.ndarray]:
        shifted_unflattened_tcb = outputs_dataset["tcb"][:-1]
        num_periods = np.asarray(inputs_dataset_tos["tos"]).shape[0] - 1
        intpp = np.asarray(inputs_dataset_intpp["intpp"][1:].reshape(num_periods, -1, 1))
        tos = np.asarray(inputs_dataset_tos["tos"][1:].reshape(num_periods, -1, 1))
        shifted_tcb = shifted_unflattened_tcb.reshape(num_periods, -1, 1)  # num_months -1, lat * long, 1
        tcb = outputs_dataset["tcb"][1:].reshape(num_periods, -1, 1)
        delta_tcb = tcb - shifted_tcb
        features = np.concatenate([tos, intpp, shifted_tcb], axis= -1)

        latitude_features_single_period  = np.repeat(np.expand_dims(np.expand_dims(np.arange(90, -89.5, -1), axis=1), axis=0), 360, axis=-1)
        latitude_features = np.repeat(np.expand_dims(latitude_features_single_period, axis=0), num_periods, axis=0).reshape(num_periods, -1, 1)
        print(latitude_features.shape)
        longitude_features_single_period = np.repeat(np.expand_dims(np.expand_dims(np.arange(-179.5, 180, 1), axis=0), axis=0), 180, axis=0)
        longitude_features = np.repeat(np.expand_dims(longitude_features_single_period, axis=0), num_periods, axis=0).reshape(num_periods, -1, 1)

        features = np.concatenate([features, latitude_features, longitude_features], axis= -1)
        labels = delta_tcb if predict_delta else tcb
        fill_values = [inputs_dataset_tos["tos"]._FillValue, inputs_dataset_intpp["intpp"]._FillValue, outputs_dataset["tcb"]._FillValue, inputs_dataset_intpp["intpp"]._FillValue, outputs_dataset["tcb"]._FillValue]
        features, labels = filter_fill_by_period(features, labels, fill_values = fill_values,
            label_fill_value = outputs_dataset["tcb"]._FillValue)
        return features, labels