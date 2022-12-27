"""
Module to load and process the input data and predictions of MACROECOLOGICAL model,
for use with ML models for prediction.
MACROECOLOGICAL Paper:
"""
from dataclasses import dataclass
from typing import List, Tuple
import netCDF4
import numpy as np
from sklearn.model_selection import train_test_split
from pca import pca

from emulator.dataloading.dataloader import Dataloader, filter_fill, filter_fill_by_period, replace_fill

INPUTS_PATH_TOS_BOATS = "../Emulator/Inputs/BOATS/gfdl-esm4_r1i1p1f1_historical_tos_60arcmin_global_monthly_1950_2014.nc"
INPUTS_PATH_INTPP_BOATS = "../Emulator/Inputs/BOATS/gfdl-esm4_r1i1p1f1_historical_intpp_60arcmin_global_monthly_1950_2014.nc"
OUTPUTS_PATH_BOATS = "../Emulator/Outputs/BOATS/boats_gfdl-esm4_nobasd_historical_nat_default_tcb_global_monthly_1950_2014.nc"

TEST_INPUTS_PATH_TOS_BOATS = "../Emulator/Inputs/BOATS/gfdl-esm4_r1i1p1f1_ssp585_tos_60arcmin_global_monthly_2015_2100.nc"
TEST_INPUTS_PATH_INTPP_BOATS = "../Emulator/Inputs/BOATS/gfdl-esm4_r1i1p1f1_ssp585_intpp_60arcmin_global_monthly_2015_2100.nc"
TEST_OUTPUTS_PATH_BOATS = "../Emulator/Outputs/BOATS/boats_gfdl-esm4_nobasd_ssp585_nat_default_tcb_global_monthly_2015_2100.nc"

@dataclass
class BoatsConfig:
    inputs_path_tos: str
    inputs_path_intpp: str
    outputs_path: str
    mask_tos: bool = False
    mask_intpp: bool = False
    debug: bool = False
    by_period: bool = False
    predict_delta: bool = False
    periodic_spatial: bool = False
    flat: bool = False
    contextual: bool = False

    # def validate(train_config: "BoatsConfig", other: "BoatsConfig"):
    #     count = 0
    #     if train_config.by_period:
    #         count += 1
    #     if train_config.contextual:
    #         count += 1
    #     if train_config.flat:
    #         count += 1
    #     if train_config.


class BoatsDataloader(Dataloader):
    def __init__(self, boats_config: BoatsConfig) -> None:
        """
        Read NetCDF datasets.
        """
        super().__init__()
        inputs_dataset_tos = netCDF4.Dataset(boats_config.inputs_path_tos)
        # print(inputs_dataset_tos.variables)

        inputs_dataset_intpp = netCDF4.Dataset(boats_config.inputs_path_intpp)
        outputs_dataset = netCDF4.Dataset(boats_config.outputs_path)

        self.pca = pca(n_components=20)
        

        if boats_config.by_period:
            features_array, labels = self.get_features_by_period(inputs_dataset_tos, inputs_dataset_intpp, outputs_dataset, boats_config.predict_delta)
        elif boats_config.flat:
            features_array, labels = self.get_features_flat(inputs_dataset_tos, inputs_dataset_intpp, outputs_dataset, boats_config.predict_delta)
        elif boats_config.contextual:
            features_array, labels = self.get_contextual_features(inputs_dataset_tos, inputs_dataset_intpp, outputs_dataset, boats_config.predict_delta)

        
        self.features = features_array
        self.labels = labels
        
        if boats_config.debug:
            self.features = self.features[0:240]
            self.labels = self.labels[0:240]
        super()._print_stats()

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
        # latitude_features_single_period = np.repeat(np.expand_dims(np.expand_dims(np.arange(-89.5, 90, 1), axis=1), axis=0), 360, axis=0)
        latitude_features = np.repeat(np.expand_dims(latitude_features_single_period, axis=0), num_periods, axis=0).reshape(num_periods, -1, 1)
        print(latitude_features.shape)
        longitude_features_single_period = np.repeat(np.expand_dims(np.expand_dims(np.arange(-179.5, 180, 1), axis=0), axis=0), 180, axis=0)
        longitude_features = np.repeat(np.expand_dims(longitude_features_single_period, axis=0), num_periods, axis=0).reshape(num_periods, -1, 1)
        # x = np.cos(np.deg2rad(latitude_features)) * np.cos(np.deg2rad(longitude_features))
        # y = np.cos(np.deg2rad(latitude_features)) * np.sin(np.deg2rad(longitude_features))
        # z = np.sin(np.deg2rad(latitude_features))
        # features = np.concatenate([features, x, y, z], axis= -1)
        features = np.concatenate([features, latitude_features, longitude_features], axis= -1)
        labels = delta_tcb if predict_delta else tcb
        fill_values = [inputs_dataset_tos["tos"]._FillValue, inputs_dataset_intpp["intpp"]._FillValue, outputs_dataset["tcb"]._FillValue, inputs_dataset_intpp["intpp"]._FillValue, outputs_dataset["tcb"]._FillValue]
        features, labels = filter_fill_by_period(features, labels, fill_values = fill_values,
            label_fill_value = outputs_dataset["tcb"]._FillValue)
        return features, labels

    def get_features_flat(self, inputs_dataset_tos, inputs_dataset_intpp, outputs_dataset, predict_delta: bool) -> Tuple[np.ndarray, np.ndarray]:
        shifted_unflattened_tcb = outputs_dataset["tcb"][:-1]
        tos = np.asarray(inputs_dataset_tos["tos"][1:]).flatten().reshape(-1, 1) # tos is temperature of surface (input feature)
        intpp = np.asarray(inputs_dataset_intpp["intpp"][1:]).flatten().reshape(-1, 1) #primary production (input feature)
        shifted_tcb = shifted_unflattened_tcb.flatten().reshape(-1, 1)
        tcb = np.asarray(outputs_dataset["tcb"][1:]).flatten().reshape(-1, 1) # tcb is the main output to predict
        delta_tcb = tcb - shifted_tcb
        features = np.concatenate([tos, intpp, shifted_tcb], axis= -1)
        latitude_features_single_period = np.repeat(np.expand_dims(np.expand_dims(np.arange(90, -89.5, -1), axis=1), axis=0), 360, axis=0)
        latitude_features = np.repeat(np.expand_dims(latitude_features_single_period, axis=0), inputs_dataset_tos["tos"][1:].shape[0], axis=0).flatten().reshape(-1, 1)
        longitude_features_single_period = np.repeat(np.expand_dims(np.expand_dims(np.arange(-179.5, 180, 1), axis=0), axis=0), 180, axis=0)
        longitude_features = np.repeat(np.expand_dims(longitude_features_single_period, axis=0), inputs_dataset_tos["tos"][1:].shape[0], axis=0).flatten().reshape(-1, 1)
        # x = np.cos(np.deg2rad(latitude_features)) * np.cos(np.deg2rad(longitude_features))
        # y = np.cos(np.deg2rad(latitude_features)) * np.sin(np.deg2rad(longitude_features))
        # z = np.sin(np.deg2rad(latitude_features))
        # features = np.concatenate([features, x, y, z], axis= -1)
        features = np.concatenate([features, latitude_features, longitude_features], axis= -1)

        # latitude_feature = np.repeat(np.sin(np.deg2rad(np.arange(-89.5, 90, 1))).reshape(-1, 1),
        # longitude_feature = np.sin(np.deg2rad(np.arange(-179.5, 180, 1))).reshape(-1, 1)
        # features = np.concatenate([features, latitude_features, longitude_features], axis= -1)


        labels = delta_tcb if predict_delta else tcb
        fill_values = [inputs_dataset_tos["tos"]._FillValue, inputs_dataset_intpp["intpp"]._FillValue, outputs_dataset["tcb"]._FillValue]
        features, labels = filter_fill(features, labels, fill_values = fill_values,
                label_fill_value = outputs_dataset["tcb"]._FillValue)
        return features, labels
        
    def get_contextual_features(self, inputs_dataset_tos, inputs_dataset_intpp, outputs_dataset, predict_delta: bool) -> Tuple[np.ndarray, np.ndarray]:
        shifted_tcb = outputs_dataset["tcb"][:-1]
        tos = np.asarray(inputs_dataset_tos["tos"][1:])
        intpp = np.asarray(inputs_dataset_intpp["intpp"][1:])
        tcb = np.asarray(outputs_dataset["tcb"][1:])
        tos = tos.reshape(tos.shape[0], -1)
        intpp = intpp.reshape(intpp.shape[0], -1)
        reshaped_shifted_tcb = shifted_tcb.reshape(shifted_tcb.shape[0], shifted_tcb.shape[1] * shifted_tcb.shape[2])
        features = np.concatenate([tos, intpp], axis= -1)
        labels = tcb
        labels = labels.reshape(labels.shape[0], labels.shape[1] * labels.shape[2])
        print(f"shapes are: autoregressive {shifted_tcb.shape}, intpp {intpp.shape}, tos {tos.shape} all {features.shape}, labels start {tcb.shape}, labels {labels.shape}")
        fill_values = [inputs_dataset_tos["tos"]._FillValue, inputs_dataset_intpp["intpp"]._FillValue, outputs_dataset["tcb"]._FillValue]
        features, labels = replace_fill(features, labels, fill_values = fill_values)
        return features, labels
        # pca_results = self.pca.fit_transform(features)
        # print(pca_results)
        # print(features.shape)

        # print(pca_results.keys())
        # pca_features = pca_results["PC"].to_numpy()
        # print(type(pca_features))
        # print(pca_results["topfeat"])
        # print(f"pca features shape is", pca_features.shape)
        # print("labels shape: ", labels.shape)
        # return pca_features, labels
