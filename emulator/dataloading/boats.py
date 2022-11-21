"""
Module to load and process the input data and predictions of MACROECOLOGICAL model,
for use with ML models for prediction.
MACROECOLOGICAL Paper:
"""
from typing import List
import netCDF4
import numpy as np
from sklearn.model_selection import train_test_split

from emulator.dataloading.dataloader import Dataloader


INPUTS_PATH_TOS_BOATS = "../Inputs/BOATS/gfdl-esm4_r1i1p1f1_historical_tos_60arcmin_global_monthly_1950_2014.nc"
INPUTS_PATH_INTPP_BOATS = "../Inputs/BOATS/gfdl-esm4_r1i1p1f1_historical_intpp_60arcmin_global_monthly_1950_2014.nc"
OUTPUTS_PATH_BOATS = "../Outputs/BOATS/boats_gfdl-esm4_nobasd_historical_nat_default_tcb_global_monthly_1950_2014.nc"

TEST_INPUTS_PATH_TOS_BOATS = "../Inputs/BOATS/gfdl-esm4_r1i1p1f1_ssp585_tos_60arcmin_global_monthly_2015_2100.nc"
TEST_INPUTS_PATH_INTPP_BOATS = "../Inputs/BOATS/gfdl-esm4_r1i1p1f1_ssp585_intpp_60arcmin_global_monthly_2015_2100.nc"
TEST_OUTPUTS_PATH_BOATS = "../Outputs/BOATS/boats_gfdl-esm4_nobasd_ssp585_nat_default_tcb_global_monthly_2015_2100.nc"

class BoatsDataloader(Dataloader):
    def __init__(self, inputs_path_tos: str, outputs_path: str, inputs_path_intpp, mask_tos: bool = False, mask_intpp: bool = False) -> None:
        """
        Read NetCDF datasets.
        """
        inputs_dataset_tos = netCDF4.Dataset(inputs_path_tos)
        inputs_dataset_intpp = netCDF4.Dataset(inputs_path_intpp)
        outputs_dataset = netCDF4.Dataset(outputs_path)
        print(outputs_dataset.variables)

        tos = np.asarray(inputs_dataset_tos["tos"]).flatten().reshape(-1, 1) # tos is temperature of surface (input feature)
        intpp = np.asarray(inputs_dataset_intpp["intpp"]).flatten().reshape(-1, 1) #primary production (input feature)
        tcb = np.asarray(outputs_dataset["tcb"]).flatten().reshape(-1, 1) # tcb is the main output to predict

        labels = tcb
        if mask_intpp:
            features_array = tos
            fill_values = [inputs_dataset_tos["tos"]._FillValue]
        elif mask_tos:
            features_array = intpp
            fill_values = [inputs_dataset_intpp["intpp"]._FillValue]
        else:
            features_array = np.concatenate([tos, intpp], axis=1)
            fill_values = [inputs_dataset_tos["tos"]._FillValue, inputs_dataset_intpp["intpp"]._FillValue]

        features_array, self.labels = filter(features_array, labels, fill_values = fill_values,
            label_fill_value = outputs_dataset["tcb"]._FillValue)

        self.features = features_array.reshape(-1, 2 if not mask_tos and not mask_intpp else 1)
        self.labels = self.labels.reshape(-1, 1)

        print(f"features shape: {self.features.shape}")
        print(f"labels shape: {self.labels.shape}")
        print(f"labels max and min: {max(self.labels)} and {min(self.labels)}")
        print(f"features max and min: {self.features.min()} and {self.features.min()}")

