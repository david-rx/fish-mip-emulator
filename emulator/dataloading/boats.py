"""
Module to load and process the input data and predictions of MACROECOLOGICAL model,
for use with ML models for prediction.
MACROECOLOGICAL Paper:
"""
from typing import List
import netCDF4
import numpy as np
from sklearn.model_selection import train_test_split

from emulator.dataloading.dataloader import Dataloader, filter_fill, filter_fill_by_period

INPUTS_PATH_TOS_BOATS = "../Inputs/BOATS/gfdl-esm4_r1i1p1f1_historical_tos_60arcmin_global_monthly_1950_2014.nc"
INPUTS_PATH_INTPP_BOATS = "../Inputs/BOATS/gfdl-esm4_r1i1p1f1_historical_intpp_60arcmin_global_monthly_1950_2014.nc"
OUTPUTS_PATH_BOATS = "../Outputs/BOATS/boats_gfdl-esm4_nobasd_historical_nat_default_tcb_global_monthly_1950_2014.nc"

TEST_INPUTS_PATH_TOS_BOATS = "../Inputs/BOATS/gfdl-esm4_r1i1p1f1_ssp585_tos_60arcmin_global_monthly_2015_2100.nc"
TEST_INPUTS_PATH_INTPP_BOATS = "../Inputs/BOATS/gfdl-esm4_r1i1p1f1_ssp585_intpp_60arcmin_global_monthly_2015_2100.nc"
TEST_OUTPUTS_PATH_BOATS = "../Outputs/BOATS/boats_gfdl-esm4_nobasd_ssp585_nat_default_tcb_global_monthly_2015_2100.nc"

class BoatsDataloader(Dataloader):
    def __init__(self, inputs_path_tos: str, outputs_path: str, inputs_path_intpp, mask_tos: bool = False, mask_intpp: bool = False, debug = False, by_period = False, predict_delta: bool = False) -> None:
        """
        Read NetCDF datasets.
        """
        inputs_dataset_tos = netCDF4.Dataset(inputs_path_tos)
        inputs_dataset_intpp = netCDF4.Dataset(inputs_path_intpp)
        outputs_dataset = netCDF4.Dataset(outputs_path)

        unflattened_tcb = outputs_dataset["tcb"] # (num_months, lat, long, 1)
        starting_shape = inputs_dataset_tos["tos"].shape
        shifted_unflattened_tcb = unflattened_tcb[:-1] # drop first since boundary condition is unknown

        if by_period:
            num_periods = np.asarray(inputs_dataset_tos["tos"]).shape[0] - 1
            intpp = np.asarray(inputs_dataset_intpp["intpp"][1:].reshape(num_periods, -1, 1))
            tos = np.asarray(inputs_dataset_tos["tos"][1:].reshape(num_periods, -1, 1))
            shifted_tcb = shifted_unflattened_tcb.reshape(num_periods, -1, 1)
            tcb = outputs_dataset["tcb"][1:].reshape(num_periods, -1, 1)
            delta_tcb = tcb - shifted_tcb

        else:
            tos = np.asarray(inputs_dataset_tos["tos"][1:]).flatten().reshape(-1, 1) # tos is temperature of surface (input feature)
            intpp = np.asarray(inputs_dataset_intpp["intpp"][1:]).flatten().reshape(-1, 1) #primary production (input feature)

            shifted_tcb = shifted_unflattened_tcb.flatten().reshape(-1, 1)

            tcb = np.asarray(outputs_dataset["tcb"][1:]).flatten().reshape(-1, 1) # tcb is the main output to predict
            delta_tcb = tcb - shifted_tcb

        if debug:
            if by_period:
                keep_shape = 10
            else:
                keep_shape = starting_shape[1] * starting_shape[2] * 2
            tos = tos[:keep_shape, :]
            intpp = intpp[:keep_shape, :]
            shifted_tcb = shifted_tcb[:keep_shape, :]
            tcb = tcb[:keep_shape, :]
            delta_tcb = delta_tcb[:keep_shape, :]
            print(tos.shape)
            print(tcb.shape)


        # print(outputs_dataset.variables)

        labels = delta_tcb if predict_delta else tcb
        if mask_intpp:
            features_array = tos
            fill_values = [inputs_dataset_tos["tos"]._FillValue]
        elif mask_tos:
            features_array = intpp
            fill_values = [inputs_dataset_intpp["intpp"]._FillValue]
        else:
            features_array = np.concatenate([tos, intpp, shifted_tcb], axis= -1)
            fill_values = [inputs_dataset_tos["tos"]._FillValue, inputs_dataset_intpp["intpp"]._FillValue, outputs_dataset["tcb"]._FillValue]

        if by_period:
            features_array, self.labels = filter_fill_by_period(features_array, labels, fill_values = fill_values,
            label_fill_value = outputs_dataset["tcb"]._FillValue)
            self.features = features_array
        else:
            features_array, self.labels = filter_fill(features_array, labels, fill_values = fill_values,
                label_fill_value = outputs_dataset["tcb"]._FillValue)

            #TODO: needed?
            self.features = features_array.reshape(-1, 3 if not mask_tos and not mask_intpp else 1)
            self.labels = self.labels.reshape(-1, 1)

        print(f"features shape: {self.features.shape}")
        print(f"labels shape: {self.labels.shape}")
        print(f"labels max and min: {self.labels.max()} and {self.labels.min()}")
        print(f"features max and min: {self.features.max()} and {self.features.min()}")

