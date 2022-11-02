"""
Module to process the MACROECOLOGICAL
data for training and evaluation
"""
import netCDF4
import numpy as np
from sklearn.model_selection import train_test_split
import xarray as xr
import nctoolkit as nc


INPUTS_PATH = "../Inputs/MACROECOLOGICAL/gfdl-esm4_r1i1p1f1_historical_tos_onedeg_global_annual_1950_2014.nc"
INPUTS_PATH_INTPP = "../Inputs/MACROECOLOGICAL/gfdl-esm4_r1i1p1f1_historical_intpp_onedeg_global_annual_1950_2014.nc"
OUTPUTS_PATH = "../Outputs/MACROECOLOGICAL/macroecological_gfdl-esm4_nobasd_historical_nat_default_tcb_global_annual_1950_2014.nc"

TEST_INPUTS_PATH = "../Inputs/MACROECOLOGICAL/gfdl-esm4_r1i1p1f1_ssp585_tos_onedeg_global_annual_2015_2100.nc"
TEST_OUTPUTS_PATH = "../Outputs/MACROECOLOGICAL/macroecological_gfdl-esm4_nobasd_ssp585_nat_default_tcb_global_annual_2015_2100.nc"



class MacroecologicalDataLoader:
    def __init__(self, inputs_path: str, outputs_path: str, inputs_path_intpp) -> None:
        # inputs_dataset = netCDF4.Dataset(inputs_path)
        # inputs_dataset = xr.open_dataset(inputs_path)
        # monthly_dataset = inputs_dataset.resample(time="y")
        inputs_dataset = nc.open_data(inputs_path, checks=False)
        inputs_dataset.tmean(["year", "month"])
        # print(inputs_dataset)
        inputs_dataset_intpp = netCDF4.Dataset(inputs_path_intpp)
        outputs_dataset = netCDF4.Dataset(outputs_path)
        tos = np.asarray(inputs_dataset["tos"]) # tos is temperature of surface (input var)
        tcb = np.asarray(outputs_dataset["tcb"]) # tcb is the main output to predict
        # print("outputs", outputs_dataset.variables)
        # ["vdariables"])
        # print("inputs---------------------------------", inputs_dataset.variables)
        self.flattened_tos = tos.flatten()
        print(f"surface temp shape: {tos.shape}")
        self.flattened_tcb = tcb.flatten()
        print(f"tcb output shape {tcb.shape}")

    def load_train_eval(self):
        """
        Gets the train and eval splits for inputs and outputs
        """
        tos_train, tos_eval, tcb_train, tcb_eval = train_test_split(self.flattened_tos, self.flattened_tcb)
        return tos_train, tos_eval, tcb_train, tcb_eval


if __name__ == "__main__":
    loader = MacroecologicalDataLoader(INPUTS_PATH, OUTPUTS_PATH, INPUTS_PATH_INTPP)
    loader.load_train_eval()


