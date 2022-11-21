from dataclasses import dataclass


@dataclass
class PredictionRequest:
    intpp: float
    surface_temp: float

@dataclass
class PredictionResponse:
    net_biomass: float