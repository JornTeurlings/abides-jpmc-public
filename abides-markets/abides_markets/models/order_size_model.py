import json

import numpy as np
from pomegranate.gmm import GeneralMixtureModel

order_size = {
    "class": "GeneralMixtureModel",
    "distributions": [
        {
            "class": "Distribution",
            "name": "LogNormalDistribution",
            "parameters": [2.9, 1.2],
            "frozen": False,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [100.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [200.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [300.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [400.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [500.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [600.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [700.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [800.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [900.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [1000.0, 0.15],
            "frozen": True,
        },
    ],
    "weights": [
        0.2,
        0.7,
        0.06,
        0.004,
        0.0329,
        0.001,
        0.0006,
        0.0004,
        0.0005,
        0.0003,
        0.0003,
    ],
}

order_size_heavy_orders_model = {
    "class": "GeneralMixtureModel",
    "distributions": [
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [800.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [900.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [1000.0, 0.15],
            "frozen": True,
        },
    ],
    "weights": [
        0.33,
        0.33,
        0.33
    ],
}


class OrderSizeModel:
    def __init__(self, heavy_model=False) -> None:
        self.model = GeneralMixtureModel.from_json(
            json.dumps(order_size if not heavy_model else order_size_heavy_orders_model))

    def sample(self, random_state: np.random.RandomState) -> float:
        return round(self.model.sample(random_state=random_state))
