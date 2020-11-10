import os
import json

import numpy as np

from datetime import date, datetime, timedelta

from jump import Jump
from pso import ParticleSwarmOptimizer

jump = Jump()

BEG_OF_TIME = '2016-06-01'
END_OF_TIME = '2020-09-30'

def get_all_assets(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            result = json.load(f)

    else:
        assets, portfolio_id = jump.get_assets_with_all_informations()
        result = {
            "assets": assets,
            "portfolio_id": portfolio_id,
        }

        with open(file_name, "w") as f:
            json.dump(result, f)

    return result["assets"], result["portfolio_id"]


def get_asset_returns(asset):
    returns = []

    prev_date = datetime.strptime(BEG_OF_TIME, '%Y-%m-%d').date() - timedelta(days=1)
    prev_value = 0.0

    for i, values in enumerate(asset["values"]):
        current_date = datetime.strptime(values["date"], '%Y-%m-%d').date()

        nb_days = (current_date - prev_date).days
        diff = (values["return"] - prev_value) / nb_days

        for i in range(nb_days):
            returns.append(prev_value + diff * i)

        prev_value = values[i - 1]# Get the previous value
        prev_date = current_date

    return returns


asset_dict, portfolio_id = get_all_assets("assets_cache.json")

keys = list(asset_dict.keys())
assets = list(asset_dict.values())

returns_list = []

for _, asset in asset_dict.items():
    returns = get_asset_returns(asset)
    returns_list.append(returns)

returns_list = np.array(returns_list)
mean_returns = np.mean(returns_list, axis=1)

returns_covariance = np.cov(returns_list)

def compute_fitness(x):
    return (x.T @ mean_returns - 0) / (returns_covariance @ x @ x.T) # FIXME replace 0


PSO = ParticleSwarmOptimizer(
    params={
        "max_it": 100,
        "pop_size": 1000,
        "hyperparameters": {
            "inertia": 0.09,
            "inertia_dampening": 0.99,
            "cognitive_acceleration": 1.0,
            "social_acceleration": 1.0,
        }
    },
    constraints={
        "n_var": len(asset_dict),
        "var_min": 0.00,
        "var_max": 0.10,
    },
    fitness_function=compute_fitness,
)

PSO.initialize()
PSO.run()
