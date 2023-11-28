import pandas as pd
from pathlib import Path


DATASETS = 'all'
datasets_paths = []

if (DATASETS == 'oropharynx') or (DATASETS=='all'):
    datasets_paths.append(Path("../data/datasets/2022-CLB-oropharynx.csv"))
    datasets_paths.append(Path("../data/datasets/2022-USZ-oropharynx.csv"))
if (DATASETS == 'multisite') or (DATASETS=='all'):
    datasets_paths.append(Path("../data/datasets/2022-CLB-multisite.csv"))
    datasets_paths.append(Path("../data/datasets/2022-ISB-multisite.csv"))


dataset = pd.DataFrame({})
for dataset_path in datasets_paths:
    dataset_new= pd.read_csv(dataset_path, header=[0, 1, 2])
    dataset = pd.concat([dataset, dataset_new], ignore_index=True)


locations = ['oropharynx', 'oral cavity', 'hypopharynx', 'larynx']

oropharynx = dataset["tumor"]["1"]["location"] == 'oropharynx'
oral_cavity = dataset["tumor"]["1"]["location"] == 'oral cavity'
hypopharynx = dataset["tumor"]["1"]["location"] == 'hypopharynx'
larynx = dataset["tumor"]["1"]["location"] == 'larynx'


oropharynx = dataset[oropharynx]
oral_cavity = dataset[oral_cavity]
hypopharynx = dataset[hypopharynx]
larynx = dataset[larynx]