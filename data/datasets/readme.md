# Tutorial on how to create enhanced datasets

1. Store the raw dataset in the `raw/` folder
2. (activate the venv?) idk if this is needed.
3. In the lysubsite directory, run ```python lyscripts/data join --inputs data/datasets/raw/2023-usz-hypopharynx-larynx.csv --output data/joined.csv

```
4.  Run `python lyscripts data enhance data/joined.csv data/enhanced.csv --modalities CT MRI PET FNA diagnostic_consensus pathology pCT`

Make sure the params.yml file is in the root directory
```
