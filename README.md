# Molecule solubility
We tried to predict water solubility of molecules based on molecular structure. Graph models used are from machine learning library [deepchem](https://github.com/deepchem/deepchem). We compared:
- Graph Convolutional Model
- Message Passing Neural Network
- Random Forest Regressor.

## Results
Dataset was split into training (80%) and testing (20%) set. Results were measured on testing set. Best performance was achieved by graph models especially MPNN.
| Model | RMSE  |  MAE  |  R2   |
|:-----:|:-----:|:-----:|:-----:|
|  GCM  | 0.799 | 0.618 | 0.854 |
| MPNN  | 0.593 | 0.454 | 0.924 |
|  RFR  | 1.142 | 0.872 | 0.701 |
