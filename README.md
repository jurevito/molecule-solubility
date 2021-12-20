# Molecule solubility
We tried to predict water solubility of molecules based on molecular structure. Graph models used are from machine learning library [deepchem](https://github.com/deepchem/deepchem). We compared:
- Graph Convolutional Model
- Message Passing Neural Network
- Random Forest Regressor.

## Results
Dataset was split into training (80%) and testing (20%) set. Results were measured on testing set.
| Model | RMSE | MAE  | R2   |
|:-----:|:----:|:----:|:----:|
| GCM   | 0.00 | 0.00 | 0.00 |
| MPNN  | 0.00 | 0.00 | 0.00 |
| RFR   | 0.00 | 0.00 | 0.00 |
