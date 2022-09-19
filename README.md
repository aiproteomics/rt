# Experiment with a transformer architecture for retention time prediction


![Transformer for retention time prediction](images/graphical-abstract.png)


This repository contains our experiments with the transformer architecture for retention time prediction in liquid chromatography mass spectrometry-based proteomics.

## Data

- Prosit: data/prosit*

- AutoRT: data/PXD006109*

- DeepDIA: data/PXD005573*

- Independent test set: data/PXD019038*

## Trained models

- Prosit: models/prosit/

- AutoRT: models/autort/

- DeepDIA: models/deepdia/

## Example usages

We used Python 3.9 and Tensorflow 2.6.

### Tunning
```
python rt.py tune -data autort -logs logs-random-search-autort -epochs 50 -n_layers 6 8 10 12 -n_heads 2 4 8 -dropout 0.05 0.1 0.15 -batch_size 64 128 256 512 1024 -d_model 256 512 768 -d_ff 256 512 768 1024 -seed 0 -n_random_samples 100
```
This command will perform a random search for hyperparameter settings. The result is a text output containing epoch, loss, and validation loss for further analysis in R. There is also a log folder for TensorBoard visualization.

### Training

```
python rt.py train -data autort -logs logs-train-autort -epochs 2000 -n_layers 10 -n_heads 8 -dropout 0.1 -batch_size 256 -d_model 256 -d_ff 1024
```
This command will produce a trained model with the specified hyperparameters.

### Testing on holdout data

```
python rt.py predict [model_dir/] -data [prosit|autort|deepdia] -output output.txt
```
The predicted values are in column 'y_pred' in the output tab-separated-value file. The measured values are in column 'y'. One has to use the holdout data corresponding to the trained model.

### Predicting retention time for a new set of peptides

```
python rt.py predict [model_dir/] -input peptides.csv -output peptides-rt.txt
```
The input peptides.csv is a comma-separated-value text file containing two columns named 'sequence' and 'rt'. When measured rt values are not available, one can fill the rt column with 0. The predicted values are in column 'y_pred' in the output tab-separated-value peptides-rt.txt file. The input rt values are copied to column 'y' in the output.
