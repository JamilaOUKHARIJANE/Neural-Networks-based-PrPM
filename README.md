# Neural Networks-based Predictive process monitoring
This repository contains the source code of a Neural Network-based predictive process monitoring system that provides:
- prediction of next activities of a process instance
- prediction of next activities with allocated resources of a process instance 
- prediction of a process instance outcome
## Requirements
The following Python packages are required:

-   [keras]() tested with version 3.4.1;
-   [tensorflow]() tested with version 2.17.0;
-   [jellyfish]() tested with version 0.9.0;
-   [Distance]() tested with version 0.1.3.
-   [pm4py]() tested with version 2.5.2.
-   [matplotlib](https://matplotlib.org/) tested with version 3.6.3;
-   [numpy]() tested with version 1.26.4;
-   [pandas]() tested with version 1.5.3;
-   [keras-nlp]() tested with version 0.14.0.


## Usage
The system has been tested with Python 3.10 After installing the requirements, please download this repository.

## Repository Structure
- `data/input` contains the input logs in`.xes`,`.xes.gz`or`.csv` format and the BPMN models of these logs;
- `media/output` contains the trained models and results of predictive process monitoring;
- `src/commons` contains the code defining the main settings for the experiments (training and evaluation);
- `src/training` contains the code for Neural Networks model training;
- `src/evaluation` contains the code for evaluating the trained model and generating predictions; 
- `experiments_runner.py` is the main Python script for running the experiments;
- `results_aggregator.py` is a Python script for aggregating the results of each dataset and presenting in a more 
  understandable format.
  

## Running the code
### (1) Training
To train a Neural Networks model: **LSTM**: `--model="LSTM"` or **transformer** `--model="keras_trans"` for a given dataset (event log), type: 
```
python run_experiments.py --log='helpdesk.csv' --model="LSTM" --train
```
The categorical data is used in training model as defined in the event log. 
if you need to use one-hot encoding of categorical data, so you just need to add`--one_hot_encoding`
```
$ python run_experiments.py --log='helpdesk.csv' --model="LSTM" --train --one_hot_encoding
```
By default, the given dataset is split in 90% for the training and 10% for the testing. 
If you need to use a variant-based sampling split of training and testing datasets, then, just add`--use_variant_split` 
```
python run_experiments.py --log='helpdesk.csv' --model="LSTM" --train --use_variant_split
```
### (2) Evaluation
To run the evaluation for a given (pretrained) dataset, you need to specify the prediction algorithm: baseline `--algo="baseline"` to select the best prediction or `--algo="beamsearch"` to use a [Beam Search](https://towardsdatascience.com/foundations-of-nlp-explained-visually-beam-search-how-it-works-1586b9849a24) algorithm:

```
python run_experiments.py --log='helpdesk.csv' --model="LSTM" --algo="baseline" --evaluate
```
if you want to minimise the prediction of redundant activities/resources, 
you need to apply the probability reduction for repetitive activities/resources
in a trace prefix by adding`--use_Prob_reduction`:
```
python run_experiments.py --log='helpdesk.csv' --model="LSTM" --algo="baseline" --evaluate --use_Prob_reduction
```
### Training and evaluation
if you want to train and evaluate your model in the same experiment, you need to set the `--full_run` option instead of using `--train` and then `--evaluate` :
```
$ python run_experiments.py --log='helpdesk.csv' --model="LSTM" --algo="baseline" --full_run
```

### Gathering the results
After running the experiments, type:
```
$ python results_aggregator.py 
```
to aggregate the Damerau-Levenshtein distance of activities and resources for all datasets. The results will be in the 
file`aggregated_results_performance.csv` in `media/output`folder. 

Type
```
$ python plot_pred_performance.py
```
to have a plot of aggregated results per process prefix length for each dataset. The plot will be in the file `aggregated_distances_per_prefix.pdf` in the `media/output` 
folder.
