## Description
This is the Capstone Project of Machine Learning Course at RS School.<br>
The goal is to implement ML project comprasing of model training, selection and evaluation for prediction of the forest cover type. The dataset [Forest Cover Type](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) from Kaggle is used in this project.
<img src="./images/Experiment_results_training.jpg", width="350">

## Usage
1. Clone this repository to your machine.<br>
2. Download [Forest Cover Type](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset, save csv locally (default path is *data/train.csv* in repository's root).<br>
3. Make sure Python 3.9 and Poetry are installed on your machine.<br>
4. Install the project dependencies with the following command:
```sh
poetry install --no-dev
```
5. Pandas profiling report for training data can be created using the following command:
```sh
poetry run report
```
As a result, html file with the report will be created in the */data* directory.<br>
6. Model can be trained with the following command:
```sh
poetry run train 
```
Default model is Decision Tree with maximum depth = 10. You can select model and define hyperparameters in the CLI. For instance:
```sh
poetry run train --model="Logistic Regression" --regularization=2.5 --max_iter=1000 --scaling=True
```
To get a full list of available models and hyperparameters, use *--help*:
```sh
poetry run train --help
```
7. To determine the optimum parameters for a model, the following command can be used:
```sh
poetry run tune --model="Random Forest" --max_depth=[10,20,30,40] --n_estimators=[50,100,150,200]
```
The command requires selection of the model of interest and lists of hyperparameters to tune. To get a full list of tunable models and hyperparameters use *--help*:
```sh
poetry --help
```
8. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```

## Testing
To test the code install all requirements:
```sh
poetry install
```
To run the existing tests, use the following command:
```sh 
poetry run pytest
```
Additionally, to run all sessions of testing and formatting the following command can be used:
```sh
nox
```