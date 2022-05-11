## Description
This is the Capstone Project of Machine Learning Course at RS School.<br>
The goal is to implement ML project comprasing of model training, selection and evaluation for classification of the forest cover type. The dataset [Forest Cover Type](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) from Kaggle is used in this project.<br>

Detailed information about training set can be found in the following [report](profile_report.html), which was created using pandas profiling module.

Three models with different hyperparameters were trained to predict the class of forest cover type. K-fold cross validation was employed to split the data into training and validation sets. For evaluation of the classifier, three metrics - precision, recall and f1 score were used. The results are summaries in the picture below:
<img src="./images/Experiment_results_training.png" width="900">

Automatic hyperparameters tuning by means of GridSearchCV was implemented to determine best hyperparameters for the three machine learning models. Nested cross validation was employed to select and evaluate the models. F1 score was used as optimization metric for hyperparameters selection. Generalized performance of the classifier was evaluated with three metrics - precision, recall and f1 score. The results are presented in the picture below:
<img src="./images/Experiment_results_tuning.png" width="900">

Unit and integration tests were implemented to check code corectness. All functions were type annotated. Code formatting were edited using Black and checked using Flake8. The results of the testing and linting is shown on the picture below:
<img src="./images/nox.png" width="900">

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
As a result, html file with the report will be created in repository's root.<br>
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
7. To tune a model for determination of the optimum parameters, the following command can be used:
```sh
poetry run tune --model="Random Forest" --max_depth=[10,20,30,40] --n_estimators=[50,100,150,200]
```
The command requires selection of the model of interest and lists of hyperparameters to tune. To get a full list of tunable models and hyperparameters use *--help*:
```sh
poetry run tune --help
```
8. Run MLflow UI to see the information about conducted experiments:
```sh
poetry run mlflow ui
```

## Testing
To test the code install all dependencies using the command:
```sh
poetry install
```
To run the existing tests, use the following command:
```sh 
poetry run pytest
```
To format and check code lining and type annotation the following three commands can be used:
```sh
black .
flake8 --max_line_length=88
mypy .
```
Alternatively, to run all sessions of testing and formatting the following command can be used:
```sh
nox
```
<br>