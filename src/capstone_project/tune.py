import os
from pathlib import Path
from joblib import dump
import mlflow
import mlflow.sklearn
import click
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score as f1, make_scorer
from sklearn.model_selection import cross_validate, KFold, GridSearchCV
from .data import get_data
from .pipeline import preprocess

def eval_metric(true, pred, average: str):
    precision = precision_score(true, pred, average = average)
    recall = recall_score(true, pred, average = average)
    f1_score = f1(true, pred, average = average)
    return precision, recall, f1_score

class IntList(click.Option):
    def type_cast_value(self, ctx, value) -> list():
        if value:
            value = str(value)
            list_as_str = value.lstrip('[').rstrip(']')
            list_of_items = [int(item) for item in list_as_str.split(',')]
            return list_of_items
        else:
            return []

class FloatList(click.Option):
    def type_cast_value(self, ctx, value) -> list():
        if value:
            value = str(value)
            list_as_str = value.lstrip('[').rstrip(']')
            list_of_items = [float(item) for item in list_as_str.split(',')]
            return list_of_items
        else:
            return []

class StringList(click.Option):
    def type_cast_value(self, ctx, value) -> list():
        if value:
            value = str(value)
            list_as_str = value.lstrip('[').rstrip(']')
            list_of_items = [item for item in list_as_str.split(',')]
            return list_of_items
        else:
            return []

@click.command()
@click.option('--model', type = click.Choice(['Decision Tree', 'Logistic Regression', 'Random Forest'], case_sensitive = False), help = 'The model to be trained')
@click.option('--kf_n_outer', default = 5, help = 'Number of folds for outer loop of cross-validation')
@click.option('--kf_n_inner', default = 5, help = 'Number of folds for inner loop of cross-validation')
@click.option('--scaling', default = False, help = 'Numeric features scaling')
@click.option('--variance_threshold', default = False, help = 'Variance threshold feature selection')
@click.option('--threshold', default = 0.0, help = 'Threshold for variance threshold feature selection')
@click.option('--pca', default = False, help = 'PCA dimensionality reduction')
@click.option('--n_components', default = 2, help = 'Number of  components for PCA dimensionality reduction')

@click.option('--max_depth', cls=IntList, default = [], help = 'Tree maximum depth for decision tree')
@click.option('--max_iter', cls=IntList, default = [], help = 'Maximum number of iterations for logistic regression')
@click.option('--regularization', cls=FloatList, default = [], help = 'Inverse of regularization strength for logistic regression')
@click.option('--n_estimators', cls=IntList, default = [], help = 'Number of trees in random forest')

@click.option('--save_model_path', default = os.path.join(Path.cwd(), 'data/model.joblib'))
@click.option('--dataset_path', default = os.path.join(Path.cwd(), 'data'))
@click.option('--average', default = 'macro')
@click.option('--random_state', default = 42, help = 'Random state')

def tune(model: str, save_model_path: Path, variance_threshold: bool, threshold: bool, scaling: bool, pca: bool, n_components: list, max_depth: list, penalty: list, max_iter: list, regularization: list, n_estimators: list, dataset_path: Path, kf_n_inner: int, kf_n_outer: int, average: str, random_state: int) -> None:

    X, y = get_data(dataset_path)

    X_prep, params_prep = preprocess(X, variance_threshold = variance_threshold, threshold = threshold, scaling = scaling, pca = pca, n_components = n_components)

    all_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_iter': max_iter, 'C': regularization}

    model_params = {}

    for param in all_params.items():
        if param[1]:    
            model_params[param[0]] = param[1]
            click.echo('Model parameter: {0} - {1}'.format(param[0], param[1]))

    if model == 'Decision Tree':
        train_model = DecisionTreeClassifier(random_state = random_state)  
    elif model == 'Logistic Regression':
        train_model = LogisticRegression(random_state = random_state)
    elif model == 'Random Forest':
        train_model = RandomForestClassifier(random_state = random_state)

    cv_outer = KFold(n_splits = kf_n_outer, random_state = random_state, shuffle = True)

    outer_scores = []
    outer_models = []
    outer_params = []

    for train_index, test_index in cv_outer.split(X_prep):
        with mlflow.start_run(experiment_id = 1, run_name = model + '_inner result'):
            X_train, X_test = X_prep[train_index, :], X_prep[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            cv_inner = KFold(n_splits = kf_n_inner, random_state = random_state, shuffle = True)

            gs_metric = make_scorer(f1, average = average)

            gs = GridSearchCV(train_model, model_params, scoring = gs_metric, cv = cv_inner, refit = True)
            result = gs.fit(X_train, y_train)

            best_model = result.best_estimator_
            outer_models.append(best_model)

            y_pred = best_model.predict(X_test)

            precision, recall, f1_score = eval_metric(y_test, y_pred, average = average)
            outer_scores.append((precision, recall, f1_score))

            mlflow.sklearn.log_model(best_model, model)

            best_params = result.best_params_
            outer_params.append(best_params)

            params = {**params_prep, **best_params}
            params['kf_n_outer'] = kf_n_outer
            params['kf_n_inner'] = kf_n_inner

            for param in params.items():
                click.echo('parameter {0} is {1}.'.format(param[0], param[1]))
                mlflow.log_param(param[0], param[1])

            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1_score', f1_score)

    with mlflow.start_run(experiment_id = 1, run_name = model + '_outer result'):

        score_sums = {'avg_precision': 0, 'avg_recall': 0, 'avg_f1_score': 0}
        max_score = []
        for scores in outer_scores:
            score_sums['avg_precision'] += scores[0]
            score_sums['avg_recall'] += scores[1]
            score_sums['avg_f1_score'] += scores[2]
            max_score.append(scores[2])

        for score_sums in score_sums.items():
            mlflow.log_metric(score_sums[0], score_sums[1] / kf_n_outer)
            click.echo('{0} is {1}.'.format(score_sums[0], score_sums[1] / kf_n_outer))

        best_model = outer_models[np.argmax(max_score)]
        mlflow.sklearn.log_model(best_model, model)
        dump(best_model, save_model_path)

        best_params = outer_params[np.argmax(max_score)]

        for param in params.items():
            click.echo('parameter {0} is {1}.'.format(param[0], param[1]))
            mlflow.log_param(param[0], param[1])




        