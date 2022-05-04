from pathlib import Path
from joblib import dump
import mlflow
import mlflow.sklearn
import click
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from .data import get_data

def evaluation_metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred, average = 'weighted')
    return acc, f1

print('See path: ', Path)
max_depth = 5
#experiment_id = mlflow.create_experiment('First experiment')

@click.command()
@click.option('--max_depth', default = 5, help = 'Tree maximum depth')
@click.option('--dataset_path', default = r'C:\Users\anast\Documents\GitHub\Capstone-Project\data', type = click.Path(exists = True, dir_okay = False, path_type = Path))
def train(dataset_path, max_depth):

    train_X, train_y, test_X = get_data(dataset_path)

    with mlflow.start_run(experiment_id = 3):
        model = DecisionTreeClassifier(random_state = 42, max_depth = max_depth)
        model.fit(train_X, train_y)
        (acc, f1) =  evaluation_metrics(train_y, model.predict(train_X))

        click.echo('Accuracy: {0}. \nF1 score: {1}.'.format(acc, f1))

        mlflow.log_param('max_depth', max_depth)
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('f1_score', f1)
        mlflow.sklearn.log_model(model, 'model')
        print('Max_depth:', max_depth)


