from click.testing import CliRunner
import pytest
import pandas as pd
import os
import joblib
from pathlib import Path
from capstone_project.train import train


@pytest.fixture
def input_labels():
    test_inputs = pd.read_csv(
        os.path.join(Path.cwd(), "tests", "fixtures", "fixture.csv")
    )
    return test_inputs.iloc[:, -1]


@pytest.fixture
def input_features():
    test_inputs = pd.read_csv(
        os.path.join(Path.cwd(), "tests", "fixtures", "fixture.csv")
    )
    return test_inputs.iloc[:, :-1]


@pytest.fixture
def input_path():
    path = os.path.join(Path.cwd(), "tests", "fixtures", "fixture.csv")
    return path


@pytest.fixture
def output_path():
    path = os.path.join(Path.cwd(), "tests", "test_model.joblib")
    return path


def test_train(input_path, output_path, input_features, input_labels):
    runner = CliRunner()
    with runner.isolated_filesystem():

        tree = runner.invoke(
            train,
            [
                "--dataset_path",
                input_path,
                "--save_model_path",
                output_path,
                "--model",
                "Decision Tree",
            ],
        )
        assert tree.exit_code == 0
        loaded_tree = joblib.load(output_path)
        loaded_tree.fit(input_features, input_labels)
        tree_score = loaded_tree.score(input_features, input_labels)
        assert loaded_tree.get_depth() >= 1
        assert loaded_tree.get_n_leaves() >= 2
        assert tree_score >= 0.5

        tree = runner.invoke(
            train,
            [
                "--dataset_path",
                input_path,
                "--save_model_path",
                output_path,
                "--model",
                "Random Forest",
            ],
        )
        assert tree.exit_code == 0
        loaded_forest = joblib.load(output_path)
        loaded_forest.fit(input_features, input_labels)
        forest_score = loaded_tree.score(input_features, input_labels)
        assert forest_score >= 0.5
        assert (
            forest_score >= tree_score
        ), "Random forest score is expected to be higher than \
            Decision tree score."
