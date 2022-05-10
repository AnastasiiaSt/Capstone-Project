import pytest
from click.testing import CliRunner
import os
from pathlib import Path
import pandas as pd
from capstone_project.train import train
from capstone_project.data import get_data


@pytest.fixture
def input_path():
    path = os.path.join(Path.cwd(), "tests", "fixtures", "fixture.csv")
    return path


def test_get_data(input_path):
    X, y = get_data(path=input_path)
    assert X.shape[0] > 1, "Size of the dataset is insufficient."
    assert (
        X.shape[0] == y.shape[0]
    ), "Number of features is not equal to number of labels."


def test_pca_components(input_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            train, ["--dataset_path", input_path, "--pca", True, "--n_components", 100]
        )
        assert (
            result.exit_code == 1
        ), "The value of pca components should be between 0 and number of features in dataset."


def test_variance_threshold(input_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            train,
            [
                "--dataset_path",
                input_path,
                "--variance_threshold",
                True,
                "--threshold",
                -1,
            ],
        )
        assert (
            result.exit_code == 1
        ), "The value of variance threshold should be between 0 and 1."
