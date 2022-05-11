import os
from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle


def get_extract(
    input_path: str = os.path.join(Path.cwd(), "data"),
    output_path: str = os.path.join(Path.cwd(), "tests", "fixtures"),
    input_file_name: str = "train.csv",
    output_file_name: str = "fixture.csv",
):
    input_file_path = os.path.join(input_path, input_file_name)
    dataset = pd.read_csv(input_file_path, index_col="Id")

    dataset = shuffle(dataset, random_state=42)

    output_file_path = os.path.join(output_path, output_file_name)

    dataset[:99].to_csv(output_file_path, index_label="Id", sep=",")


get_extract()
