import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path
from data import get_data


# Create preprocessing pipeline


def preprocess(dataset):

    num_attribs = dataset.select_dtypes(include = ['number']).columns.to_list()
    cat_attribs = dataset.select_dtypes(include = ['category']).columns.to_list()

    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])

    full_pipeline = ColumnTransformer([
        ('numerical', num_pipeline, num_attribs),
        ('categorical', LabelEncoder(), cat_attribs)
    ])

    preprocess_dataset = full_pipeline.fit_transform(dataset)

    return preprocess_dataset






