import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from typing import Tuple


# Create preprocessing pipeline


def preprocess(
    dataset: pd.DataFrame,
    variance_threshold: bool,
    threshold: int,
    scaling: bool,
    pca: bool,
    n_components: int,
) -> Tuple[pd.DataFrame, dict]:

    params_prep = {}

    num_attribs = dataset.select_dtypes(include=["number"]).columns.to_list()
    cat_attribs = dataset.select_dtypes(include=["category"]).columns.to_list()

    num_pipeline_steps = []
    cat_pipeline_steps = []

    cat_pipeline_steps.append(("encoder", LabelEncoder()))
    num_pipeline_steps.append(("imputer", SimpleImputer()))

    if variance_threshold:
        num_pipeline_steps.append(
            ("variance_threshold", VarianceThreshold(threshold=threshold))
        )
        params_prep["variance_threshold"] = True
        params_prep["threshold"] = threshold

    if pca:
        num_pipeline_steps.append(("pca", PCA(n_components=n_components)))
        params_prep["pca"] = True
        params_prep["n_components"] = n_components

    if scaling:
        num_pipeline_steps.append(("scaling", StandardScaler()))
        params_prep["scaling"] = True

    full_pipeline = ColumnTransformer(
        [
            ("numerical", Pipeline(steps=num_pipeline_steps), num_attribs),
            ("categorical", Pipeline(steps=cat_pipeline_steps), cat_attribs),
        ]
    )

    preprocess_dataset = full_pipeline.fit_transform(dataset)

    return preprocess_dataset, params_prep
