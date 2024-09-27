from enum import Enum

import numpy as np
import pandas as pd
from tsfresh import extract_features


class FusionMode(Enum):
    DATA_MEAN = 0
    DATA_NORM = 1
    FEATURE_MEAN = 2
    FEATURE_NORM = 3


def wrap_extract_features(df: pd.DataFrame):
    selected_columns = [
        f"{header}__{stat}"
        for header in df.columns
        if header != "id"
        for stat in ["maximum", "minimum", "median", "sample_entropy", "skewness"]
    ]
    feat = extract_features(
        df,
        column_id="id",
        column_kind=None,
        column_value=None,
        default_fc_parameters={
            "maximum": None,
            "minimum": None,
            "median": None,
            "sample_entropy": None,
            "skewness": None,
        },
        kind_to_fc_parameters=None,
        disable_progressbar=True,
    )
    return feat[selected_columns]


def calculate_extract_fusion_futures(
    pair_data_1: pd.DataFrame,
    pair_data_2: pd.DataFrame,
    mode: FusionMode,
) -> pd.DataFrame:

    pd1 = pair_data_1.copy()
    pd2 = pair_data_2.copy()
    if "time" in pair_data_1.columns:
        pd1 = pd1.drop(columns=["time"])
    if "time" in pair_data_2.columns:
        pd2 = pd2.drop(columns=["time"])

    # fusion before feature extraction
    if mode == FusionMode.DATA_MEAN:
        fusion_df = (pd1 + pd2) / 2
    elif mode == FusionMode.DATA_NORM:
        fusion_df = np.sqrt(pd1.applymap(lambda x: x**2) + pd2.applymap(lambda x: x**2))

    # feature extraction
    if mode == FusionMode.DATA_MEAN or mode == FusionMode.DATA_NORM:
        feature = wrap_extract_features(fusion_df)
    elif mode == FusionMode.FEATURE_MEAN or mode == FusionMode.FEATURE_NORM:
        fusion_df1 = wrap_extract_features(pd1).reset_index().drop("index", axis=1)
        fusion_df2 = wrap_extract_features(pd2).reset_index().drop("index", axis=1)

    # fusion after feature extraction
    if mode == FusionMode.FEATURE_MEAN:
        feature = (fusion_df1 + fusion_df2) / 2
    elif mode == FusionMode.FEATURE_NORM:
        feature = np.sqrt(
            fusion_df1.applymap(lambda x: x**2) + fusion_df2.applymap(lambda x: x**2)
        )

    return feature
