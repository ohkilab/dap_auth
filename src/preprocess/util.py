import pandas as pd
import numpy as np


def removal_gravitational_acceleration(
    df: pd.DataFrame, alpha: float = 0.8
) -> pd.DataFrame:

    acc_label = "acc"
    acc_column_names = [acc_label + s for s in ["X", "Y", "Z"]]

    removed_df = df.copy()

    for acc_column_name in acc_column_names:
        removed_df.loc[acc_column_name] = high_pass_filter(
            removed_df[acc_column_name].to_numpy(), alpha
        )
    return removed_df


def high_pass_filter(
    time_series_data: np.typing.ArrayLike, alpha: float = 0.8
) -> np.typing.ArrayLike:

    filtered_data = time_series_data.copy()
    offset = time_series_data[0]

    for i in range(0, len(time_series_data) - 1):
        filtered_data[i] -= offset
        offset = alpha * offset + (1 - alpha) * filtered_data[i + 1]

    return filtered_data
