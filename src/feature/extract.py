import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def standardization(df: pd.DataFrame) -> pd.DataFrame:

    standard = StandardScaler().fit_transform(df)
    standard_df = pd.DataFrame(standard)

    return standard_df


def triaxial_attributes_l2norm(
    df: pd.DataFrame, triaxial_attributes_colnames=["acc", "gyro"]
) -> pd.DataFrame:
    triaxial_label = ["X", "Y", "Z"]
    triaxial_l2norm_labels = [l + "_l2norm" for l in triaxial_attributes_colnames]

    l2norm = np.zeros((len(triaxial_attributes_colnames), len(df)))

    for att_idx, triaxial_attribute in enumerate(triaxial_attributes_colnames):

        for label in triaxial_label:
            l2norm[att_idx] += df[triaxial_attribute + label] ** 2

        l2norm[att_idx] = np.sqrt(l2norm[att_idx])

    l2norm_df = pd.DataFrame(l2norm.T, columns=triaxial_l2norm_labels)
    return l2norm_df
