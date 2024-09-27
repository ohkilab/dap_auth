import pandas as pd

from preprocess.pair_data_extraction import pair_extraction
from feature.fusion import FusionMode, calculate_extract_fusion_futures
from preprocess.util import removal_gravitational_acceleration
from feature.extract import standardization, triaxial_attributes_l2norm


def preprocessing(
    device1_data: pd.DataFrame, device2_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    extracted_device1_data, extracted_device2_data = pair_extraction(
        device1_data, device2_data
    )
    rm_gravity_device1_data = removal_gravitational_acceleration(extracted_device1_data)
    rm_gravity_device2_data = removal_gravitational_acceleration(extracted_device2_data)
    return rm_gravity_device1_data, rm_gravity_device2_data


def feature_extraction(
    device1_data: pd.DataFrame, device2_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Calculation of mid-level features
    standard_device1_data = standardization(device1_data.drop("time", axis=1))
    standard_device2_data = standardization(device2_data.drop("time", axis=1))
    standard_device1_data.columns = device1_data.columns[1:]
    standard_device2_data.columns = device2_data.columns[1:]
    l2norm_device1_data = triaxial_attributes_l2norm(device1_data)
    l2norm_device2_data = triaxial_attributes_l2norm(device2_data)

    device1_middle_feat = pd.concat(
        [standard_device1_data, l2norm_device1_data], axis=1
    )
    device2_middle_feat = pd.concat(
        [standard_device2_data, l2norm_device2_data], axis=1
    )
    # groupbyで特徴量算出するため参照列を追加する
    # 別々に特徴量算出するためidはダミー列
    device1_middle_feat["id"] = 0
    device2_middle_feat["id"] = 1
    # Calculation of final features
    feat = calculate_extract_fusion_futures(
        device1_middle_feat, device2_middle_feat, FusionMode.FEATURE_MEAN
    )

    return feat
