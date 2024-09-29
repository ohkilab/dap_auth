import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig

from preprocess.pair_data_extraction import pair_extraction
from feature.fusion import FusionMode, calculate_extract_fusion_futures
from preprocess.util import removal_gravitational_acceleration
from feature.extract import standardization, triaxial_attributes_l2norm
from dataset.sensordata import MaeSoDatasetMode, MaeSoIndivisualDataset, PairDataDataset
from sampling.device_handler import MotionSegmentDeterminator


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


def extract_feature_from_old_data(cfg: DictConfig):

    dataset = MaeSoIndivisualDataset(
        cfg.dataset_path,
        (cfg.correct_user1, cfg.correct_user2),
        MaeSoDatasetMode.NORMAL,
    )

    feat_df = pd.DataFrame()
    label_list = list()
    pair_list = list()
    for device1_data, device2_data, label, data_info in tqdm(dataset):
        # Calculation of statistical features
        device1_data, device2_data = pair_extraction(
            device1_data=device1_data, device2_data=device2_data
        )
        feat = calculate_extract_fusion_futures(
            device1_data, device2_data, FusionMode.FEATURE_MEAN
        )
        feat_df = pd.concat([feat_df, feat], axis=0)
        label_list.append(label)
        pair_list.append((data_info["user1_id"], data_info["user2_id"]))

    feat_df = feat_df.reset_index().drop("index", axis=1)
    return feat_df, label_list, pair_list


def extract_feature(cfg: DictConfig):
    dataset = PairDataDataset(cfg.dataset_path, [cfg.correct_user1, cfg.correct_user2])

    def split_sensor_data(sensor_data: pd.DataFrame) -> list[pd.DataFrame]:
        df_list = []
        segment_determinator = MotionSegmentDeterminator()
        for i in range(len(sensor_data)):
            current_gyro = sensor_data.loc[i, ["gyroX", "gyroY", "gyroZ"]]
            segment_determinator.updateData(current_gyro, i)
            if segment_determinator.finished:
                start_idx = segment_determinator.start_idx
                end_idx = segment_determinator.end_idx
                df_list.append(sensor_data.iloc[start_idx:end_idx])
                segment_determinator.clear()
        return df_list

    feat_df = pd.DataFrame()
    label_list = list()
    pair_list = list()
    for device1_data, device2_data, label, data_info in tqdm(dataset):
        # Calculation of statistical features
        device1_motion_data_list = split_sensor_data(device1_data)
        device2_motion_data_list = split_sensor_data(device2_data)

        min_length = min(len(device1_motion_data_list), len(device2_motion_data_list))
        device1_motion_data_list = device1_motion_data_list[:min_length]
        device2_motion_data_list = device2_motion_data_list[:min_length]

        for device1_motion_data, device2_motion_data in zip(
            device1_motion_data_list, device2_motion_data_list
        ):
            preprocessed_device1_data, preprocessed_device2_data = preprocessing(
                device1_motion_data, device2_motion_data
            )
            feat = feature_extraction(
                preprocessed_device1_data, preprocessed_device2_data
            )
            feat_df = pd.concat([feat_df, feat], axis=0)
            label_list.append(label)
            pair_list.append((data_info["user1_name"], data_info["user2_name"]))

    feat_df = feat_df.reset_index().drop("index", axis=1)
    return feat_df, label_list, pair_list
