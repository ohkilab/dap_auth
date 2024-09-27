import os
import yaml
import pickle

import pandas as pd
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from sampling.device_handler import MotionSegmentDeterminator
from preprocess.pair_data_extraction import pair_extraction
from preprocess.util import removal_gravitational_acceleration
from feature.extract import standardization, triaxial_attributes_l2norm
from feature.fusion import FusionMode, calculate_extract_fusion_futures
from model.load import load_model
from dataset.sensordata import PairDataDataset, MaeSoIndivisualDataset, MaeSoDatasetMode


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
    standard_device1_data = standardization(device1_data)
    standard_device2_data = standardization(device2_data)
    l2norm_device1_data = triaxial_attributes_l2norm(device1_data)
    l2norm_device2_data = triaxial_attributes_l2norm(device2_data)

    device_1_middle_feat = pd.concat(
        [standard_device1_data, l2norm_device1_data], axis=1
    )
    device2_middle_feat = pd.concat(
        [standard_device2_data, l2norm_device2_data], axis=1
    )
    # Calculation of final features
    feat = calculate_extract_fusion_futures(
        device_1_middle_feat, device2_middle_feat, FusionMode.FEATURE_MEAN
    )

    return feat


def train(cfg: DictConfig, output_dir_path: str):
    classifier = load_model(cfg.model.param_dict_path, cfg.model.modelname)
    dataset = PairDataDataset(cfg.dataset_path, [cfg.correct_user1, cfg.correct_user2])

    feat_df = pd.DataFrame()
    label_list = list()
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

    feat_df = feat_df.reset_index().drop("index", axis=1)
    feat_df.to_csv(os.path.join(output_dir_path, "feat_df.csv"), index=False)
    label_list = pd.Series(label_list)
    label_list.to_csv(os.path.join(output_dir_path, "label_list.csv"), index=False)

    classifier = load_model(cfg.model.param_dict_path, cfg.model.modelname)
    classifier.fit(feat_df, label_list)
    return classifier


def old_data_format_train(cfg: DictConfig, output_dir_path: str):

    dataset = MaeSoIndivisualDataset(
        cfg.dataset_path,
        [cfg.correct_user1, cfg.correct_user2],
        MaeSoDatasetMode.NORMAL,
    )

    feat_df = pd.DataFrame()
    label_list = list()
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

    feat_df = feat_df.reset_index().drop("index", axis=1)
    feat_df.to_csv(os.path.join(output_dir_path, "feat_df.csv"), index=False)
    label_list = pd.Series(label_list)
    label_list.to_csv(os.path.join(output_dir_path, "label_list.csv"), index=False)

    classifier = load_model(cfg.model.param_dict_path, cfg.model.modelname)
    classifier.fit(feat_df, label_list)
    return classifier


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    assert (
        cfg.correct_user1 is not None
    ), "Please specify correct_user1. how to use: correct_user1=xxx"
    assert (
        cfg.correct_user2 is not None
    ), "Please specify correct_user2. how to use: correct_user2=xxx"

    output_dir_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # model = train(cfg, output_dir_path)
    model = old_data_format_train(cfg, output_dir_path)

    with open(
        os.path.join(
            output_dir_path,
            f"{cfg.model.modelname}_{cfg.correct_user1}_and_{cfg.correct_user2}.yaml",
        ),
        "w",
    ) as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
