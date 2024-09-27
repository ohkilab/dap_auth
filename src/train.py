import os
import yaml
import pickle

import pandas as pd
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from encapsulate_preprocess import preprocessing, feature_extraction
from sampling.device_handler import MotionSegmentDeterminator
from preprocess.pair_data_extraction import pair_extraction
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


def extract_feature(cfg: DictConfig):
    dataset = PairDataDataset(cfg.dataset_path, [cfg.correct_user1, cfg.correct_user2])

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


def extract_feature_from_old_data(cfg: DictConfig):

    dataset = MaeSoIndivisualDataset(
        cfg.dataset_path,
        [cfg.correct_user1, cfg.correct_user2],
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


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    assert (
        cfg.correct_user1 is not None
    ), "Please specify correct_user1. how to use: correct_user1=xxx"
    assert (
        cfg.correct_user2 is not None
    ), "Please specify correct_user2. how to use: correct_user2=xxx"

    output_dir_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # feat, label_list, pair_list = extract_feature(cfg)
    feat, label_list, pair_list = extract_feature_from_old_data(cfg)

    feat.to_csv(os.path.join(output_dir_path, "feat_df.csv"), index=False)
    pair_list = pd.Series(pair_list)
    pair_list.to_csv(os.path.join(output_dir_path, "pair_list.csv"), index=False)

    classifier = load_model(cfg.model.param_dict_path, cfg.model.modelname)
    classifier.fit(feat, label_list)

    with open(
        os.path.join(
            output_dir_path,
            f"{cfg.model.modelname}_{cfg.correct_user1}_and_{cfg.correct_user2}.pickle",
        ),
        "wb",
    ) as f:
        pickle.dump(classifier, f)


if __name__ == "__main__":
    main()
