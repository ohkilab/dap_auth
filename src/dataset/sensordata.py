import os
from enum import Enum
from glob import glob

import pandas as pd


class BasePairDataset:
    def __init__(self, directory_path: str, correct_pair_names: tuple[str, str]):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def _validate_pair_names(self):
        pass

    def _get_labels(self):
        pass


class PairDataDataset(BasePairDataset):
    def __init__(self, directory_path: str, correct_pair_names: tuple[str, str]):

        if os.path.exists(directory_path):
            self.directory_path = directory_path
        else:
            raise ValueError(f"Dataset path does not exist : {directory_path} ")

        self.dataset_info_path = os.path.join(
            self.directory_path, "sensor_data_info.csv"
        )

        self.dataset_info: pd.DataFrame = pd.read_csv(self.dataset_info_path)

        self.correct_pair_names = correct_pair_names
        self._validate_correct_pair_names()

    def _validate_correct_pair_names(self):
        def df_contain_name(df, name):
            return (name in df.loc[:, "user1_name"].values) or (
                name in df.loc[:, "user2_name"].values
            )

        error_msg = ""
        if not df_contain_name(self.dataset_info, self.correct_pair_names[0]):
            error_msg += self.correct_pair_names[0]
        if not df_contain_name(self.dataset_info, self.correct_pair_names[1]):
            if error_msg:
                error_msg = "(" + error_msg + ", " + self.correct_pair_names[1] + ")"
            else:
                error_msg = self.correct_pair_names[1]

        if error_msg != "":
            raise ValueError(f"User name {error_msg} not found in dataset")

    def __len__(self):
        return len(self.dataset_info)

    def _get_label(self, data_info: pd.Series) -> int:

        if (data_info["user1_name"] in self.correct_pair_names) and (
            data_info["user2_name"] in self.correct_pair_names
        ):
            return 1
        return 0

    def __getitem__(self, idx):
        if idx != 0:
            data_info = self.dataset_info.iloc[idx % len(self)]
        else:
            data_info = self.dataset_info.iloc[idx]

        user1_sensor_data_path = os.path.join(
            self.directory_path, data_info["user1_data_path"]
        )
        user2_sensor_data_path = os.path.join(
            self.directory_path, data_info["user2_data_path"]
        )

        label = self._get_label(data_info)

        user1_sensor_data = pd.read_csv(user1_sensor_data_path)
        user2_sensor_data = pd.read_csv(user2_sensor_data_path)

        return user1_sensor_data, user2_sensor_data, label, data_info
