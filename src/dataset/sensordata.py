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
        if idx >= len(self):
            raise IndexError("Index out of range")

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


class MaeSoDatasetMode(Enum):
    NORMAL = 0
    MASTERY = 1
    SPOOF = 2
    COLLAB_SPOOF = 3


class MaeSoIndivisualDataset(BasePairDataset):
    def __init__(
        self,
        directory_path: str,
        correct_pair_names: tuple[str, str],
        mode: MaeSoDatasetMode = MaeSoDatasetMode.NORMAL,
    ):
        self.correct_pair_names = correct_pair_names
        self.mode = mode

        self.id_file_path = os.path.join(directory_path, "id.csv")
        if not os.path.exists(self.id_file_path):
            raise ValueError(f"ID file does not exist : {self.id_file_path}")

        # 識別子とユーザー名の対応表
        # カラム名はid,name
        # idは{pair_number}_{user_number}の形式
        self.id_file = pd.read_csv(self.id_file_path)

        self._validate_correct_pair_names()

        self.user1_file_path_list = []
        self.user2_file_path_list = []
        self._generate_file_list(directory_path)
        if len(self.user1_file_path_list) != len(self.user2_file_path_list):
            raise ValueError("The number of user1 and user2 data is different")

    def _generate_file_list(self, directory_path):

        mode_directory_path = os.path.join(directory_path, self.mode.name.lower())

        pair_dir_name_list = [
            name
            for name in os.listdir(mode_directory_path)
            if os.path.isdir(os.path.join(mode_directory_path, name))
        ]

        pair_number_list = [
            item.split("_")[0]
            for item in pair_dir_name_list
            if item.split("_")[1] == "1"
        ]
        for pair_number in pair_number_list:
            user1_file_path = os.path.join(mode_directory_path, f"{pair_number}_0")
            user2_file_path = os.path.join(mode_directory_path, f"{pair_number}_1")

            if os.path.exists(user1_file_path) and os.path.exists(user2_file_path):
                user1_file_path_list = glob(user1_file_path + "/*.csv")
                user2_file_path_list = glob(user2_file_path + "/*.csv")
                file_path_list_length = min(
                    len(user1_file_path_list), len(user2_file_path_list)
                )
                user1_file_path_list = user1_file_path_list[:file_path_list_length]
                user2_file_path_list = user2_file_path_list[:file_path_list_length]

                self.user1_file_path_list += user1_file_path_list
                self.user2_file_path_list += user2_file_path_list
            else:
                raise ValueError(
                    f"Pair data file does not exist : {user1_file_path} or {user2_file_path}"
                )

    def __len__(self):
        return len(self.user1_file_path_list)

    def _validate_correct_pair_names(self):
        def df_contain_name(df, name):
            return name in df.loc[:, "id"].values

        error_msg = ""
        if not df_contain_name(self.id_file, self.correct_pair_names[0]):
            error_msg += self.correct_pair_names[0]
        if not df_contain_name(self.id_file, self.correct_pair_names[1]):
            if error_msg:
                error_msg = "(" + error_msg + ", " + self.correct_pair_names[1] + ")"
            else:
                error_msg = self.correct_pair_names[1]

        if error_msg != "":
            raise ValueError(f"User name {error_msg} not found in dataset")

    def path2id(self, path):
        return os.path.basename(os.path.dirname(path))

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range")

        if idx != 0:
            user1_file_path = self.user1_file_path_list[idx % len(self)]
            user2_file_path = self.user2_file_path_list[idx % len(self)]
        else:
            user1_file_path = self.user1_file_path_list[idx]
            user2_file_path = self.user2_file_path_list[idx]

        if os.path.basename(os.path.basename(user1_file_path)) != os.path.basename(
            os.path.basename(user2_file_path)
        ):
            raise ValueError("User data does not match")

        user1_sensor_data = pd.read_csv(user1_file_path)
        user2_sensor_data = pd.read_csv(user2_file_path)

        user1_id = self.path2id(user1_file_path)
        user2_id = self.path2id(user2_file_path)

        if user1_id.split("_")[0] != user2_id.split("_")[0]:
            raise ValueError("Pair id does not match")

        data_info = pd.Series(
            {
                "user1_id": user1_id,
                "user2_id": user2_id,
                "user1_data_path": user1_file_path,
                "user2_data_path": user2_file_path,
            }
        )
        label = self._get_label(data_info)

        return user1_sensor_data, user2_sensor_data, label, data_info

    def _get_label(self, data_info: pd.Series) -> int:
        user1_id = data_info["user1_id"]
        user2_id = data_info["user2_id"]

        if (user1_id in self.correct_pair_names) and (
            user2_id in self.correct_pair_names
        ):
            return 1
        return 0


if __name__ == "__main__":
    path = "/Users/okanoshinkuu/Workspace/lab/dev/dap_auth/dap_auth_demo/data/tmp_data"
    correct_pair_names = ("0_0", "0_1")
    dataset = MaeSoIndivisualDataset(path, correct_pair_names)
    for hoge in dataset:
        print(type(hoge))
