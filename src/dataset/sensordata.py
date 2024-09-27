import os
from enum import Enum
from glob import glob
import random

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
        scenario_mode: MaeSoDatasetMode = MaeSoDatasetMode.NORMAL,
        is_train: bool = True,
    ):
        self.correct_pair_names = correct_pair_names
        self.scenario_mode = scenario_mode
        self.is_train = is_train

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

        mode_directory_path = os.path.join(
            directory_path, self.scenario_mode.name.lower()
        )
        self.train_test_idx_ref_column_name = [
            "pair_id",
            "path1",
            "path2",
            "train/test",
        ]
        self.train_test_idx_ref_file_path = os.path.join(
            mode_directory_path, "train_test_split.csv"
        )
        train_test_idx_ref = None
        if os.path.exists(self.train_test_idx_ref_file_path):
            train_test_idx_ref = pd.read_csv(self.train_test_idx_ref_file_path)

        self._generate_file_list(mode_directory_path, train_test_idx_ref)
        if len(self.user1_file_path_list) != len(self.user2_file_path_list):
            raise ValueError("The number of user1 and user2 data is different")

    def _generate_file_list(
        self, directory_path: str, train_test_split_info: pd.DataFrame = None
    ):
        generate_split_info = train_test_split_info is None

        pair_dir_name_list = [
            name
            for name in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, name))
        ]

        pair_id_list = [
            item.split("_")[0]
            for item in pair_dir_name_list
            if item.split("_")[1] == "1"
        ]

        if generate_split_info:
            train_test_split_info = pd.DataFrame(
                columns=self.train_test_idx_ref_column_name
            )

        for pair_id in pair_id_list:
            user1_file_path = os.path.join(directory_path, f"{pair_id}_0")
            user2_file_path = os.path.join(directory_path, f"{pair_id}_1")

            if not (
                os.path.exists(user1_file_path) and os.path.exists(user2_file_path)
            ):
                raise ValueError(
                    f"Pair data file does not exist : {user1_file_path} or {user2_file_path}"
                )

            user1_file_path_list = glob(user1_file_path + "/*.csv")
            user2_file_path_list = glob(user2_file_path + "/*.csv")
            file_path_list_length = min(
                len(user1_file_path_list), len(user2_file_path_list)
            )
            user1_file_path_list = user1_file_path_list[:file_path_list_length]
            user2_file_path_list = user2_file_path_list[:file_path_list_length]

            # trainとtestの分割
            if not generate_split_info:
                pair_train_test_split_info = train_test_split_info[
                    train_test_split_info.loc[:, "pair_id"] == int(pair_id)
                ]
                train_info = pair_train_test_split_info[
                    pair_train_test_split_info.loc[:, "train/test"] == "train"
                ]
                test_info = pair_train_test_split_info[
                    pair_train_test_split_info.loc[:, "train/test"] == "test"
                ]
                train_idx = [
                    i
                    for i in range(len(user1_file_path_list))
                    if os.path.relpath(user1_file_path_list[i], directory_path)
                    in train_info.loc[:, "path1"].values
                ]
                test_idx = [
                    i
                    for i in range(len(user1_file_path_list))
                    if os.path.relpath(user1_file_path_list[i], directory_path)
                    in test_info.loc[:, "path1"].values
                ]

            else:
                # 分割情報が存在しない場合、情報を作成し、分割
                train_idx, test_idx = self._generate_train_test_info(
                    [i for i in range(len(user1_file_path_list))]
                )
                for i in range(len(user1_file_path_list)):
                    train_test_split_info = pd.concat(
                        [
                            train_test_split_info,
                            pd.DataFrame(
                                {
                                    "pair_id": [pair_id],
                                    "path1": [
                                        os.path.relpath(
                                            user1_file_path_list[i], directory_path
                                        )
                                    ],
                                    "path2": [
                                        os.path.relpath(
                                            user2_file_path_list[i], directory_path
                                        )
                                    ],
                                    "train/test": [
                                        "train" if i in train_idx else "test"
                                    ],
                                }
                            ),
                        ],
                        axis=0,
                    )

            if self.is_train:
                user1_file_path_list = [user1_file_path_list[idx] for idx in train_idx]
                user2_file_path_list = [user2_file_path_list[idx] for idx in train_idx]
            else:
                user1_file_path_list = [user1_file_path_list[idx] for idx in test_idx]
                user2_file_path_list = [user2_file_path_list[idx] for idx in test_idx]

            self.user1_file_path_list += user1_file_path_list
            self.user2_file_path_list += user2_file_path_list

        train_test_split_info.to_csv(self.train_test_idx_ref_file_path, index=False)

    def _generate_train_test_info(
        self, idx_list, test_rate: float = 0.2
    ) -> pd.DataFrame:
        test_num = int(len(idx_list) * test_rate)
        tmp_idx_list = random.sample(idx_list, len(idx_list))
        train_idx_list = tmp_idx_list[test_num:]
        test_idx_list = tmp_idx_list[:test_num]
        return train_idx_list, test_idx_list

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


from datetime import datetime
from tqdm import tqdm


def str2datetime(time: str) -> datetime:
    # マイクロ秒の記述指定子%fは6桁までしか対応していないため整形する
    if "." in time:
        time_part, microsecond_part = time.split(".")
        # マイクロ秒部分が6桁を超える場合は切り捨てる
        microsecond_part = microsecond_part[:6].ljust(6, "0")  # 6桁に調整
        time = f"{time_part}.{microsecond_part}"
    return datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")


if __name__ == "__main__":
    path = "/Users/okanoshinkuu/Workspace/lab/dev/dap_auth/dap_auth_demo/data/maeda_sensor_data"
    correct_pair_names = ("0_0", "0_1")
    dataset = MaeSoIndivisualDataset(path, correct_pair_names)
    for data1, data2, _, data_info in tqdm(dataset):
        pass

        # for i in range(len(data1)):
        #     try:
        #         str2datetime(data1["time"].iloc[i])
        #     except Exception as e:
        #         print(e)
        #         print(f"data1 path : {data_info['user1_data_path']}")
        #         print(f"data1 idx : {i}")
        # for i in range(len(data2)):
        #     try:
        #         str2datetime(data2["time"].iloc[i])
        #     except Exception as e:
        #         print(e)
        #         print(f"data1 path : {data_info['user2_data_path']}")
        #         print(f"data1 idx : {i}")
