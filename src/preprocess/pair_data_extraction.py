import pandas as pd
from datetime import datetime


def str2datetime(time: str) -> datetime:
    return datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")


def search_near_time_idx(ref_time: datetime, df: pd.DataFrame) -> int:
    near_time_idx = 0
    min_time_diff = None
    for df_idx in range(0, df.shape[0]):
        target_time = str2datetime(df["time"].iloc[df_idx])
        time_diff = abs(target_time - ref_time)
        if (min_time_diff is None) or (time_diff < min_time_diff):
            min_time_diff = time_diff
            near_time_idx = df_idx
        else:
            break
    return near_time_idx


def pair_extraction(
    device1_data: pd.DataFrame, device2_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    start_idx1 = 0
    start_idx2 = 0
    end_idx1 = device1_data.shape[0]
    end_idx2 = device2_data.shape[0]

    start_time1 = str2datetime(device1_data["time"].iloc[0,])
    start_time2 = str2datetime(device2_data["time"].iloc[0,])
    end_time1 = str2datetime(device1_data["time"].iloc[-1,])
    end_time2 = str2datetime(device2_data["time"].iloc[-1,])

    # 開始時間を遅い方のデバイスに合わせる
    if start_time1 > start_time2:
        start_idx2 = search_near_time_idx(start_time1, device2_data)
    elif start_time1 < start_time2:
        start_idx1 = search_near_time_idx(start_time2, device1_data)

    # 終了時間を早い方のデバイスに合わせる
    if end_time1 > end_time2:
        end_idx1 = search_near_time_idx(end_time2, device1_data)
    elif end_time1 < end_time2:
        end_idx2 = search_near_time_idx(end_time1, device2_data)

    extracted_device1_data = device1_data.iloc[start_idx1:end_idx1,]
    extracted_device2_data = device2_data.iloc[start_idx2:end_idx2,]

    return extracted_device1_data, extracted_device2_data
