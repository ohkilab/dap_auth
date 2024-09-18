import os

import hydra
from omegaconf import DictConfig

from datetime import datetime

import faulthandler
import tracemalloc

from sampling.data_sampler import PairDataSampler, SamplingMode


def sampling(user1_name, user2_name, device1_address, device2_address):

    sampler = PairDataSampler(
        user1_name,
        user2_name,
        device1_address,
        device2_address,
        mode=SamplingMode.SAMPLING,
    )
    sampler.run()
    device1_data, device2_data = sampler.get_data()
    return device1_data, device2_data


def output_sampling_data(
    output_dir_path: str,
    device1_data: str,
    device2_data: str,
    cfg: DictConfig,
    start_date: datetime,
    user1_name: str,
    user2_name: str,
    remark: str,
):
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    formatted_date = start_date.strftime("%Y%m%d%H%M%S")
    device1_data_filename = f"{formatted_date}_{user1_name}.csv"
    device2_data_filename = f"{formatted_date}_{user2_name}.csv"
    device1_data_output_path = os.path.join(output_dir_path, device1_data_filename)
    device2_data_output_path = os.path.join(output_dir_path, device2_data_filename)

    device1_data.to_csv(device1_data_output_path)
    device2_data.to_csv(device2_data_output_path)

    info_filename = "sensor_data_info.csv"
    column = [
        "start_date",
        "user1_name",
        "user1_data_path",
        "user1_device_address",
        "user2_name",
        "user2_data_path",
        "user2_device_address",
        "remark",
    ]
    infofile_output_path = os.path.join(output_dir_path, info_filename)

    info_text = f"{formatted_date},{user1_name},{device1_data_filename},{cfg.devices.device1.address},{user2_name},{device2_data_filename},{cfg.devices.device2.address},{remark}\n"
    if os.path.exists(infofile_output_path):
        with open(infofile_output_path, "a") as f:
            f.write(info_text)
    else:
        with open(infofile_output_path, "w") as f:
            f.write(",".join(column) + "\n")
            f.write(info_text)


@hydra.main(version_base=None, config_path="../conf", config_name="data_sampling")
def main(cfg: DictConfig):
    print("plese input user1_name: ", end="")
    user1_name = input()
    print("plese input user2_name: ", end="")
    user2_name = input()
    print("please input remark: ", end="")
    remark_data = input()

    # black band
    device1_address = cfg.devices.device1.address
    # blue band
    device2_address = cfg.devices.device2.address

    output_dir_path = os.path.join(os.getcwd(), cfg.output_dir_path)

    start_date = datetime.now()

    device1_data, device2_data = sampling(
        user1_name,
        user2_name,
        device1_address,
        device2_address,
    )
    output_sampling_data(
        output_dir_path,
        device1_data,
        device2_data,
        cfg,
        start_date,
        user1_name,
        user2_name,
        remark_data,
    )


if __name__ == "__main__":

    tracemalloc.start()
    faulthandler.enable()

    main()

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("traceback")

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
        for line in stat.traceback.format():
            print(line)
        print("=====")
