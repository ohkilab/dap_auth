import hydra
from omegaconf import DictConfig

import pandas as pd

import faulthandler
import tracemalloc

from sampling.data_sampler import PairDataSampler, SamplingMode
from preprocess.pair_data_extraction import pair_extraction
from preprocess.util import removal_gravitational_acceleration


def sampling(
    user1_name, user2_name, device1_address, device2_address
) -> tuple[pd.DataFrame, pd.DataFrame]:

    sampler = PairDataSampler(
        user1_name,
        user2_name,
        device1_address,
        device2_address,
        mode=SamplingMode.DEMO,
    )
    sampler.run()
    device1_data, device2_data = sampler.get_data()
    return device1_data, device2_data


def preprocessing(
    device1_data: pd.DataFrame, device2_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    extracted_device1_data, extracted_device2_data = pair_extraction(
        device1_data, device2_data
    )
    rm_gravity_device1_data = removal_gravitational_acceleration(extracted_device1_data)
    rm_gravity_device2_data = removal_gravitational_acceleration(extracted_device2_data)
    return rm_gravity_device1_data, rm_gravity_device2_data


@hydra.main(version_base=None, config_path="../conf", config_name="data_sampling")
def main(cfg: DictConfig):
    print("plese input user1_name: ", end="")
    user1_name = input()
    print("plese input user2_name: ", end="")
    user2_name = input()
    # black band
    device1_address = cfg.devices.device1.address
    # blue band
    device2_address = cfg.devices.device2.address

    # Sampling only in the operating section
    device1_data, device2_data = sampling(
        user1_name,
        user2_name,
        device1_address,
        device2_address,
    )
    # TODO: Pair data operating section alignment
    # TODO: preprocessing ( Gravitational acceleration removal / Calculation of L2Norm )
    # TODO: Feature Extraction
    # TODO: authentication


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
