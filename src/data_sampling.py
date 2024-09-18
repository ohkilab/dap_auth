import os

import hydra
from omegaconf import DictConfig

import faulthandler
import tracemalloc

from sampling.data_sampler import PairDataSampler, SamplingMode


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

    sampler = PairDataSampler(
        user1_name,
        user2_name,
        device1_address,
        device2_address,
        mode=SamplingMode.SAMPLING,
    )
    sampler.run()
    sampler.output_sampling_data(output_dir_path, remark_data)


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
