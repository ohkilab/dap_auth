import faulthandler
import tracemalloc

import pandas as pd
import hydra
from omegaconf import DictConfig

from sampling.data_sampler import PairDataSampler, SamplingMode
from encapsulate_preprocess import preprocessing, feature_extraction
from preprocess.pair_data_extraction import pair_extraction
from preprocess.util import removal_gravitational_acceleration
from feature.extract import standardization, triaxial_attributes_l2norm
from feature.fusion import FusionMode, calculate_extract_fusion_futures
from model.load import load_model


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


@hydra.main(version_base=None, config_path="../conf", config_name="demo")
def main(cfg: DictConfig):
    print("plese input user1_name: ", end="")
    user1_name = input()
    print("plese input user2_name: ", end="")
    user2_name = input()
    # black band
    device1_address = cfg.devices.device1.address
    # blue band
    device2_address = cfg.devices.device2.address

    device1_data, device2_data = sampling(
        user1_name,
        user2_name,
        device1_address,
        device2_address,
    )
    preprocessed_device1_data, preprocessed_device2_data = preprocessing(
        device1_data, device2_data
    )
    feat = feature_extraction(preprocessed_device1_data, preprocessed_device2_data)

    classifier = load_model(cfg.model.param_dict_path, cfg.model.modelname)
    pred = classifier.predict_proba(feat)[0]
    print(pred)


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
