import os
import pickle

import pandas as pd
import hydra
from omegaconf import DictConfig

from model.load import load_model
from encapsulate_preprocess import extract_feature_from_old_data


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def train(cfg: DictConfig):
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
    train()
