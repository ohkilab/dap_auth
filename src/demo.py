import faulthandler
import tracemalloc
from datetime import datetime
from threading import Thread

import pandas as pd
import hydra
from omegaconf import DictConfig

from sampling.data_sampler import PairDataSampler, SamplingMode
from encapsulate_preprocess import preprocessing, feature_extraction
from model.load import load_model
from visualize.demo_visualizer import DemoSite, DemoPageStat
import logging

# DashのログレベルをWARNING以上のレベルで抑制
logging.getLogger("dash").setLevel(logging.WARNING)
logging.getLogger("werkzeug").setLevel(
    logging.WARNING
)  # Dash内部のHTTPサーバ（werkzeug）のログも抑制


def sampling(
    user1_name,
    user2_name,
    device1_address,
    device2_address,
    on_update=None,
    on_terminate=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    sampler = PairDataSampler(
        user1_name,
        user2_name,
        device1_address,
        device2_address,
        mode=SamplingMode.DEMO,
        on_update=on_update,
        on_terminated=on_terminate,
    )
    sampler.run()
    device1_data, device2_data = sampler.get_data()
    return device1_data, device2_data


@hydra.main(version_base=None, config_path="../conf", config_name="demo")
def main(cfg: DictConfig):

    user1_name = "demo1"
    user2_name = "demo2"

    # black band
    device1_address = cfg.devices.device1.address
    # blue band
    device2_address = cfg.devices.device2.address

    visualizer = DemoSite()
    print(id(visualizer))

    def on_device_update(
        sensor_name: str,
        time: datetime,
        acc: list[float],
        gyro: list[float],
        angle: list[float],
        mag: list[float],
    ):

        if sensor_name == user1_name:
            target_comp = visualizer.sampling_page.device1_graph_component
        elif sensor_name == user2_name:
            target_comp = visualizer.sampling_page.device2_graph_component
        else:
            raise ValueError("Invalid sensor name")

        target_comp.update_data(acc, gyro, angle, mag)

    def on_device_terminate():
        visualizer.state = DemoPageStat.AUTHORIZE
        visualizer.sampling_page.is_terminated = True

    def on_authorization_complete(result):
        visualizer.authorize_page.result = result

    sampler = PairDataSampler(
        user1_name,
        user2_name,
        device1_address,
        device2_address,
        mode=SamplingMode.DEMO,
        on_update=on_device_update,
        on_terminated=on_device_terminate,
    )

    thread = Thread(target=authorize, args=(cfg, sampler, on_authorization_complete))
    thread.start()
    visualizer.run()


def authorize(
    cfg: DictConfig, sampler: PairDataSampler, on_authorization_complete: callable
):

    sampler.run()
    device1_data, device2_data = sampler.get_data()

    preprocessed_device1_data, preprocessed_device2_data = preprocessing(
        device1_data, device2_data
    )
    feat = feature_extraction(preprocessed_device1_data, preprocessed_device2_data)

    classifier = load_model(cfg.model.param_dict_path, cfg.model.modelname)
    pred = classifier.predict_proba(feat)[0]
    other_pred, target_pred = pred

    auth_result = target_pred >= cfg.pred_threshold
    on_authorization_complete(auth_result)
    if auth_result:
        print(f"authrized!")
    else:
        print(f"unauthrized...")
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
