import os

import pandas as pd

from enum import Enum
from datetime import datetime

from .util.app import App
from .device_handler import BaseDeviceHandler, DemoDeviceHandler


class SamplingMode(Enum):
    SAMPLING = 0
    DEMO = 1


class PairDataSampler:
    def __init__(
        self,
        device1_name: str,
        device2_name: str,
        device1_address: str,
        device2_address: str,
        mode: SamplingMode = SamplingMode.SAMPLING,
    ):
        self.app = App()
        self.device1_name = device1_name
        self.device2_name = device2_name
        self.device1_address = device1_address
        self.device2_address = device2_address
        self.mode = mode

        self.start_date = None

        self.device1_finished = False
        self.device2_finished = False

        if mode == SamplingMode.DEMO:
            handler = DemoDeviceHandler
        elif mode == SamplingMode.SAMPLING:
            handler = BaseDeviceHandler
        else:
            raise ValueError("Invalid sampling mode")

        self.device1_handler = handler(
            self.app,
            device1_name,
            device1_address,
            self.on_sensor_update,
            self.on_device1_terminated,
        )
        self.device2_handler = handler(
            self.app,
            device2_name,
            device2_address,
            self.on_sensor_update,
            self.on_device2_terminated,
        )

    def run(self):
        try:
            self.start_date = datetime.now()
            self.device1_handler.start()
            self.device2_handler.start()
            self.app.add_event(self._check_finished)
            self.app.run()

        except KeyboardInterrupt:
            # This is a provisional termination condition
            self.app.stop()
            print("stop sampling")

        finally:
            # Device normally terminates when each of its own termination conditions are met
            # If the device terminates abnormally, have the device follow normal termination procedures.
            if not self.device1_finished:
                self.device1_handler.stop()
            if not self.device2_finished:
                self.device2_handler.stop()

    def _check_finished(self):
        # Check if two devices are terminated
        if self.device1_finished and self.device2_finished:
            self.app.stop()
        else:
            self.app.add_event(self._check_finished)

    def on_sensor_update(
        self,
        sensor_name: str,
        time: datetime,
        acc: list[float],
        gyro: list[float],
        angle: list[float],
        mag: list[float],
    ):
        pass

    def on_device1_terminated(self, sensor_name: str):
        self.device1_finished = True

    def on_device2_terminated(self, sensor_name: str):
        self.device2_finished = True

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not (self.device1_finished and self.device2_finished):
            raise ValueError("Data sampling is not finished")

        return (
            self.device1_handler.get_sensor_data(),
            self.device2_handler.get_sensor_data(),
        )

    def output_sampling_data(
        self,
        output_dir_path: str,
        remark: str = "",
    ):
        if not (self.device1_finished and self.device2_finished):
            raise ValueError("Data sampling is not finished")

        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        formatted_date = self.start_date.strftime("%Y%m%d%H%M%S")
        device1_data_filename = f"{formatted_date}_{self.device1_name}.csv"
        device2_data_filename = f"{formatted_date}_{self.device2_name}.csv"
        device1_data_output_path = os.path.join(output_dir_path, device1_data_filename)
        device2_data_output_path = os.path.join(output_dir_path, device2_data_filename)

        device1_data = self.device1_handler.get_sensor_data()
        device2_data = self.device2_handler.get_sensor_data()
        device1_data.to_csv(device1_data_output_path, index=False)
        device2_data.to_csv(device2_data_output_path, index=False)

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

        info_text = f"{formatted_date},{self.device1_name},{device1_data_filename},{self.device1_address},{self.device2_name},{device2_data_filename},{self.device2_address},{remark}\n"
        if os.path.exists(infofile_output_path):
            with open(infofile_output_path, "a") as f:
                f.write(info_text)
        else:
            with open(infofile_output_path, "w") as f:
                f.write(",".join(column) + "\n")
                f.write(info_text)
