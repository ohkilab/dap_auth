import numpy as np
import pandas as pd

from itertools import product
from datetime import datetime

from util.app import App
from device_handler import DeviceHandler


class PairDataSampler:
    def __init__(
        self,
        device1_name: str,
        device2_name: str,
        device1_address: str,
        device2_address: str,
    ):
        self.app = App()
        self.device1_name = device1_name
        self.device2_name = device2_name
        self.device1_address = device1_address
        self.device2_address = device2_address

        self.device1_finished = False
        self.device2_finished = False

        self.device1_handler = DeviceHandler(
            self.app,
            device1_name,
            device1_address,
            self.on_sensor_update,
            self.on_sensor_terminated,
        )
        self.device2_handler = DeviceHandler(
            self.app,
            device2_name,
            device2_address,
            self.on_sensor_update,
            self.on_sensor_terminated,
        )

    def run(self):
        try:
            self.device1_handler.start()
            self.device2_handler.start()
            self.app.add_event(self._check_finished)
            self.app.run()

        except KeyboardInterrupt:
            # This is a provisional termination condition
            self.app.stop()
            print("stop sampling")

        finally:
            if not self.device1_finished:
                self.device1_handler.stop()
            if not self.device2_finished:
                self.device2_handler.stop()

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (
            self.device1_handler.get_sensor_data(),
            self.device2_handler.get_sensor_data(),
        )

    def _check_finished(self):
        if self.device1_finished and self.device2_finished:
            self.app.stop()
        else:
            self.app.add_event(self._check_finished)

    # DeviceHandlerでセンサデータが更新された時にuser_dataを更新
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

    def on_sensor_terminated(self, sensor_name: str):
        pass
