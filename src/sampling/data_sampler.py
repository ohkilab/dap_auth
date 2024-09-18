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

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (
            self.device1_handler.get_sensor_data(),
            self.device2_handler.get_sensor_data(),
        )

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
