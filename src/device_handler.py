import asyncio
from threading import Thread
import datetime

from util.app import App
from util.app_notifier import AppNotifierBase
from device_model import DeviceModel

from typing import List, Callable


class DeviceHandler(AppNotifierBase):
    def __init__(
        self,
        app: App,
        name: str,
        device_adress: str,
        on_update: Callable[
            [
                List[float],
                List[float],
                List[float],
                List[float],
            ],
            None,
        ],
        on_terminated: Callable[[], None],
    ) -> None:
        super().__init__(app)
        # TODO: Check address
        self.name = name
        self.device = DeviceModel(name, device_adress, self.updateData)
        self.on_update = on_update
        self.on_terminated = on_terminated

        self.thread = Thread(target=self._run_thread)

        # "{}-{}-{} {}:{}:{}:{}".format(year, mon, day, hour, minute, sec, mils),
        self.current_time: str = ""  # Date and time of sensor data acquisition

        # (x,y,z)
        self.current_acc: List[float] = [0, 0, 0]  # acceleration
        self.current_gyro: List[float] = [0, 0, 0]  # angular velocity
        self.current_angle: List[float] = [0, 0, 0]  # angle
        self.current_mag: List[float] = [0, 0, 0]  # magnetic

    def notify(self):
        self.on_update(
            self.name,
            self.current_time,
            self.current_acc,
            self.current_gyro,
            self.current_angle,
            self.current_mag,
        )

    def updateData(self, device: DeviceModel):
        # time_key = "time"
        # self.current_time = device.deviceData[time_key]
        acc_key = "Acc"
        gyro_key = "As"
        angle_key = "Angle"
        mag_key = "H"
        xyz_key = ["X", "Y", "Z"]
        expected_data_key = (
            [acc_key + t for t in xyz_key]
            + [gyro_key + t for t in xyz_key]
            + [angle_key + t for t in xyz_key]
            + [mag_key + t for t in xyz_key]
        )

        # NOTE: Check for missing attributes. Because some attributes are missing when data reception starts, etc.
        # NOTE: If missing occurs after reception starts, consider value completion, etc.
        if bool(set(expected_data_key) - device.deviceData.keys()):
            print("Missing data !")
            return

        self.current_time = datetime.datetime.now()

        for i in range(len(xyz_key)):
            self.current_acc[i] = device.deviceData[acc_key + xyz_key[i]]
            self.current_gyro[i] = device.deviceData[gyro_key + xyz_key[i]]
            self.current_angle[i] = device.deviceData[angle_key + xyz_key[i]]
            self.current_mag[i] = device.deviceData[mag_key + xyz_key[i]]
        self.event.set()

    def start(self):
        super().start()
        self.thread.start()

    def stop(self):
        self.device.closeDevice()
        self.on_terminated(self.name)

    def _run_thread(self):
        # データ取得は別スレッドで
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.device.openDevice())
