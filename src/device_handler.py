import asyncio
from threading import Thread

from itertools import product

import numpy as np
import pandas as pd
import datetime

from typing import List, Callable

from util.app import App
from util.app_notifier import AppNotifierBase
from device_model import DeviceModel


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

        self.sensor_data_basic_labels = [
            "time",
        ]
        self.sensor_data_triaxial_labels = [
            "acc",
            "gyro",
            "mag",
            "angle",
        ]
        self.sensor_data_labels = self.sensor_data_basic_labels + [
            "".join(pair)
            for pair in product(self.sensor_data_triaxial_labels, ["X", "Y", "Z"])
        ]
        self.sensor_data = np.empty((0, len(self.sensor_data_labels)))

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

        # 現在取得したデータの更新
        for i in range(len(xyz_key)):
            self.current_acc[i] = device.deviceData[acc_key + xyz_key[i]]
            self.current_gyro[i] = device.deviceData[gyro_key + xyz_key[i]]
            self.current_angle[i] = device.deviceData[angle_key + xyz_key[i]]
            self.current_mag[i] = device.deviceData[mag_key + xyz_key[i]]

        # 時系列データへの結合
        # print(f"Sensor name: {sensor_name}, time: {time}")
        row = np.array([])
        # ラベル順の結合
        for basic_label in self.sensor_data_basic_labels:
            if basic_label == "time":
                row = np.append(row, self.current_time)
        for triaxial_label in self.sensor_data_triaxial_labels:
            if triaxial_label == "acc":
                row = np.append(row, self.current_acc)
            elif triaxial_label == "gyro":
                row = np.append(row, self.current_gyro)
            elif triaxial_label == "angle":
                row = np.append(row, self.current_angle)
            elif triaxial_label == "mag":
                row = np.append(row, self.current_mag)
        self.sensor_data = np.vstack([self.sensor_data, row])

        self.event.set()

    def get_sensor_data(self):
        df = pd.DataFrame(self.sensor_data, columns=self.sensor_data_labels)
        return df

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
