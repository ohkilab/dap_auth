from collections import deque

import numpy as np
from dash import dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go


class DeviceComponent:
    def __init__(self, device_id: str, parent_interval_id: str):
        self.device_id = device_id
        self.interval_id = parent_interval_id
        self.is_updated = False
        # Dashアプリケーションの作成
        # 表示項目やデータの初期化
        self.init_id_data()
        self.init_view_data()
        # レイアウトの設定、データ更新関数の登録
        self.app_layout = self._create_layout()
        # self.register_callbacks()

    def get_layout(self):
        return self.app_layout

    def clear_page(self):
        self.is_updated = False
        self.x_data = deque(maxlen=self.max_data_length)
        self.graph1_data1 = deque(maxlen=self.max_data_length)
        self.graph1_data2 = deque(maxlen=self.max_data_length)
        self.graph1_data3 = deque(maxlen=self.max_data_length)
        self.graph2_data = deque(maxlen=self.max_data_length)
        for i in range(self.max_data_length):
            self.x_data.append(i)
            self.graph1_data1.append(0)
            self.graph1_data2.append(0)
            self.graph1_data3.append(0)
            self.graph2_data.append(0)
            self.x_data_idx += 1

    def init_id_data(self):
        self.device_name = f"device{self.device_id}"
        self.graph1_id = self.device_name + "-graph1"
        self.graph2_id = self.device_name + "-graph2"
        self.shared_store_id = self.device_name + "-shared-data-store"

    def init_view_data(self):

        self.max_data_length = 50
        self.x_data_idx = 0

        self.x_data = deque(maxlen=self.max_data_length)
        self.x_axis_label = "time"

        self.graph1_title = "Sensor Data"
        self.graph1_y_axis_label = "value"
        self.graph1_data1_label = "acceleration"
        self.graph1_data1 = deque(maxlen=self.max_data_length)
        self.graph1_data2_label = "angle"
        self.graph1_data2 = deque(maxlen=self.max_data_length)
        self.graph1_data3_label = "magnetic"
        self.graph1_data3 = deque(maxlen=self.max_data_length)

        self.graph2_title = "L2 norm of triaxial angular velocity"
        self.graph2_y_axis_label = "l2 norm"
        self.graph2_data = deque(maxlen=self.max_data_length)

        for i in range(self.max_data_length):
            self.x_data.append(i)
            self.graph1_data1.append(0)
            self.graph1_data2.append(0)
            self.graph1_data3.append(0)
            self.graph2_data.append(0)
            self.x_data_idx += 1

    def update_data(
        self,
        acc: list[float],
        gyro: list[float],
        angle: list[float],
        mag: list[float],
    ):
        if not self.is_updated:
            self.is_updated = True

        self.x_data.append(self.x_data_idx)
        acc_l2 = np.linalg.norm(acc)
        angle_l2 = np.linalg.norm(angle)
        mag_l2 = np.linalg.norm(mag)
        gyro_l2 = np.linalg.norm(gyro)

        self.graph1_data1.append(acc_l2)
        self.graph1_data2.append(angle_l2)
        self.graph1_data3.append(mag_l2)
        self.graph2_data.append(gyro_l2)

        self.x_data_idx += 1

    def register_callbacks(self, app):

        @app.callback(
            Output(self.shared_store_id, "data"),
            [Input(self.interval_id, "n_intervals")],
        )
        def update_data_store(n):
            return {
                "x_data": list(self.x_data),
                "graph1_data1": list(self.graph1_data1),
                "graph1_data2": list(self.graph1_data2),
                "graph1_data3": list(self.graph1_data3),
                "graph2_data": list(self.graph2_data),
            }

        @app.callback(
            Output(self.graph1_id, "figure"),
            [Input(self.shared_store_id, "data")],
        )
        def update_graph_1(data):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=data["x_data"],
                    y=data["graph1_data1"],
                    mode="lines",
                    name=self.graph1_data1_label,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=data["x_data"],
                    y=data["graph1_data2"],
                    mode="lines",
                    name=self.graph1_data2_label,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=data["x_data"],
                    y=data["graph1_data3"],
                    mode="lines",
                    name=self.graph1_data3_label,
                )
            )
            fig.update_layout(
                title=self.graph1_title,
                xaxis_title=self.x_axis_label,
                yaxis_title=self.graph1_y_axis_label,
                margin=dict(l=40, r=40, t=40, b=20),
            )
            return fig

        @app.callback(
            Output(self.graph2_id, "figure"),
            [Input(self.shared_store_id, "data")],
        )
        def update_graph_2(data):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=data["x_data"],
                    y=data["graph2_data"],
                    mode="lines",
                    name=self.graph2_title,
                )
            )
            fig.update_layout(
                title=self.graph2_title,
                xaxis_title=self.x_axis_label,
                yaxis_title=self.graph2_y_axis_label,
                margin=dict(l=40, r=40, t=40, b=20),
            )
            return fig

    def _create_layout(self):
        return (
            dcc.Graph(
                id=self.graph1_id,
                config={"responsive": True},
            ),
            dcc.Graph(
                id=self.graph2_id,
                config={"responsive": True},
            ),
            dcc.Store(id=self.shared_store_id),
        )
