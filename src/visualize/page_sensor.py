from dash import dcc, html
from dash.dependencies import Input, Output

from .component.device_component import DeviceComponent


class SensorDataVisualizer:
    def __init__(self):
        self.is_terminated = False
        # 表示項目やデータの初期化
        self.init_view_data()
        self.device1_graph_component = DeviceComponent("0", self.interval_id)
        self.device2_graph_component = DeviceComponent("1", self.interval_id)
        # レイアウトの設定、データ更新関数の登録
        self.layout = self._create_layout()
        # self.register_callbacks()
        self.status_message = "sensor connecting..."

    def get_layout(self):
        return self.layout

    def init_view_data(self):
        self.title = "Sensor Data Visualizer"
        self.status_message = "initializing..."
        self.interval_id = "sensor-vis-graph-update"

    def register_callbacks(self, app):
        self.device1_graph_component.register_callbacks(app)
        self.device2_graph_component.register_callbacks(app)

        @app.callback(
            Output("status-message", "children"),
            [Input("sensor-vis-graph-update", "n_intervals")],
        )
        def update_status_message(n):
            if (
                self.device1_graph_component.is_updated
                and self.device2_graph_component.is_updated
            ):
                self.status_message = "sensor connected"
            if self.is_terminated:
                self.status_message = (
                    "data extraction for the operation section is complete"
                )
            return self.status_message

    def _create_layout(self):
        graph1_layout = self.device1_graph_component.get_layout()
        graph1_shard_store = graph1_layout[2]
        graph2_layout = self.device2_graph_component.get_layout()
        graph2_shard_store = graph2_layout[2]
        return html.Div(
            style={
                "height": "100vh",
                "padding": "20px",
                "box-sizing": "border-box",
                "overflow": "hidden",
            },
            children=[
                html.H1(
                    self.title,
                    style={
                        "text-align": "center",
                        "margin": "0",
                        "padding": "0",
                        "margin-bottom": "10px",
                    },
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "grid-template-columns": "1fr 1fr",
                        "gap": "10px",
                        "height": "80vh",
                    },
                    children=[
                        graph1_layout[0],
                        graph2_layout[0],
                        graph1_layout[1],
                        graph2_layout[1],
                    ],
                ),
                graph1_shard_store,
                graph2_shard_store,
                dcc.Interval(
                    id=self.interval_id, interval=1000, n_intervals=0
                ),  # 1秒ごとに更新
                html.Div(
                    id="status-message",
                    style={
                        "text-align": "center",
                        "margin-top": "10px",
                        "font-size": "18px",
                    },
                    children=self.status_message,
                ),
            ],
        )
