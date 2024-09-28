from enum import Enum

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from .page_sensor import SensorDataVisualizer
from .page_authorize_result import AuthrizeResultVisualizer


class DemoPageStat(Enum):
    SAMPLING = 0
    AUTHORIZE = 1


class DemoSite:
    def __init__(self):
        # Dashアプリケーションの作成
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.init_page()
        # レイアウトの設定、データ更新関数の登録
        self.app_layout = self._create_layout()
        self.app.layout = self.app_layout
        self.register_callbacks()

    def run(self):
        # self.app.run_server(debug=True)
        self.app.run_server()

    def init_page(self):
        self.state = DemoPageStat.SAMPLING
        self.old_state = self.state
        self.sampling_page = SensorDataVisualizer()
        self.authorize_page = AuthrizeResultVisualizer(False)

    def register_callbacks(self):
        self.sampling_page.register_callbacks(self.app)
        self.authorize_page.register_callbacks(self.app)

        @self.app.callback(
            Output("page-content", "children"), Input("page_counter", "n_intervals")
        )
        def display_page(n):
            if self.state == self.old_state:
                # 描画なし
                raise dash.exceptions.PreventUpdate

            self.old_state = self.state
            if self.state == DemoPageStat.SAMPLING:
                return self.sampling_page.get_layout()
            elif self.state == DemoPageStat.AUTHORIZE:
                return self.authorize_page.get_layout()
            else:
                return self.sampling_page.get_layout()

    def _create_layout(self):
        page_layout = self.sampling_page.get_layout()
        return html.Div(
            [
                html.Div(id="page-content", children=page_layout),
                dcc.Interval(id="page_counter", interval=10000, n_intervals=0),
            ]
        )


if __name__ == "__main__":
    visualizer = DemoSite()
    import threading
    import time

    class DummyHoge:
        def __init__(self, me):
            self.me = me

        def aaaa(self):
            self.me()

    def update_data():
        idx = 0
        while True:
            hoge = visualizer.sampling_page.device1_graph_component
            hoge.update_data([idx, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])
            idx += 1
            time.sleep(1)

    def hogehuga():
        aaa = DummyHoge(update_data)
        aaa.aaaa()

    threading.Thread(target=hogehuga).start()
    visualizer.run()
